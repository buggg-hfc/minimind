"""Prompt-parallel generation for the native eval_llm entry point."""

import json
import os
import tempfile
import time
from contextlib import contextmanager
from itertools import islice
from pathlib import Path

import torch
import torch.distributed as dist


def read_prompts(source):
    for line_no, line in enumerate(source, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"invalid JSON on input line {line_no}: {exc}") from exc
        if not isinstance(record, dict) or not isinstance(record.get("prompt"), str):
            raise ValueError(f"input line {line_no} requires a string 'prompt'")
        if "response" in record or "generated_tokens" in record:
            raise ValueError(f"input line {line_no} contains a reserved output field")
        yield record


@contextmanager
def ordered_output(path, rank):
    if rank:
        yield None
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, mode="w", encoding="utf-8", delete=False) as sink:
        temp = Path(sink.name)
        try:
            yield sink
            sink.close()
            os.replace(temp, path)
        finally:
            temp.unlink(missing_ok=True)


@torch.inference_mode()
def generate_batch(records, model, tokenizer, args):
    texts = []
    for record in records:
        prompt = record["prompt"]
        if "pretrain" in args.weight:
            text = (tokenizer.bos_token or "") + prompt
        else:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, open_thinking=bool(args.open_thinking),
            )
        texts.append(text)
    inputs = tokenizer(texts, padding=True, return_tensors="pt")
    width = inputs.input_ids.shape[1]
    if not inputs.attention_mask.bool().any(1).all():
        raise ValueError("a prompt produced zero input tokens")
    context = getattr(model.config, "max_position_embeddings", None)
    if context and width + args.max_new_tokens - 1 > context:
        raise ValueError(f"padded prompt length {width} and --max_new_tokens {args.max_new_tokens} exceed context {context}")
    inputs = inputs.to(args.device)
    sequences = model.generate(
        inputs=inputs.input_ids, attention_mask=inputs.attention_mask, use_cache=True,
        do_sample=False, max_new_tokens=args.max_new_tokens,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    results = []
    for ids in sequences[:, width:].tolist():
        if tokenizer.eos_token_id in ids:
            ids = ids[:ids.index(tokenizer.eos_token_id) + 1]
        results.append((tokenizer.decode(ids, skip_special_tokens=True), len(ids)))
    return results


def run_batches(args, model, tokenizer, rank, world):
    source_path, target = Path(args.prompts_jsonl), Path(args.output_jsonl)
    if source_path.resolve() == target.resolve():
        raise ValueError("input and output must be different files")
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    started = time.monotonic()
    tokens = requests = 0
    with source_path.open(encoding="utf-8") as source, ordered_output(target, rank) as sink:
        records = read_prompts(source)
        # Several local batches per gather amortize coordination while bounding memory.
        while window := list(islice(records, world * args.batch_size * 8)):
            local = window[rank::world]
            answers = []
            for start in range(0, len(local), args.batch_size):
                answers.extend(generate_batch(local[start:start + args.batch_size], model, tokenizer, args))
            gathered = [None] * world if rank == 0 else None
            if world > 1:
                dist.gather_object(answers, gathered, dst=0)
            else:
                gathered = [answers]
            if rank == 0:
                for i, record in enumerate(window):
                    response, count = gathered[i % world][i // world]
                    sink.write(json.dumps(dict(record, response=response, generated_tokens=count), ensure_ascii=False) + "\n")
                    tokens += count
                requests += len(window)
    if rank == 0:
        elapsed = time.monotonic() - started
        print(f"{requests} prompts, {tokens} generated tokens in {elapsed:.2f}s ({tokens / elapsed:.1f} generated tokens/s)")


def run_offline(args, load_model):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and torch.device(args.device).type == "cuda":
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
    if world > 1:
        # Gloo transports decoded results on CPU. Generation requires no GPU collectives.
        dist.init_process_group("gloo")
    try:
        rank = dist.get_rank() if world > 1 else 0
        model, tokenizer = load_model(args)
        run_batches(args, model, tokenizer, rank, world)
    finally:
        if world > 1:
            dist.destroy_process_group()
