"""Paired post-training diagnostics for native MiniMind-MoE V100 pilot exports."""
import argparse
import csv
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from dataset.lm_dataset import PretrainDataset
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.quant_linear import install_quantization
from model.quant_policy import MIXED_SCHEMES, MODEL_SCHEMES, projection_policy, scheme_artifact_name
from scripts.moe_diagnostic_kl import compare_forwards, weight_kl_rows
from scripts.moe_diagnostic_ops import OperatorProbe


def file_sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint(args, scheme):
    tag = scheme_artifact_name(scheme)
    name = f"pilot32k_s{args.seed}_{tag}_{args.hidden_size}_moe.pth"
    searched = [root / "out" / name for root in (args.root, args.baseline_root)]
    for path in searched:
        if path.is_file():
            return path
    raise FileNotFoundError("Missing pilot weights: " + " or ".join(map(str, searched)))


def load_model(path, config, device):
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    model = MiniMindForCausalLM(config)
    model.load_state_dict(state, strict=True)
    return install_quantization(model.float(), "bf16").to(device), state


def save_table(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as dest:
        writer = csv.DictWriter(dest, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def probe_operators(model, scheme, loader, args):
    install_quantization(model, "bf16").train()
    probe = OperatorProbe(model, scheme, args.layers, max_elements=args.max_elements, seed=args.sample_seed)
    with probe:
        for ids, labels in loader:
            model.zero_grad(set_to_none=True)
            if not labels[:, 1:].ne(-100).any():
                raise ValueError("No prediction targets in operator microbatch")
            out = model(ids.to(args.device), labels=labels.to(args.device), use_cache=False)
            ((out.loss + out.aux_loss) / args.accumulation_steps).backward()
            del out
    model.zero_grad(set_to_none=True)
    model.eval()
    return probe


def diagnose(args):
    schemes = list(dict.fromkeys(s for item in args.schemes for s in (MIXED_SCHEMES if item == "grid" else (item,))))
    if args.layers is None:
        args.layers = list(range(args.num_hidden_layers))
    if not args.layers or min(args.layers) < 0 or max(args.layers) >= args.num_hidden_layers:
        raise ValueError("Invalid --layers for this checkpoint architecture")
    if min(args.documents, args.operator_documents, args.max_elements, args.batch_size,
           args.accumulation_steps, args.token_chunk) < 1 or args.seq_len < 2:
        raise ValueError("Counts must be positive and --seq-len must be >= 2")
    torch.manual_seed(args.sample_seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    paths = {s: checkpoint(args, s) for s in dict.fromkeys(["bf16", *schemes])}
    split = args.root / "runs/v100-pilot/data/valid.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    data = PretrainDataset(str(split), tokenizer, max_length=args.seq_len)
    if not len(data):
        raise ValueError("Empty validation split")
    # Keep this order stable across schemes, and use its prefix in the backward probe.
    ids = list(range(len(data)))
    random.Random(args.sample_seed).shuffle(ids)
    ids = ids[:args.documents]
    op_ids = ids[:args.operator_documents]
    loader = DataLoader(Subset(data, ids), batch_size=args.batch_size)
    op_loader = DataLoader(Subset(data, op_ids), batch_size=args.batch_size)
    if args.output is None:
        parent = args.root / "runs/v100-pilot/diagnostics"
        parent.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(prefix=f"moe_s{args.seed}_", dir=parent))
    else:
        folder = args.output.resolve()
        folder.mkdir(parents=True, exist_ok=True)
        if any(folder.iterdir()):
            raise FileExistsError(f"Choose an empty output directory: {folder}")
    print(f"Output: {folder.resolve()}", flush=True)
    manifest = dict(status="running", train_seed=args.seed, sample_seed=args.sample_seed,
                    valid_file=str(split.resolve()), valid_sha256=file_sha(split), document_ids=ids,
                    operator_document_ids=op_ids, batch_size=args.batch_size, seq_len=args.seq_len,
                    hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
                    operator_anchor=args.operator_anchor, layers=args.layers,
                    max_elements=args.max_elements, token_chunk=args.token_chunk,
                    accumulation_steps=args.accumulation_steps, torch=str(torch.__version__),
                    device=str(args.device), route_lock=args.route_lock,
                    policies={s: projection_policy(s) for s in schemes},
                    checkpoints={s: {"path": str(p.resolve()), "sha256": file_sha(p)} for s, p in paths.items()})
    manifest_path = folder / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    config = MiniMindConfig(hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers, use_moe=True)
    ref, ref_state = load_model(paths["bf16"], config, args.device)
    summary = []
    for scheme in schemes:
        print(f"Diagnosing {scheme}", flush=True)
        candidate, state = load_model(paths[scheme], config, args.device)
        case = folder / scheme
        case.mkdir()
        save_table(case / "weight_hist_kl.csv", weight_kl_rows(ref_state, state, layers=args.layers))
        del state
        anchor = ref if args.operator_anchor == "bf16" else candidate
        probe = probe_operators(anchor, scheme, op_loader, args)
        save_table(case / "operators.csv", probe.rows())
        save_table(case / "components.csv", probe.component_rows())
        del probe
        comparisons = [
            ("same_weights", None, scheme),
            ("trained_weights", ref, "bf16"),
            ("end_to_end", ref, scheme),
        ]
        if args.route_lock:
            comparisons.append(("same_weights_fixed_routes", None, scheme))
        for kind, reference, forward in comparisons:
            print(f"  {kind}", flush=True)
            options = {"lock_routes": True} if kind == "same_weights_fixed_routes" else {}
            data_rows = compare_forwards(candidate, loader, forward, reference_model=reference,
                                         token_chunk=args.token_chunk, **options)
            summary.append({"scheme": scheme, "comparison": kind, **data_rows["summary"]})
            for pos, row in enumerate(data_rows["documents"]):
                row["validation_index"] = ids[pos]
            for table in ("documents", "routing", "expert_loads", "margin_bins"):
                save_table(case / f"{kind}_{table}.csv", data_rows[table])
        del anchor, candidate
        save_table(folder / "summary.csv", summary)
    manifest["status"] = "complete"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Report: {folder / 'summary.csv'}", flush=True)
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO, help="Pilot experiment worktree")
    parser.add_argument("--baseline-root", type=Path, default=Path("/home/lz/minimind"))
    parser.add_argument("--tokenizer", type=Path, default=REPO / "model")
    parser.add_argument("--schemes", nargs="+", choices=(*MODEL_SCHEMES, "grid"),
                        help="Pilot formats; grid expands to the 21 mixed MoE schemes",
                        default=["mxfp8", "hif4_thr_3mm", "nvfp4_3mm", "nvint4_3mm",
                                 "hif4_thr_3mm_ffn_gate_down_mxfp8"])
    parser.add_argument("--seed", type=int, default=42, help="Training seed used in pilot weight filenames")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--documents", type=int, default=16, help="Number of validation documents for paired KL/CE")
    parser.add_argument("--operator-documents", type=int, default=4, help="Prefix of the same sample for backward probes")
    parser.add_argument("--operator-anchor", choices=("bf16", "candidate"), default="bf16",
                        help="Source of frozen weights for both sides of the operator replay")
    parser.add_argument("--layers", type=int, nargs="+", help="Zero-based operator/weight diagnostic layers (default all)")
    parser.add_argument("--max-elements", type=int, default=4096, help="Max paired operator output coordinates per call")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--accumulation-steps", type=int, default=4, help="Loss divisor for each operator microbatch")
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--num-hidden-layers", type=int, default=8)
    parser.add_argument("--token-chunk", type=int, default=64, help="Positions per full-vocabulary KL reduction")
    parser.add_argument("--route-lock", action="store_true", help="Also replay with BF16 top-1 expert assignments")
    parser.add_argument("--output", type=Path, help="Empty report directory (default: a new directory per invocation)")
    diagnose(parser.parse_args())


if __name__ == "__main__":
    main()
