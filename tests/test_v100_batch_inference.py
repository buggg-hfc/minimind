import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
from model.quant_linear import install_quantization
from scripts.batch_inference import run_batches


class BatchInferenceTest(unittest.TestCase):
    def test_left_padded_prefill_and_decode_use_unpadded_rope_positions(self):
        torch.manual_seed(41)
        config = MiniMindConfig(hidden_size=32, num_hidden_layers=1, vocab_size=128,
                                num_attention_heads=4, num_key_value_heads=2, intermediate_size=64,
                                max_position_embeddings=32)
        model = install_quantization(MiniMindForCausalLM(config).eval(), "hif4_thr_3mm")
        padded = torch.tensor([[0, 0, 4, 5], [1, 2, 3, 6]])
        seen = []
        handle = model.lm_head.register_forward_hook(lambda module, inputs, logits: seen.append(logits.detach().clone()))
        try:
            for cached in (False, True):
                options = dict(max_new_tokens=3, do_sample=False, use_cache=cached, eos_token_id=None)
                seen.clear()
                model.generate(padded, padded.ne(0), **options)
                batched = list(seen)
                seen.clear()
                model.generate(padded[:1, 2:], **options)
                self.assertEqual(len(seen), 3)
                for one, both in zip(seen, batched):
                    self.assertEqual(both.shape, (2, 1, 128))
                    torch.testing.assert_close(one, both[:1], rtol=0, atol=0)
        finally:
            handle.remove()

    def test_strided_workers_preserve_order_and_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "input.jsonl")
            target = Path(directory, "output.jsonl")
            opts = SimpleNamespace(prompts_jsonl=source, output_jsonl=target, batch_size=2)
            world = 3
            for size in (0, 2, 53):  # empty, fewer than ranks, and a partial final gather window
                with self.subTest(size=size):
                    rows = [{"id": i, "prompt": f"测试 {i % 7}"} for i in range(size)]
                    source.write_text("".join(json.dumps(x) + "\n" for x in rows), encoding="utf-8")
                    gathered = {}
                    seen = []
                    for rank in (2, 1, 0):
                        window = 0

                        def generate(records, *_):
                            seen.extend(row['id'] for row in records)
                            return [(row["prompt"] + "!", 1) for row in records]

                        def gather(answers, output, dst):
                            nonlocal window
                            gathered[rank, window] = answers
                            if output is not None:
                                output[:] = [gathered[r, window] for r in range(world)]
                            window += 1

                        with patch("scripts.batch_inference.generate_batch", generate), patch(
                            "scripts.batch_inference.dist.gather_object", gather
                        ):
                            run_batches(opts, None, SimpleNamespace(pad_token_id=0), rank, world)
                    written = [json.loads(row) for row in target.read_text(encoding="utf-8").splitlines()]
                    self.assertEqual(sorted(seen), list(range(size)))
                    self.assertEqual(written, [dict(x, response=x["prompt"] + "!", generated_tokens=1) for x in rows])

    def test_bad_input_preserves_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory, "input.jsonl")
            target = Path(directory, "output.jsonl")
            target.write_text("previous result", encoding="utf-8")
            opts = SimpleNamespace(prompts_jsonl=source, output_jsonl=target, batch_size=1)
            # The malformed record follows a whole successfully generated window.
            source.write_text('{"prompt":"ok"}\n' * 8 + '{"id":1}\n', encoding="utf-8")
            with patch("scripts.batch_inference.generate_batch", return_value=[("ok", 1)]):
                with self.assertRaisesRegex(ValueError, "input line 9"):
                    run_batches(opts, None, SimpleNamespace(pad_token_id=0), 0, 1)
            self.assertEqual(target.read_text(encoding="utf-8"), "previous result")
            self.assertCountEqual(Path(directory).iterdir(), [source, target])


if __name__ == "__main__":
    unittest.main()
