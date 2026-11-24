"""MiniMind MoE: compare early versus late MXFP8 in the expert down projection."""
import argparse
import csv
import fcntl
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil

from early_prefix import make_anchor
from early_halves import finish_pair
from early_results import write_four_cells


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for data in iter(lambda: f.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_history(base, torch):
    original = base / "hif4-prefix-1024"
    old = base / "pair-1024-to-1424"
    final = base / "pair-1024-to-2048"
    old_spec = read_json(old / "experiment.json")
    cont = read_json(final / "continuation.json")
    snapshot = read_json(original / "inputs.json")
    if (cont["source_manifest_sha256"] != digest(old / "experiment.json")
            or cont["source_summary_sha256"] != digest(old / "summary.csv")
            or cont["source_dir"] != str(old) or cont["target_update"] != 256
            or old_spec["prefix"]["input_manifest_sha256"] != digest(original / "inputs.json")
            or old_spec["prefix"]["path"] != str(original)):
        raise ValueError("原 HiF4 前缀、100 步分支及其续训记录不属于同一套实验")
    environment = {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                   "datasets": version("datasets"), "transformers": version("transformers")}
    if old_spec["environment"] != environment:
        raise ValueError(f"请使用原训练环境：{old_spec['environment']}")
    train_hash = digest(original / "data/train.jsonl")
    valid_hash = digest(old / "valid.jsonl")
    if (train_hash != old_spec["prefix"]["train_sha256"]
            or train_hash != snapshot["files"]["data/train.jsonl"]
            or valid_hash != old_spec["valid_sha256"] or valid_hash != cont["valid_sha256"]):
        raise ValueError("实际数据与原实验的数据指纹不匹配")
    with (final / "summary.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if ([int(r["relative_update"]) for r in rows] != list(range(257))
            or int(rows[-1]["optimizer_update"]) != 512 or int(rows[-1]["microbatch"]) != 2048):
        raise ValueError("原两条分支尚未完成 512 次优化器更新")
    for i, row in enumerate(rows):
        if i % 20 == 0 or i == 256:
            if int(row["eval_valid_tokens"]) != 236442 or not all(
                    math.isfinite(float(row[k])) for k in ("eval_ce_a", "eval_ce_b")):
                raise ValueError(f"原分支第 {i} 次更新缺少完整验证")
    print(f"历史 2 组已完成；train.sha256={train_hash}，valid.sha256={valid_hash}", flush=True)
    return original, old, final


def check_batch_order(previous, current, *, updates=256, world=8):
    # Each pair summary already checked both arms against each other. A comparison
    # between the A arms thus verifies batch identity across all four trajectories.
    import numpy as np
    for rank in range(world):
        for step in range(1, updates + 1):
            rel = Path("hif4", "metrics", f"rank-{rank:02d}", f"update-{step:06d}.npz")
            with np.load(previous / rel, allow_pickle=False) as a, np.load(current / rel, allow_pickle=False) as b:
                if a["input_sha256"].item() != b["input_sha256"].item():
                    raise ValueError(f"四组间的训练输入不一致，rank {rank}，分叉后更新 {step}")
    print(f"已配对核实四组后半程 {updates} 次更新 × {world} 个 rank 的输入及标签", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path(os.environ.get("MINIMIND_FORK_ROOT", "/data/lz/minimind-moe-fork-s42")),
                        help="已有 HiF4 公共前缀及两次配对实验的根目录")
    parser.add_argument("--check", action="store_true", help="只核对现有实验、环境和数据")
    args = parser.parse_args()
    import torch
    root = args.root.expanduser().resolve()
    original, old, final = check_history(root, torch)
    if args.check:
        return
    if torch.cuda.device_count() != 8:
        raise SystemExit("请在 8 张 V100 全部可见时执行")
    output = root / "reciprocal-down-s42"
    output.mkdir(exist_ok=True)
    with (output / ".experiment.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("本实验正在运行")
        print(f"输出目录 {output}，当前可用 {shutil.disk_usage(output).free / 2**30:.1f} GiB", flush=True)
        anchor = output / "mixed-prefix-1024"
        pair = output / "pair-1024-to-2048"
        make_anchor(original, anchor, old / "scripts")
        report = finish_pair(original, anchor, old, pair, old / "valid.jsonl")
        check_batch_order(final, pair)
        cells = write_four_cells(final / "summary.csv", report, output / "four_cells.csv")
        print(f"CROSSOVER_READY {cells}", flush=True)


if __name__ == "__main__":
    main()
