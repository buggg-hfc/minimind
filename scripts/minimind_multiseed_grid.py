"""Train MiniMind-3 Dense and MoE and validate each model in its training format."""
import argparse
import csv
import fcntl
from pathlib import Path

from minimind_grid_reuse import previous_moe_scores
from minimind_grid_train import ARCHITECTURES, SCHEMES, prepare, score_arm
from minimind_moe_paired_seeds import DEFAULT_DATA, DEFAULT_OUTPUT as PAIRED_OUTPUT, record_pairs


def run(args):
    seeds = args.seeds
    if not seeds or len(set(seeds)) != len(seeds) or any(not 0 <= s <= 2**32 - 8 for s in seeds):
        raise ValueError("种子必须互异，并为八个 rank 的 seed+rank 留出 uint32 范围")
    output = args.output.expanduser().resolve()
    previous = args.previous_paired.expanduser().resolve() if args.previous_paired else None
    if previous and (previous == output or previous in output.parents or output in previous.parents):
        raise ValueError("本次输出和历史配对任务请使用两个独立目录")
    output.mkdir(parents=True, exist_ok=True)
    report = output / "same_format.csv"
    with (output / ".grid.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"已有训练任务在使用 {output}") from exc
        work = prepare(output, args.data.expanduser().resolve(), seeds)
        cases = [(seed, arch, scheme) for seed in seeds for arch in ARCHITECTURES for scheme in SCHEMES]
        # Keep already published rows visible while revalidating them after a restart.
        rows = {}
        if report.is_file():
            with report.open(encoding="utf-8-sig", newline="") as src:
                for row in csv.DictReader(src):
                    rows[int(row["seed"]), row["architecture"], row["train_scheme"]] = row
        with previous_moe_scores(previous, work, seeds) as cached:
            import torch
            if torch.cuda.device_count() != 8 or any("V100" not in torch.cuda.get_device_name(i) for i in range(8)):
                raise RuntimeError("本次预训练需要 8 张可见 V100")
            for i, case in enumerate(cases, 1):
                seed, arch, scheme = case
                reuse = arch == "moe" and (seed, scheme) in cached
                print(f"[{i}/{len(cases)}] seed={seed} 模型={arch} 训推格式={scheme} "
                      f"{'复用已验证的历史配对结果' if reuse else '运行或恢复'}", flush=True)
                if reuse:
                    ce, ppl = cached[seed, scheme]
                    source = previous / "work"
                else:
                    ce, ppl = score_arm(work, arch, seed, scheme)
                    source = work
                rows[case] = dict(seed=seed, architecture=arch, train_scheme=scheme, eval_scheme=scheme,
                                  optimizer_updates=512, valid_tokens=236442, mean_ce=ce, ppl=ppl,
                                  source_work=str(source))
                record_pairs(report, [rows[key] for key in cases if key in rows])
                print(f"  CE={ce:.6f}  PPL={ppl:.4f}", flush=True)
        print(f"GRID_READY {report}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=[43, 44, 45, 46])
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path("/data/lz/minimind-five-format-grid"))
    parser.add_argument("--previous-paired", type=Path, default=PAIRED_OUTPUT,
                        help="上一次 MoE 双格式配对训练的目录，用于等待并复用已完成的相同训练")
    run(parser.parse_args())
