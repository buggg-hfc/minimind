"""Run the paired seed-42 MoE warmup experiment at one or both base learning rates."""
import argparse
import csv
import fcntl
from importlib.metadata import version
import math
import os
from pathlib import Path
import sys

from minimind_moe_noise_floor import ENDPOINTS, inputs, prepare_arm
from minimind_moe_paired_seeds import DEFAULT_DATA, digest, record_pairs


REPO = Path(__file__).resolve().parents[1]
BASE_LR, LOW_LR = 5e-4, 1e-4
LR_DIRS = {BASE_LR: "lr5e-4", LOW_LR: "lr1e-4"}
FP4_SCHEMES = ("hif4_thr_3mm", "nvint4_3mm", "nvfp4_3mm")
SCHEMES = ("mxfp8", *FP4_SCHEMES)
DEFAULT_REFERENCE = Path("/data/lz/minimind-moe-noise-s42/noise_runs.csv")


def default_schemes(lr):
    return FP4_SCHEMES if lr == BASE_LR else SCHEMES


def endpoint_ce(row):
    ce, ppl = float(row["mean_ce"]), float(row["ppl"])
    if (row["eval_scheme"] != row["train_scheme"] or int(row["optimizer_updates"]) != 512
            or int(row["valid_tokens"]) != 236442 or not math.isfinite(ce) or ce < 0
            or not math.isfinite(ppl) or ppl < 1 or not math.isclose(math.log(ppl), ce, abs_tol=1e-9)):
        raise ValueError(f"Invalid same-format validation endpoint for {row['train_scheme']}")
    return ce


def historical_ce(path):
    scores = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            scheme = row["train_scheme"]
            if (row["seed"], row["architecture"]) == ("42", "moe") and scheme in SCHEMES:
                if scheme in scores:
                    raise ValueError(f"Duplicate historical {scheme} in {path}")
                scores[scheme] = endpoint_ce(row)
    if set(scores) != set(SCHEMES):
        raise ValueError(f"Missing historical controls in {path}")
    return scores


def mxfp8_warmup_ce(path):
    if not path.is_file():
        return None
    with path.open(encoding="utf-8-sig", newline="") as stream:
        values = [endpoint_ce(r) for r in csv.DictReader(stream)
                  if (r["seed"], r["architecture"], r["train_scheme"], r["variant"])
                  == ("42", "moe", "mxfp8", "warmup26")]
    if len(values) > 1:
        raise ValueError(f"Duplicate seed-42 MXFP8 warmup26 result in {path}")
    return values[0] if values else None


def comparison_rows(scores, historical, reference, lr=BASE_LR):
    control = reference if lr == BASE_LR else None
    if "mxfp8" in scores:
        control = scores["mxfp8"][0]
    rows = []
    for scheme, (ce, ppl) in scores.items():
        original = historical[scheme] if lr == BASE_LR else None
        original_gap = original - historical["mxfp8"] if original is not None else None
        gap = ce - control if control is not None else None
        rows.append(dict(seed=42, architecture="moe", train_scheme=scheme, eval_scheme=scheme,
                         base_lr=lr, warmup_updates=26, optimizer_updates=512, valid_tokens=236442,
                         mean_ce=ce, ppl=ppl, mxfp8_warmup_ce=control, gap_vs_mxfp8_warmup=gap,
                         no_warmup_ce=original,
                         warmup_minus_no_warmup_ce=ce-original if original is not None else None,
                         gap_vs_mxfp8_no_warmup=original_gap,
                         change_in_gap=gap-original_gap if gap is not None and original_gap is not None else None))
    return rows


def save_report(path, measured, historical, reference):
    rows = []
    high = measured[BASE_LR].copy()
    if reference is not None:
        high.setdefault("mxfp8", (reference, math.exp(reference)))
    for lr, scores in ((BASE_LR, high), (LOW_LR, measured[LOW_LR])):
        ordered = {s: scores[s] for s in SCHEMES if s in scores}
        for row in comparison_rows(ordered, historical, reference, lr):
            s = row["train_scheme"]
            row["low_lr_minus_base_lr_warmup_ce"] = (
                row["mean_ce"] - high[s][0] if lr == LOW_LR and s in high else None)
            rows.append(row)
    if rows:
        record_pairs(path, rows)


def run(args):
    import torch
    sys.path.insert(0, str(REPO))
    from minimind_fp4_warmup_arm import cached_score, run_arm

    if len(args.lr) != len(set(args.lr)) or args.schemes and len(args.schemes) != len(set(args.schemes)):
        raise ValueError("Duplicate learning rates or schemes")
    if torch.cuda.device_count() != 8 or any("V100" not in torch.cuda.get_device_name(i) for i in range(8)):
        raise RuntimeError("Expected the eight V100 GPUs used by the seed-42 pilot")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".warmup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        paths, hashes = inputs(args.data.expanduser().resolve())
        historical = historical_ce(REPO / ENDPOINTS)
        reference = mxfp8_warmup_ce(args.noise_runs.expanduser())
        if reference is None and BASE_LR in args.lr:
            print("未找到原 MXFP8 warmup26 对照。将保存绝对 CE；可用 --noise-runs 指定旧 noise_runs.csv。", flush=True)
        os.environ.setdefault("HF_DATASETS_CACHE", str(output / "hf-datasets-cache"))
        scripts = ("minimind_moe_fp4_warmup.py", "minimind_fp4_warmup_arm.py",
                   "minimind_moe_noise_floor.py", "minimind_moe_paired_seeds.py", "minimind_grid_train.py")
        plan = dict(seed=42, architecture="moe", world_size=8, seq_len=512, batch_per_device=2,
                    accumulation_steps=4, microbatches=2048, optimizer_updates=512,
                    warmup_updates=26, variant="warmup26", files=hashes,
                    controllers={n: digest(REPO / "scripts" / n) for n in scripts},
                    versions={n: version(n) for n in ("torch", "transformers", "datasets")})
        work = {lr: output / name / "work" for lr, name in LR_DIRS.items()}
        measured = {lr: {} for lr in LR_DIRS}
        for lr, root in work.items():
            if lr not in args.lr and not root.is_dir():
                continue
            prepare_arm(root, paths, dict(plan, base_lr=lr))
            for scheme in SCHEMES:
                result = cached_score(root, scheme, lr)
                if result is not None:
                    measured[lr][scheme] = result
        report = output / "warmup_comparison.csv"
        for lr in args.lr:
            for scheme in args.schemes or default_schemes(lr):
                if scheme not in measured[lr]:
                    print(f"seed=42 MoE {scheme} lr={lr:g} warmup=26 -> {work[lr]}", flush=True)
                    measured[lr][scheme] = run_arm(work[lr], scheme, lr)
                ce, ppl = measured[lr][scheme]
                save_report(report, measured, historical, reference)
                print(f"{scheme} lr={lr:g}: valid CE={ce:.8f}, PPL={ppl:.6f}", flush=True)
        print(f"WARMUP_READY {report}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lr", nargs="+", type=float, choices=tuple(LR_DIRS), default=[BASE_LR])
    parser.add_argument("--schemes", nargs="+", choices=SCHEMES,
                        help="Default: three 4-bit formats at 5e-4; add matched MXFP8 at 1e-4")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path("/data/lz/minimind-moe-4bit-warmup-s42"))
    parser.add_argument("--noise-runs", type=Path, default=DEFAULT_REFERENCE,
                        help="Previous seed-42 noise_runs.csv for the 5e-4 MXFP8 warmup26 reference")
    run(parser.parse_args())
