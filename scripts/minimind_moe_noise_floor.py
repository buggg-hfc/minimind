"""Measure seed-42 MoE sensitivity with matched BF16 and MXFP8 perturbation runs."""
import argparse
import csv
import fcntl
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile

from minimind_moe_paired_seeds import DEFAULT_DATA, SPLITS, digest, record_pairs


REPO = Path(__file__).resolve().parents[1]
SCHEMES = ("bf16", "mxfp8")
DEFAULT_SCENARIOS = ("repeat", "lr_x1p0001", "lr_x0p9999")
PURE_PERTURBATIONS = (*DEFAULT_SCENARIOS[1:], "init_1ulp", "swap12")
WARMUP_CONTROLS = {"warmup26": "repeat", "warmup26_lr_x1p0001": "lr_x1p0001",
                   "warmup26_lr_x0p9999": "lr_x0p9999"}
SCENARIOS = ("repeat", *PURE_PERTURBATIONS, *WARMUP_CONTROLS)
SEED = 42
ENDPOINTS = "experiments/five-format-multiseed/same_format_validation.csv"


def historical_controls(path):
    controls = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            scheme = row["train_scheme"]
            if (int(row["seed"]), row["architecture"]) != (SEED, "moe") or scheme not in SCHEMES:
                continue
            ce, ppl = float(row["mean_ce"]), float(row["ppl"])
            if (scheme in controls or row["eval_scheme"] != scheme or int(row["optimizer_updates"]) != 512
                    or int(row["valid_tokens"]) != 236442 or not math.isfinite(ce) or ce < 0
                    or not math.isfinite(ppl) or ppl < 1 or not math.isclose(math.log(ppl), ce, abs_tol=1e-10)):
                raise ValueError(f"Invalid historical seed-42 control for {scheme}: {path}")
            controls[scheme] = (ce, ppl)
    if set(controls) != set(SCHEMES):
        raise ValueError(f"Missing historical BF16/MXFP8 controls in {path}")
    return controls


def inputs(data):
    code = [*sorted((REPO / "model").glob("*.py")), *sorted((REPO / "model").glob("*.json"))]
    paths = {p.relative_to(REPO).as_posix(): p for p in code}
    for name in ("dataset/__init__.py", "dataset/lm_dataset.py", "trainer/train_pretrain.py",
                 "trainer/trainer_utils.py", "tests/test_v100_batch_inference.py",
                 "scripts/minimind_v100_pilot.py", "scripts/minimind_noise_arm.py", "scripts/minimind_noise_swap.py",
                 "scripts/minimind_noise_lr.py"):
        paths[name] = REPO / name
    for split in SPLITS:
        paths[f"runs/v100-pilot/data/{split}.jsonl"] = data / f"{split}.jsonl"
    hashes = {name: digest(path) for name, path in paths.items()}
    for split, expected in SPLITS.items():
        if hashes[f"runs/v100-pilot/data/{split}.jsonl"] != expected:
            raise ValueError(f"{split}.jsonl differs from the published 32k pilot split")
    return paths, hashes


def prepare_arm(work, paths, protocol):
    """Create a complete, isolated training directory before publishing it for use."""
    files = protocol["files"]
    if work.exists():
        if json.loads((work / "noise_protocol.json").read_text(encoding="utf-8")) != protocol:
            raise ValueError(f"Protocol or arm identity changed: {work}")
        for name, expected in files.items():
            if digest(work / name) != expected:
                raise ValueError(f"Staged source or input changed: {work / name}")
        return
    work.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work.parent) as tmp:
        stage = Path(tmp) / "work"
        for name, src in paths.items():
            dest = stage / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            if digest(dest) != files[name]:
                raise ValueError(f"File changed while staging: {src}")
        (stage / "noise_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
        stage.rename(work)


def report(output, controls, measured):
    rows, summaries = [], []
    for scheme in SCHEMES:
        old_ce, old_ppl = controls[scheme]
        repeat = measured.get((scheme, "repeat"))
        repeat_ce, repeat_ppl = repeat or (None, None)
        changes, changes_vs_old, warmup_deltas = {}, {}, {}
        for variant in SCENARIOS:
            point = measured.get((scheme, variant))
            if point is None:
                continue
            ce, ppl = point
            delta = ce - repeat_ce if repeat else None
            if variant in PURE_PERTURBATIONS:
                changes_vs_old[variant] = ce - old_ce
                if repeat:
                    changes[variant] = delta
            # Warmup is a schedule intervention. Match its LR multiplier to the ordinary schedule.
            normal_variant = WARMUP_CONTROLS.get(variant)
            normal_point = measured.get((scheme, normal_variant))
            if normal_point:
                warmup_deltas[variant] = ce - normal_point[0]
            schedule_control = "warmup26" if normal_variant else "repeat"
            schedule_point = measured.get((scheme, schedule_control))
            rows.append(dict(seed=SEED, architecture="moe", train_scheme=scheme, eval_scheme=scheme,
                             variant=variant, optimizer_updates=512, valid_tokens=236442, mean_ce=ce,
                             ppl=ppl, historical_ce=old_ce, historical_ppl=old_ppl, repeat_ce=repeat_ce,
                             repeat_ppl=repeat_ppl, delta_ce_vs_historical=ce-old_ce,
                             abs_delta_ce_vs_historical=abs(ce-old_ce), delta_ce_vs_repeat=delta,
                             abs_delta_ce_vs_repeat=abs(delta) if repeat else None,
                             ppl_change_pct_vs_repeat=100*(ppl/repeat_ppl-1) if repeat else None,
                             same_schedule_control=schedule_control,
                             delta_ce_vs_base_lr_same_schedule=ce-schedule_point[0] if schedule_point else None,
                             no_warmup_same_lr_control=normal_variant,
                             warmup_minus_no_warmup_same_lr_ce=warmup_deltas.get(variant)))
        if not any(key[0] == scheme for key in measured):
            continue
        up, down = measured.get((scheme, "lr_x1p0001")), measured.get((scheme, "lr_x0p9999"))
        summaries.append(dict(seed=SEED, scheme=scheme, perturbations_measured=len(changes_vs_old),
                              historical_ce=old_ce, repeat_ce=repeat_ce,
                              repeat_minus_historical_ce=repeat_ce-old_ce if repeat else None,
                              **{f"{v}_minus_repeat_ce": changes.get(v) for v in PURE_PERTURBATIONS},
                              lr_up_minus_down_ce=up[0]-down[0] if up and down else None,
                              max_abs_perturbation_delta_ce_vs_repeat=max(map(abs, changes.values()), default=None),
                              max_abs_perturbation_delta_ce_vs_historical=max(map(abs, changes_vs_old.values()), default=None),
                              **{f"{v}_minus_no_warmup_same_lr_ce": warmup_deltas.get(v) for v in WARMUP_CONTROLS}))
    record_pairs(output / "noise_runs.csv", rows)
    record_pairs(output / "noise_summary.csv", summaries)


def existing_scores(output, paths, plan):
    """Keep earlier panels in the report when adding a scenario to an existing run."""
    from minimind_grid_train import arm_paths, read_score

    measured = {}
    for scheme in SCHEMES:
        for variant in SCENARIOS:
            work = output / "arms" / scheme / variant
            checkpoint, export, score = arm_paths(work, "moe", SEED, scheme)
            if not all(p.is_file() for p in (checkpoint, export, score)):
                continue
            if score.stat().st_mtime_ns <= max(checkpoint.stat().st_mtime_ns, export.stat().st_mtime_ns):
                print(f"Skipping stale result: {score}", flush=True)
                continue
            prepare_arm(work, paths, dict(plan, scheme=scheme, variant=variant))
            try:
                measured[scheme, variant] = read_score(score, "moe", SEED, scheme)
            except (ValueError, KeyError, TypeError):
                print(f"Skipping invalid result: {score}", flush=True)
    return measured


def run(args):
    sys.path.insert(0, str(REPO))
    import torch
    from minimind_noise_arm import run_arm

    if len(set(args.scenarios)) != len(args.scenarios):
        raise ValueError("--scenarios must be distinct")
    if torch.cuda.device_count() != 8 or any("V100" not in torch.cuda.get_device_name(i) for i in range(8)):
        raise ValueError("Expected eight visible V100 GPUs, as in the historical pilot")
    output, data = args.output.expanduser().resolve(), args.data.expanduser().resolve()
    if output == data:
        raise ValueError("Use separate directories for experiment output and input data")
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".noise-floor.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another driver owns {output}") from exc
        paths, hashes = inputs(data)
        controls = historical_controls(REPO / ENDPOINTS)
        controller_files = ("minimind_moe_noise_floor.py", "minimind_grid_train.py", "minimind_moe_paired_seeds.py")
        plan = dict(seed=SEED, architecture="moe", optimizer_updates=512, microbatches=2048, world_size=8,
                    sequence_length=512, per_device_batch_size=2, gradient_accumulation=4,
                    historical_controls={s: list(v) for s, v in controls.items()}, files=hashes,
                    controllers={f: digest(REPO / "scripts" / f) for f in controller_files},
                    versions={name: version(name) for name in ("torch", "transformers", "datasets")})
        os.environ.setdefault("HF_DATASETS_CACHE", str(output / "hf-datasets-cache"))
        measured = existing_scores(output, paths, plan)
        for variant in SCENARIOS:
            if variant not in args.scenarios:
                continue
            for scheme in SCHEMES:
                work = output / "arms" / scheme / variant
                prepare_arm(work, paths, dict(plan, scheme=scheme, variant=variant))
                print(f"MoE seed 42: {scheme} / {variant} -> {work}", flush=True)
                ce, ppl = run_arm(work, scheme, variant)
                measured[scheme, variant] = (ce, ppl)
                report(output, controls, measured)
                print(f"{scheme} / {variant}: CE={ce:.6f}  PPL={ppl:.5f}", flush=True)
    print(f"NOISE_FLOOR_READY {output / 'noise_summary.csv'}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="32k pilot directory with train.jsonl and valid.jsonl")
    p.add_argument("--output", type=Path, default=Path("/data/lz/minimind-moe-noise-s42"))
    p.add_argument("--scenarios", choices=SCENARIOS, nargs="+", default=DEFAULT_SCENARIOS,
                   help="run both formats for each selected scenario; default: unperturbed repeat and LR ±0.01%%")
    run(p.parse_args())
