"""Measure final learned-weight histogram KL against a same-seed BF16 model on CPU."""
import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from model.quant_policy import RESIDUAL_POLICY, RESIDUAL_SCHEMES
from minimind_grid_train import ARCHITECTURES, SCHEMES, _completed_state, arm_paths
from minimind_moe_paired_seeds import SPLITS, record_pairs


@dataclass(frozen=True)
class Endpoint:
    path: Path
    ce: float
    ppl: float


def metrics(row):
    ce, ppl = float(row["mean_ce"]), float(row["ppl"])
    if (not math.isfinite(ce) or ce < 0 or not math.isfinite(ppl)
            or ppl < 1 or not math.isclose(math.log(ppl), ce, rel_tol=1e-10, abs_tol=1e-14)):
        raise ValueError(f"Invalid CE/PPL pair: {ce}, {ppl}")
    return ce, ppl


def grid_index(path):
    if not path.is_file():
        return {}
    result = {}
    report_time = path.stat().st_mtime_ns
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = int(row["seed"]), row["architecture"], row["train_scheme"]
            if (key in result or row["eval_scheme"] != key[2] or int(row["optimizer_updates"]) != 512
                    or int(row["valid_tokens"]) != 236442):
                raise ValueError(f"{path}: duplicate or incompatible grid row {key}")
            metrics(row)
            row["report_mtime_ns"] = report_time
            result[key] = row
    return result


def native_score(path, seed, arch, scheme, export_name):
    row = json.loads(path.read_text(encoding="utf-8-sig"))
    expected = dict(architecture=arch, train_scheme=scheme, eval_scheme=scheme,
                    checkpoint=export_name, valid_sha256=SPLITS["valid"], valid_tokens=236442,
                    documents=1024, seq_len=512, batch_size=2)
    if seed != 42 or "seed" in row:
        expected["seed"] = seed  # original seed-42 eval JSON predates this field
    if scheme in RESIDUAL_SCHEMES:
        expected.update(residual_policy=RESIDUAL_POLICY, eval_residual_policy=RESIDUAL_POLICY)
    for field, value in expected.items():
        if row.get(field) != value:
            raise ValueError(f"{path}: {field}={row.get(field)!r}, expected {value!r}")
    return metrics(row)


def legacy_scores():
    # The published seed-42 native same-format rows identify the historical final run.
    path = REPO / "experiments/moe-fp4-seed42/dense_moe_pilot_endpoints.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {(row["architecture"], row["train_scheme"]): metrics(row)
                for row in csv.DictReader(stream) if row["eval_scheme"] == row["train_scheme"]}


def available_endpoint(grid, legacy, previous, index_row, seed, arch, scheme, historical=None):
    roots = []
    if index_row and index_row.get("source_work"):
        roots.append(Path(index_row["source_work"]).expanduser().resolve())
    roots.append(grid / "work")
    if previous and arch == "moe" and scheme in ("mxfp8", "hif4_thr_3mm"):
        roots.append(previous / "work")
    if seed == 42:
        roots.append(legacy)
    attempted = []
    for root in dict.fromkeys(roots):
        resume, export, score = arm_paths(root, arch, seed, scheme)
        # The native trainer writes matching plain model exports to out/ and checkpoints/.
        export_copy = resume.parent / export.name
        if not export.is_file() and export_copy.is_file():
            export = export_copy
        if not (export.is_file() or resume.is_file()):
            attempted.extend(str(p) for p in (export, export_copy, resume))
            continue

        expected = metrics(index_row) if index_row else None
        if expected is None and root == legacy and seed == 42:
            expected = historical
        if score.is_file():
            measured = native_score(score, seed, arch, scheme, export.name)
            if expected and not all(math.isclose(x, y, rel_tol=1e-12) for x, y in zip(measured, expected)):
                raise ValueError(f"{score}: endpoint CE/PPL disagree with the recorded completed run")
            # An indexed final evaluation measured before a file was overwritten cannot attest that file.
            if expected and export.is_file() and export.stat().st_mtime_ns <= score.stat().st_mtime_ns:
                return Endpoint(export, *measured)
        else:
            measured = expected
            if (measured and index_row and index_row.get("source_work") and export.is_file()
                    and root == Path(index_row["source_work"]).expanduser().resolve()
                    and export.stat().st_mtime_ns <= index_row["report_mtime_ns"]):
                return Endpoint(export, *measured)

        if measured and resume.is_file():
            # Use the final resume model itself. Its validated cursor cannot authenticate a separate export.
            return Endpoint(resume, *measured)
        if measured is not None and export.is_file():
            attempted.append(f"Cannot verify final update for {export}; missing matching {resume}")
        else:
            attempted.append(f"Missing final CE/PPL for {export}; expected {score}")
    raise FileNotFoundError(f"Missing verified endpoint for {arch}/seed{seed}/{scheme}:\n  " + "\n  ".join(attempted))


def load_endpoint(endpoint, seed, scheme):
    from minimind_weight_kl import load_weight_state, validate_weight_state

    if endpoint.path.name.endswith("_resume.pth"):
        state = _completed_state(endpoint.path, seed, scheme)["model"]
        return validate_weight_state(state, endpoint.path)
    return load_weight_state(endpoint.path)


def run(args):
    import torch
    from minimind_weight_kl import compare_weight_states

    if args.threads <= 0:
        raise ValueError("--threads must be positive")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must be distinct, nonnegative integers")
    if len(set(args.architectures)) != len(args.architectures):
        raise ValueError("Duplicate architecture")
    torch.set_num_threads(args.threads)
    grid = args.grid_output.expanduser().resolve()
    legacy = args.legacy_root.expanduser().resolve()
    previous = args.previous_paired.expanduser().resolve() if args.previous_paired else None
    index = grid_index(grid / "same_format.csv")
    historical = legacy_scores() if 42 in args.seeds else {}
    endpoints = {}
    for seed in args.seeds:
        for arch in args.architectures:
            for scheme in SCHEMES:
                key = (seed, arch, scheme)
                endpoints[key] = available_endpoint(grid, legacy, previous, index.get(key), *key,
                                                    historical=historical.get((arch, scheme)))

    summary, details = [], []
    for seed in args.seeds:
        for arch in args.architectures:
            ref = endpoints[seed, arch, "bf16"]
            ref_state = load_endpoint(ref, seed, "bf16")
            for scheme in SCHEMES:
                target = endpoints[seed, arch, scheme]
                is_self = scheme == "bf16"
                state = ref_state if is_self else load_endpoint(target, seed, scheme)
                stats, tensors = compare_weight_states(ref_state, state)
                del state
                identity = dict(seed=seed, architecture=arch, reference_scheme="bf16", train_scheme=scheme)
                summary.append(dict(**identity, optimizer_updates=512, reference_checkpoint=str(ref.path),
                                    candidate_checkpoint=str(target.path), valid_tokens=236442,
                                    reference_mean_ce=ref.ce, reference_ppl=ref.ppl,
                                    eval_scheme=scheme, mean_ce=target.ce, ppl=target.ppl, **stats))
                details.extend(dict(**identity, **row) for row in tensors)
                print(f"{arch} seed={seed} bf16 -> {scheme}: KL={stats['weight_kl_forward']:.9g}", flush=True)
            del ref_state

    output = args.output.expanduser().resolve() if args.output else grid / "weight_kl"
    output.mkdir(parents=True, exist_ok=True)
    record_pairs(output / "weight_kl_by_tensor.csv", details)
    record_pairs(output / "weight_kl_summary.csv", summary)
    print(f"WEIGHT_KL_READY {output} ({len(summary)} models, {len(details)} tensor comparisons)", flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid-output", type=Path, default=Path("/data/lz/minimind-five-format-grid"),
                   help="five-format grid directory containing work/ and same_format.csv")
    p.add_argument("--legacy-root", type=Path, default=Path("/home/lz/minimind"),
                   help="original training checkout containing seed 42 out/ and runs/")
    p.add_argument("--previous-paired", type=Path, help="optional relocated MoE two-format run directory")
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 100, 2026])
    p.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    p.add_argument("--threads", type=int, default=4, help="CPU threads available to PyTorch")
    p.add_argument("--output", type=Path, help="output directory; default: GRID_OUTPUT/weight_kl")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
