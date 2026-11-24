"""Summarize final low-precision validations in the two-stage down_proj experiment."""
import csv
import math
import os
import tempfile
from pathlib import Path


_LAST = 256
_NAMES = {"H": "hif4_thr_3mm", "M": "mxfp8"}
_EVAL = ("eval_valid_tokens", "eval_ce_a", "eval_ppl_a", "eval_ce_b", "eval_ppl_b")


def _load(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [int(r["relative_update"]) for r in rows] != list(range(_LAST + 1)):
        raise ValueError(f"{path} requires one row per relative update, from 0 to {_LAST}")
    return rows


def _compare_inputs(first, second):
    eval_tokens = None
    for update, (old, new) in enumerate(zip(first, second)):
        if update:
            for key, expected in (("optimizer_update", 256 + update), ("microbatch", 1024 + 4 * update)):
                if int(old[key]) != expected or int(new[key]) != expected:
                    raise ValueError(f"Unexpected {key} at relative update {update}")
            n = int(old["train_valid_tokens"])
            lr = float(old["lr"])
            if n <= 0 or n != int(new["train_valid_tokens"]):
                raise ValueError(f"Training token counts differ or are invalid at relative update {update}")
            if not math.isfinite(lr) or lr < 0 or lr != float(new["lr"]):
                raise ValueError(f"Learning rates differ or are invalid at relative update {update}")
        scores = [row[key] for row in (old, new) for key in _EVAL]
        if not any(scores) and update not in (0, _LAST):
            continue
        if not all(scores):
            raise ValueError(f"Incomplete paired validation at relative update {update}")
        count = int(old["eval_valid_tokens"])
        if count <= 0 or count != int(new["eval_valid_tokens"]) or eval_tokens not in (None, count):
            raise ValueError(f"Validation token count differs at relative update {update}")
        eval_tokens = count
        for row in (old, new):
            for side in "ab":
                ce, ppl = (float(row[f"eval_{key}_{side}"]) for key in ("ce", "ppl"))
                if not (ce >= 0 and math.isfinite(ce) and ppl > 0 and math.isfinite(ppl)
                        and math.isclose(ce, math.log(ppl), rel_tol=1e-8, abs_tol=1e-10)):
                    raise ValueError(f"Inconsistent validation CE/PPL at relative update {update}")


def write_four_cells(previous: Path, current: Path, out: Path):
    """Read the H-prefix and M-prefix reports, in that order.

    Within each input, arm A uses H and arm B uses M in the second half.
    H/M specify the expert down_proj precision; other projection policies are held fixed.
    """
    previous, current, out = map(Path, (previous, current, out))
    if previous.resolve() == current.resolve():
        raise ValueError("Expected separate H-prefix and M-prefix summaries")
    first, second = _load(previous), _load(current)
    _compare_inputs(first, second)
    if out.suffix.lower() != ".csv":
        out = out / "four_cells.csv"
    if out.resolve() in (previous.resolve(), current.resolve()):
        raise ValueError("Output would overwrite an input summary")
    cells = []
    ce = {}
    for early, summary, matched in (("H", first, "a"), ("M", second, "b")):
        midpoint, final = summary[0], summary[-1]
        for late, side in (("H", "a"), ("M", "b")):
            case = early + late
            ce[case] = float(final[f"eval_ce_{side}"])
            cells.append(dict(case=case, first_half_down_proj=_NAMES[early], second_half_down_proj=_NAMES[late],
                              optimizer_update=int(final["optimizer_update"]), valid_tokens=int(final["eval_valid_tokens"]),
                              mean_ce=ce[case], ppl=float(final[f"eval_ppl_{side}"]),
                              midpoint_prefix_own_ce=float(midpoint[f"eval_ce_{matched}"]),
                              midpoint_prefix_own_ppl=float(midpoint[f"eval_ppl_{matched}"])))

    out.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".four_cells_", dir=out.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as dst:
            writer = csv.DictWriter(dst, fieldnames=cells[0].keys())
            writer.writeheader()
            writer.writerows(cells)
        os.replace(tmp, out)
    finally:
        Path(tmp).unlink(missing_ok=True)

    mid_h, mid_m = cells[0], cells[-1]
    print("H=HiF4，M=仅专家 down_proj 使用 MXFP8。两位字母依次表示前半程和后半程。")
    print("中点验证使用各前缀自己的推理格式")
    for name, row in (("H", mid_h), ("M", mid_m)):
        print(f"  {name} 前缀  CE={row['midpoint_prefix_own_ce']:.8f}  PPL={row['midpoint_prefix_own_ppl']:.6f}")
    print(f"  中点 CE 差 M − H = {mid_m['midpoint_prefix_own_ce'] - mid_h['midpoint_prefix_own_ce']:+.8f}")
    print("终点验证 CE 差值，前四行中负数表示改用 MXFP8 后 CE 更低。")
    for label, candidate, reference in (
        ("仅改前半程，后半程为 H", "MH", "HH"),
        ("仅改前半程，后半程为 M", "MM", "HM"),
        ("仅改后半程，前半程为 H", "HM", "HH"),
        ("仅改后半程，前半程为 M", "MM", "MH"),
    ):
        print(f"  {label}  {candidate} − {reference} = {ce[candidate] - ce[reference]:+.8f}")
    interaction = (ce["MM"] - ce["MH"]) - (ce["HM"] - ce["HH"])
    print(f"  两阶段交互差值  (MM − MH) − (HM − HH) = {interaction:+.8f}")
    return out
