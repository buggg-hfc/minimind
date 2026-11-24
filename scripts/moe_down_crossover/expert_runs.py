"""Run the two early single-expert interventions and compare their all-HiF4 endpoints."""
import csv
import fcntl
import math
import os
from pathlib import Path
import tempfile

from early_results import _load, _compare_inputs
from expert_late import finish_expert
from expert_prefix import make_expert_anchor


def controls(first, reciprocal):
    # Reuse both completed trajectories without a new GPU evaluation.
    h, m = _load(first / "summary.csv"), _load(reciprocal / "summary.csv")
    _compare_inputs(h, m)
    result = []
    for case, history, early, side in (("HH", h, "none", "a"), ("MH", m, "all", "b")):
        mid, end = history[0], history[-1]
        result.append(dict(case=case, early_mxfp8_down=early, final_down="hif4_thr_3mm",
                           optimizer_update=int(end["optimizer_update"]), valid_tokens=int(end["eval_valid_tokens"]),
                           mean_ce=float(end["eval_ce_a"]), ppl=float(end["eval_ppl_a"]),
                           midpoint_prefix_own_ce=float(mid[f"eval_ce_{side}"]),
                           midpoint_prefix_own_ppl=float(mid[f"eval_ppl_{side}"])))
    return result


def write_report(out, rows):
    hh, mh = rows[:2]
    for row in rows:
        if (row["optimizer_update"] != 512 or row["valid_tokens"] != 236442 or
                not math.isclose(row["mean_ce"], math.log(row["ppl"]), rel_tol=1e-8) or
                not math.isclose(row["midpoint_prefix_own_ce"], math.log(row["midpoint_prefix_own_ppl"]), rel_tol=1e-8)):
            raise ValueError(f"Invalid final or midpoint validation in {row['case']}")
    scored = [dict(row, delta_ce_vs_HH=row["mean_ce"]-hh["mean_ce"], delta_ce_vs_MH=row["mean_ce"]-mh["mean_ce"])
              for row in rows]
    fd, tmp = tempfile.mkstemp(dir=out.parent, prefix=".expert-cases-")
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=scored[0].keys())
            writer.writeheader()
            writer.writerows(scored)
        os.replace(tmp, out)
    finally:
        Path(tmp).unlink(missing_ok=True)


def run_experts(root, original, old, final, check_batch_order):
    reciprocal = root / "reciprocal-down-s42" / "pair-1024-to-2048"
    rows = controls(final, reciprocal)
    dest = root / "single-expert-down-s42"
    dest.mkdir(exist_ok=True)
    report = dest / "expert_cases.csv"
    with (dest / ".experiment.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"两组单专家实验正在运行：{dest}") from exc
        for expert in (2, 1):
            name = f"L6E{expert}"
            arm = dest / name
            arm.mkdir(exist_ok=True)
            anchor, late = arm / "prefix-1024", arm / "late-1024-to-2048"
            print(f"{name}：前 256 次 AdamW 更新只切换该专家的 down_proj，后 256 次按 HiF4 分支训练。", flush=True)
            make_expert_anchor(original, anchor, old / "scripts", expert)
            scores = finish_expert(original, anchor, old, late, expert)
            check_batch_order(final, late)
            rows.append(dict(case=name, early_mxfp8_down=f"layer6.expert{expert}", final_down="hif4_thr_3mm", **scores))
            print(f"{name}  CE={scores['mean_ce']:.6f}  PPL={scores['ppl']:.4f}"
                  f"  ΔCE 对 HH={scores['mean_ce'] - rows[0]['mean_ce']:+.6f}", flush=True)
        write_report(report, rows)
        print(f"SINGLE_EXPERT_READY {report}", flush=True)
