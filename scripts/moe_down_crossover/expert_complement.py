"""Test early MXFP8 on the other 31 down projections while L6E1 uses HiF4."""
import fcntl

from expert_late import _validation, finish_expert
from expert_prefix import ALL_EXCEPT_L6E1, expert_scheme, make_expert_anchor
from expert_runs import controls, write_report


def reference_rows(root, final):
    rows = controls(final, root / "reciprocal-down-s42" / "pair-1024-to-2048")
    run = root / "single-expert-down-s42" / "L6E1" / "late-1024-to-2048" / "hif4"
    mid = _validation(run, 0, expert_scheme(1))
    end = _validation(run, 256, "hif4")
    rows.append(dict(case="L6E1", early_mxfp8_down="layer6.expert1", final_down="hif4_thr_3mm",
                     optimizer_update=512, valid_tokens=end["valid_tokens"], mean_ce=end["mean_ce"],
                     ppl=end["ppl"], midpoint_prefix_own_ce=mid["mean_ce"], midpoint_prefix_own_ppl=mid["ppl"]))
    return rows


def run_complement(root, original, old, final, check_batch_order):
    rows = reference_rows(root, final)
    dest = root / "leave-l6e1-out-s42"
    dest.mkdir(exist_ok=True)
    with (dest / ".experiment.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"已有任务正在运行此组实验：{dest}") from exc
        anchor = dest / "prefix-1024"
        late = dest / "late-1024-to-2048"
        print("前半程以 MXFP8 训练其余 31 个 down_proj，L6E1 的 down_proj 使用 HiF4。", flush=True)
        make_expert_anchor(original, anchor, old / "scripts", ALL_EXCEPT_L6E1)
        scores = finish_expert(original, anchor, old, late, ALL_EXCEPT_L6E1)
        check_batch_order(final, late)
        rows.append(dict(case="ALL_EXCEPT_L6E1", early_mxfp8_down="all_except_layer6.expert1",
                         final_down="hif4_thr_3mm", **scores))
        report = dest / "expert_factorial.csv"
        write_report(report, rows)
        hh, mh, e1, other = rows
        print("L6E1 改用 MXFP8 带来的验证 CE 降幅（正值表示改善）")
        print(f"其余 31 个 down 为 HiF4 时：{hh['mean_ce'] - e1['mean_ce']:+.6f}")
        print(f"其余 31 个 down 为 MXFP8 时：{other['mean_ce'] - mh['mean_ce']:+.6f}")
        print(f"COMPLEMENT_READY {report}", flush=True)
