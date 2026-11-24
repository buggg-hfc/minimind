"""Compare the seed-42 H/M prefixes and their four second-half validation curves (CPU)."""
import argparse
import csv
import math
import os
from pathlib import Path
import re
from statistics import mean, median
import tempfile

from moe_down_crossover.early_results import _compare_inputs, _load


DEFAULT_ROOT = Path("/data/lz/minimind-moe-fork-s42")
PREFIX_END = 1024
LOG = re.compile(r"Epoch:\[1/1\]\((\d+)/2048\),\s*loss:\s*([^,]+),\s*"
                 r"logits_loss:\s*([^,]+),\s*aux_loss:\s*([^,]+),\s*lr:\s*([^,\s]+)")
RESUME = re.compile(r"跳过前\s*(\d+)\s*个step")
FIELDS = ("loss", "logits_loss", "aux_loss", "lr")


def read_log(path, *, last_microbatch=PREFIX_END, allow_empty=False):
    records = {}
    removed = 0
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            cursor = RESUME.search(line)
            if cursor or ("torch.distributed.run" in line and "--from_resume 0 " in line):
                start = int(cursor[1]) if cursor else 0
                stale = [step for step in records if step > start]
                for step in stale:
                    del records[step]
                removed += len(stale)
                continue
            match = LOG.search(line)
            if not match:
                continue
            step = int(match[1])
            if not 1 <= step <= last_microbatch or step % 4:
                raise ValueError(f"{path}: 意外的 microbatch {step}")
            values = dict(zip(FIELDS, map(float, match.groups()[1:])))
            if not math.isfinite(values["lr"]) or values["lr"] <= 0:
                raise ValueError(f"{path}: microbatch {step} 学习率无效")
            if step in records and records[step] != values:
                raise ValueError(f"{path}: microbatch {step} 有冲突记录且缺少回退游标")
            records[step] = values
    if not records and not allow_empty:
        raise ValueError(f"未找到训练记录：{path}")
    if removed:
        print(f"{path.name} ({path.parent.name})：忽略 {removed} 条被后续重启覆盖的记录。")
    return records


def prefix_rows(h, m):
    result = []
    # Include known missing slots explicitly, as well as any additional actual measurements.
    ticks = set(range(20, PREFIX_END + 1, 20)) | set(h) | set(m)
    for step in sorted(ticks):
        a, b = h.get(step, {}), m.get(step, {})
        if a and b and a["lr"] != b["lr"]:
            raise ValueError(f"microbatch {step} 两个前缀学习率不同")
        paired = bool(a and b)
        finite = paired and all(math.isfinite(row["logits_loss"]) for row in (a, b))
        row = {"microbatch": step, "adamw_update": step // 4, "paired": int(paired),
               "finite_logits_pair": int(finite)}
        for key in FIELDS:
            row[key+"_H"], row[key+"_M"] = a.get(key), b.get(key)
        for key in ("logits_loss", "aux_loss"):
            row[key+"_H_minus_M"] = round(a[key]-b[key], 4) if paired and all(
                math.isfinite(v[key]) for v in (a, b)) else None
        result.append(row)
    if not any(r["finite_logits_pair"] for r in result):
        raise ValueError("两份前缀日志没有共同的有限 logits_loss 观测")
    return result


def windows(pairs):
    result = []
    for low in range(1, 257, 32):
        group = [r for r in pairs if low <= r["adamw_update"] < low + 32]
        observed = [r for r in group if r["finite_logits_pair"]]
        gap = [r["logits_loss_H_minus_M"] for r in observed]
        max_row = max(observed, key=lambda r: abs(r["logits_loss_H_minus_M"])) if observed else {}
        result.append(dict(update_from=low, update_to=low+31, log_slots=len(group),
                           paired=sum(r["paired"] for r in group), finite_pairs=len(observed),
                           mean_gap=mean(gap) if gap else None, median_gap=median(gap) if gap else None,
                           largest_absolute_gap_signed=max_row.get("logits_loss_H_minus_M"),
                           peak_microbatch=max_row.get("microbatch")))
    return result


def validations(first, second):
    h, m = _load(first), _load(second)
    _compare_inputs(h, m)
    result = []
    for a, b in zip(h, m):
        if not a["eval_ce_a"]:
            continue
        ntokens = int(a["eval_valid_tokens"])
        if ntokens != 236442:
            raise ValueError("验证集的有效 token 数与 seed-42 实验不符")
        hh, hm, mh, mm = (float(r["eval_ce_"+arm]) for r in (a, b) for arm in ("a", "b"))
        result.append(dict(adamw_update=256+int(a["relative_update"]), valid_tokens=ntokens,
                           HH=hh, HM=hm, MH=mh, MM=mm, fixed_H_HH_minus_MH=hh-mh, fixed_M_HM_minus_MM=hm-mm))
    return result


def write_csv(path, rows):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".audit-")
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        Path(tmp).unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="seed-42 分叉实验目录")
    parser.add_argument("--out", type=Path, help="默认写入 ROOT/prefix-audit")
    args = parser.parse_args()
    root = args.root
    reciprocal = root / "reciprocal-down-s42"
    pairs = prefix_rows(read_log(root / "hif4-prefix-1024/prefix.log"),
                        read_log(reciprocal / "mixed-prefix-1024/prefix.log"))
    groups = windows(pairs)
    rel = "pair-1024-to-2048/summary.csv"
    valid = validations(root / rel, reciprocal / rel)
    out = args.out or root / "prefix-audit"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("prefix_pairs", pairs), ("prefix_windows", groups), ("validation_gaps", valid)):
        write_csv(out / (name+".csv"), rows)
    print("前缀 H − M：正值表示 M 的 CE 较低。训练日志取 rank 0 当前 microbatch，损失打印至四位小数。")
    for row in groups:
        med = row["median_gap"]
        label = f"{med:+.4f}" if med is not None else "无"
        print(f"更新 {row['update_from']:3d}–{row['update_to']:3d}  "
              f"有效配对 {row['finite_pairs']}/{row['log_slots']}  采样 CE 差中位数 {label}")
    print("验证集同一后半程格式的前缀差（中点用相同推理格式）：")
    for row in valid:
        print(f"更新 {row['adamw_update']}  HH−MH={row['fixed_H_HH_minus_MH']:+.6f}  "
              f"HM−MM={row['fixed_M_HM_minus_MM']:+.6f}")
    print(f"PREFIX_AUDIT_READY {out.resolve()}")


if __name__ == "__main__":
    main()
