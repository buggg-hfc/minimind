"""Plot measured MiniMind-MoE seed-42 training and validation cross-entropy.

    python scripts/plot_moe_loss.py --output loss.png
    python scripts/plot_moe_loss.py --run-root /data/lz/minimind-moe-fork-s42
    python scripts/plot_moe_loss.py --train-log BF16=runs/v100-pilot/logs/pilot32k_s42_moe_bf16.log

H/M in a two-letter label describe expert down_proj precision before/after
update 256. M means only down_proj uses MXFP8; the full MXFP8 baseline is separate.
"""
import argparse
import csv
import json
import math
import re
from collections import defaultdict, deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


DATA = Path(__file__).resolve().parents[1] / "experiments" / "moe-fp4-seed42"
FORK = 256
VALID_TOKENS = 236442
STYLE = {
    "HH": ("#2166ac", "o"), "HM": ("#d55e00", "x"),
    "MH": ("#00856a", "^"), "MM": ("#85459b", "+"),
    "bf16": ("#353535", "s"), "mxfp8": ("#997000", "D"),
    "NVFP4 3MM": ("#53636d", "P"), "NVINT4 3MM": ("#6c5c30", "*"),
    "L6E1": ("#c13c70", "v"), "L6E2": ("#52752a", "<"),
    "ALL_EXCEPT_L6E1": ("#007e9b", ">"),
}
PILOT_CASES = {
    "bf16": "pilot BF16", "mxfp8": "pilot MXFP8",
    "hif4_thr_3mm_r2": "pilot HiF4 3MM", "nvfp4_3mm_r2": "pilot NVFP4 3MM",
    "nvint4_3mm_r2": "pilot NVINT4 3MM",
    "hif4_thr_3mm_ffn_down_mxfp8_r2": "pilot HiF4 + MXFP8 down",
}
LOG = re.compile(r"Epoch:\[1/1\]\((\d+)/2048\).*?\blogits_loss:\s*([^,\s]+)")


def rows(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)


def record(series, case, step, ce, tokens, source):
    point = (float(ce), int(tokens))
    if not math.isfinite(point[0]) or point[0] < 0 or point[1] <= 0:
        raise ValueError(f"Invalid CE or token count for {case}, step {step} in {source}")
    if step in series[case]:
        old_ce, old_count = series[case][step]
        if point[1] != old_count or not math.isclose(point[0], old_ce, abs_tol=1e-8):
            raise ValueError(f"Conflicting measurements for {case}, step {step} in {source}")
    series[case][step] = point


def validation(series, case, step, ce, tokens, source):
    if int(tokens) != VALID_TOKENS:
        raise ValueError(f"Expected {VALID_TOKENS} validation tokens in {source}")
    record(series, case, step, ce, tokens, source)


def load_pair(path, arms, train, valid):
    for row in rows(path):
        step = FORK + int(row["relative_update"])
        if row["optimizer_update"] and int(row["optimizer_update"]) != step:
            raise ValueError(f"Unexpected optimizer update in {path} at step {step}")
        for side, arm in zip("ab", arms):
            if row[f"train_ce_{side}"]:
                record(train, arm, step, row[f"train_ce_{side}"], row["train_valid_tokens"], path)
            if row[f"eval_ce_{side}"]:
                validation(valid, arm, step, row[f"eval_ce_{side}"], row["eval_valid_tokens"], path)


def read_prefix(path):
    # The native trainer prints rank 0's current microbatch CE, excluding aux loss.
    points = {}
    with path.open(encoding="utf-8", errors="replace") as stream:
        for line in stream:
            match = LOG.search(line)
            if match:
                micro, ce = int(match[1]), float(match[2])
                if not 1 <= micro <= 2048 or not math.isfinite(ce) or ce < 0:
                    raise ValueError(f"Invalid training measurement in {path}")
                points[micro / 4] = ce
    return points


def load_runs(root, train, valid, logs):
    pairs = [("pair-1024-to-2048", ("HH", "HM")),
             ("reciprocal-down-s42/pair-1024-to-2048", ("MH", "MM"))]
    sources = []
    for directory, arms in pairs:
        path = root / directory / "summary.csv"
        if path.is_file():
            load_pair(path, arms, train, valid)
            sources.append(path)
    arms = {"L6E1": root / "single-expert-down-s42/L6E1",
            "L6E2": root / "single-expert-down-s42/L6E2",
            "ALL_EXCEPT_L6E1": root / "leave-l6e1-out-s42"}
    prefixes = {"H prefix": root / "hif4-prefix-1024/prefix.log",
                "M prefix": root / "reciprocal-down-s42/mixed-prefix-1024/prefix.log"}
    for case, directory in arms.items():
        prefixes[case] = directory / "prefix-1024/prefix.log"
        late = directory / "late-1024-to-2048/hif4"
        for path in sorted((late / "train").glob("update-*.json")):
            point = json.loads(path.read_text())
            record(train, case, int(point["optimizer_update"]), point["mean_ce"],
                   point["valid_tokens"], path)
            sources.append(path)
        for path in sorted((late / "eval").glob("step-*.json")):
            point = json.loads(path.read_text())
            validation(valid, case, FORK + int(point["updates_since_fork"]),
                       point["mean_ce"], point["valid_tokens"], path)
            sources.append(path)
    for path in (root / "single-expert-down-s42/expert_cases.csv",
                 root / "leave-l6e1-out-s42/expert_factorial.csv"):
        if path.is_file():
            for row in rows(path):
                case = row["case"]
                # MH's prefix-own score belongs to M evaluation, shared with MM.
                prefix_case = "MM" if case == "MH" else case
                validation(valid, prefix_case, FORK, row["midpoint_prefix_own_ce"], row["valid_tokens"], path)
                validation(valid, case, int(row["optimizer_update"]), row["mean_ce"], row["valid_tokens"], path)
            sources.append(path)
    for name, path in prefixes.items():
        if logs is not None and path.is_file():
            points = read_prefix(path)
            if points:
                logs[name] = points
                sources.append(path)
    if not sources and logs is not None:
        raise ValueError(f"No seed-42 fork measurements found in {root}")
    print(f"Loaded {len(sources)} local result files from {root}")


def rolling_ce(steps, points, window):
    history = deque()
    nll = count = 0
    result = []
    for i, step in enumerate(steps):
        if i and step != steps[i - 1] + 1:
            history.clear()
            nll = count = 0
        ce, tokens = points[step]
        entry = (ce * tokens, tokens)
        history.append(entry)
        nll += entry[0]
        count += tokens
        if len(history) > window:
            loss, n = history.popleft()
            nll -= loss
            count -= n
        result.append(nll / count)
    return result


def plot(train, valid, logs, window, output):
    panels = 4 if logs else 3
    fig, axes = plt.subplots(panels, 1, figsize=(10.5, 2.65 * panels + .6),
                             constrained_layout=True)
    tr, val, diff = axes[-3:]
    if logs:
        ax = axes[0]
        ax.set_title("Training CE  |  logged rank-0 microbatches")
        for name, points in logs.items():
            steps = sorted(points)
            ax.plot(steps, [points[s] for s in steps], linewidth=1, label=name)

    tr.set_title(f"Training CE  |  faint: each global update, line: {window}-update token-weighted mean")
    for case, points in train.items():
        color, marker = STYLE[case]
        steps = sorted(points)
        tr.scatter(steps, [points[s][0] for s in steps], s=5, alpha=.16, color=color)
        tr.plot(steps, rolling_ce(steps, points, window), linewidth=1.6, color=color,
                label=case, marker=marker if len(steps) == 1 else None)

    val.set_title("Validation CE  |  full validation set")
    for case, points in valid.items():
        steps = sorted(points)
        color, marker = STYLE[case]
        val.plot(steps, [points[s][0] for s in steps], linestyle="-" if len(steps) > 2 else "none",
                 marker=marker, markersize=7, markerfacecolor="none", linewidth=1.2,
                 color=color, label=f"{case} ({points[steps[-1]][0]:.4f})")
    diff.set_title("Paired validation CE difference  |  negative favors late MXFP8 down")
    diff.axhline(0, color="#666666", linewidth=.8)
    for early, late in (("HH", "HM"), ("MH", "MM")):
        common = sorted(valid.get(early, {}).keys() & valid.get(late, {}).keys())
        if common:
            delta = [valid[late][s][0] - valid[early][s][0] for s in common]
            diff.plot(common, delta, marker="o", markersize=4, color=STYLE[late][0],
                      linestyle="-" if len(common) > 2 else "none", label=f"{late} - {early}")
    for ax in axes:
        ax.axvline(FORK, color="#777777", linestyle=":", linewidth=1)
        ax.set_xlabel("AdamW update   (fork at 256)")
        ax.set_ylabel("CE difference (nats)" if ax is diff else "CE (nats)")
        ax.grid(axis="y", alpha=.2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
        ax.legend(fontsize=8, ncol=4)
    fig.suptitle("MiniMind-MoE  /  seed 42", fontsize=15)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=175)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=DATA, help="Published seed-42 CSV directory")
    parser.add_argument("--run-root", type=Path, help="Local minimind-moe-fork-s42 directory")
    parser.add_argument("--pilot-logs", type=Path, action="append", default=[], metavar="DIR",
                        help="Load available seed-42 baseline logs under DIR (repeatable)")
    parser.add_argument("--train-log", action="append", default=[], metavar="NAME=PATH",
                        help="Add a native seed-42 MoE pilot log (repeatable)")
    parser.add_argument("--window", type=int, default=10, help="Trailing training window, 1 for raw CE")
    parser.add_argument("--output", type=Path, default=Path("moe_loss.png"), help="PNG, PDF or SVG path")
    args = parser.parse_args()
    if args.window < 1:
        parser.error("--window must be positive")
    if args.output.suffix.lower() not in (".png", ".pdf", ".svg"):
        parser.error("--output must end in .png, .pdf or .svg")
    train, valid, logs = defaultdict(dict), defaultdict(dict), {}
    try:
        load_pair(args.data_dir / "paired_updates_HH_HM.csv", ("HH", "HM"), train, valid)
        points = args.data_dir / "validation_points.csv"
        for row in rows(points):
            case = row["trajectory"]
            if case == "M_prefix":
                case = "MM"  # At the midpoint the M prefix was evaluated with M precision.
            validation(valid, case, int(row["optimizer_update"]), row["mean_ce"], row["valid_tokens"], points)
        grid = args.data_dir / "moe_ffn_grid_seed42.csv"
        formats = {"nvfp4_3mm": "NVFP4 3MM", "nvint4_3mm": "NVINT4 3MM"}
        for row in rows(grid):
            scheme = row["训练方案"]
            if scheme in formats:
                validation(valid, formats[scheme], 512, row["本组验证CE"], VALID_TOKENS, grid)
        if args.run_root:
            load_runs(args.run_root, train, valid, logs)
        for directory in args.pilot_logs:
            found = False
            for scheme, name in PILOT_CASES.items():
                path = directory / f"pilot32k_s42_moe_{scheme}.log"
                if path.is_file():
                    points = read_prefix(path)
                    if points:
                        if name in logs and logs[name] != points:
                            raise ValueError(f"Conflicting pilot logs for {name} in {directory}")
                        logs[name] = points
                        found = True
            if not found:
                raise ValueError(f"No matching seed-42 MoE pilot logs with CE in {directory}")
        for spec in args.train_log:
            name, sep, path = spec.partition("=")
            if not name or not sep or not path or name in logs:
                raise ValueError(f"Expected a unique NAME=PATH for --train-log: {spec}")
            logs[name] = read_prefix(Path(path))
            if not logs[name]:
                raise ValueError(f"No seed-42 pilot-format logits_loss observations in {path}")
        plot(train, valid, logs, args.window, args.output)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"{parser.prog}: {error}\n")
    print(f"Saved {args.output.resolve()}")
    for case, points in valid.items():
        print(f"{case}: {len(train.get(case, {}))} global training CE, {len(points)} validation CE")


if __name__ == "__main__":
    main()
