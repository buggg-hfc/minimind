"""Plot all available seed-42 MoE loss measurements on the original 512-update schedule."""
import argparse
import math
from collections import defaultdict
from pathlib import Path

from minimind_moe_prefix_audit import DEFAULT_ROOT, read_log
from plot_moe_loss import DATA, FORK, load_pair, load_runs, plt, rows, validation


END = 512
PILOT_DIRS = [Path("/home/lz/minimind/runs/v100-pilot/logs"),
              Path("/data/lz/minimind-moe-ffn-mxfp8/runs/v100-pilot/logs")]
PREFIX_PATHS = {
    "H": "hif4-prefix-1024/prefix.log",
    "M": "reciprocal-down-s42/mixed-prefix-1024/prefix.log",
    "L6E1": "single-expert-down-s42/L6E1/prefix-1024/prefix.log",
    "L6E2": "single-expert-down-s42/L6E2/prefix-1024/prefix.log",
    "ALL_EXCEPT_L6E1": "leave-l6e1-out-s42/prefix-1024/prefix.log",
}
LOSS = ("logits_loss", "loss", "aux_loss")
COLORS = ["#176aa1", "#c95b25", "#008470", "#844ea4", "#a57200", "#bd507b", "#485674", "#608135"]
PREFIX_LABELS = {"H": "H prefix (HH, HM)", "M": "M prefix (MH, MM)"}
FAMILIES = {"hif4_thr_3mm": "HiF4-thr 3MM", "nvfp4_3mm": "NVFP4 3MM",
            "nvint4_3mm": "NVINT4 3MM", "bf16": "BF16", "mxfp8": "MXFP8"}


def logged(path, end):
    records = read_log(path, last_microbatch=end * 4, allow_empty=True)
    if not records:
        print(f"No logged loss yet: {path}")
        return {}
    print(f"{path.name}: updates {min(records) // 4}–{max(records) // 4}, {len(records)} logged microbatches")
    return {micro // 4: row for micro, row in records.items()}


def prefixes(root):
    data = {}
    for name, rel in PREFIX_PATHS.items():
        path = root / rel
        if path.is_file():
            records = logged(path, FORK)
            if records:
                data[name] = records
    path = root / "prefix-audit/prefix_pairs.csv"
    if path.is_file():
        # The audit table contains blank cells for expected but unobserved steps.
        for name in ("H", "M"):
            if name in data:
                continue
            records = {}
            for row in rows(path):
                if any(row[f"{key}_{name}"] for key in LOSS):
                    records[int(row["adamw_update"])] = {
                        k: float(row[f"{k}_{name}"] or "nan") for k in LOSS}
            if records:
                data[name] = records
    return data


def fork_data(root, data_dir):
    train, valid = defaultdict(dict), defaultdict(dict)
    load_pair(data_dir / "paired_updates_HH_HM.csv", ("HH", "HM"), train, valid)
    for row in rows(data_dir / "validation_points.csv"):
        case = row["trajectory"]
        if case == "M_prefix":
            case = "MM"  # Shared M prefix, evaluated in M precision.
        validation(valid, case, int(row["optimizer_update"]), row["mean_ce"], row["valid_tokens"], data_dir)
    load_runs(root, train, valid, None)
    audit = root / "prefix-audit/validation_gaps.csv"
    if audit.is_file():
        for row in rows(audit):
            for case in ("HH", "HM", "MH", "MM"):
                validation(valid, case, int(row["adamw_update"]), row[case], row["valid_tokens"], audit)
    return train, valid


def panels(title):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.7), constrained_layout=True)
    fig.suptitle(title, fontsize=15)
    flat = list(axes.flat)
    for ax, heading in zip(flat, ("Training  /  next-token CE", "Validation  /  next-token CE",
                                  "Training  /  total loss", "Training  /  router auxiliary loss")):
        ax.set_title(heading, loc="left", fontsize=12)
        ax.set_ylabel("Loss")
        ax.set_xlabel("AdamW update")
        ax.set_xlim(0, END + 4)
        ax.set_xticks([1, 64, 128, 192, FORK, 320, 384, 448, END])
        ax.grid(axis="y", alpha=.23)
    return fig, flat


def trace(ax, points, label, color, spacing, *, dashed=False):
    if not points:
        return
    steps, values = [], []
    previous = None
    for step in sorted(points):
        val = points[step]
        if previous is not None and step - previous > spacing:
            steps.append(float("nan"))
            values.append(float("nan"))
        steps.append(step)
        values.append(val if math.isfinite(val) else float("nan"))
        previous = step
    ax.plot(steps, values, color=color, label=label, linestyle="--" if dashed else "-",
            marker=".", markersize=3.3, linewidth=1.0, alpha=.93, clip_on=False)


def native_curves(axes, records, name, color, *, dashed=False):
    for ax, key in zip((axes[0], axes[2], axes[3]), LOSS):
        trace(ax, {s: r[key] for s, r in records.items()}, name, color, 5, dashed=dashed)


def save(fig, axes, path):
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles, labels, loc="best", fontsize=8, ncol=2)
        else:
            ax.text(.5, .5, "No measurements", transform=ax.transAxes, ha="center", color=".4")
    fig.savefig(path, dpi=155)
    plt.close(fig)
    print(f"Saved {path}")


def fork_plot(out, names, early_names, early, train, valid, filename):
    fig, axes = panels("MiniMind MoE  /  seed 42  /  " + filename.replace("_", " "))
    for ax in axes:
        ax.axvline(FORK, color=".5", linestyle=":", linewidth=.9)
    axes[0].set_title("Training CE  /  dashed: rank-0 microbatch; solid: global update", loc="left")
    color_map = dict(zip(names, COLORS))
    color_map["H"] = color_map["HH"]
    color_map["M"] = color_map["MH"]
    for name in early_names:
        if name in early:
            native_curves(axes, early[name], PREFIX_LABELS.get(name, name),
                          color_map[name], dashed=True)
    for case in names:
        color = color_map[case]
        trace(axes[0], {s: v[0] for s, v in train.get(case, {}).items()}, case, color, 1)
        trace(axes[1], {s: v[0] for s, v in valid.get(case, {}).items()}, case, color, 20)
    save(fig, axes, out / (filename + ".png"))


def pilot_plots(directories, grid, out):
    # One panel set for the five baselines, one per 4-bit projection ablation grid.
    catalog = defaultdict(list)
    missing = []
    for row in grid:
        scheme, family = row["训练方案"], row["基础格式"]
        base = scheme == family
        tag = scheme if scheme in ("bf16", "mxfp8") else scheme + "_r2"
        filename = f"pilot32k_s42_moe_{tag}.log"
        found = [d / filename for d in directories if (d / filename).is_file()]
        entries = []
        for path in found:
            data = logged(path, END)
            # Copies in the two training worktrees may refer to the same trajectory.
            if data and not any(old == data for _, old in entries):
                entries.append((path, data))
        if not entries:
            missing.append(scheme)
        names = [("baselines", FAMILIES[family])] if base else []
        if family not in ("bf16", "mxfp8"):
            names.append((family, "no fallback" if base else row["使用MXFP8的专家投影"]))
        for group, label in names:
            catalog[group].append((label, entries, float(row["本组验证CE"])))
    for group, entries in catalog.items():
        fig, axes = panels("MiniMind MoE  /  seed 42  /  " + FAMILIES.get(group, group))
        axes[0].set_title("Training CE  /  rank-0 logged microbatch", loc="left")
        axes[1].set_title("Validation CE  /  published own-format endpoint", loc="left")
        for i, (label, paths, end_ce) in enumerate(entries):
            color = COLORS[i % len(COLORS)]
            for path, data in paths:
                suffix = f" [{path.parent}]" if len(paths) > 1 else ""
                native_curves(axes, data, label + suffix, color)
            trace(axes[1], {END:end_ce}, f"{label} ({end_ce:.4f})", color, 1)
        save(fig, axes, out / f"pilot_{group}.png")
    if missing:
        print("Pilot training logs missing (only measured validation endpoints shown): " + ", ".join(missing))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--pilot-logs", type=Path, action="append", metavar="DIR",
                        help="Seed-42 pilot log directory (repeatable; default: both original worktrees)")
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, help="Defaults to RUN_ROOT/full-loss")
    args = parser.parse_args()
    if args.pilot_logs:
        for path in args.pilot_logs:
            if not path.is_dir():
                parser.error(f"Pilot log directory not found: {path}")
    if not args.run_root.is_dir():
        parser.error(f"Fork experiment directory not found: {args.run_root}")
    out = args.out or args.run_root / "full-loss"
    out.mkdir(parents=True, exist_ok=True)
    early = prefixes(args.run_root)
    train, valid = fork_data(args.run_root, args.data_dir)
    fork_plot(out, ["HH", "HM", "MH", "MM"], ["H", "M"], early, train, valid, "crossover")
    experts = [e for e in PREFIX_PATHS if e not in ("H", "M") and (e in early or e in train or e in valid)]
    if experts:
        fork_plot(out, ["HH", "MH", *experts], ["H", "M", *experts], early, train, valid, "experts")
    missing = [k for k in ("H", "M") if k not in early]
    if missing:
        print("No first-half measurements for: " + ", ".join(missing))
    for case in ("HH", "HM", "MH", "MM", *experts):
        steps = sorted(train.get(case, {}))
        print(f"{case}: {len(steps)} global training updates" + (f" ({steps[0]}–{steps[-1]})" if steps else ""))
    pilot_plots(args.pilot_logs or PILOT_DIRS, rows(args.data_dir / "moe_ffn_grid_seed42.csv"), out)


if __name__ == "__main__":
    main()
