"""Plot the last 60 AdamW updates of the published four-seed training Lce panel."""
import argparse
import csv
import math
from pathlib import Path

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from minimind_plot_lce import ARCHITECTURES, LOG_BATCHES, SEEDS, STYLES, chinese_font


START, END = 452, 512
UPDATES = sorted(batch // 4 for batch in LOG_BATCHES if START <= batch // 4 <= END)


def read_window(path):
    points = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            update = int(row["optimizer_update"])
            if not START <= update <= END:
                continue
            key = row["architecture"], int(row["seed"]), row["train_scheme"], update
            value = float(row["logits_loss"])
            if (key in points or update * 4 != int(row["microbatch"])
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"Invalid or duplicate training sample {key}")
            points[key] = value
    expected = {(arch, seed, scheme, u) for arch in ARCHITECTURES for seed in SEEDS
                for scheme in STYLES for u in UPDATES}
    if points.keys() != expected:
        raise ValueError("The late training window is incomplete or includes unexpected samples")
    return points


def render(points, arch, directory, font):
    # One scale per architecture enables direct comparisons across its four seeds.
    deltas = {(seed, scheme): [
        points[arch, seed, scheme, u] - points[arch, seed, "mxfp8", u] for u in UPDATES
    ] for seed in SEEDS for scheme in STYLES}
    limit = max(.005, 1.08 * max(abs(x) for values in deltas.values() for x in values))
    fig = Figure(figsize=(10.2, 7.05), dpi=145, facecolor="white")
    FigureCanvasAgg(fig)
    axes = fig.subplots(2, 2, sharex=True, sharey=True)
    fig.subplots_adjust(left=.08, right=.977, top=.765, bottom=.125, hspace=.31, wspace=.12)
    fig.text(.08, .934, f"{ARCHITECTURES[arch]} · 训练后段的格式差异", fontsize=19,
             fontproperties=font, color="#182631")
    fig.text(.08, .878, f"第 {START}–{END} 次更新  /  纵轴为各格式训练交叉熵减去同一步 MXFP8 的值（nat/token）",
             fontsize=10, fontproperties=font, color="#455260")
    for seed, ax in zip(SEEDS, axes.flat):
        ax.set_title(f"随机种子 {seed}", loc="left", pad=7, fontsize=12, fontproperties=font)
        for scheme, (label, color, style, marker) in STYLES.items():
            ax.plot(UPDATES, deltas[seed, scheme], label=label, color=color, linestyle=style,
                    linewidth=1.2, marker=marker, markersize=3.2, alpha=.95,
                    zorder=2 if scheme == "mxfp8" else 3)
        ax.set_xlim(START, END + 1)
        ax.set_xticks((455, 465, 475, 485, 495, 505, 512))
        ax.set_ylim(-limit, limit)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, symmetric=True))
        ax.grid(axis="y", linewidth=.7, color="#DDE3E8")
        ax.set_axisbelow(True)
        ax.tick_params(axis="both", length=0, labelsize=9, colors="#455260")
        for edge in ("top", "left", "right"):
            ax.spines[edge].set_visible(False)
        ax.spines["bottom"].set_color("#A4B0BA")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(.078, .857), frameon=False,
               ncol=5, borderaxespad=0, fontsize=9, columnspacing=1.8)
    fig.text(.52, .062, "AdamW 更新步", ha="center", fontsize=10, fontproperties=font)
    fig.text(.08, .015, f"各面板取相同的 {len(UPDATES)} 个 rank 0 日志采样点，负值表示该步 Lce 低于 MXFP8。",
             fontsize=9, fontproperties=font, color="#455260")
    target = directory / f"zoom_{arch}.png"
    fig.savefig(target, dpi=145, metadata={"Software": "MiniMind"})
    fig.clear()
    return target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--font", type=Path)
    args = p.parse_args()
    points = read_window(args.csv)
    font = chinese_font(args.font)
    args.out.mkdir(parents=True, exist_ok=True)
    for arch in ARCHITECTURES:
        print(render(points, arch, args.out, font))


if __name__ == "__main__":
    main()
