"""Render the four-seed MiniMind training log panel. Matplotlib is needed only for rendering."""
import argparse
from collections import defaultdict
import csv
import math
import os
from pathlib import Path
from tempfile import NamedTemporaryFile


SEEDS = (42, 43, 100, 2026)
LOG_BATCHES = frozenset((*range(20, 2048, 20), 2048))
CSV_FIELDS = {"seed", "architecture", "train_scheme", "microbatch", "optimizer_update", "logits_loss"}
ARCHITECTURES = {"dense": "稠密模型（Dense）", "moe": "混合专家模型（MoE）"}
# Keep the format's color, line pattern and marker consistent across all eight figures.
STYLES = {
    "bf16": ("BF16", "#343D49", ":", "s"),
    "mxfp8": ("MXFP8", "#0072B2", "-", "o"),
    "hif4_thr_3mm": ("HiF4 THR 3MM", "#BA487E", "--", "^"),
    "nvfp4_3mm": ("NVFP4 3MM", "#C85100", "-.", "D"),
    "nvint4_3mm": ("NVINT4 3MM", "#00866C", "-", "v"),
}
X_LABEL = "AdamW 更新步"
CE_LABEL = "训练交叉熵（nat/token）"
DIFF_LABEL = "交叉熵差值（nat/token）"
DIFF_TITLE = "相对 MXFP8 的差值"
FOOTNOTE = "每个点取 rank 0 当前 microbatch 的 logits_loss。同一 microbatch 配对求差，负值表示低于 MXFP8。"


def chinese_font(font_path):
    from matplotlib import font_manager
    from matplotlib.ft2font import FT2Font

    required_text = "".join(ARCHITECTURES.values()) + X_LABEL + CE_LABEL + DIFF_LABEL + DIFF_TITLE + FOOTNOTE + "随机种子"
    required = {ord(c) for c in required_text if ord(c) > 127}
    if font_path:
        candidates = [Path(font_path).expanduser()]
    else:
        def preference(entry):
            name = entry.name.lower()
            preferred = ("noto sans cjk sc", "source han sans", "noto sans cjk", "wenquanyi",
                         "microsoft yahei", "simhei", "pingfang")
            priority = next((i for i, text in enumerate(preferred) if text in name), len(preferred))
            regular = entry.weight == "normal" or entry.weight == 400
            return priority, not regular, entry.style != "normal", entry.fname

        candidates = (Path(entry.fname) for entry in sorted(font_manager.fontManager.ttflist, key=preference))
    for path in dict.fromkeys(candidates):
        if path.is_file():
            try:
                supported = FT2Font(str(path)).get_charmap()
            except (OSError, RuntimeError, ValueError):
                continue
            if required.issubset(supported):
                return font_manager.FontProperties(fname=str(path))
    raise ValueError("未找到能显示全部中文标注的字体。请使用 --font 指定 NotoSansCJK-Regular.ttc 或其他中文字体。")


def render_curves(rows, output_dir, font_path=None):
    """Write seed{seed}_{dense|moe}.png and return the eight paths in seed order."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.ticker import MaxNLocator

    panel = defaultdict(dict)
    for row in rows:
        seed, arch, scheme = int(row["seed"]), row["architecture"], row["train_scheme"]
        if seed not in SEEDS or arch not in ARCHITECTURES or scheme not in STYLES:
            raise ValueError(f"Unexpected training run: {seed}/{arch}/{scheme}")
        batch, update, ce = int(row["microbatch"]), int(row["optimizer_update"]), float(row["logits_loss"])
        series = panel[seed, arch, scheme]
        if (batch in series or batch not in LOG_BATCHES or update != batch // 4
                or not math.isfinite(ce) or ce < 0):
            raise ValueError(f"Invalid or duplicate log point: {seed}/{arch}/{scheme}/{batch}")
        series[batch] = (update, ce)

    loss_min, loss_max, max_diff = math.inf, -math.inf, 0.0
    for seed in SEEDS:
        for arch in ARCHITECTURES:
            reference = panel[seed, arch, "mxfp8"]
            for scheme in STYLES:
                samples = panel[seed, arch, scheme]
                if samples.keys() != LOG_BATCHES or reference.keys() != LOG_BATCHES:
                    raise ValueError(f"Missing logged microbatches for {seed}/{arch}/{scheme} or its MXFP8 reference")
                for batch, (update, ce) in samples.items():
                    ref_ce = reference[batch][1]
                    loss_min, loss_max = min(loss_min, ce), max(loss_max, ce)
                    max_diff = max(max_diff, abs(ce - ref_ce))

    font = chinese_font(font_path)
    pad = max((loss_max - loss_min) * 0.04, 0.01)
    diff_limit = max(max_diff * 1.06, 0.001)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for seed in SEEDS:
        for arch, title in ARCHITECTURES.items():
            fig = Figure(figsize=(10, 6.7), dpi=160, facecolor="#FFFFFF")
            FigureCanvasAgg(fig)
            grid = fig.add_gridspec(2, 1, left=.104, right=.973, top=.82, bottom=.147,
                                    hspace=.29, height_ratios=(2.1, 1))
            loss_ax = fig.add_subplot(grid[0])
            delta_ax = fig.add_subplot(grid[1], sharex=loss_ax)
            reference = panel[seed, arch, "mxfp8"]
            positions = sorted(reference)
            xs = [reference[batch][0] for batch in positions]
            for ax in (loss_ax, delta_ax):
                ax.set_axisbelow(True)
                ax.grid(axis="y", linewidth=.65, color="#DAE0E5")
                for edge in ("right", "top", "left"):
                    ax.spines[edge].set_visible(False)
                ax.spines["bottom"].set_color("#ABB4BD")
                ax.tick_params(axis="both", labelsize=9, length=0, pad=5, colors="#414B55")
            for scheme, (label, color, style, marker) in STYLES.items():
                ys = [panel[seed, arch, scheme][batch][1] for batch in positions]
                line = dict(label=label, color=color, linestyle=style, marker=marker, markersize=2.5,
                            linewidth=1.2, alpha=.91)
                loss_ax.plot(xs, ys, **line)
                if scheme != "mxfp8":
                    delta_ax.plot(xs, [y - reference[b][1] for b, y in zip(positions, ys)], **line)
            delta_ax.axhline(0, color=STYLES["mxfp8"][1], linewidth=1.05)
            loss_ax.tick_params(labelbottom=False)
            loss_ax.set_ylim(max(0, loss_min - pad), loss_max + pad)
            loss_ax.set_ylabel(CE_LABEL, fontproperties=font, fontsize=11)
            delta_ax.set_ylim(-diff_limit, diff_limit)
            delta_ax.yaxis.set_major_locator(MaxNLocator(nbins=4, symmetric=True))
            delta_ax.set_ylabel(DIFF_LABEL, fontproperties=font, fontsize=10)
            delta_ax.set_xlabel(X_LABEL, fontproperties=font, fontsize=11, labelpad=7)
            delta_ax.set_title(DIFF_TITLE, fontproperties=font, fontsize=10, loc="left", color="#414B55", pad=5)
            delta_ax.set_xlim(0, 516)
            delta_ax.set_xticks(range(0, 513, 64))
            fig.text(.104, .944, f"MiniMind {title}", fontproperties=font, fontsize=20, color="#182631")
            fig.text(.973, .945, f"随机种子 {seed}", ha="right", fontproperties=font, fontsize=14, color="#414B55")
            fig.legend(*loss_ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.103, .918),
                       ncol=5, frameon=False, borderaxespad=0, handlelength=2.35, columnspacing=1.8,
                       fontsize=9.5)
            fig.text(.104, .031, FOOTNOTE, fontproperties=font, fontsize=9, color="#505963")
            path = directory / f"seed{seed}_{arch}.png"
            with NamedTemporaryFile(dir=directory, suffix=".png", delete=False) as tmp:
                tmp_path = Path(tmp.name)
            try:
                fig.savefig(tmp_path, format="png")
                os.replace(tmp_path, path)
            finally:
                tmp_path.unlink(missing_ok=True)
                fig.clear()
            paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser(description="从 training_lce.csv 绘制四个随机种子的八张训练交叉熵图。")
    parser.add_argument("--csv", required=True, type=Path, help="原始日志采样表 training_lce.csv")
    parser.add_argument("--out", required=True, type=Path, help="输出八张 PNG 的目录")
    parser.add_argument("--font", type=Path, help="可选的中文字体路径")
    args = parser.parse_args()
    try:
        with args.csv.open(encoding="utf-8-sig", newline="") as stream:
            rows = csv.DictReader(stream)
            missing = CSV_FIELDS.difference(rows.fieldnames or ())
            if missing:
                raise ValueError("CSV 缺少字段 " + ", ".join(sorted(missing)))
            paths = render_curves(rows, args.out, args.font)
    except (ImportError, OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print("\n".join(str(path) for path in paths))


if __name__ == "__main__":
    main()
