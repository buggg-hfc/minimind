"""Extract the 40 original MiniMind training CE traces and optionally plot them (CPU only)."""
import argparse
import csv
import math
import os
from pathlib import Path
import re
import shlex
import tempfile

SEEDS = (42, 43, 100, 2026)
ARCHITECTURES = ("dense", "moe")
SCHEMES = ("bf16", "mxfp8", "hif4_thr_3mm", "nvfp4_3mm", "nvint4_3mm")
TICKS = (*range(20, 2048, 20), 2048)
FIELDS = ("seed", "architecture", "train_scheme", "microbatch", "optimizer_update", "logits_loss")
PILOT_LOGS = Path("runs/v100-pilot/logs")
GRID = Path("/data/lz/minimind-five-format-grid")
HISTORICAL = Path("/home/lz/minimind") / PILOT_LOGS
PAIRED = Path("/data/lz/minimind-moe-paired-seeds/work") / PILOT_LOGS
RECORD = re.compile(r"Epoch:\[1/1\]\((\d+)/2048\),\s*loss:\s*[^,]+,\s*"
                    r"logits_loss:\s*([^,]+),\s*aux_loss:\s*[^,]+,\s*lr:\s*([^,\s]+)")
RESUME = re.compile(r"跳过前\s*(\d+)\s*个step")


def tag(scheme):
    return scheme + "_r2" if scheme.endswith("_3mm") else scheme


def filename(seed, arch, scheme):
    return f"pilot32k_s{seed}_{arch}_{tag(scheme)}.log"


def check_command(line, seed, arch, scheme):
    args = shlex.split(line)
    # Reject fork/perturbation entry points even if their native trainer arguments match.
    ddp = args.index("torch.distributed.run")
    trainer = next((i for i in range(ddp + 1, len(args)) if not args[i].startswith("-")), None)
    if trainer is None or Path(args[trainer]).name != "train_pretrain.py":
        raise ValueError("日志启动的是其他实验入口")
    opts = {}
    for i, token in enumerate(args):
        if token.startswith("--"):
            name, eq, value = token.partition("=")
            opts[name] = value if eq else (args[i + 1] if i + 1 < len(args) else "")
    expected = {"--nproc-per-node": "8", "--nnodes": "1", "--use_moe": str(int(arch == "moe")),
                "--epochs": "1", "--seed": str(seed), "--save_weight": f"pilot32k_s{seed}",
                "--quant_sim": scheme, "--hidden_size": "768", "--num_hidden_layers": "8",
                "--max_seq_len": "512", "--batch_size": "2", "--accumulation_steps": "4",
                "--log_interval": "20", "--from_weight": "none"}
    for key, value in expected.items():
        if opts.get(key) != value:
            raise ValueError(f"训练参数 {key} 应为 {value}，实际为 {opts.get(key)}")
    if float(opts.get("--learning_rate", "nan")) != 0.0005 or opts.get("--from_resume") not in ("0", "1"):
        raise ValueError("学习率或断点参数与历史 pilot 不符")
    return opts["--from_resume"] == "1"


def read_log(path, seed, arch, scheme):
    records = {}
    command_seen = resumed = False
    try:
        with path.open(encoding="utf-8", errors="replace") as handle:
            for number, line in enumerate(handle, 1):
                if "torch.distributed.run" in line and "--quant_sim" in line:
                    resumed = check_command(line, seed, arch, scheme)
                    command_seen = True
                    if not resumed:
                        records.clear()
                    continue
                cursor = RESUME.search(line)
                if cursor:
                    step = int(cursor[1])
                    if not resumed or not 0 < step <= 2048 or step % 4:
                        raise ValueError(f"断点游标异常 {step}")
                    records = {k: v for k, v in records.items() if k <= step}
                    resumed = False
                    continue
                if "Epoch:[" not in line:
                    continue
                match = RECORD.search(line)
                if not command_seen or not match:
                    raise ValueError("缺少完整 pilot 启动头，或训练日志字段与预期不同")
                step = int(match[1])
                if resumed:
                    # With no saved checkpoint, the native trainer starts afresh.
                    if step != TICKS[0]:
                        raise ValueError("续训没有记录断点游标")
                    records.clear()
                    resumed = False
                loss, lr = map(float, match.groups()[1:])
                expected_lr = 0.0005 * (0.1 + 0.45 * (1 + math.cos(math.pi * step / 2048)))
                if step not in TICKS or not math.isfinite(loss) or loss < 0:
                    raise ValueError(f"无效的训练记录 microbatch={step}, logits_loss={loss}")
                if not math.isclose(lr, expected_lr, rel_tol=0, abs_tol=5.01e-9):
                    raise ValueError(f"microbatch={step} 的学习率 {lr} 与原余弦日程不符")
                if step in records and records[step] != loss:
                    raise ValueError(f"microbatch={step} 有未注明重启的冲突记录")
                records[step] = loss
    except (ValueError, IndexError) as exc:
        raise ValueError(f"{path}:{number}: {exc}") from exc
    missing = sorted(set(TICKS) - records.keys())
    if missing:
        raise ValueError(f"{path}: 缺少 {len(missing)} 个采样点，首个缺失 microbatch={missing[0]}")
    return records


def grid_sources(report):
    by_run = {}
    with report.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            seed, arch, scheme = int(row["seed"]), row["architecture"], row["train_scheme"]
            if seed not in SEEDS or arch not in ARCHITECTURES or scheme not in SCHEMES:
                continue
            key = (seed, arch, scheme)
            if (key in by_run or not row.get("source_work") or row["eval_scheme"] != scheme
                    or int(row["optimizer_updates"]) != 512 or int(row["valid_tokens"]) != 236442):
                raise ValueError(f"{report}: 运行条目不符 {key}")
            by_run[key] = Path(row["source_work"]) / PILOT_LOGS
    return by_run


def collect(log_dirs=None, historical_logs=HISTORICAL, grid=GRID):
    report = grid / "same_format.csv"
    sources = grid_sources(report) if not log_dirs and report.is_file() else {}
    rows, errors = [], []
    for seed in SEEDS:
        for arch in ARCHITECTURES:
            for scheme in SCHEMES:
                key = (seed, arch, scheme)
                name = filename(*key)
                if log_dirs:
                    dirs = log_dirs
                elif seed == 42:
                    dirs = [historical_logs]
                elif report.is_file():
                    dirs = [sources[key]] if key in sources else []
                else:
                    dirs = [grid / "work" / PILOT_LOGS, PAIRED]
                paths = sorted({p.resolve() for folder in dirs if (p := folder / name).is_file()})
                if not paths:
                    errors.append(f"{name}: 未找到（来源目录 {', '.join(map(str, dirs)) or str(report)}）")
                    continue
                try:
                    trace = read_log(paths[0], *key)
                    for other in paths[1:]:
                        if read_log(other, *key) != trace:
                            raise ValueError(f"{paths[0]} 与 {other} 有不同的训练轨迹，请明确来源")
                except ValueError as exc:
                    errors.append(str(exc))
                    continue
                rows.extend(dict(seed=seed, architecture=arch, train_scheme=scheme, microbatch=m,
                                 optimizer_update=m // 4, logits_loss=trace[m]) for m in TICKS)
    if errors:
        raise ValueError("\n".join(errors))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-logs", default=HISTORICAL, type=Path, help="原始 seed 42 日志目录")
    parser.add_argument("--grid", default=GRID, type=Path, help="多 seed 网格输出根目录")
    parser.add_argument("--log-dir", action="append", type=Path, help="直接指定日志目录。可重复，替代默认查找")
    parser.add_argument("--out", type=Path, required=True, help="training_lce.csv 与可选图像的目录")
    parser.add_argument("--plot", action="store_true", help="同时绘制八张中文 PNG")
    parser.add_argument("--font", type=Path, help="显式选择中文字体文件")
    args = parser.parse_args()
    try:
        rows = collect(args.log_dir, args.historical_logs, args.grid)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"{exc}\n")
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=args.out,
                                     delete=False) as f:
        temp = Path(f.name)
        try:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.replace(temp, args.out / "training_lce.csv")
        finally:
            temp.unlink(missing_ok=True)
    print(f"{len(rows)} 个已记录的 Lce 采样点，40 组训练：{args.out / 'training_lce.csv'}")
    if args.plot:
        from minimind_plot_lce import render_curves
        render_curves(rows, args.out, font_path=args.font)


if __name__ == "__main__":
    main()
