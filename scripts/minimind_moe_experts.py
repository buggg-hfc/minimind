"""Aggregate existing MiniMind MoE fork traces on CPU by layer and expert."""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np


DEFAULT_PAIR = Path("/data/lz/minimind-moe-fork-s42/pair-1024-to-2048")
BRANCHES = ("hif4", "hif4_down_mxfp8")  # A, B: late expert down_proj policy.
VECTORS = (("down_grad", "clipped_grad"), ("down_delta", "adamw_update"))


def _paired(a, b, key):
    value = a[key]
    if not np.array_equal(value, b[key]):
        raise ValueError(f"分支的 {key} 不一致")
    return value


def _routes(a, b, rank, world, update):
    for key, expected in (("rank", rank), ("world", world), ("update", update)):
        if _paired(a, b, key).item() != expected:
            raise ValueError(f"文件内容的 {key} 与路径不匹配")
    if not _paired(a, b, "input_sha256").item():
        raise ValueError("缺少训练输入指纹")
    counts = _paired(a, b, "micro_valid")
    shapes = _paired(a, b, "micro_shapes")
    experts = int(_paired(a, b, "expert_count").item())
    ar, br = a["routes"], b["routes"]
    if (ar.shape != br.shape or ar.ndim != 3 or ar.shape[0] == 0 or not 1 <= ar.shape[2] <= experts
            or counts.ndim != 1 or shapes.shape != (counts.size, 2) or np.any(counts < 0)
            or np.any(shapes[:, 0] < 1) or np.any(shapes[:, 1] < 2)
            or np.any(counts > shapes[:, 0]*(shapes[:, 1]-1)) or ar.shape[1] != counts.sum()):
        raise ValueError("路由维度或有效预测 token 数不匹配")
    for routes in (ar, br):
        if (routes.dtype.kind not in "iu" or (routes.size and (routes.min() < 0 or routes.max() >= experts))
                or np.any(routes[..., 1:] <= routes[..., :-1])):
            raise ValueError("路由结果应为排序且互斥的有效专家编号")
    return ar, br, experts


def _weights(a, b, shape, moments, identity):
    indices = _paired(a, b, "down_indices")
    shapes = _paired(a, b, "down_shapes")
    seed = int(_paired(a, b, "sample_seed").item())
    if (indices.ndim != 3 or indices.shape[:2] != shape or indices.shape[-1] < 1
            or shapes.shape != (*shape, 2) or indices.dtype.kind not in "iu" or np.any(shapes <= 0)
            or np.any(indices < 0) or np.any(indices[..., -1] >= shapes.prod(axis=2))
            or np.any(indices[..., 1:] <= indices[..., :-1])):
        raise ValueError("down_proj 固定采样坐标无效")
    if identity is not None:
        if identity[0] != seed or not np.array_equal(identity[1], indices) or not np.array_equal(identity[2], shapes):
            raise ValueError("不同更新的 down_proj 采样坐标或权重形状不一致")
    else:
        identity = (seed, indices, shapes)
    for j, (key, _) in enumerate(VECTORS):
        x, y = a[key].astype(np.float64), b[key].astype(np.float64)
        if x.shape != indices.shape or y.shape != x.shape or not (np.isfinite(x).all() and np.isfinite(y).all()):
            raise ValueError(f"无效的 {key} 采样值")
        for i, terms in enumerate((x*x, y*y, x*y, (y-x)**2)):
            moments[j, i] += terms.sum(axis=-1)
    return identity


def _fraction(n, d):
    return n/d if d else None


def analyze(pair=DEFAULT_PAIR, out=None):
    pair = Path(pair)
    dest = (Path(out) if out is not None else pair / "expert-analysis") / "experts.csv"
    with (pair / "summary.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) < 2 or [int(r["relative_update"]) for r in rows] != list(range(len(rows))):
        raise ValueError("summary.csv 中的更新编号不连续")
    steps = len(rows)-1
    for file, field in (("continuation.json", "target_update"), ("second_half.json", "relative_updates"),
                        ("experiment.json", "settings")):
        if (pair / file).is_file():
            config = json.loads((pair / file).read_text(encoding="utf-8"))[field]
            expected = config["optimizer_updates"] if field == "settings" else config
            if expected != steps:
                raise ValueError(f"summary.csv 覆盖 {steps} 次更新，运行配置要求 {expected} 次")
            break
    world = int(rows[1]["world_size"])
    if world < 1:
        raise ValueError("无效的 world_size")
    layout = samples = None
    tokens = 0
    for update, row in enumerate(rows[1:], 1):
        n = changed = 0
        if (int(row["world_size"]), int(row["route_paired_ranks"])) != (world, world):
            raise ValueError(f"更新 {update} 缺少完整的 DDP 配对")
        for rank in range(world):
            rel = Path("metrics", f"rank-{rank:02d}", f"update-{update:06d}.npz")
            with (np.load(pair/BRANCHES[0]/rel, allow_pickle=False) as a,
                  np.load(pair/BRANCHES[1]/rel, allow_pickle=False) as b):
                ar, br, experts = _routes(a, b, rank, world, update)
                current = (ar.shape[0], experts, ar.shape[-1])
                if layout is None:
                    layout = current
                    counts = np.zeros((3, *layout[:2]), dtype=np.int64)
                    moments = np.zeros((len(VECTORS), 4, *layout[:2]), dtype=np.float64)
                if layout != current:
                    raise ValueError(f"rank {rank} 更新 {update} 的层数、专家数或 top-k 改变")
                n += ar.shape[1]
                changed += int(np.count_nonzero(np.any(ar != br, axis=-1)))
                for expert in range(experts):
                    x, y = (ar == expert).any(axis=-1), (br == expert).any(axis=-1)
                    counts[0, :, expert] += x.sum(axis=1)
                    counts[1, :, expert] += y.sum(axis=1)
                    counts[2, :, expert] += (x & y).sum(axis=1)
                if rank == 0:
                    samples = _weights(a, b, layout[:2], moments, samples)
        if ((n, changed, layout[0]*n, samples[1].size) != tuple(int(row[k]) for k in
                ("train_valid_tokens", "route_changed", "route_layer_token_observations", "down_sampled_coordinates"))):
            raise ValueError(f"更新 {update} 的原始数据与 summary.csv 统计不一致")
        tokens += n
    result = []
    for layer, expert in np.ndindex(layout[:2]):
        a, b, both = (int(c) for c in counts[:, layer, expert])
        lost, gained = a-both, b-both
        union = a+b-both
        record = dict(layer=layer, expert=expert, optimizer_updates=steps, top_k=layout[2],
                      valid_token_positions=tokens, load_a=a, load_b=b, load_fraction_a=_fraction(a, tokens),
                      load_fraction_b=_fraction(b, tokens), lost=lost, gained=gained, union=union,
                      membership_changed=lost+gained, turnover=_fraction(lost+gained, union),
                      sampled_coordinates=samples[1].shape[2])
        for j, (_, label) in enumerate(VECTORS):
            aa, bb, ab, diff = moments[j, :, layer, expert]
            norm_a, norm_b = math.sqrt(aa), math.sqrt(bb)
            cos = _fraction(ab, norm_a*norm_b)
            record.update({label+"_norm_a": norm_a, label+"_norm_b": norm_b,
                           label+"_difference_sq_sum": diff, label+"_relative_l2": _fraction(math.sqrt(diff), norm_a),
                           label+"_cosine": None if cos is None else float(np.clip(cos, -1., 1.))})
        update_diff = record["adamw_update_difference_sq_sum"]
        record["adamw_update_layer_mismatch_share"] = _fraction(update_diff, moments[1, 3, layer].sum())
        result.append(record)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".experts-", dir=dest.parent)
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result[0].keys())
            writer.writeheader()
            writer.writerows(result)
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)
    print(f"{dest}  |  {steps} 次更新，{layout[0]} 层 × {layout[1]} 专家，{tokens} 个有效预测位置。")
    print("A/B 分别为后半程 down_proj 使用 HiF4/MXFP8。负载分母为有效位置数。lost/gained 按 A→B 计；"
          "turnover 为改变归属的位置数 ÷ 至少一侧选择该专家的位置数。")
    print(f"每个专家固定抽取 {samples[1].shape[2]} 个 down_proj 坐标，梯度和实际更新跨步拼接计算。"
          "层内贡献为该专家更新差平方和占本层采样总和的比例。零分母留空。")
    return dest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair", nargs="?", type=Path, default=DEFAULT_PAIR, help="含 summary.csv 的历史分支目录")
    parser.add_argument("--out", type=Path, help="汇总输出目录（默认 <pair>/expert-analysis）")
    args = parser.parse_args()
    analyze(args.pair, args.out)
