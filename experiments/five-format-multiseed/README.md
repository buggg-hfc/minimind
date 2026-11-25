# MiniMind 五种训推格式的多 seed 绝对验证 CE

本实验在 MiniMind-3 Dense 和 MoE 上从头训练。每次运行完成 512 次 AdamW 更新，在同一份 1,024 篇文档、236,442 个有效预测 token 的验证集上按训练格式计算平均 next-token 交叉熵，单位为 nat/token，数值越低越好。CE 按有效预测 token 加权，不含路由辅助损失。每次运行的训练量为 7,427,366 个有效预测 token。

## 40 次运行的终点

[完整精度 CSV](same_format_validation.csv) 每行对应一次独立训练的终点评估，保留原始 CE 和 PPL（PPL = exp(CE)）。下表保留六位小数。HiF4 为 thr 版本，NVINT4 为整数格式，三种 4-bit 均使用 R2 残差 3MM。MXFP8 指所有可切换投影均采用 MXFP8。

| 架构 | Seed | MXFP8 | BF16 | HiF4 3MM | NVFP4 3MM | NVINT4 3MM |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 42 | 4.329319 | 4.305833 | 4.316963 | 4.290002 | 4.281307 |
| Dense | 43 | 4.309593 | 4.285785 | 4.297457 | 4.297352 | 4.283074 |
| Dense | 100 | 4.230604 | 4.203001 | 4.225828 | 4.233132 | 4.222616 |
| Dense | 2026 | 4.315347 | 4.253925 | 4.279694 | 4.262388 | 4.284923 |
| MoE | 42 | 4.295454 | 4.263755 | 4.330955 | 4.316045 | 4.325956 |
| MoE | 43 | 4.355023 | 4.383419 | 4.250588 | 4.391722 | 4.310518 |
| MoE | 100 | 4.319434 | 4.339387 | 4.305688 | 4.361529 | 4.308389 |
| MoE | 2026 | 4.260391 | 4.277753 | 4.282136 | 4.265992 | 4.311837 |

seed 42 的十条数据来自[历史 pilot 终点评估](../moe-fp4-seed42/dense_moe_pilot_endpoints.csv)中训练格式等于评估格式的记录。三种 4-bit 取 R2 完整训练。seed 43、100、2026 来自五格式网格导出的 `same_format.csv`。两种来源使用同一验证切分，SHA256 为 `72bb2daa3e43457bebd118119f335fcebf6433077ab56ae2ab6f6f9f28f60a04`。网格代码版本为 [56723b2](https://github.com/buggg-hfc/minimind/blob/56723b230647242a1cecd271e7edd629ca1bb84a/scripts/minimind_multiseed_grid.py)。

### MXFP8 基线和 BF16 参照

下表先在相同架构和 seed 内计算「候选 CE 减参照 CE」，再对四个差值取算术平均。小于零表示候选格式平均 CE 更低。

| 架构 | 4-bit 格式 | 平均 ΔCE 对 MXFP8 | 平均 ΔCE 对 BF16 | CE 低于 BF16 的 seed 数 |
|---|---|---:|---:|---:|
| Dense | HiF4 3MM | -0.016230 | +0.017850 | 0/4 |
| Dense | NVFP4 3MM | -0.025497 | +0.008583 | 1/4 |
| Dense | NVINT4 3MM | -0.028236 | +0.005844 | 2/4 |
| MoE | HiF4 3MM | -0.015234 | -0.023737 | 2/4 |
| MoE | NVFP4 3MM | +0.026246 | +0.017743 | 1/4 |
| MoE | NVINT4 3MM | +0.006600 | -0.001903 | 2/4 |

MoE 的 MXFP8 在 seed 2026 得到自身四次中的最低 CE（4.260391），且在该 seed 的五种格式中排名第一。seed 42 的 MXFP8 为 4.295454，BF16 为 4.263755。四个 seed 上 MXFP8 优于 BF16 三次，平均 CE 差为 −0.008503。HiF4 在 seed 43 取得较大的对 BF16 优势（−0.132831），四个 seed 中有两次优于 BF16。NVFP4 在四次中均落后于 MXFP8，在其中三次也落后于 BF16。四个配对结果可用于描述本轮的波动；基线在其他种子中的分布仍有不确定性。

## 第 6 层专家 1 和 2 的选择与干预

模型的层号和专家号都从 0 开始；L6E1 和 L6E2 位于第七个 Transformer 块。单专家运行器在运行前固定了候选与顺序 `(2, 1)`。现存代码和实验材料没有记载在 8 层中挑出层 6 的阈值、随机抽样过程或全网筛选公式。从已记录的实验设计能确认的是在同一层比较两位专家。[固定候选的运行器](https://github.com/buggg-hfc/minimind/blob/eb822233c32f964177b1c0f05eafdf579920e3e7/scripts/moe_down_crossover/expert_runs.py#L60-L69) · [模块精度配置](../../scripts/moe_down_crossover/expert_prefix.py)

已有的 seed 42 后半程诊断说明这对专家适合检验什么。HH 和 HM 从同一 HiF4 中点分叉，后 256 次更新里 HH 的专家 down 保持 HiF4，HM 的专家 down 使用 MXFP8。下表给出训练 token 的路由负载，以及两条轨迹在每位专家 1,024 个固定 down 权重坐标上的实际 AdamW 更新分歧。

| 专家 | HH 负载 | HM 负载 | 更新相对 L2 分歧 | 更新余弦 |
|---|---:|---:|---:|---:|
| L6E1 | 24.90% | 24.82% | 0.2421（层内最低） | 0.9708 |
| L6E2 | 25.31% | 25.34% | 0.3071（层内最高） | 0.9530 |

两者在 HH 中的负载相差约 0.40 个百分点，在同一层内覆盖了较低与较高的相对更新分歧。这给出了一个负载接近的探索性对照。相对 L2 为跨步采样更新差的 L2 范数除以 HH 更新范数，覆盖后半程 3,715,155 个配对有效位置。[32 位专家的完整统计](../moe-fp4-seed42/expert_updates_HH_HM.csv) · [字段与分母](../moe-fp4-seed42/COLUMNS.md)

在后续单专家干预中，前 256 次更新仅将目标专家的 `down_proj` 改为 MXFP8，后 256 次及终点评估统一使用 HiF4 策略。seed 42 的终点 CE 为 L6E1 **4.262189**、L6E2 **4.303534**；全程 HiF4 的分叉对照 HH 为 **4.330909**，前半程全部专家 down 用 MXFP8 的对照 MH 为 **4.264549**。本组中，后半程诊断里相对更新分歧较小的 E1，其前半程干预带来的终点 CE 改善反而较大。这些干预只覆盖一个 seed、两个单专家。40 组主表的 seed 42 MoE HiF4 取历史完整训练的 **4.330955**，与分叉对照 HH 分开记录。[四组干预原值](expert_interventions_seed42.csv)

## 训练终点的权重直方图 KL

在保有终点权重的八卡 V100 训练机上，从包含本脚本的仓库目录运行下面的命令。计算使用 CPU，可以与后续 GPU 实验分开运行。

```bash
/root/anaconda3/envs/lz/bin/python scripts/minimind_multiseed_weight_kl.py \
  --grid-output /data/lz/minimind-five-format-grid \
  --legacy-root /home/lz/minimind \
  --seeds 42 43 100 2026 --threads 4
```

默认同时处理 Dense 与 MoE 的全部五种方案。程序按网格 CSV 的 `source_work` 查找已评估的权重并支持历史 seed 42 的 R2 导出。每一组取相同架构、相同 seed 的 BF16 训练终点作为参照，直接比较训练器保存的 FP32 权重。权重 KL 的计算只依赖模型参数。输出中的验证 CE 和 PPL 来自已有终点评估。

| 文件（默认在 `--grid-output` 的 `weight_kl/` 子目录） | 内容 |
|---|---|
| `weight_kl_summary.csv` | 每个模型一行。主指标 `weight_kl_forward` 为 BF16 → 本组的权重直方图 KL。另给出反向 KL、Jeffreys 双向之和、参与统计的张量数和参数数、两组检查点路径与同格式验证 CE/PPL。 |
| `weight_kl_by_tensor.csv` | 每个权重张量一行，带模块名、层号、专家号、参数数与双向 KL。可以从这里单看专家的 `down_proj`，或按组件以 `elements` 加权。 |

对每个同名权重张量，使用两组权重的联合最小值与最大值构造 128 个等宽桶，保留正负号，每个桶加 0.5 再归一化。以自然对数计算 KL，再按每个张量的元素数加权得到全模型指标。统计包含嵌入、归一化、注意力、FFN 和 MoE 路由器，共享的输入嵌入与输出头计一次。BF16 自比较应为 0。同形张量即使逐坐标不同，也可能有相同的值分布；解释模型行为可结合独立的验证 CE 或下一 token KL。

若单独跑某个子集，可用 `--seeds 42 --architectures moe --output /data/lz/minimind-five-format-grid/weight_kl_s42_moe` 保留独立输出。完整的四个 seed 预期有 40 行模型汇总，其中 BF16 自比较占 8 行。

### 四个 seed 的终点权重 KL 实测

[40 组逐次测量](weight_kl_summary.csv)包含正向、反向和 Jeffreys 权重 KL、同格式验证 CE/PPL，以及对应的 BF16 验证 CE/PPL。训练与评估配置和上文相同。为便于共享，两个检查点字段保留导出文件名。Dense 覆盖 90 个权重张量、63,912,192 个元素；MoE 覆盖 170 个权重张量、198,416,640 个元素。8 个 BF16 自比较在三个 KL 列上均为 0。

下表按 seed 42、43、100、2026 等权取平均。正向 KL 的参照是同 seed、同架构的 BF16。两个计数列分别比较候选与同 seed 的 MXFP8。ΔCE 为候选验证 CE 减去 MXFP8 验证 CE。

| 架构 | 格式 | 平均正向权重 KL（×10⁻⁴ nat） | KL 低于 MXFP8 的 seed 数 | 平均 ΔCE | CE 低于 MXFP8 的 seed 数 |
|---|---|---:|---:|---:|---:|
| Dense | MXFP8 | 1.4256 | — | 0 | — |
| Dense | HiF4 3MM | 1.3369 | 4/4 | −0.016230 | 4/4 |
| Dense | NVINT4 3MM | 1.3147 | 4/4 | −0.028236 | 4/4 |
| Dense | NVFP4 3MM | 1.3436 | 4/4 | −0.025497 | 3/4 |
| MoE | MXFP8 | 1.8791 | — | 0 | — |
| MoE | HiF4 3MM | 1.5288 | 3/4 | −0.015234 | 2/4 |
| MoE | NVINT4 3MM | 1.6909 | 3/4 | +0.006600 | 2/4 |
| MoE | NVFP4 3MM | 1.4665 | 3/4 | +0.026246 | 0/4 |

MoE 的 NVFP4 在 seed 42、43、2026 同时出现了比 MXFP8 更低的权重直方图 KL 和更高的验证 CE。四个 seed 上其正向 KL 均值比 MXFP8 低 21.96%，验证 CE 却全部高于 MXFP8。这个排序说明，训练终点的全模型权重值分布接近程度，不能单独解释这些模型的验证 CE 差距。定位具体专家或训练阶段还需要逐张量结果或配对干预。

### 逐张量核查：MoE 的 NVFP4 与 MXFP8

[逐张量原值（5,200 行）](weight_kl_by_tensor.csv)包括四个 seed 的 Dense 和 MoE、全部五种训练格式。按每个张量的 `elements` 加权，40 组的正向 KL、反向 KL 和 Jeffreys 散度均与[模型级汇总](weight_kl_summary.csv)吻合（绝对误差小于 1e−15）。[按组件汇总](weight_kl_by_component.csv)提供各架构各 seed 的完整组件表；[按层汇总](moe_weight_kl_by_layer.csv)按层分别保留四位专家的 up/gate/down、专家 FFN 合计和独立路由器。FFN 合计与其三个投影覆盖同一批参数。

以下 ΔK = KL(BF16 权重分布 ∥ NVFP4 权重分布) − KL(BF16 权重分布 ∥ MXFP8 权重分布)，参照均取相同 seed、相同架构的 BF16 训练终点。正值表示该组 NVFP4 权重的直方图离 BF16 更远。每个张量沿用各自与参考配对的联合极值分桶，各层及组件均按参数元素数加权。

| MoE 组件或层，ΔK 单位为 10⁻⁴ nat | seed 42 | seed 43 | seed 100 | seed 2026 |
|---|---:|---:|---:|---:|
| 全模型 | −0.2508 | −0.6307 | +0.1089 | −0.8776 |
| 层 2 的专家 FFN | +0.1660 | +0.2244 | +0.1206 | +1.3985 |
| 层 2 的专家 gate_proj | +0.3324 | +0.1657 | +0.1239 | +1.4549 |
| 层 2 的专家 up_proj | +0.1686 | +0.2449 | +0.1729 | +1.1565 |
| 层 2 的专家 down_proj | −0.0031 | +0.2626 | +0.0650 | +1.5843 |
| 层 6 的专家 FFN | −0.5549 | −0.1015 | −0.4949 | −0.4798 |

- 专家 up/gate/down 共占 MoE 唯一参数的 90.37%。将三个投影的组内 ΔK 分别乘该组参数量占比、求和，再对四个 seed 等权平均，所得有符号差额占全模型平均 ΔK 的 95.31%。这个比例描述权重直方图 KL 净差的来源。
- 层号从 0 起算。层 2 是唯一一个四个 seed 的专家 FFN 聚合 ΔK 均为正的层；其中门控投影和上投影各为 4/4 正，下投影为 3/4 正。将该层每位专家的三个投影分别合并，没有一个固定专家编号四次均为正。层 2 独立的路由器权重 `mlp.gate` 也为 4/4 正。路由器在线性计算中始终使用 BF16 模拟，其训练所得权重仍可随上游轨迹改变。
- 层 6（此前单专家 HiF4 回退所在层）的三个专家投影分别都是 4/4 负。上述 NVFP4 与 MXFP8 的终点比较，与早前 HiF4 的前半程精度干预衡量不同问题。
- 横向核对其他格式，HiF4 3MM 的层 2 专家 FFN 相对 MXFP8 的正向权重 KL 差同样四次为正（依 seed 为 +0.2410、+0.1514、+0.0699、+0.0686，单位 10⁻⁴ nat），但 HiF4 在其中两个 seed 的验证 CE 反而更低。NVINT4 的层 2 专家 FFN 则仅在 100 和 2026 高于 MXFP8。仅用该层 KL 差的正负，无法区分不同格式的 CE 表现。
- 层 2 是在检查全部八层后发现的候选。其 ΔK 最大的 seed 2026 恰好有最小的验证 CE 差（NVFP4 − MXFP8 为 +0.00560 nat/token）。要核查这层对 CE 的影响，可从同一初始化与数据顺序配对训练：在 NVFP4 基础上，仅将层 2 全部专家的 up_proj 和 gate_proj 换成 MXFP8，以原 NVFP4 为基准，并用层 6 的同样回退作为同规模对照。保留相同验证集和终点评估步数。

## Seed 42 的 MoE 微扰与学习率 warmup

在 8 张 V100 上，从相同种子分别复跑 BF16 和 MXFP8，每组完成 512 次 AdamW 更新。验证前向沿用该组训练格式。主指标是 236,442 个有效预测 token 上的平均交叉熵（nat/token），PPL 为其指数，路由辅助损失仅用于训练。

在检出本版本的仓库根目录执行：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /root/anaconda3/envs/lz/bin/python \
  scripts/minimind_moe_noise_floor.py \
  --data /home/lz/minimind/runs/v100-pilot/data \
  --output /data/lz/minimind-moe-noise-s42 \
  --scenarios repeat lr_x1p0001 lr_x0p9999 warmup26
```

| 组别 | 基础 LR | 前 26 次更新 | 比较对象 |
|---|---:|---|---|
| `repeat` | 0.00050000 | 原余弦曲线 | 同格式历史 seed 42 |
| `lr_x1p0001` | 0.00050005 | 原余弦曲线整体乘 1.0001 | 同格式本轮 repeat |
| `lr_x0p9999` | 0.00049995 | 原余弦曲线整体乘 0.9999 | 同格式本轮 repeat |
| `warmup26` | 0.00050000 | 线性升至原曲线在第 26 次更新的 LR | 同格式本轮 repeat |

每 4 个 microbatch 更新一次。以第 1 至 512 次更新为序号，基础 LR 组的 warmup 第 1 次更新取 0.00001912088，第 26 次取 0.00049714282。第 26 次起与同倍率原曲线逐步相同。中断后用相同命令恢复至最近保存的训练进度，完成的训练和有效评估会复用。

可在相同输出目录追加 `--scenarios warmup26_lr_x1p0001 warmup26_lr_x0p9999`，凑齐每种格式的 3 个 LR 倍率 × 2 种日程。另可追加 `swap12`（每个 rank 交换第一个累积窗口里的前两个 microbatch）或 `init_1ulp`（第 0 层注意力 q_proj 权重的 [0,0] 增加一个朝正无穷方向的 BF16 ULP）。每个场景增加两次训练。

每完成一组就更新输出目录中的两个 CSV：

- `noise_runs.csv` 给出每组绝对验证 CE、PPL。对同一格式，`delta_ce_vs_repeat` 为该组 CE 减本轮原样复跑 CE，`warmup_minus_no_warmup_same_lr_ce` 为相同 LR 倍率下 warmup 减无 warmup，负值表示 warmup 改善。
- `noise_summary.csv` 给出本轮复跑与历史值的差，以及所测纯微扰相对本轮复跑的最大绝对 CE 差。warmup 单独汇总。这些幅度描述 seed 42 上实际测得的敏感性，估计跨次复跑的随机方差还需要独立重复。

## 四个 seed 的逐步训练交叉熵

用原始训练日志绘制 seed 42、43、100、2026 的 Dense 和 MoE 曲线，每个 seed 两张图，每张图叠加 BF16、MXFP8、HiF4 THR 3MM、NVFP4 3MM 和 NVINT4 3MM。上图显示五种格式的训练交叉熵，下图显示相同 microbatch 与 MXFP8 的差。八张图共用坐标范围，未作平滑。

横轴为 AdamW 更新步。每条曲线在第 5、10、…、510、512 次更新各有一个观测，记录 rank 0 当前 microbatch 的 `logits_loss`（nat/token，四位小数）。这个字段已扣除 MoE 路由辅助损失。终点表 `same_format_validation.csv` 记录模型在验证集上的 CE，采用不同的数据和聚合方式。

在训练机的本版本仓库根目录执行：

```bash
/root/anaconda3/envs/lz/bin/python scripts/minimind_collect_lce.py \
  --out /data/lz/minimind-training-lce --plot
```

省略 `--plot` 可以仅用 Python 标准库导出 `training_lce.csv`（40 组 × 103 个采样点）。脚本从原 pilot 读取 seed 42 日志，其他三个 seed 按多 seed 网格报告的 `source_work` 查找。4-bit 日志使用 R2 版本。若已把日志复制到其他机器，可重复使用 `--log-dir <日志目录>`。

在已安装 Matplotlib 和中文字体的工作机上，也可以直接从 CSV 绘图：

```bash
python scripts/minimind_plot_lce.py \
  --csv /data/lz/minimind-training-lce/training_lce.csv \
  --out /data/lz/minimind-training-lce
```

图名为 `seed42_dense.png`、`seed42_moe.png` 等。可用 `--font /path/to/NotoSansCJK-Regular.ttc` 指定字体。
