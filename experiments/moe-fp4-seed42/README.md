# MiniMind-MoE 预训练精度对照 · seed 42

这组实验比较 HiF4-thr 3MM、NVFP4 3MM、NVINT4 3MM 和 MXFP8。前三种均为 4-bit 残差补偿计算，NVINT4 是整数格式。主指标为 1,024 篇验证文档、236,442 个有效预测 token 的平均交叉熵（CE，单位 nat）和 PPL。下列数据来自同一个训练种子。

## P0 · 文件和口径

| 数据 | 可回答的问题 |
|---|---|
| [26 组训练策略与两种验证精度](moe_ffn_grid_seed42.csv) | 每组终点权重分别按 BF16 和本组格式推理。含三个 4-bit 格式各自的 8 种专家投影组合及两条基线。 |
| [端到端、投影和逐专家诊断工作簿](minimind_moe_s42_matched.xlsx) | 四张表覆盖全量验证、16 篇下一 token KL、4 篇固定操作数误差及逐层逐专家结果。 |
| [后半程逐步配对数据](paired_updates_HH_HM.csv) | A=HH，B=HM。从同一 HiF4 第 256 次 AdamW 更新断点（microbatch 1024，共用模型、AdamW 和各卡 RNG）分叉。含起点及 256 次更新。 |
| [后半程按专家汇总](expert_updates_HH_HM.csv) | 同一对 HH/HM、同一训练 token 顺序。每层 4 个专家，共 32 行。统计覆盖后半程 3,715,155 个有效预测位置。 |
| [四格交叉实验原值](four_cells.csv) | 用户提供的 CSV 数值。H/M 仅指所有专家的 down_proj 精度。两个字母依次代表前、后 256 次更新。 |
| [各文件字段、分母与复算方法](COLUMNS.md) | 包括原 CSV 的 ΔCE 列所用的验证格式、路由 turnover 的分母和梯度采样范围。 |

前四份文件保留了交付时的字节。两份训练统计的原附件是 CSV 内容的文本文件，在这里使用 .csv 扩展名。工作簿可视为同批数据的整理版本。复算时可以直接读取同目录的 CSV。

### 训练和验证配置

| 项目 | 设置与来源 |
|---|---|
| 模型 | 8 层，隐藏维 768，8 个 query 头、4 个 KV 头，头维 96。每层独立 4 个 SwiGLU 专家、Top-1，无共享专家，专家中间维 2432。词表 6400，输入嵌入和输出头共享权重。 |
| 参数 | 按该配置和共享权重去重，总参数 198,416,640。按训练器的 Top-1 活跃参数口径为 63,936,768（非专家参数 + 每层一个专家）。 |
| 日程 | 从头训练，seed 42。8 × V100，1 epoch，每卡每 microbatch 2 篇，累积 4 次，全局每次更新 64 篇。最大序长 512，2048 个 microbatch，512 次 AdamW 更新。每步有效预测 token 数随 padding 而变。 |
| 优化器 | 训练器仅显式设置 AdamW 的学习率。PyTorch 2.8.0 默认 β₁=0.9、β₂=0.999、ε=1e−8、weight decay=0.01。全局梯度范数裁剪阈值 1.0。 |
| 学习率 | 按 microbatch 编号 t 调度，lr(t) = 0.0005 × [0.1 + 0.45 × (1 + cos(πt/2048))]，无 warmup。第 256 次更新为 0.000275，第 512 次为 0.00005。 |
| 训练目标 | next-token CE 加各层路由辅助损失，各层 router_aux_loss_coef=0.0005，再按累积数缩放。全量验证表报告有效预测 token 的 CE。 |
| 数值实现 | V100 上用 FP32 承载和 GEMM 模拟数值格式，模拟模式关闭 AMP。Q/K/V/O 与专家 gate/up/down 按策略切换。路由线性层、LM head 和注意力 QK/PV 乘法保持 BF16 模拟。选中专家 down 的回退同时作用于 FWD、dX、dW。 |
| 残差 | R2 在 BF16 做差，残差重新量化。3MM = A₀B₀ + A₁B₀ + A₀B₁。分组沿各逻辑乘法的归约维，HiF4 组宽 64，NVFP4/NVINT4 为 16，MXFP8 为 32。在 NV 反向路径，dY 的主值和残差共享 FP32 tensor scale，重算组 scale。 |
| 数据指纹 | 训练 JSONL SHA256 为 `4d691136c467563ebea032e972747e957bdc2d06919b9f1a8baea05ff466ffb6`，验证 JSONL 为 `72bb2daa3e43457bebd118119f335fcebf6433077ab56ae2ab6f6f9f28f60a04`。历史 HiF4 与 down 回退组已在训练机核对相同。 |

仓库可复核的实现位于 [模型与路由](../../model/model_minimind.py)、[训练器](../../trainer/train_pretrain.py)、[学习率和参数口径](../../trainer/trainer_utils.py)、[格式实现](../../model/quant_formats.py)、[模拟 GEMM 和回退](../../model/quant_linear.py) 及 [AdamW 默认值](https://github.com/pytorch/pytorch/blob/v2.8.0/torch/optim/adamw.py)。

代码版本可以追溯到以下提交。原始结果没有逐进程的 Git HEAD；这些 SHA 指向对应实现。交叉运行器另用文件 SHA 锁定训练机上的前缀快照和分支脚本。

| 实现 | 固定提交 |
|---|---|
| R2 残差、专家投影组合 | [19d21c6](https://github.com/buggg-hfc/minimind/commit/19d21c621fbefcbeef62c1b454f65243c02c5438)、[017c69f](https://github.com/buggg-hfc/minimind/commit/017c69fb64f3a887bad2b7fc1d3e5c94b72f2df0) |
| 回放诊断、配对续训 | [de5576c](https://github.com/buggg-hfc/minimind/commit/de5576ce2a2d7b4b19965f356d02137e66c1c588)、[054a592](https://github.com/buggg-hfc/minimind/commit/054a592a95a22398da0210eb87fd919344f47435) |
| 四格交叉、逐专家汇总 | [5eb51c3](https://github.com/buggg-hfc/minimind/commit/5eb51c36c7fb96f779bd0dc3b9a8b679e1a009f5)、[c3e180a](https://github.com/buggg-hfc/minimind/commit/c3e180a911288395c7688025c093b1cebf56f0f1) |

### 4 篇文档的 BF16 回放用了哪份权重？各专家分到多少 token？

清单中的 BF16 权重名是 `pilot32k_s42_bf16_768_moe.pth`，SHA256 为 `1ab058fc1a31b519d412d85b9a31ee0aeea11d4b5d0cba417ff39b37bb58ba30`。它与原始 BF16 终点评测使用同一文件名。训练器在各次保存时覆盖同名导出，导出只含 state_dict，诊断清单也未记录 step。按完整训练流程推定为第 512 次更新；确证步数需将相同 SHA 的导出张量与含 step 的 BF16 resume 对上。两个第 256 步的交叉前缀另有独立断点。

回放使用验证文档编号 964、18、184、392（零起编号），有效预测 token 分别为 334、132、511、188，共 1,165。每层实际有 4 × 512 = 2,048 个输入位置进入专家。下面从原始 down_by_expert 取一个方案、FWD、applied 的 input_rows 得到，已验证 5 个回放方案的计数完全一致。四篇中每个专家合计收到的输入行数包含 padding 和没有下一个预测目标的位置。现有文件没有逐专家的有效 token 掩码，也没有逐篇的专家交叉表。

| 层号 | 专家 0 | 专家 1 | 专家 2 | 专家 3 |
|---:|---:|---:|---:|---:|
| 0 | 570 | 590 | 609 | 279 |
| 1 | 406 | 629 | 535 | 478 |
| 2 | 489 | 648 | 493 | 418 |
| 3 | 279 | 641 | 415 | 713 |
| 4 | 622 | 472 | 396 | 558 |
| 5 | 252 | 661 | 596 | 539 |
| 6 | 453 | 506 | 726 | 363 |
| 7 | 467 | 651 | 580 | 350 |

可直接读 [32 个专家的行数](operator_routed_rows.csv)、[4 篇回放的逐方向误差](operator_metrics_applied.csv)、[16 篇端到端评测](diagnostic_16docs.csv) 和 [逐文档 NLL](diagnostic_documents.csv)。[紧凑诊断 JSON](operator_replay.json.xz) 保留交付的 raw/applied、各分母、清单及权重 SHA，清单中的训练机路径仅保留文件名。读取方式为 Python 的 `json.load(lzma.open("operator_replay.json.xz", "rt"))`。

## P1 · 已有的验证曲线、中点与其他种子

[验证点长表](validation_points.csv) 收录有实测依据的 33 条记录。HH 和 HM 在全局第 256、276、296、316、336、356、376、396、416、436、456、476、496、512 步都有验证。四格实验的终点见下表。MXFP8 基线对全部可切换投影使用 MXFP8；MM 只让专家 down 使用 MXFP8。

| 训练策略 | 第 256 步 PPL（前缀自身格式） | 第 512 步 CE | 第 512 步 PPL（终点自身格式） |
|---|---:|---:|---:|
| HH | 158.05251 | 4.330909 | 76.01338 |
| HM | 158.05251（共用 H 前缀） | 4.331348 | 76.04676 |
| MH | 134.99132（共用 M 前缀） | 4.264549 | 71.13286 |
| MM | 134.99132 | 4.263745 | 71.07569 |
| 全 MXFP8 | 缺 | 4.295454 | 73.36552 |
| BF16 | 缺 | 4.263755 | 71.07637 |

第 256 步再用 M 精度评测同一个 H 前缀得到 PPL 158.06417，CE 5.06300106。这是 HM 的即时切换评测，与 four_cells 中按 H 评出的共享前缀列是两次不同评测。M、H 前缀自身格式的中点 CE 相差 −0.15771687，包含推理格式和训练权重两项差别。终点 MH 和 HH 使用相同的后半程及验证格式，CE 差为 −0.06636001。

| 对方还需要的材料 | 当前覆盖与最直接的补充来源 |
|---|---|
| 前半程逐步记录、六组完整曲线 | 两种前缀各有一个已上交的中点。训练机上的 `hif4-prefix-1024/prefix.log` 和 `reciprocal-down-s42/mixed-prefix-1024/prefix.log` 提供训练日志，其中的训练 loss 不能填进验证曲线。M 前缀两组的后半程定期验证可从 `reciprocal-down-s42/pair-1024-to-2048/summary.csv` 补入。这里的相对路径以训练实验根目录为起点。 |
| 两个中点断点或验证集路由、权重统计 | H 前缀文件为 `hif4-prefix-1024/checkpoints/fork1024_s42_hif4_thr_3mm_r2_768_moe_resume.pth`。M 前缀文件为 `reciprocal-down-s42/mixed-prefix-1024/checkpoints/reciprocal_s42_hif4_thr_3mm_ffn_down_mxfp8_r2_768_moe_resume.pth`。本次材料仅含验证 CE/PPL。逐专家训练表覆盖之后 256 步；第 256 步两种前缀在验证集上的路由熵、同 token 一致率、权重范数和离群值尚缺。 |
| 其他训练 seed 或单专家干预 | 当前提供的 MoE 和 Dense 实测均为 seed 42。四格分支、重复运行、同层四个专家均不能用作独立训练种子。单专家实验有[运行器](../../scripts/moe_down_crossover/expert_runs.py)，本次材料中没有终点结果。 |

### 分析时可先验算的三项结果

1. **统一 BF16 推理后，训练结果之间仍有差距。** 以 HiF4 对全 MXFP8 为例，各自格式验证的 ΔCE 为 +0.03550062，统一 BF16 验证为 +0.03587017。对 NVFP4 两个值为 +0.02059084、+0.02058148。原始网格也可检查其他投影，例如 HiF4 仅 up 回退时 PPL 69.95，仅 gate 回退时 69.87，仅 down 回退时 71.13。单个组合的改善不足以证明 down 是唯一敏感位置。[同格式和 BF16 验证](moe_ffn_grid_seed42.csv)
2. **早期作用在相同的后半程格式下仍可观察到。** 前半程效应 MH−HH、MM−HM 分别为 −0.06636001、−0.06760295；后半程效应 HM−HH、MM−MH 为 +0.00043896、−0.00080399；两阶段交互 (MM−MH)−(HM−HH) 为 −0.00124294。差值均为 CE。不同 seed 的不确定性尚待估计。[四格原值](four_cells.csv)
3. **已有数据可以约束 H1–H4。** 在固定 BF16 权重的 4 篇回放中，down 的 applied dW 翻号率为 MXFP8 0.926%、HiF4 0.326%、NVFP4 0.402%；三个方案的采样输出置零率均为 0。这些是回放输出的统计，尚无早期训练坐标的清零和翻号轨迹。AdamW 的偏差校正使第 1 步梯度项近似符号更新，推及后续 256 步需要各坐标的一、二阶矩。3MM 省略的残差乘积可取两种符号，系统性收缩需要有符号测量。路由梯度还受 softmax 导数和辅助损失影响。后期 HH/HM 已出现路由分歧且 CE 接近；要检验前期的因果机制，还缺前缀诊断和多 seed。[回放输出](operator_metrics_applied.csv) · [逐步配对](paired_updates_HH_HM.csv)

历史 Dense 对照也使用同一验证集和 seed。Dense 的 HiF4、NVFP4、MXFP8 以自身格式验证时，PPL 分别为 74.96、72.97、75.89。它支持同一 pilot 下存在架构间排序差异，不能替代 MoE 内的路由干预。[Dense 与 MoE 的原始终点评测摘录](dense_moe_pilot_endpoints.csv)
