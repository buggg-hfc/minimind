# 字段与复算口径

文件均为 UTF-8。用 Python 标准库读 CSV 时，可统一使用 `encoding="utf-8-sig", newline=""`。CE 为自然对数单位，PPL = exp(CE)。合并批次或文档时，先用有效预测 token 数加权 CE，再取指数。

## 26 组终点与四格实验

| 字段 | 含义 |
|---|---|
| 基础格式、使用MXFP8的专家投影、训练方案 | 训练配置。投影集合适用于每一层的全部专家。all 表示专家的 up、gate、down。专家 gate_proj 和选择专家的路由层 gate 是不同模块。 |
| BF16验证CE、BF16验证PPL | 用这一行训练出的权重，将模拟器切到 BF16 验证。 |
| 本组验证CE、本组验证PPL | 用同一行权重和训练时的组件精度验证。 |
| ΔCE_对原全4bit_R2、ΔCE_对全MXFP8 | 按原 CSV 的 BF16验证CE 计算，分别减去同基础格式无回退行、全 MXFP8 行。使用自身精度时需用本组验证CE 重算。工作簿第一张表使用自身精度的差值。 |
| case、first_half_down_proj、second_half_down_proj | 两个半程各 256 次 AdamW 更新。H 为 HiF4-thr 3MM，M 为仅专家 down_proj 使用 MXFP8。 |
| optimizer_update、valid_tokens、mean_ce、ppl | 四格终点的全局更新编号、验证有效 token 数、平均 CE 及 PPL。 |
| midpoint_prefix_own_ce、midpoint_prefix_own_ppl | 各自前缀在第 256 步使用前半程格式验证。HH/HM 共用一次 H 测量；MH/MM 共用一次 M 测量。 |

[验证点长表](validation_points.csv) 的 trajectory 标识训练轨迹，eval_policy 标识该次实际使用的推理配置，source 标识原表。M_prefix 只代表 MH/MM 的共同中点。H 前缀在同一步有 H、M 两次即时评测。全 MXFP8 与 BF16 行是历史 pilot 终点；交叉实验 MM 与历史 down 全程回退行也应保留各自的运行身份。长表只列实际测量。历史 Dense/MoE 终点摘录中的 architecture、train_scheme、eval_scheme 分别是架构、训练和推理策略，checkpoint 为导出权重名，18 行仅保留共享基线和 R2。

## 后半程训练轨迹

[逐步表](paired_updates_HH_HM.csv) 的 a、b 分别表示 HH、HM。训练和验证均按各分支当时的格式。除验证点之外的空白表示该步未验证；第 0 行是分叉点，有些训练字段为空。

| 字段 | 含义 |
|---|---|
| relative_update、optimizer_update、microbatch、lr | 分叉后更新数、全局更新数、已处理的 microbatch 数和本轮更新所用学习率。相对步 0 对应全局更新 256。 |
| train_valid_tokens、train_ce_a/b、train_ppl_a/b、train_ce_b_minus_a | 该次更新对应的八卡有效训练 token 数、两分支的语言模型 CE/PPL、B−A。训练 CE 不含路由辅助损失。 |
| eval_valid_tokens、eval_ce_a/b、eval_ppl_a/b、eval_ce_b_minus_a | 同一全量验证集上的有效 token 数、CE/PPL 以及验证 CE 的 B−A。 |
| delta_ce_change_since_fork | 当次验证的 B−A 减去分叉时同权重、不同推理格式的 B−A。 |
| route_paired_ranks、world_size | 已按相同输入配对的 DDP rank 数和总 rank 数。 |
| route_layer_token_observations、route_changed、route_changed_fraction | 训练时配对的层 × 有效预测位置数、Top-1 专家不同的位置数及两者之比。跨步总分歧率用计数之和重算。 |
| down_sampled_coordinates、down_delta_norm_a/b | 8 × 4 个专家各 1,024 个固定 down 参数坐标，总计 32,768。后两列为当步两个实际参数更新向量的 L2 范数。 |
| down_delta_relative_l2、down_delta_cosine | 当步实际更新在这些坐标上的 ‖B−A‖₂/‖A‖₂ 与余弦相似度。 |
| down_clipped_grad_relative_l2、down_clipped_grad_cosine | 相同坐标、相同公式，使用进入优化器的裁剪后梯度。梯度与参数更新各有自己的分母。 |

## 按专家汇总

[逐专家表](expert_updates_HH_HM.csv) 的层号、专家号均从 0 开始。跨层的同号专家具有不同参数。负载使用后半程实际训练的有效预测位置。

| 字段 | 含义 |
|---|---|
| layer、expert、optimizer_updates、top_k、valid_token_positions | 层号、专家号、覆盖更新数（256）、每位置专家数（1）、该层配对有效位置总数（3,715,155）。 |
| load_a/b、load_fraction_a/b | A/B 选中该专家的位置数及其占该层有效位置的比例。 |
| lost、gained、union | 仅 A 选、仅 B 选、任一方选该专家的位置数。 |
| membership_changed、turnover | lost + gained 及其除以 union 的比例。Top-1 下一个 token 换专家会在原专家和新专家处各计一次成员变化。 |
| sampled_coordinates | 每位专家固定采样的 down_proj 参数坐标数。 |
| clipped_grad_*、adamw_update_* | 分别为裁剪后梯度和实际参数更新。将该专家所有步的采样向量拼接，norm_a/b 是两分支的范数，difference_sq_sum 是差向量平方和，relative_l2 为 ‖B−A‖₂/‖A‖₂，cosine 为余弦相似度。 |
| adamw_update_layer_mismatch_share | 本专家实际更新差平方和 / 本层四位专家的实际更新差平方和。它描述更新分歧在一层内的分布，验证 CE 贡献需要单独干预衡量。 |

## 算子回放与诊断样本

[逐投影和逐专家 CSV](operator_metrics_applied.csv) 在一张表中保留了两种粒度。scope=components 按投影汇总，scope=down_by_expert 按层、专家列出 down 投影；同一次观测出现在两种粒度中，分析时选取其一。[原始数值导出](operator_replay.json.xz) 中 cases 下的各表按列存储，列表相同下标组成一行，数字可能用字符串编码。

| 字段 | 含义 |
|---|---|
| layer、expert、projection、scheme、primitive、gemm | 层、专家、投影名、整套待测策略、该投影实际格式、乘法方向（FWD、dX、dW）。按投影聚合时层和专家为空。 |
| stage | raw 比较 GEMM 的 FP32 承载输出。applied 比较参考与待测均经过 BF16 输出舍入后的值。可直接读的 CSV 选用 applied；压缩 JSON 包含两者。 |
| calls、input_rows、output_elements、sampled_count、finite_count | 调用数、原线性层的输入位置总数、输出元素总量、抽取的配对输出元素数、两侧均有限的采样数。每次调用/方向最多取 4,096 个输出坐标。input_rows 在不同 stage、方向、待测格式会重复。 |
| ref_zero_count、ref_nonzero_count、ref_zero_est_nonzero_count | 有限样本中参考为零、参考非零、参考为零但待测非零的个数。 |
| signed_mean、mae、ulp_mean、relative_mae | δ=待测−参考；前两者为样本均值及平均绝对误差。ulp_mean 是以参考 BF16 ULP 为单位的平均绝对误差。relative_mae 是参考非零坐标上 abs(δ)/abs(参考) 的均值。 |
| operator_hist_kl、variance_retention | 每次调用共用 128 个有符号桶，每桶加 0.5，计算参考到待测的 KL，再按有限采样数加权。方差留存率为待测方差/参考方差。 |
| same_direction_shrink/equal/expand_rate、zeroed_rate、sign_flip_rate | 分母均为参考非零的有限采样数。五类分别表示同号幅值缩小、相同、变大，待测变成零，异号且两侧非零。对应的 count 存于 JSON。这些是 GEMM 输出坐标统计，不能当作输入量化器清零率。 |

[专家输入行数](operator_routed_rows.csv) 中 routed_positions_including_padding 是 4 篇回放的专家输入行数，positions_per_layer 为各层全部输入行数 2,048。[16 篇端到端表](diagnostic_16docs.csv) 的 next_token_kl_bf16_to_candidate 是下一 token 全词表分布的 KL，按 3,929 个有效 token 聚合。[逐文档表](diagnostic_documents.csv) 的 validation_index 是原验证集零起编号，candidate_nll_sum 是该文档所有有效预测 token 的 NLL 之和。对同一方案，sum(candidate_nll_sum)/sum(valid_tokens) 可还原 16 篇的 mean_ce。
