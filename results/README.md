# results/ 结果数据说明

论文全部实验结果（实验脚本的原始输出 JSON）。每个文件都能在论文里找到对应位置，
`code/checks/final_data_check.py`（75 项，作者本地自检）与 `code/checks/verify_results.py`
（179 项，只读本目录、无需论文源文件，可直接运行）就是拿这些文件跟论文数字逐项对拍的；
汇总文件可由 `code/checks/aggregate_results.py` 从逐种子文件再生。

## baselines/ 源域基线

| 文件 | 对应论文 |
|---|---|
| baseline_3seed_final.json | 表 1，四骨干 5 折组交叉验证（3 种子；± 为种子间样本标准差 ddof=1） |
| lobo_tcn.jsonl | 表 2 的 TCN 行（原始运行，119 折，每行一折） |
| lobo_tcn_s42/43/44.jsonl | 图 1 的 357 折合并分布 + 4.2 节的跨种子复核（复现运行，每行一折） |
| lobo_lstm.jsonl / lobo_transformer.jsonl | 表 2 的 LSTM / Transformer 行（各 119 折，原运行） |
| lobo_final_multiseed.json | 4.2 节跨种子复核（TCN 3 次重复 92.45±0.98；LSTM/Transformer 仅原运行 n_runs=1） |
| lobo_3model_final.csv | 4.8 节双模型互补性（r=0.699）与择优上界的逐电芯数据 |

> 版本说明：早期版本汇总里的 "LSTM/Transformer/集成各 2 次重复"（96.21±1.92 等）
> 无逐折原始文件支撑，已从汇总与论文中撤下（aggregate_results.py 只再生有 jsonl
> 支撑的部分）。

## transfer/ 跨化学体系迁移

| 文件 | 对应论文 |
|---|---|
| transfer_multiseed_v2.json | 表 3 + 图 2，三机制对照（5 种子） |
| raw_protocol_multiseed.json | 表 4 首行口径，原始协议漂移（5 种子 42-46） |
| t3_soh_tcn_s42-46.json | 表 4 逐种子原始输出（原始协议，t3_transfer_local.py 产物） |
| t3b_tcn/lstm_s42-46.json | 表 3 逐种子原始输出（逐数据集标准化协议，t3b_std_local.py 产物） |
| t3b_tcn_s42_repeat1/2.json | 4.8 节同种子两次独立执行的重复对（CALCE 零样本 RMSE 相差 14%，GPU 非确定性实证） |
| t3_rul_tcn_s42.json | RUL 口径对照（3.1 节末尾提到 SOH 作迁移评估目标的依据） |

> 口径提示：表 3（论文主表）= 逐数据集标准化协议 = t3b 文件；表 4 首行 = 原始协议
> = t3_soh 文件。早期版本 README 把两者写反过，以本版为准。

## ablation/ 特征丰富度消融

| 文件 | 对应论文 |
|---|---|
| ablation_v3c_multiseed.json | 表 6 + 图 4（v3c 版建模表，论文采用） |
| ablation_multiseed.json / ablation_multiseed_v5.json | 数据版本对照存档（v3 / v5 建模表；论文未直接引用其数值） |
| t3c/t3d/t3e_*_s42-46.json | 三个数据版本的逐种子原始输出（base7 / curve14 两组） |

## conformal/ 保形区间

| 文件 | 对应论文 |
|---|---|
| conformal_multiseed_summary.json | 表 5 + 图 3，双路由对照（5 种子） |
| t4c_multiseed_summary.json | 4.5 节条件化收窄（Mondrian / 加权，负结果） |
| t4_tcn/lstm_s42-46.json | 表 5 的逐种子原始输出（含逐电芯残差向量） |
| t4c_mondrian_s42-46.json / t4c_weighted_s42-46.json | 4.5 节扩展实验的逐种子输出 |
| t4d_per_cell_tcn/lstm_s42-46.json | 4.5 节末补充诊断（逐电芯覆盖率、逐循环残差、聚合保形变体） |

> 版本说明（2026-10）：t4 与 t4d 的旧版实现误用"微调前模型"的残差计算目标域校准
> 分位数、却在微调后模型上评估覆盖，违反拆分保形的同源前提，旧结果（目标校准
> PICP≈1.00）作废。现版 t4_*.json / t4d_*.json 为修复后重跑结果：校准分位数由
> 部署模型自身在校准电芯上的残差计算，逐电芯残差向量已随 JSON 保存。
> t4c_weighted_*.json 同批重跑（移除测试真值泄漏 + 修正加权分位数 off-by-one）；
> t4c_mondrian_*.json 无缺陷、保留原始运行。

## 补充实验（2026-10-02，论文 4.5/4.9 节引用）

| 文件 | 说明 |
|---|---|
| t4_split_sweep/ | 覆盖率-校准电芯数扫描（`t4_conformal_local.py --n-cal`）：固定微调电芯数（CALCE 4 / NASA 1），校准电芯 1→3 档，2 骨干 × 5 种子；`sweep_summary.json` 为聚合 |
| t4_gru_s42-46.json | GRU 骨干的稳健性复核（双路由，5 种子，源域预训练同协议重跑） |
| diag_deterministic/ | 跨管线复现性诊断：t3b/t3e 同种子 `--deterministic` 开关重跑对照（正文 4.9 引用） |

## 其他

- `early_pred_summary.json` / `early_pred_results.csv`：4.7 节早期寿命预测（ΔQ 特征 + 岭回归，逐电芯预测明细 119 颗；log10 RMSE 0.117 / 循环 RMSE 141.7 / MAPE 19.5%；脚本 `code/early_pred/t5_early_pred.py`）

## 口径提示

- 逐种子文件用种子号命名（s42-s46）；论文表格除 LOBO 单次运行外均报 5 种子（42-46）
- 全文 ± 为样本标准差（ddof=1）
- 表 3 的增益比是逐种子（基线/微调）再平均，不是两列均值相除；重尾分布下中位数口径更保守（4.3 有并列披露）
- 表 5 的划分硬编码（CALCE 3/2/3，NASA 2/1/1）；Mondrian / 加权用 1/3 三分协议；
  每种子的划分随种子重新抽取，± 同时反映训练随机性与划分差异
- 同种子重跑会有小幅浮动（GPU 非确定性），零点几到几个百分点的差异属正常范围；
  两套脚本管线的源域预训练不可逐位复现（同种子微调 RMSE 相差最高 57%，见论文 4.8）
