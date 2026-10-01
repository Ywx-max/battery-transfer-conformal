# results/ 结果数据说明

论文全部实验结果（实验脚本的原始输出 JSON）。每个文件都能在论文里找到对应位置，
`code/checks/final_data_check.py`（75 项，作者本地自检）与 `code/checks/verify_results.py`
（155 项，只读本目录、无需论文源文件，可直接运行）就是拿这些文件跟论文数字逐项核对的。

## baselines/ 源域基线

| 文件 | 对应论文 |
|---|---|
| baseline_3seed_final.json | 表 1，四骨干 5 折组交叉验证（3 种子） |
| lobo_tcn.jsonl | 表 2 的 TCN 行（原始运行，119 折，每行一折） |
| lobo_tcn_s42/43/44.jsonl | 图 1 的 357 折合并分布 + 4.2 节的跨种子复核（复现运行，每行一折） |
| lobo_lstm.jsonl / lobo_transformer.jsonl | 表 2 的 LSTM / Transformer 行（各 119 折） |
| lobo_final_multiseed.json | 4.2 节跨种子复核（TCN 92.45±0.98、集成 70.52±0.27 等） |
| lobo_3model_final.csv | 4.8 节双模型互补性（r=0.699）的逐电芯数据 |

## transfer/ 跨化学体系迁移

| 文件 | 对应论文 |
|---|---|
| transfer_multiseed_v2.json | 表 3 + 图 2，三机制对照（5 种子） |
| raw_protocol_multiseed.json | 表 4 首行，原始协议漂移（4 种子） |
| t3_soh_tcn_s42-46.json | 表 3 的逐种子原始输出 |
| t3b_tcn/lstm_s42-46.json | 表 4 的逐种子原始输出（逐数据集标准化协议） |
| t3b_tcn_s42_repeat1/2.json | 4.8 节同种子两次独立执行的重复对（CALCE 零样本 RMSE 相差 14%，GPU 非确定性实证） |
| t3_rul_tcn_s42.json | RUL 口径对照（3.1 节末尾提到 SOH 作迁移评估目标的依据） |

## ablation/ 特征丰富度消融

| 文件 | 对应论文 |
|---|---|
| ablation_v3c_multiseed.json | 表 6 + 图 4（v3c 版建模表，论文采用） |
| ablation_multiseed.json / ablation_multiseed_v5.json | 表 6 注明的前两个数据版本对照（v3 / v5） |
| t3c/t3d/t3e_*_s42-46.json | 三个数据版本的逐种子原始输出（base7 / curve14 两组） |

## conformal/ 保形区间

| 文件 | 对应论文 |
|---|---|
| conformal_multiseed_summary.json | 表 5 + 图 3，双路由对照（5 种子） |
| t4c_multiseed_summary.json | 4.5 节条件化收窄（Mondrian / 加权，负结果） |
| t4_tcn/lstm_s42-46.json | 表 5 的逐种子原始输出 |
| t4c_mondrian_s42-46.json / t4c_weighted_s42-46.json | 4.5 节扩展实验的逐种子输出 |
| t4d_per_cell_tcn/lstm_s42-46.json | 4.5 节末补充诊断（逐电芯覆盖率、聚合保形变体、逐循环残差） |

## 其他

- `early_pred_summary.json` / `early_pred_results.csv`：4.7 节早期寿命预测（ΔQ 特征 + 岭回归，逐电芯预测明细 119 颗，相对误差 19.5%；脚本 `code/early_pred/t5_early_pred.py`）

## 口径提示

- 逐种子文件用种子号命名（s42-s46）；表 3 报 5 种子，表 4 的漂移分解为 4 种子（s42-45）
- 表 3 的增益比是逐种子（基线/微调）再平均，不是两列均值相除
- 表 5 的划分硬编码（CALCE 3/2/3，NASA 2/1/1）；Mondrian / 加权用 1/3 三分协议
- 同种子重跑会有小幅浮动（GPU 非确定性），零点几到几个百分点的差异属正常范围
