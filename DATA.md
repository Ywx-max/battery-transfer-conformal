# 数据来源与获取

本仓库不重新分发原始数据，请从官方渠道下载（均为公开数据集）：

1. **MIT-Stanford**（124 颗磷酸铁锂，源域）
   - Severson et al., *Data-driven prediction of battery cycle life before capacity degradation*, Nature Energy 4, 283-291 (2019). DOI: 10.1038/s41560-019-0356-8
   - 下载：https://data.matr.io/1
2. **CALCE CS2 + CX2 系列**（各 8 颗、共 16 颗钴酸锂方壳，目标域）
   - 马里兰大学 CALCE 电池研究组：https://calce.umd.edu/battery-data
   - 论文纳入：CS2_8 / CS2_21 / CS2_33–38，以及 CX2_16 / CX2_31 / CX2_33–38
   - 未纳入：CX2_3（脉冲放电）、CX2_4（温度循环 25/35/45/55 °C）、CX2_8（3C 放电）、
     CX2_32（脉冲负载）——工况与 CS2 不一致。纳入准则见论文 4.1 节，
     逐电芯筛选指标（滑窗数 / SOH max / ICA 非空率与峰位 / 贴窗比例）见
     论文 4.1 节与 `results_cx2/cx2_cell_qc.csv`
   - 注意：CX2 额定容量 1.35 Ah（CS2 为 1.1 Ah），解析器按电芯系列设定 RATED；
     CX2_31 为 CADEX txt 格式（其余为 Arbin excel）
3. **NASA PCoE**（4 颗钴酸锂 18650，目标域）
   - NASA Prognostics Center of Excellence Data Repository:
     https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/

下载后放到 `data/raw/` 下（calce 和 nasa 各建一个子目录），解析脚本会从那里读；
再把输出的特征表合并成建模表放到 `data/` 下，合并方法见 `code/README.md` 的运行顺序。

## 随仓库提供的派生数据（`data/`）

原始数据不转存，但论文实验用到的中间表随仓库提供，省去从头解析：

| 文件 | 说明 |
|---|---|
| `data/建模表_v3.csv.gz` | 建模表 v3（13 列，105 287 行）；基线、迁移、保形实验的输入，`gunzip` 后即为下游脚本读取的 `data/建模表_v3.csv` |
| `data/建模表_v3c.csv.gz` | 建模表 v3c = v3 + 7 列放电曲线特征（20 列）；消融实验的输入 |
| `data/建模表_v5.csv.gz` | 建模表 v5（数据版本对照，`t3d_ablation_v5.py` 的输入） |
| `data/mit_dq_early.csv` | MIT 早期预测的 ΔQ 特征表（124 行，含寿命标签）；4.8 节脚本的输入 |
| `data/建模表_v3_cx2.csv.gz` | CX2 扩充版建模表（CALCE 16 颗；论文投稿版实验的输入） |
| `data/建模表_v3c_cx2.csv.gz` | CX2 扩充版 + 7 列曲线特征 |

解压：`gunzip data/建模表_v3.csv.gz`（Windows 下用 7-Zip 或 `gzip -d` 同样可以）。
这些建模表由 `code/data_prep/` 的解析脚本产出的分数据集特征表合并而成（`_cx2` 两张由 `build_modeling_table_cx2.py` 追加生成），
合并规则见 `code/data_prep/build_modeling_table.py`（注意：MIT 侧的分数据集中间产物
未随仓库发布，从原始数据重建建模表需先补齐这些输入，见 code/README.md 的"复现边界"）。

`data/mit_dq_early.csv` 为论文使用的**冻结版本**（作者本地管线提取）；
`code/data_prep/extract_mit_dq_early.py` 是其特征定义的独立实现，与冻结版的
关系（哪些统计量可对齐、哪些存在实现差异）见该脚本的 docstring。
