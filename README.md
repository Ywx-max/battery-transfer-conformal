# Cross-Chemistry Pre-training + Conformal Recalibration for Li-ion Battery Lifetime Prediction

> 中文导读见下文「项目简介」；每个脚本头部都有中文注释，说明设计取舍和踩过的坑。

Code and analysis pipeline for the paper:

> 杨王兴 (Yang Wangxing). 基于跨化学体系预训练与保形校准的锂离子电池寿命概率预测
> (Probabilistic remaining useful life prediction of lithium-ion batteries via
> cross-chemistry pre-training and conformal recalibration). 2026.

## 项目简介

跨化学体系（磷酸铁锂 → 钴酸锂）的电池寿命概率预测：用大规模 LFP 数据预训练，
迁移到只有几颗电芯的 LCO 目标域微调，再用保形预测给出带覆盖保证的区间。
核心发现与仓库的对应关系：

| 论文中的结论 | 代码在哪儿 |
|---|---|
| 预训练价值随目标域数据减少而放大（NASA 1/14.5） | `code/transfer/t3_transfer_local.py` |
| 源域校准区间全面崩塌、目标域再校准恢复（表 5） | `code/conformal/t4_conformal_local.py` |
| 逐电芯覆盖诊断 + 按电芯聚合保形变体（4.5 末） | `code/conformal/t4d_per_cell_diag.py` |
| 漂移分解：量纲漂移 vs 关系漂移（表 4） | `code/transfer/t3b_std_local.py` |
| 特征消融：曲线特征跨域有害（表 6） | `code/ablation/` |
| 早期寿命预测：ΔQ 特征 + 岭回归（4.7） | `code/early_pred/t5_early_pred.py` |
| 论文数字逐项自检（75 项；路径绑定作者本机布局，仓库内不可直接运行） | `code/checks/final_data_check.py` |
| 结果一键复算（155 项，只读本仓库数据） | `code/checks/verify_results.py` |

## 仓库结构

```
code/
  data_prep/   原始数据解析（CALCE / NASA → 每循环特征表）+ 建模表合并
  early_pred/  早期寿命预测（ΔQ 特征 + 岭回归，论文 4.7）
  baselines/   源域基线：4 骨干 5 折组交叉验证 + LOBO 留一电芯
  transfer/    跨化学体系迁移（三机制对照 + 逐数据集标准化协议）
  ablation/    特征丰富度消融（3 个数据版本）
  conformal/   保形区间：双路由对照 / Mondrian / 加权 / 逐电芯诊断
  checks/      数据一致性自检（本地核对 + 一键复算）
data/                派生数据：两张建模表（.csv.gz）与 ΔQ 特征表，见 DATA.md
results/             论文全部实验结果的原始 JSON（对应关系见 results/README.md）
figures_reproduce/   论文 4 张图的复现脚本（直接读 results/）
```

脚本之间没有复杂的包依赖：单个文件拷出去，配上数据路径就能跑。
完整的运行顺序和依赖说明在 `code/README.md`。

## 结果一键复算

    python code/checks/verify_results.py

只读本仓库 `results/` 里的结果文件，把论文中的 155 个关键数字重算一遍
（表 1-6、摘要增益比与变异系数、4.2 跨种子复核、4.5 逐电芯诊断、4.6 Welch 检验、
4.7 早期预测），不需要原始数据集、不需要 GPU、也不需要论文源文件，纯标准库约 1 秒。
逐行给出「论文位置 | 论文数值 | 由本仓库数据重算」，全部一致时返回码 0。（`final_data_check.py` 是同一核对的作者本地版，依赖作者机器的目录布局，仅供参照。）

## 数据来源（均为公开数据集，本仓库不转存原始数据）

| Dataset | Cells | Chemistry | Source |
|---|---|---|---|
| MIT-Stanford | 124 | LFP | Severson et al., *Nature Energy* 2019 ([data](https://data.matr.io/1)) |
| CALCE CS2 | 8 | LCO | [CALCE Battery Group, U. Maryland](https://calce.umd.edu/battery-data) |
| NASA PCoE | 4 | LCO | [NASA PCoE Data Repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/) |

获取方式与目录结构见 `DATA.md`。

论文自己的实验结果放在 `results/`，是各实验脚本的原始输出，哪个文件对应论文哪张表
在 `results/README.md` 里列了。`figures_reproduce/make_figures.py` 直接从 `results/`
读数据重画论文全部 4 张图。

## 运行环境

- Python 3.11+，PyTorch 2.x（CUDA 可选，全部实验在单卡 8GB 笔记本 GPU 上也能跑），
  pandas / numpy / scikit-learn / matplotlib。
- `pip install -r requirements.txt`
- 每个脚本顶部有一个 `DATA = ...` 路径常量，改成你的本地路径再跑；
  完整运行顺序见 `code/README.md`。

## AI 使用声明（AI Usage Statement）

研究过程中使用生成式 AI 工具辅助完成了部分代码实现与文字表述润色；研究设计、全部实验、
数据分析与结论均由作者本人独立完成，作者对全部内容负责。

Generative AI tools were used to assist with part of the code implementation and language
polishing. The research design, all experiments, data analysis and conclusions were completed
independently by the author, who takes full responsibility for all content.

## 许可与版权声明（License & Copyright）

**本仓库采用 CC BY-NC-SA 4.0（署名-非商业性使用-相同方式共享）许可，详见 [LICENSE](LICENSE)。**

- **欢迎**：学术引用、教学使用、复现和改进本代码。需要署名，成果按相同方式共享。
- **不允许**：商业用途；不署名使用代码、方法或图表；把本仓库内容写进论文但不引用
  （学术不端，作者保留追究权利）
- 引用格式见 [CITATION.cff](CITATION.cff)（GitHub 右上角 "Cite this repository" 按钮可直接导出 BibTeX）

论文《基于跨化学体系预训练与保形校准的锂离子电池寿命概率预测》的文本不在本仓库中，
其著作权归作者（杨王兴）所有；本仓库代码仅供学习、复现与学术交流。

**联系方式**：2024212097@bupt.cn（授权咨询、学术交流、侵权举报）
