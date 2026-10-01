# 管线导读

按运行顺序读。每个脚本头部都有中文注释，说明设计取舍；这里只给全局地图和常用命令。

## 运行顺序

```
1. data_prep/          原始数据 → 每循环特征表
   parse_calce.py        CALCE 8 颗（CADEX txt + Arbin xlsx 双格式）→ calce_features.csv
   parse_calce_v2.py     CALCE 重构统一版（电流积分容量 + v_q 曲线特征）→ calce_full_v2.csv
   parse_nasa.py         NASA 4 颗 → nasa_capacity.csv（含 EOL/RUL 定义）
   features_nasa.py      NASA 循环级特征（含 ICA）→ nasa_features.csv
   （MIT 数据已有官方特征表，直接用 mit_*.csv）
        ↓ 合并成一张建模表（三个数据集统一列名：dataset/battery_id/cycle/...）
   build_modeling_table.py  按上一步的产物合并出 建模表_v3.csv / 建模表_v3c.csv
        （论文实验用的那两张表已随仓库以 .csv.gz 提供，解压即可，见 DATA.md）

2. baselines/          源域基线（选出迁移骨干）
   t2_train_local.py     4 骨干 × 5 折组交叉验证（--smoke 先做过拟合自检）
   t2_lobo_local.py      LOBO 留一电芯 119 折（断点续跑，表 2 / 图 1 的数据源）

3. transfer/           跨化学体系迁移
   t3_transfer_local.py  zero-shot / fine-tune / target-only 三机制（表 3）
   t3b_std_local.py      逐数据集标准化协议（表 4 漂移分解的另一半）

4. ablation/           特征消融（表 6）
   t3c_ablation_local.py base7 vs curve14（v3 建模表）
   t3d_ablation_v5.py    同协议，v5 建模表（数据版本对照）
   t3e_ablation_v3c.py   同协议，v3c 建模表（论文最终采用）

5. conformal/          保形区间
   t4_conformal_local.py 双路由对照（表 5，最核心）
   t4c_mondrian_local.py 条件化收窄之一：SOH 分箱
   t4c_weighted_local.py 条件化收窄之二：密度比加权
   t4d_per_cell_diag.py  逐电芯覆盖率与按电芯聚合变体（论文 4.5 末）

6. early_pred/          早期寿命预测（论文 4.7）
   t5_early_pred.py      ΔQ 特征 + 岭回归，留一电芯 119 颗（输入 data/mit_dq_early.csv）

7. checks/
   final_data_check.py   实验结果 JSON vs 论文数字，75 项逐项核对（约 3 秒，需论文源文件）
   verify_results.py     结果复算：只读 results/ 把论文数字重算一遍 155 项（无需论文源文件、无 GPU）
```

## 结果数据

仓库 `results/` 里是论文全部实验结果的原始 JSON（各脚本的输出格式）。重跑脚本后
把输出覆盖到对应文件即可；`results/README.md` 列了每个文件对应论文的哪张表。
`checks/final_data_check.py` 是作者本机的核对脚本（路径按工作目录写死、需要论文
LaTeX 源），克隆仓库跑不了它，但结果文件本身可以直接看。

## 论文口径备忘

- **种子**：迁移/保形/消融 = 42-46 共 5 次独立运行；漂移分解 = 42-45 共 4 次；
  报"均值±标准差"，摘要里的增益比是**逐种子比值再平均**（不是均值相除）。
- **划分**：表 5 保形 = 硬编码 (CALCE: 3/2/3, NASA: 2/1/1)；
  Mondrian/加权 = 1/3 三分协议（CALCE 2/2/4, NASA 1/1/2）。
  `t4d_per_cell_diag.py` 与表 5 用同一划分（脚本内 assert 校验）。
- **标准化**：t3/t3c/t3d/t3e = 源域 scaler 直接用于目标域（原始协议）；
  t3b/t4*/t4d = 源域、目标域各用各的 scaler（逐数据集协议）。
  这个区别就是论文表 4"漂移分解"的两行。
- **GPU 非确定性**：同种子重跑指标有小幅浮动（论文 4.8 有讨论），
  所以任何"小数第 3 位"的比较都不该当真。

## 常见改造点

- 换数据集：改各脚本顶部的 `DATA` 路径 + `FEATS` 列名列表；
  建模表需含 `dataset`（MIT/CALCE/NASA）、`battery_id`、`cycle`、`soh`、`rul` 列。
- 加新骨干：在 `t2_train_local.py` 的 `new_model()` 里注册，
  其余脚本的模型选择逻辑都是薄封装。
- 调保形失配率：`ALPHA`（0.10 → 0.90 名义覆盖），各脚本顶部。
- 加诊断指标：`t4d_per_cell_diag.py` 的 json 里存了逐循环残差，
  改评估逻辑不用重训。
