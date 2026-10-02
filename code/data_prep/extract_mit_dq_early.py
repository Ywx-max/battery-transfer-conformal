# -*- coding: utf-8 -*-
"""从 MIT-Stanford 原始循环数据提取前 100 循环 ΔQ 特征（论文 4.7 的输入）。

特征定义（与论文 4.7 节一致）：对每颗电芯取循环 10 与循环 100 的放电段，
把两轮的放电容量-电压曲线插值到公共电压网格（2.0--3.5 V，1000 点），
取差分 ΔQ(V) = Q_100(V) - Q_10(V)（含循环间总放电容量的常数偏移修正），
提取 6 维：方差、均值、线性拟合斜率、截距、最小值、最大值。

与随仓库数据的关系（重要）：
  results 同款输入 `data/mit_dq_early.csv` 是论文使用的**冻结版本**，由作者本地
  管线提取。本脚本是该特征定义的独立实现：在 b1c0 上验证，方差与均值两统计量
  与冻结版偏差 < 1%，但斜率/截距/极值对放电段边界与电压网格的细节敏感，存在
  实现层面的差异（原始提取管线未随仓库发布）。因此：
    * 要逐位复现论文 4.7 的输入 -> 直接用随仓库的 data/mit_dq_early.csv；
    * 要在自己提取的特征上重跑基线 -> 运行本脚本后执行 code/early_pred/t5_early_pred.py。

用法：python code/data_prep/extract_mit_dq_early.py --mit-dir <含 b1c*.parquet 的目录> \
            --out data/mit_dq_early_extracted.csv
输出：与 data/mit_dq_early.csv 同列结构（battery_id, cycle_10, cycle_100, dq_var,
dq_mean, dq_slope, dq_intercept, dq_min, dq_max, cycle_life, n_cycles_total），
其中 cycle_life 需要各数据集的 EOL 标注（README/DATA.md）；本脚本若无法确定
EOL 则写 -1，由调用方补充。
"""
import argparse
import glob
import os
import numpy as np
import pandas as pd

GRID_LO, GRID_HI, GRID_N = 2.0, 3.5, 1000
CYCLE_A, CYCLE_B = 10, 100


def dq_features(parquet_path):
    df = pd.read_parquet(parquet_path)
    dis = df[df["current_A"] < -0.01]  # 放电段
    curves = {}
    for cyc in (CYCLE_A, CYCLE_B):
        c = dis[dis["cycle_number"] == cyc].sort_values("time_s")
        if len(c) < 10:
            return None
        curves[cyc] = (c["voltage_V"].values, c["capacity_Ah"].values,
                       float(c["capacity_Ah"].max()))
    (v_a, q_a, m_a), (v_b, q_b, m_b) = curves[CYCLE_A], curves[CYCLE_B]
    # capacity_Ah 在放电段是"剩余容量"计数器（满电时最大、放电结束时归零），
    # 直接相减会混入两轮总容量的差；补上常数偏移后 ΔQ 才是"放电容量之差"
    offset = m_a - m_b
    grid = np.linspace(GRID_LO, GRID_HI, GRID_N)
    qa = np.interp(grid, np.sort(v_a), q_a[np.argsort(v_a)])
    qb = np.interp(grid, np.sort(v_b), q_b[np.argsort(v_b)])
    dq = (qb - qa) + offset
    slope, intercept = np.polyfit(grid, dq, 1)
    n_cycles = int(df["cycle_number"].max())
    return {"dq_var": float(np.var(dq)), "dq_mean": float(np.mean(dq)),
            "dq_slope": float(slope), "dq_intercept": float(intercept),
            "dq_min": float(dq.min()), "dq_max": float(dq.max()),
            "n_cycles_total": n_cycles}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mit-dir", required=True, help="含 b1c*.parquet 的目录")
    ap.add_argument("--out", default="data/mit_dq_early_extracted.csv")
    args = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(os.path.join(args.mit_dir, "b1c*.parquet"))):
        bid = os.path.splitext(os.path.basename(p))[0]
        f = dq_features(p)
        if f is None:
            print("skip %s (cycle 10/100 数据不足)" % bid)
            continue
        f.update({"battery_id": bid, "cycle_10": CYCLE_A, "cycle_100": CYCLE_B,
                  "cycle_life": -1})  # EOL 标注不在原始 parquet 中，需按 DATA.md 补
        rows.append(f)
    out = pd.DataFrame(rows)[["battery_id", "cycle_10", "cycle_100", "dq_var",
                              "dq_mean", "dq_slope", "dq_intercept", "dq_min",
                              "dq_max", "cycle_life", "n_cycles_total"]]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print("wrote %s (%d cells)" % (args.out, len(out)))


if __name__ == "__main__":
    main()
