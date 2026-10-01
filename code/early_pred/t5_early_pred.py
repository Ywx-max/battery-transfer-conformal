# -*- coding: utf-8 -*-
"""早期寿命预测（论文 4.7）：只用前 100 个循环的信息预测整条寿命。

思路照 Severson 等 2019 的差分容量曲线：取循环 10 与循环 100 的放电电压-容量曲线相减，
在 ΔQ(V) 上取 6 个统计量（方差、均值、斜率、截距、最小、最大）作特征，
岭回归回归 log10(寿命)，评估用留一电芯——119 颗有寿命标签的电芯逐颗留出。

固定设置：特征先标准化，Ridge(alpha=1.0)，目标 log10(cycle_life)。这套参数是在公开数据上
试出来的默认值（没有做超参搜索），因此本基线是确定性单次运行，不涉及随机种子。

输入：data/mit_dq_early.csv（ΔQ 特征表，由公开 MIT-Stanford 数据按上式提取，随仓库提供）
输出：results/early_pred_summary.json、results/early_pred_results.csv（逐电芯预测明细）

    python t5_early_pred.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = Path("data/mit_dq_early.csv")
OUT_DIR = Path("results")
FEATURES = ["dq_var", "dq_mean", "dq_slope", "dq_intercept", "dq_min", "dq_max"]


def main():
    df = pd.read_csv(DATA)
    df = df[df["cycle_life"].notna() & (df["cycle_life"] > 0)].reset_index(drop=True)
    X = df[FEATURES].to_numpy(dtype=float)
    y_log = np.log10(df["cycle_life"].to_numpy(dtype=float))

    # 留一电芯：每颗电芯当一次测试集，其余 118 颗训练
    pred_cycles = np.empty(len(df))
    for i in range(len(df)):
        train = np.arange(len(df)) != i
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(X[train], y_log[train])
        pred_cycles[i] = 10 ** model.predict(X[i : i + 1])[0]

    true_cycles = df["cycle_life"].to_numpy(dtype=float)
    resid = pred_cycles - true_cycles
    summary = {
        "rmse_log": float(np.sqrt(np.mean((np.log10(pred_cycles) - y_log) ** 2))),
        "rmse_cycles": float(np.sqrt(np.mean(resid ** 2))),
        "mae_cycles": float(np.mean(np.abs(resid))),
        "rel_err_pct": float(np.mean(np.abs(resid) / true_cycles) * 100),
        "n": int(len(df)),
        "features": FEATURES,
    }

    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / "early_pred_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    pd.DataFrame({
        "battery_id": df["battery_id"],
        "cycle_life": true_cycles,
        "pred_cycle_life": pred_cycles,
    }).to_csv(OUT_DIR / "early_pred_results.csv", index=False)

    print("电芯数: %d" % summary["n"])
    print("log10 空间 RMSE : %.4f" % summary["rmse_log"])
    print("循环数空间 RMSE : %.1f" % summary["rmse_cycles"])
    print("逐电芯相对误差  : %.2f%%" % summary["rel_err_pct"])


if __name__ == "__main__":
    main()
