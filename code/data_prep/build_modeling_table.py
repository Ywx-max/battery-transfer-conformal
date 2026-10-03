# -*- coding: utf-8 -*-
"""建模表构建：三张分数据集特征表 → 一张统一列名的建模表（下游全部实验的输入）。

合并规则就是"列名对齐 + 打 dataset 标签后纵向拼接"，各数据集能提供的量并不相同，
缺的列留空而不是补 0（下游按列取用时自动跳过空值）：

  NASA : nasa_features.csv 直接可用（字段最全，但没有平均放电电压）
  MIT  : mit_capacity.csv（容量/放电时长/SOH/EOL）+ mit_features.csv（电压统计量与 ICA）
         按 (电池, 循环) 对齐；MIT 原始数据未给出充电段时长，charge_dur_s 留空
  CALCE: calce_features.csv（容量取清洗列；该数据集没有放电时长与电压统计量，
         EOL 标签也不适用——CALCE 电芯未跑到寿命终点，eol_cycle/rul 留空）

v3c = v3 + 7 列放电曲线特征（v_q10…v_q90、ICA 次峰 ica2_peak、半高宽 ica_fwhm），
消融实验用的就是这一版。曲线表里 CALCE 的键叫 cycle_id、NASA 的列名带 _dch 后缀，
都在这里统一。

输入：data/ 下 mit_capacity.csv、mit_features.csv、calce_features.csv、nasa_features.csv；
      做 v3c 还需要 mit_curve_features.csv、calce_curve_features.csv、nasa_curve_features.csv
输出：data/modeling_table_v3.csv、data/modeling_table_v3c.csv

    python build_modeling_table.py

数据版本说明：v3 与 v5 两版建模表在解析口径上存在差异（详见 results/README.md），
因此本脚本按论文所述规则重建的结果，与论文实验所用的那一版表在个别电芯的离群点/EOL 标注上
可能有一行级差别。论文实验实际使用的两张表随仓库提供（data/建模表_v3.csv.gz、
data/建模表_v3c.csv.gz），解压后可直接替代本脚本的输出。
"""
from pathlib import Path
import pandas as pd

# 与其它脚本一致：数据放 data/ 下（建模表也写回这里）
DATA = Path("data")
OUT = DATA

FEATS = ["capacity_Ah", "soh", "discharge_dur_s", "v_mean_V", "v_min_V",
         "ica_peak", "ica_peak_V", "charge_dur_s", "eol_cycle", "rul"]
CURVE = ["v_q10", "v_q30", "v_q50", "v_q70", "v_q90", "ica2_peak", "ica_fwhm"]


def read(name):
    return pd.read_csv(DATA / name)


def mit_part():
    cap = read("mit_capacity.csv")
    feat = read("mit_features.csv")
    df = cap.merge(feat, on=["battery_id", "cycle"], how="left", suffixes=("", "_f"))
    df["v_mean_V"] = df["v_mean_V"]
    df["v_min_V"] = df["v_min_V"]
    df["ica_peak"] = df["ica_peak"]
    df["ica_peak_V"] = df["ica_peak_V"]
    df["charge_dur_s"] = pd.NA          # MIT 无充电段时长
    return df


def calce_part():
    # 容量取列 capacity_Ah（与其它数据集同口径的原始解析值；capacity_Ah_clean 只作对照留档）
    df = read("calce_features.csv")
    for col in ["discharge_dur_s", "v_mean_V", "v_min_V", "eol_cycle", "rul"]:
        df[col] = pd.NA
    return df


def nasa_part():
    df = read("nasa_features.csv")
    df["v_mean_V"] = pd.NA              # NASA 未给平均放电电压
    return df


def build_v3():
    parts = []
    for name, df in [("NASA", nasa_part()), ("MIT", mit_part()), ("CALCE", calce_part())]:
        df = df.copy()
        df.insert(0, "dataset", name)
        parts.append(df[["dataset", "battery_id", "cycle"] + FEATS])
    return pd.concat(parts, ignore_index=True)


def curve_part(dataset):
    """7 列曲线特征；三张表的键名/列名在这里对齐"""
    if dataset == "MIT":
        df = read("mit_curve_features.csv")
    elif dataset == "CALCE":
        df = read("calce_curve_features.csv").rename(columns={"cycle_id": "cycle"})
    else:
        df = read("nasa_curve_features.csv").rename(
            columns={"ica2_peak_dch": "ica2_peak", "ica_fwhm_dch": "ica_fwhm"})
    return df[["battery_id", "cycle"] + CURVE]


def build_v3c(v3):
    parts = []
    for name in ["NASA", "MIT", "CALCE"]:
        parts.append(curve_part(name).assign(dataset=name))
    curves = pd.concat(parts, ignore_index=True)
    return v3.merge(curves, on=["dataset", "battery_id", "cycle"], how="left")


if __name__ == "__main__":
    v3 = build_v3()
    v3.to_csv(OUT / "modeling_table_v3.csv", index=False)
    v3c = build_v3c(v3)
    v3c.to_csv(OUT / "modeling_table_v3c.csv", index=False)
    print("modeling_table_v3.csv :", v3.shape)
    print("modeling_table_v3c.csv:", v3c.shape)
    print("数据集行数:", v3["dataset"].value_counts().to_dict())
