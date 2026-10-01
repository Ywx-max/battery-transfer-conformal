# -*- coding: utf-8 -*-
"""NASA PCoE 解析（T1）：4 颗 18650 钴酸锂（B0005/6/7/0018）→ 容量退化序列。

NASA 的 .mat 结构出了名的各版本不统一：字段一会儿 time 一会儿 Time。
pick() 按候选名顺序取第一个存在的字段，省得为大小写写一堆 if。
EOL 定为容量衰减到额定 2.0Ah 的 70%（1.4Ah）。注意与 CALCE/MIT 的 80%
口径不同，这是数据集自身的退役标准，论文 3.1 有说明。

本脚本只产容量级序列（nasa_capacity.csv + 退化曲线图）；
循环级特征（含 ICA）在 features_nasa.py，两步分开是因为 ICA 解析慢得多，
调特征参数时不想每次都重扫 .mat。

输入：data/raw/nasa/*.mat；输出：data/nasa_capacity.csv、results/nasa_degradation.png"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.io as sio

RAW_DIR = Path("data/raw/nasa")
OUT_DIR = Path("data")
FIG_DIR = Path("results")
RATED_AH = 2.0
EOL_AH = RATED_AH * 0.70

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

def pick(obj, *names):
    """按候选字段名取第一个存在的（.mat 字段大小写在不同批次里不一致）。"""
    for n in names:
        if hasattr(obj, n):
            return np.atleast_1d(np.asarray(getattr(obj, n))).astype(float)
    return None

def parse_battery(mat_path):
    mat = sio.loadmat(str(mat_path), struct_as_record=False, squeeze_me=True)
    key = next(k for k in mat if not k.startswith("__"))  # 排除 __header__ 等元数据键
    cycles = np.atleast_1d(mat[key].cycle)
    rows = []
    for i, c in enumerate(cycles):
        if str(c.type) != "discharge":
            continue
        d = c.data
        cap_f = pick(d, "Capacity")
        t_amb = pick(c, "ambient_temperature", "Ambient_Temperature")
        t_arr = pick(d, "time", "Time")
        v_arr = pick(d, "voltage_measured", "Voltage_measured")
        cap = float(cap_f[0]) if cap_f is not None else np.nan
        rows.append({"battery_id": key, "cycle": len(rows)+1, "cycle_idx_raw": i,
                     "capacity_Ah": cap, "soh": cap / RATED_AH,
                     "ambient_T_C": float(t_amb[0]) if t_amb is not None else np.nan,
                     "discharge_dur_s": float(t_arr[-1]) if t_arr is not None else np.nan,
                     "v_min_V": float(v_arr.min()) if v_arr is not None else np.nan})
    return pd.DataFrame(rows)

def main():
    mats = sorted(RAW_DIR.glob("B00*.mat"))
    if not mats:
        raise SystemExit(f"未找到 .mat 文件: {RAW_DIR}")
    df = pd.concat([parse_battery(m) for m in mats], ignore_index=True)
    # EOL = 首次跌破 1.4Ah 的循环；RUL = EOL 循环 - 当前循环（未达 EOL 的电芯为 NaN，
    # 后续 RUL 口径实验会自动跳过这些电芯）
    eol_map = df[df["capacity_Ah"] < EOL_AH].groupby("battery_id")["cycle"].min()
    df["eol_cycle"] = df["battery_id"].map(eol_map).fillna(-1).astype(int)
    df["rul"] = np.where(df["eol_cycle"] > 0, df["eol_cycle"] - df["cycle"], np.nan)
    out_csv = OUT_DIR / "nasa_capacity.csv"
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(df.groupby("battery_id").agg(cycles=("cycle","max"),
          cap_first=("capacity_Ah","first"), cap_last=("capacity_Ah","last"),
          eol_cycle=("eol_cycle","first")).to_string())
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    for bid, g in df.groupby("battery_id"):
        ax.plot(g["cycle"], g["capacity_Ah"], marker=".", ms=3, lw=1, label=bid)
    ax.axhline(EOL_AH, color="r", ls="--", lw=1, label="EOL 1.4 Ah")
    ax.set_xlabel("循环次数"); ax.set_ylabel("放电容量 (Ah)")
    ax.set_title("NASA PCoE 电池容量退化曲线")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "nasa_degradation.png")
    print("已输出:", out_csv)

if __name__ == "__main__":
    main()
