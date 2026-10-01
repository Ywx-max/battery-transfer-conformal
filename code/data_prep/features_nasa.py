# -*- coding: utf-8 -*-
"""NASA 每循环健康特征工程（T1 第二步）：扫描 .mat，一个放电循环产出一行。

特征 = 容量/SOH/放电时长/最低电压 + "前驱充电循环"的 ICA 三件套
（峰值、峰位、充电时长）。取前驱充电而非同循环，是因为充电发生在
放电之前，预测当前循环时它已经是发生过的事，不构成未来信息泄漏。

ICA 口径与 CALCE 略有差异：这里用 5mV 网格（CALCE 10mV）+ 0.9 倍 CC 基准
（CALCE 0.7）。NASA 采样更稀，网格粗了峰值会糊；CC 基准取前 20 点中位数，
但充电起始若含静置尾巴，基准会偏低、CC 段偏保守（已知问题，
详见 数据/解析说明_NASA特征.md 的待办）。

输入：data/raw/nasa/*.mat；输出：data/nasa_features.csv"""
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio

RAW_DIR = Path("data/raw/nasa")
OUT_DIR = Path("data")
V_LO, V_HI, DV = 3.90, 4.19, 0.005
GRID = np.arange(V_LO, V_HI, DV)

def pick(obj, *names):
    for n in names:
        if hasattr(obj, n):
            return np.atleast_1d(np.asarray(getattr(obj, n))).astype(float)
    return None

def ica_features(v, i, t):
    """CC 过滤后的充电段 → (ICA 主峰高度, 峰位置)。
    电压-容量对先按电压排序再插值到 5mV 网格：原始序列电压基本单调
    但偶有回跳，不排序的话 np.interp 会出垃圾。"""
    dt = np.diff(t)
    q = np.concatenate([[0], np.cumsum(np.abs(i[1:]) * dt)]) / 3600.0
    vm = (v[:-1] + v[1:]) / 2.0
    qm = (q[:-1] + q[1:]) / 2.0
    mm = (vm >= V_LO) & (vm <= V_HI)
    if mm.sum() < 50:
        return np.nan, np.nan
    vg, qg = vm[mm], qm[mm]
    order = np.argsort(vg)
    q_grid = np.interp(GRID, vg[order], qg[order])
    dqdv = np.diff(q_grid) / DV
    k = int(np.nanargmax(dqdv))
    return float(dqdv[k]), float(GRID[k])

def main():
    rows = []
    for mp in sorted(RAW_DIR.glob("B00*.mat")):
        mat = sio.loadmat(str(mp), struct_as_record=False, squeeze_me=True)
        key = next(k for k in mat if not k.startswith("__"))
        cycles = np.atleast_1d(mat[key].cycle)
        last_charge, n_dis = None, 0
        for c in cycles:
            ctype = str(c.type)
            d = c.data
            if ctype == "charge":
                v = pick(d, "voltage_measured", "Voltage_measured")
                i = pick(d, "current_measured", "Current_measured")
                t = pick(d, "time", "Time")
                if v is None or i is None or t is None:
                    last_charge = {"ica_peak": np.nan, "ica_peak_V": np.nan, "charge_dur_s": np.nan}
                else:
                    i_nom = np.median(np.abs(i[:20]))  # CC 基准：前 20 点中位数（含静置时偏低，见 docstring）
                    cc = np.abs(i) >= 0.9 * i_nom
                    if cc.sum() >= 50:
                        pk, pv = ica_features(v[cc], i[cc], t[cc])
                    else:
                        pk, pv = np.nan, np.nan
                    last_charge = {"ica_peak": pk, "ica_peak_V": pv, "charge_dur_s": float(t[-1])}
            elif ctype == "discharge":
                # 循环号按"放电次数"重新编（.mat 原始 index 混着内阻测量等非放电循环）
                n_dis += 1
                cap = float(pick(d, "Capacity")[0])
                t_arr = pick(d, "time", "Time")
                v_arr = pick(d, "voltage_measured", "Voltage_measured")
                rows.append({"battery_id": key, "cycle": n_dis, "capacity_Ah": cap,
                             "soh": cap / 2.0, "discharge_dur_s": float(t_arr[-1]),
                             "v_min_V": float(v_arr.min()), **(last_charge or {})})
    df = pd.DataFrame(rows)
    eol = df[df["capacity_Ah"] < 1.4].groupby("battery_id")["cycle"].min()
    df["eol_cycle"] = df["battery_id"].map(eol).fillna(-1).astype(int)
    df["rul"] = np.where(df["eol_cycle"] > 0, df["eol_cycle"] - df["cycle"], np.nan)
    out = OUT_DIR / "nasa_features.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df.groupby("battery_id").agg(n=("cycle","max"), ica_first=("ica_peak","first"),
          ica_last=("ica_peak","last"), miss=("ica_peak", lambda s: s.isna().sum())).round(3).to_string())
    print("已输出:", out)

if __name__ == "__main__":
    main()
