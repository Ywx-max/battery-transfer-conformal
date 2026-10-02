# -*- coding: utf-8 -*-
"""I5：建模表 CX2 追加（生成 建模表_v3_cx2 / v3c_cx2，冻结表一字不动）。

组成规则：
  v3_cx2  = 冻结 建模表_v3 的全部行（MIT/NASA/CS2 逐行原样） + CX2 8 颗的新行
  v3c_cx2 = 冻结 建模表_v3c 的全部行 + CX2 行的 7 列曲线特征
CX2 行来源：
  基础特征 = parse_calce.py（v1 口径）对 CX2 的解析行（capacity/soh/ica_peak/
             ica_peak_V/charge_dur_s；discharge_dur_s/v_mean_V/v_min_V/eol_cycle/
             rul 与 CS2 同口径留空）
  曲线特征 = parse_calce_v2 的"容量+特征对齐流"（xlsx 电芯：v_q10 非空特征流与
             电流积分容量流按放电循环先后位置对齐，同 parse_calce_v2.main；
             txt 电芯：process_txt_cell 单遍输出）。
  ★ 管线考古结论（2026-10-02）：冻结 v3c 的 CALCE 曲线列与 calce_full_v2.csv 按
    cycle 键连接**逐值一致**（8 电芯 0 不一致，本脚本 assert 复验），而非仓库
    build_modeling_table.py 重建版所读的 calce_curve_features.csv（该 CSV 的
    循环编号属于更早的管线时代，与 v1 键仅 ~39% 相交）。因此本脚本沿用真实机制。

自验证（脚本内 assert）：
  V1  非_CALCE 行与冻结 v3 逐行相同；CALCE 行数 = 旧 CALCE + CX2；CX2 三缺失列 100% 空；
  V2  用现行代码重建的 full_v2（CS2 部分）按 cycle 连接，与冻结 v3c 的 CALCE 曲线列
      逐值一致（等价于 I1/I4.1 的 1e-14 回归 + 装配机制验证）。

运行：python code/data_prep/build_modeling_table_cx2.py
输出：data/建模表_v3_cx2.csv.gz、data/建模表_v3c_cx2.csv.gz、data/calce_full_v2_cx2.csv
"""
import sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import parse_calce as pc
import parse_calce_v2 as pv

DATA = Path("data")
CX2_XLSX = ["CX2_16", "CX2_33", "CX2_34", "CX2_35", "CX2_36", "CX2_37", "CX2_38"]
CX2_TXT = ["CX2_31"]
CURVE = ["v_q10", "v_q30", "v_q50", "v_q70", "v_q90", "ica2_peak", "ica_fwhm"]


def v1_rows(cell):
    f = pc.calce_txt_full(pc.CAL / f"{cell}.zip", cell) if cell in CX2_TXT \
        else pc.calce_xlsx_full(pc.CAL / f"{cell}.zip", cell)
    return f.rename(columns={"cycle_id": "cycle"}).sort_values("cycle").reset_index(drop=True)


def v2_aligned_rows(cell):
    """parse_calce_v2 口径的对齐行（xlsx：特征流与容量流位置对齐；txt：单遍输出）。"""
    z = pv.CAL / f"{cell}.zip"
    with zipfile.ZipFile(z) as zf:
        has_xlsx = any(n.lower().endswith(".xlsx") for n in zf.namelist())
    if has_xlsx:
        f_feat = pv.process_xlsx_cell(z, cell)
        f_cap = pv.process_xlsx_capacity(z, cell).rename(
            columns={"gid": "cycle", "capacity_Ah_int": "capacity_Ah"})
        f_feat_d = f_feat[f_feat["v_q10"].notna()].sort_values("cycle").reset_index(drop=True)
        f_cap_s = f_cap.sort_values("cycle").reset_index(drop=True)
        n = min(len(f_feat_d), len(f_cap_s))
        if len(f_feat_d) != len(f_cap_s):
            print(f"  [对齐] {cell}: 特征循环 {len(f_feat_d)} vs 容量循环 {len(f_cap_s)}，取前 {n}")
        out = f_feat_d.iloc[:n].copy()
        out["capacity_Ah"] = f_cap_s["capacity_Ah"].values[:n]
        out["battery_id"] = cell
        out["soh"] = out["capacity_Ah"] / out["capacity_Ah"].dropna().iloc[0]
        return out
    m = pv.process_txt_cell(z, cell)
    m["soh"] = m["capacity_Ah"] / m["capacity_Ah"].dropna().iloc[0]
    return m


def main():
    t0 = time.time()
    v3 = pd.read_csv(DATA / "建模表_v3.csv.gz", compression="gzip")
    v3c = pd.read_csv(DATA / "建模表_v3c.csv.gz", compression="gzip")

    # ---- CX2 行（v1 基础特征 + v2 曲线特征）----
    cx2_base, cx2_curve, full_parts = [], [], []
    for cell in CX2_XLSX + CX2_TXT:
        f1 = v1_rows(cell)
        f2 = v2_aligned_rows(cell)
        base = pd.DataFrame({
            "dataset": "CALCE", "battery_id": cell, "cycle": f1["cycle"].values,
            "capacity_Ah": f1["capacity_Ah"].values, "soh": f1["soh"].values,
            "discharge_dur_s": np.nan, "v_mean_V": np.nan, "v_min_V": np.nan,
            "ica_peak": f1["ica_peak"].values, "ica_peak_V": f1["ica_peak_V"].values,
            "charge_dur_s": f1["charge_dur_s"].values, "eol_cycle": np.nan, "rul": np.nan})
        cx2_base.append(base)
        cx2_curve.append(f2[["battery_id", "cycle"] + CURVE])
        full_parts.append(f2)
        print(f"{cell}: v1 行 {len(f1)} | v2 对齐行 {len(f2)}", flush=True)
    cx2 = pd.concat(cx2_base, ignore_index=True)
    cx2c = pd.concat(cx2_curve, ignore_index=True)
    full_cx2 = pd.concat(full_parts, ignore_index=True)[
        ["battery_id", "cycle", "ica_main_peak", "ica_main_V", "ica2_peak", "ica_fwhm",
         "v_q10", "v_q30", "v_q50", "v_q70", "v_q90", "capacity_Ah", "ica2_peak_dch",
         "ica_fwhm_dch", "soh"]]

    # ---- V2 自验证：现行代码重建的 full_v2（CS2 部分）连接后 == 冻结 v3c 曲线列 ----
    ref_cs2_parts = []
    for cell in ["CS2_33", "CS2_34", "CS2_35", "CS2_36", "CS2_37", "CS2_38"]:
        ref_cs2_parts.append(v2_aligned_rows(cell))
    for cell in ["CS2_8", "CS2_21"]:
        ref_cs2_parts.append(v2_aligned_rows(cell))
    full_cs2 = pd.concat(ref_cs2_parts, ignore_index=True)
    frozen_cal = v3c[v3c.dataset == "CALCE"]
    m = frozen_cal.merge(full_cs2[["battery_id", "cycle"] + CURVE],
                         on=["battery_id", "cycle"], suffixes=("_fz", "_re"))
    # 未被 full_v2 覆盖的冻结行，其曲线列在冻结表中本就全空（7584-7543=41 行）
    unmatched = frozen_cal.merge(full_cs2[["battery_id", "cycle"]], on=["battery_id", "cycle"],
                                 how="left", indicator=True)
    un = unmatched[unmatched._merge == "left_only"]
    assert un[CURVE].notna().sum().sum() == 0, "未匹配行含非空曲线列，机制不符"
    mism = 0
    for c in CURVE:
        x, y = m[c + "_fz"], m[c + "_re"]
        mism += int((pd.isna(x) != pd.isna(y)).sum())
        ok = ~(pd.isna(x) | pd.isna(y))
        if ok.sum():
            mism += int((np.abs(x[ok].values - y[ok].values) > 1e-9).sum())
    print(f"V2 自验证: CS2 曲线列 {len(m)}/{len(frozen_cal)} 行比对（余 {len(un)} 行冻结侧本为空）, 不一致 {mism}")
    assert mism == 0, "重建 full_v2 与冻结 v3c 曲线列不一致，停止"

    # ---- 组装 ----
    old_cal_keys = set(map(tuple, v3[v3.dataset == "CALCE"][["battery_id", "cycle"]].values))
    assert not (old_cal_keys & set(map(tuple, cx2[["battery_id", "cycle"]].values))), "键冲突"
    v3_cx2 = pd.concat([v3, cx2[v3.columns]], ignore_index=True)
    cx2_v3c_rows = cx2.merge(cx2c, on=["battery_id", "cycle"], how="left", suffixes=("", "_c"))
    assert cx2_v3c_rows.shape[0] == len(cx2), "曲线键重复导致行爆炸"
    cx2_v3c_rows = cx2_v3c_rows.reindex(columns=v3c.columns)
    v3c_cx2 = pd.concat([v3c, cx2_v3c_rows], ignore_index=True)

    # ---- V1/V3 断言 ----
    assert v3_cx2.shape[0] == v3.shape[0] + len(cx2)
    pd.testing.assert_frame_equal(
        v3_cx2[v3_cx2.dataset != "CALCE"].reset_index(drop=True),
        v3[v3.dataset != "CALCE"].reset_index(drop=True))
    pd.testing.assert_frame_equal(
        v3_cx2[(v3_cx2.dataset == "CALCE") & (v3_cx2.battery_id.str.startswith("CS2"))].reset_index(drop=True),
        v3[v3.dataset == "CALCE"].reset_index(drop=True))
    cx2_chk = v3_cx2[v3_cx2.battery_id.str.startswith("CX2")]
    for c in ("discharge_dur_s", "v_mean_V", "v_min_V"):
        assert cx2_chk[c].notna().sum() == 0, f"CX2 行 {c} 应为空"
    assert v3c_cx2.shape[0] == v3c.shape[0] + len(cx2)
    print(f"V1/V3 断言通过: v3_cx2 {v3_cx2.shape}, v3c_cx2 {v3c_cx2.shape}, "
          f"CALCE {v3[v3.dataset=='CALCE'].shape[0]} -> {v3_cx2[v3_cx2.dataset=='CALCE'].shape[0]} 行")

    v3_cx2.to_csv(DATA / "建模表_v3_cx2.csv.gz", index=False, compression="gzip")
    v3c_cx2.to_csv(DATA / "建模表_v3c_cx2.csv.gz", index=False, compression="gzip")
    full_cx2.sort_values(["battery_id", "cycle"]).to_csv(
        DATA / "calce_full_v2_cx2.csv", index=False, encoding="utf-8-sig")
    print(f"已输出: 建模表_v3_cx2.csv.gz / 建模表_v3c_cx2.csv.gz / calce_full_v2_cx2.csv ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
