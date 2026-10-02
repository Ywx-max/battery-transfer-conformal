# -*- coding: utf-8 -*-
"""重建 calce_curve_features.csv（I1 补链路）：build_modeling_table.py 的曲线特征输入。

背景：论文表 6 的 curve14 消融依赖 calce_curve_features.csv，但产出它的驱动脚本
此前不在仓库（只有 03_实验 留档的参考产物）。本脚本基于 parse_calce_v2 的既有
特征函数（ica_curve/ica_shape/dch_curve_feats/process_xlsx_cell/process_txt_cell）
重建同一产物：每电芯跑特征流（**不做**容量对齐，含 v_q 为空的循环——这是本 CSV
与 calce_full_v2.csv 的行数差异来源），列名 cycle→cycle_id 供 build_modeling_table
 rename 使用。

列（与 2026-09 参考产物逐列一致）：
  battery_id, cycle_id, ica2_peak, ica_fwhm, ica_main_V, ica_main_peak,
  v_q10, v_q30, v_q50, v_q70, v_q90
其中 ica_main_* / ica2_peak / ica_fwhm 来自充电段 ICA（3.90-4.19 V 窗），
v_q* 来自放电段（2.0-3.5 V 定分数容量电压点）。

运行：python code/data_prep/build_calce_curve_features.py --cells cs2 [--out data/calce_curve_features.csv]
      --cells cs2 = 仅 CS2 8 颗（回归验证用）；--cells full = CS2 + CX2 16 颗
输出：data/calce_curve_features.csv"""
import argparse
import time
from pathlib import Path

import pandas as pd

from parse_calce_v2 import process_xlsx_cell, process_txt_cell, CAL

CURVE_COLS = ["ica2_peak", "ica_fwhm", "ica_main_V", "ica_main_peak",
              "v_q10", "v_q30", "v_q50", "v_q70", "v_q90"]
XLSX_CELLS = ["CS2_33", "CS2_34", "CS2_35", "CS2_36", "CS2_37", "CS2_38",
              "CX2_16", "CX2_33", "CX2_34", "CX2_35", "CX2_36", "CX2_37", "CX2_38"]
TXT_CELLS = ["CS2_8", "CS2_21", "CX2_31"]


def cell_zip(cell):
    p = CAL / f"{cell}.zip"
    if not p.exists():
        raise FileNotFoundError(f"缺少原始 zip: {p}")
    return p


def build(cells):
    frames = []
    for cell in cells:
        z = cell_zip(cell)
        # 格式判别：zip 内含 .xlsx 即 Arbin xlsx 路径（CX2 的混合 zip 里 .txt 是
        # 特殊测试的日志成员，不能据此判成 CADEX txt——CX2_16/33 都混有 txt 日志）
        import zipfile
        with zipfile.ZipFile(z) as zf:
            has_xlsx = any(n.lower().endswith(".xlsx") for n in zf.namelist())
        f = process_xlsx_cell(z, cell) if has_xlsx else process_txt_cell(z, cell)
        frames.append(f)
        print(f"{cell}: {len(f)} rows", flush=True)
    cal = pd.concat(frames, ignore_index=True).rename(columns={"cycle": "cycle_id"})
    cal = cal.sort_values(["battery_id", "cycle_id"]).reset_index(drop=True)
    return cal[["battery_id", "cycle_id"] + CURVE_COLS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", choices=["cs2", "full"], default="cs2",
                    help="cs2 = 仅 CS2 8 颗（与 2026-09 参考产物回归验证用）；full = CS2+CX2 16 颗")
    ap.add_argument("--out", default="data/calce_curve_features.csv")
    args = ap.parse_args()
    t0 = time.time()
    cells = XLSX_CELLS + TXT_CELLS if args.cells == "full" else \
        [c for c in XLSX_CELLS + TXT_CELLS if c.startswith("CS2")]
    cal = build(sorted(cells))
    cal.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"已输出 {args.out}（{len(cal)} 行, {time.time()-t0:.0f}s）")


if __name__ == "__main__":
    main()
