# -*- coding: utf-8 -*-
"""CALCE CS2 解析（T1 第一步）：8 颗钴酸锂方壳电池 → 每循环一行的特征表。

CALCE 官网的原始 zip 是两种采集设备混着的，字段和容量口径都不一样：
  CS2_8 / CS2_21  CADEX txt：Duration 才是真时间基，Capacity 列其实是
                  "放电深度百分比"，要乘额定容量才是 Ah（这个坑踩过）
  其余 6 颗       Arbin xlsx：按放电步的 Discharge_Capacity 差值算容量

ICA（增量容量 dQ/dV）取充电段 3.90-4.19V 窗口：CC 段过滤 + 电压单调化 +
10mV 分箱。分箱不是可选项：1s 采样下电压有小幅振荡，逐点差分会得到
锯齿状的假 dQ/dV，峰值位置直接废掉。

输入：data/raw/calce/CS2_*.zip；输出：data/calce_features.csv（含离点标记列）。
注意 parse_calce_v2.py 是本脚本的重构统一版（容量改电流积分、补 v_q 特征），
两版的输出不要混用。"""
import io, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

CAL = Path("data/raw/calce")   # CS2_*/CX2_*.zip 放在这里
OUT = Path("data")             # 解析结果输出目录
DV = 0.010
# 额定容量按电芯系列区分（2026-10 CX2 扩充）：CS2 = 1.1 Ah，CX2 = 1.35 Ah。
# txt 路径的容量 = 放电深度百分比 × 额定容量，RATED 拿错会把 SOH 系统性算错
# （CX2 若沿用 1.1，全新电芯只有 0.815 而非 1.0，目标域 SOH 整体下移 18.5%）
RATED_BY_PREFIX = {"CS2": 1.1, "CX2": 1.35}


def rated_for(cell):
    return RATED_BY_PREFIX[cell.split("_")[0]]

def robust_ica(v, i, t, v_lo=3.90, v_hi=4.19):
    """一段充电数据 → (ICA 主峰高度, 峰位置 V)。

    步骤：CC 过滤（|I| >= 0.7×中位数，CV 段的 dQ/dV 没有物理意义）→
    电压单调化（+0.5mV 门槛去抖动）→ 10mV 分箱平均 → 差分取最大。"""
    if len(t) < 30:
        return np.nan, np.nan
    thr = 0.7 * np.median(np.abs(i))
    m = np.abs(i) >= thr
    v, i, t = v[m], i[m], t[m]
    if len(v) < 8:
        return np.nan, np.nan
    dt = np.diff(t)
    if np.median(dt) <= 0:
        return np.nan, np.nan
    q = np.concatenate([[0], np.cumsum(np.abs(i[1:]) * dt)]) / 3600.0
    last = -1e9; kv, kq = [], []
    for vi, qi in zip(v, q):
        if vi >= last + 0.0005:
            kv.append(vi); kq.append(qi); last = vi
    kv, kq = np.array(kv), np.array(kq)
    nb = int((v_hi - v_lo) / DV)
    bins = np.floor((kv - v_lo) / DV).astype(int)
    ok = (bins >= 0) & (bins < nb)
    if ok.sum() < 8:
        return np.nan, np.nan
    bq = np.full(nb, np.nan)
    bs = pd.Series(kq[ok]).groupby(bins[ok]).mean()
    bq[bs.index.values.astype(int)] = bs.values
    dq = np.diff(bq) / DV
    centers_mid = (v_lo + (np.arange(nb) + 0.5) * DV)[:-1]
    mask = np.isfinite(dq)
    if mask.sum() < 4:
        return np.nan, np.nan
    k = int(np.argmax(np.where(mask, dq, -np.inf)))
    return float(dq[k]), float(centers_mid[k])

def calce_txt_full(zip_path, cell):
    """CADEX txt 版（CS2_8/21）。状态机逐行扫：rest→charge→discharge 切换即一轮；
    充电段攒缓冲算 ICA，放电段记容量。容量的真身是"放电深度百分比"列 × 额定容量。"""
    cap_rows, ica_rows = [], []
    n_cyc, state = 0, 'rest'
    v_buf, i_buf, t_buf = [], [], []
    with zipfile.ZipFile(zip_path) as z:
        names = sorted(n for n in z.namelist() if n.lower().endswith('.txt'))
        for name in names:
            with z.open(name) as f:
                t = pd.read_csv(f, sep='\t', encoding='utf-8', engine='python')
            t.columns = [c.strip() for c in t.columns]
            for mv, ma, du, cp in zip(t['mV'].astype(float).values, t['mA'].astype(float).values,
                                      t['Duration'].astype(float).values, t['Capacity'].astype(float).values):
                cur = 'charge' if ma > 0 else ('discharge' if ma < 0 else 'rest')
                if cur != state:
                    if state == 'charge' and len(t_buf) >= 30:
                        pk, pv = robust_ica(np.array(v_buf)/1000, np.array(i_buf)/1000, np.array(t_buf))
                        ica_rows.append((n_cyc + 1, pk, pv, t_buf[-1]-t_buf[0]))
                    if state == 'discharge':
                        n_cyc += 1
                        cap_rows.append((n_cyc, cp))
                    state, v_buf, i_buf, t_buf = cur, [], [], []
                if cur == 'charge':
                    v_buf.append(mv); i_buf.append(ma); t_buf.append(du)
        if state == 'discharge':
            n_cyc += 1
            cap_rows.append((n_cyc, cp))
    cap = pd.DataFrame(cap_rows, columns=['cycle_id', 'depth_pct'])
    cap['capacity_Ah'] = cap['depth_pct'] / 100.0 * rated_for(cell)
    cap = cap[cap['capacity_Ah'] > 0.2]
    ica = pd.DataFrame(ica_rows, columns=['cycle_id', 'ica_peak', 'ica_peak_V', 'charge_dur_s'])
    ica = ica.dropna(subset=['ica_peak']).groupby('cycle_id').agg(
        ica_peak=('ica_peak', 'mean'), ica_peak_V=('ica_peak_V', 'mean'),
        charge_dur_s=('charge_dur_s', 'sum')).reset_index()
    m = cap.merge(ica, on='cycle_id', how='left')
    m.insert(0, 'battery_id', cell)
    m['soh'] = m['capacity_Ah'] / m['capacity_Ah'].iloc[0]
    return m

def calce_xlsx_full(zip_path, cell):
    """Arbin xlsx 版（其余 6 颗）。xlsx 按 (Cycle_Index, Step_Index) 分组，
    放电步容量 = Discharge_Capacity 最大最小值之差。
    offset 累计是关键：每个 xlsx 文件的 Cycle_Index 都从 1 重新数，
    不加偏移的话后面的文件会把前面的循环覆盖掉（即旧注释"循环号=文件偏移+Cycle_Index"）。"""
    cap_rows, ica_rows, offset = [], [], 0
    with zipfile.ZipFile(zip_path) as z:
        xls = sorted(n for n in z.namelist() if n.lower().endswith('.xlsx'))
        for name in xls:
            xf = pd.ExcelFile(io.BytesIO(z.read(name)), engine='calamine')
            for sheet in [s for s in xf.sheet_names if s.lower().startswith('channel')]:
                d = xf.parse(sheet)
                d.columns = [str(c).strip() for c in d.columns]
                sub = d[['Cycle_Index','Step_Index','Current(A)','Voltage(V)','Test_Time(s)','Discharge_Capacity(Ah)']].astype(float)
                for (cyc, step), sg in sub.groupby(['Cycle_Index','Step_Index']):
                    gid = offset + int(cyc)
                    if (sg['Current(A)'] < 0).any():
                        dcap = sg.loc[sg['Current(A)'] < 0, 'Discharge_Capacity(Ah)']
                        cap_rows.append((gid, dcap.max() - dcap.min()))
                    if (sg['Current(A)'] > 0).any():
                        c = sg[sg['Current(A)'] > 0]
                        pk, pv = robust_ica(c['Voltage(V)'].values, c['Current(A)'].values, c['Test_Time(s)'].values)
                        ica_rows.append((gid, pk, pv, float(c['Test_Time(s)'].iloc[-1] - c['Test_Time(s)'].iloc[0])))
            offset = max((r[0] for r in cap_rows), default=0)
    cap = pd.DataFrame(cap_rows, columns=['cycle_id', 'c']).groupby('cycle_id')['c'].sum().reset_index()
    cap.columns = ['cycle_id', 'capacity_Ah']
    cap = cap[cap['capacity_Ah'] > 0.2]
    # 物理不可能的容量行剔除（CX2_16 的 22.7h 搁置段上 Arbin 累计计数器差值
    # 产生 12.5 Ah 行；放电不可能超过 1.5×额定）。行号保持不动，仅删行，
    # 因此 CS2（最大容量 ~1.25 Ah < 1.5×1.1）完全不受影响
    _bad = cap['capacity_Ah'] > 1.5 * rated_for(cell)
    if _bad.any():
        print(f"  [物理过滤] {cell}: 剔除 {int(_bad.sum())} 行超物理容量 "
              f"(max {cap.loc[_bad, 'capacity_Ah'].max():.2f} Ah)", flush=True)
        cap = cap[~_bad]
    ica = pd.DataFrame(ica_rows, columns=['cycle_id', 'pk', 'pv', 'dur']).dropna(subset=['pk'])
    ica = ica.groupby('cycle_id').agg(ica_peak=('pk','mean'), ica_peak_V=('pv','mean'),
                                      charge_dur_s=('dur','sum')).reset_index()
    m = cap.merge(ica, on='cycle_id', how='left')
    m.insert(0, 'battery_id', cell)
    m['soh'] = m['capacity_Ah'] / m['capacity_Ah'].iloc[0]
    return m

def main():
    frames, t0 = [], time.time()
    # 2026-10 CX2 扩充：纳入与 CS2 同条件的 8 颗（16/31/33/34/35/36/37/38），
    # 排除脉冲/温度循环/3C 工况电芯（CX2_3/4/8/32，论文 4.1 有排除准则）
    cells = ['CS2_8', 'CS2_21', 'CS2_33', 'CS2_34', 'CS2_35', 'CS2_36', 'CS2_37', 'CS2_38',
             'CX2_16', 'CX2_31', 'CX2_33', 'CX2_34', 'CX2_35', 'CX2_36', 'CX2_37', 'CX2_38']
    for cell in cells:
        f = calce_txt_full(CAL/f'{cell}.zip', cell) if cell in ('CS2_8','CS2_21','CX2_31') \
            else calce_xlsx_full(CAL/f'{cell}.zip', cell)
        frames.append(f)
        print(f"{cell}: {len(f)} 行 | {f['capacity_Ah'].iloc[0]:.3f}->{f['capacity_Ah'].iloc[-1]:.3f} Ah | ica中位 {f['ica_peak'].median():.2f}", flush=True)
    cal = pd.concat(frames, ignore_index=True).rename(columns={'cycle_id': 'cycle'})
    # 离群点清洗：11 点滚动中位数 ±25% 带外标记为离群（论文 4.1 的口径），
    # 清洗后的容量单独存 capacity_Ah_clean 列，原始列保留可查
    med = cal.groupby('battery_id')['capacity_Ah'].rolling(11, center=True, min_periods=3).median()
    cal['is_outlier'] = (cal['capacity_Ah'] - med.values).abs() > 0.25 * med.values
    cal['capacity_Ah_clean'] = cal['capacity_Ah'].where(~cal['is_outlier'])
    cal['capacity_Ah_clean'] = cal.groupby('battery_id')['capacity_Ah_clean'].transform(
        lambda s: s.interpolate(limit_direction='both'))
    cal.to_csv(OUT / 'calce_features.csv', index=False, encoding='utf-8-sig')
    print(f"已输出: {OUT / 'calce_features.csv'} ({len(cal)} 行, {time.time()-t0:.0f}s)")

if __name__ == "__main__":
    import time  # 放这里是为了 import 本模块取函数时不启动计时
    main()
