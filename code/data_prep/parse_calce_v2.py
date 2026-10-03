# -*- coding: utf-8 -*-
"""CALCE 统一提取 v2：每循环一行 = 容量(电流积分) + ICA 形状 + 放电 v_q 特征。

相比 parse_calce.py 的三点改进：
1. 容量改用放电步电流积分，不再依赖各家的容量列（txt 的 Capacity 列在状态
   切换行会取错值出 NaN；部分 xlsx 干脆缺 Discharge_Capacity 列）
2. 除 ICA 主峰外还挖了形状特征：主峰位置、次峰、半高宽。后面消融实验的
   "曲线增强 14 维"里那几个 curve 特征就是从这里来的
3. 特征与容量分开算再按放电循环顺序对齐（两者循环数偶尔不一致时取交集并告警）

本脚本产出 calce_full_v2.csv；v1 保留作中间产物出处。
xlsx 引擎用的 calamine（pip install python-calamine）：官方 openpyxl 对这批
文件又慢又偶尔报错，calamine 快一个数量级还更稳。"""
import io, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

CAL = Path("data/raw/calce")
OUT = Path("data")
V_LO, V_HI, DV = 3.90, 4.19, 0.010
# 额定容量按电芯系列区分（2026-10 CX2 扩充，与 parse_calce.py 同口径）：
# txt 路径容量 = 放电深度百分比 × 额定容量，CS2=1.1 Ah、CX2=1.35 Ah
RATED_BY_PREFIX = {"CS2": 1.1, "CX2": 1.35}
QFRACS = [0.1, 0.3, 0.5, 0.7, 0.9]


def rated_for(cell):
    return RATED_BY_PREFIX[cell.split("_")[0]]

def ica_curve(v, i, t):
    """一段充电数据 → (分箱后电压中心, dQ/dV)。CC 过滤 + 单调化 + 10mV 分箱，
    与 v1 同一套流水线；返回曲线而不是峰值，让调用方自己挖形状特征。"""
    if len(t) < 30: return None
    thr = 0.7 * np.median(np.abs(i))
    m = np.abs(i) >= thr
    v, i, t = v[m], i[m], t[m]
    if len(v) < 8: return None
    dt = np.diff(t)
    if np.median(dt) <= 0: return None
    q = np.concatenate([[0], np.cumsum(np.abs(i[1:]) * dt)]) / 3600.0
    last = -1e9; kv, kq = [], []
    for vi, qi in zip(v, q):
        if vi >= last + 0.0005:
            kv.append(vi); kq.append(qi); last = vi
    kv, kq = np.array(kv), np.array(kq)
    nb = int((V_HI - V_LO) / DV)
    bins = np.floor((kv - V_LO) / DV).astype(int)
    ok = (bins >= 0) & (bins < nb)
    if ok.sum() < 8: return None
    bq = np.full(nb, np.nan)
    bsg = pd.Series(kq[ok]).groupby(bins[ok]).mean()
    bq[bsg.index.values.astype(int)] = bsg.values
    if np.isfinite(bq).sum() < 4: return None
    ica = np.diff(bq) / DV
    centers = (V_LO + (np.arange(nb) + 0.5) * DV)[:-1]
    mask = np.isfinite(ica)
    return centers[mask], ica[mask]

def ica_shape(centers, ica):
    """从 ICA 曲线挖形状：主峰高度/位置、半高宽、次峰。
    次峰的做法：主峰 ±8 个 bin 强行置零后再取最大。简单粗暴，但对
    "主峰旁边的肩峰"足够用（我们要的是趋势特征，不是谱学精度）。"""
    pk = int(np.argmax(ica)); h = ica[pk]; half = h / 2.0
    lo = pk
    while lo > 0 and ica[lo] > half: lo -= 1
    hi = pk
    while hi < len(ica) - 1 and ica[hi] > half: hi += 1
    fwhm = (hi - lo) * DV
    ica2 = ica.copy()
    ica2[max(0, pk-8):min(len(ica), pk+8)] = 0
    return {"ica_main_peak": float(h), "ica_main_V": float(centers[pk]),
            "ica2_peak": float(ica2[int(np.argmax(ica2))]), "ica_fwhm": float(fwhm)}

def dch_curve_feats(v, i, t):
    """放电段特征：定分数容量电压点 v_q10~q90（容量走到 10%~90% 时的电压）+
    放电段 ICA（2.0-3.5V 窗、3mV 网格）的次峰与半高宽。
    v_q 系列是曲线增强特征的另一半：容量归一化后插值取电压，
    本质是"放电曲线形状的低维摘要"。"""
    dt = np.diff(t)
    q = np.concatenate([[0], np.cumsum(np.abs(i[1:]) * dt)]) / 3600.0
    if q[-1] < 0.2: return {}
    qn = q / q[-1]
    v_s = pd.Series(v).rolling(5, center=True, min_periods=1).mean().values
    d = {f"v_q{int(f*100)}": float(np.interp(f, qn, v_s)) for f in QFRACS}
    vm = (v[:-1]+v[1:])/2; qm=(q[:-1]+q[1:])/2
    mm = (vm>=2.0)&(vm<=3.5)
    if mm.sum() >= 20:
        vg,qg = vm[mm],qm[mm]; o=np.argsort(vg)
        g2 = np.arange(2.0,3.5,0.003)
        qq = np.interp(g2, vg[o], qg[o])
        ica = np.abs(np.diff(qq)/0.003)
        pk=int(np.argmax(ica)); h=ica[pk]; half=h/2
        lo=pk
        while lo>0 and ica[lo]>half: lo-=1
        hi=pk
        while hi<len(ica)-1 and ica[hi]>half: hi+=1
        ica2 = ica.copy(); ica2[max(0,pk-8):min(len(ica),pk+8)] = 0
        d["ica2_peak_dch"]=float(ica2[int(np.argmax(ica2))])
        d["ica_fwhm_dch"]=float((hi-lo)*0.003)
    return d

def process_xlsx_cell(zip_path, cell):
    rows_by_gid = {}
    order = []
    offset = 0
    with zipfile.ZipFile(zip_path) as z:
        xls = sorted(n for n in z.namelist() if n.lower().endswith('.xlsx'))
        for name in xls:
            xf = pd.ExcelFile(io.BytesIO(z.read(name)), engine='calamine')
            for sheet in [s for s in xf.sheet_names if s.lower().startswith('channel')]:
                d = xf.parse(sheet)
                d.columns = [str(c).strip() for c in d.columns]
                cols = [c for c in ['Cycle_Index','Step_Index','Current(A)','Voltage(V)','Test_Time(s)'] if c in d.columns]
                sub = d[cols].astype(float)
                cyc_col = 'Cycle_Index'
                for (cyc, step), sg in sub.groupby([cyc_col,'Step_Index']):
                    gid = offset + int(cyc)
                    rec = rows_by_gid.setdefault(gid, {"battery_id": cell, "cycle": gid})
                    if (sg['Current(A)'] < 0).any():
                        dd = sg[sg['Current(A)'] < 0]
                        rec.update(dch_curve_feats(dd['Voltage(V)'].values, dd['Current(A)'].values, dd['Test_Time(s)'].values))
                        # 此行保留占位逻辑（恒不赋值）：xlsx 容量统一由
                        # process_xlsx_capacity 用电流积分计算，见其 docstring
                        rec["capacity_Ah"] = float(dd['Test_Time(s)'].iloc[-1]*0 + dd['Current(A)'].abs().max()*0) if False else rec.get("capacity_Ah")
                    if (sg['Current(A)'] > 0).any():
                        cc = sg[sg['Current(A)'] > 0]
                        r = ica_curve(cc['Voltage(V)'].values, cc['Current(A)'].values, cc['Test_Time(s)'].values)
                        if r is not None:
                            rec.update(ica_shape(*r))
            offset = max(rows_by_gid.keys(), default=offset)
    return pd.DataFrame(list(rows_by_gid.values()))

def process_xlsx_capacity(zip_path, cell):
    """xlsx 容量：放电步电流积分（|I|·dt 累加）。
    单独一个函数而不是塞进 process_xlsx_cell，是因为两遍遍历的 groupby 键不同：
    特征按 (Cycle,Step) 挖、容量只在放电步算，混在一起会把 offset 逻辑搅乱。"""
    out = []
    offset = 0
    with zipfile.ZipFile(zip_path) as z:
        xls = sorted(n for n in z.namelist() if n.lower().endswith('.xlsx'))
        for name in xls:
            xf = pd.ExcelFile(io.BytesIO(z.read(name)), engine='calamine')
            for sheet in [s for s in xf.sheet_names if s.lower().startswith('channel')]:
                d = xf.parse(sheet)
                d.columns = [str(c).strip() for c in d.columns]
                sub = d[['Cycle_Index','Step_Index','Current(A)','Test_Time(s)']].astype(float)
                for (cyc, step), sg in sub.groupby(['Cycle_Index','Step_Index']):
                    gid = offset + int(cyc)
                    dd = sg[sg['Current(A)'] < 0]
                    if len(dd) >= 10:
                        dt = np.diff(dd['Test_Time(s)'].values)
                        q = np.concatenate([[0], np.cumsum(np.abs(dd['Current(A)'].values[1:]) * dt)]) / 3600.0
                        out.append((gid, float(q[-1])))
            offset = max((r[0] for r in out), default=offset)
    s = pd.DataFrame(out, columns=['gid','cap']).groupby('gid')['cap'].sum()
    s = s[s > 0.2].sort_index()
    return pd.DataFrame({"battery_id": cell, "gid": s.index, "capacity_Ah_int": s.values})

def process_txt_cell(zip_path, cell):
    """CADEX txt 版（CS2_8/21）。与 v1 的状态机相同，区别是容量取放电步
    Capacity 列的最后一个稳定值（/100×额定），放电 v_q 特征同步挖出。"""
    rows_by_gid = {}
    n_cyc, state = 0, 'rest'
    v_buf, i_buf, t_buf = [], [], []
    vd_buf, id_buf, td_buf = [], [], []
    cp_dis = np.nan; last_ica = None
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
                        r = ica_curve(np.array(v_buf)/1000, np.array(i_buf)/1000, np.array(t_buf))
                        if r is not None:
                            last_ica = ica_shape(*r)
                    if state == 'discharge':
                        n_cyc += 1
                        rec = rows_by_gid.setdefault(n_cyc, {"battery_id": cell, "cycle": n_cyc})
                        if np.isfinite(cp_dis):
                            rec["capacity_Ah"] = cp_dis / 100.0 * rated_for(cell)
                        if last_ica is not None:
                            rec.update(last_ica)
                        if len(vd_buf) >= 10:
                            rec.update(dch_curve_feats(np.array(vd_buf)/1000, np.array(id_buf)/1000, np.array(td_buf)))
                        last_ica = None
                    state, v_buf, i_buf, t_buf = cur, [], [], []
                    vd_buf, id_buf, td_buf = [], [], []
                if cur == 'discharge':
                    cp_dis = cp; vd_buf.append(mv); id_buf.append(ma); td_buf.append(du)
                if cur == 'charge':
                    v_buf.append(mv); i_buf.append(ma); t_buf.append(du)
        if state == 'discharge':
            n_cyc += 1
            rec = rows_by_gid.setdefault(n_cyc, {"battery_id": cell, "cycle": n_cyc})
            if np.isfinite(cp_dis):
                rec["capacity_Ah"] = cp_dis / 100.0 * rated_for(cell)
            if last_ica is not None:
                rec.update(last_ica)
            if len(vd_buf) >= 10:
                rec.update(dch_curve_feats(np.array(vd_buf)/1000, np.array(id_buf)/1000, np.array(td_buf)))
    return pd.DataFrame(list(rows_by_gid.values()))
def main():
    frames = []
    t0 = time.time()
    # 2026-10 CX2 扩充（与 parse_calce.py 同一纳入/排除清单）
    xlsx_cells = ['CS2_33', 'CS2_34', 'CS2_35', 'CS2_36', 'CS2_37', 'CS2_38',
                  'CX2_16', 'CX2_33', 'CX2_34', 'CX2_35', 'CX2_36', 'CX2_37', 'CX2_38']
    txt_cells = ['CS2_8', 'CS2_21', 'CX2_31']
    for cell in xlsx_cells:
        f_feat = process_xlsx_cell(CAL/f'{cell}.zip', cell)
        f_cap = process_xlsx_capacity(CAL/f'{cell}.zip', cell).rename(columns={"gid": "cycle", "capacity_Ah_int": "capacity_Ah"})
        # 顺序对齐：特征流与容量流都按放电循环先后产生，第 k 个对第 k 个。
        # v_q10 缺失的循环说明放电段太短挖不出特征，从特征流里剔掉再对齐
        f_feat_d = f_feat[f_feat["v_q10"].notna()].sort_values("cycle").reset_index(drop=True)
        f_cap_s = f_cap.sort_values("cycle").reset_index(drop=True)
        n = min(len(f_feat_d), len(f_cap_s))
        if len(f_feat_d) != len(f_cap_s):
            print(f"  [警告] {cell}: 特征循环 {len(f_feat_d)} vs 容量循环 {len(f_cap_s)}, 按顺序取前 {n}")
        f_feat_d = f_feat_d.iloc[:n]
        f_feat_d["capacity_Ah"] = f_cap_s["capacity_Ah"].values[:n]
        f_feat_d["battery_id"] = cell
        f_feat_d["soh"] = f_feat_d["capacity_Ah"] / f_feat_d["capacity_Ah"].dropna().iloc[0]
        frames.append(f_feat_d)
        print(f"{cell}: {n} rows | cap {f_feat_d['capacity_Ah'].iloc[0]:.3f}->{f_feat_d['capacity_Ah'].iloc[-1]:.3f} | ica {f_feat_d['ica_main_peak'].median():.2f} | v_q50 {f_feat_d['v_q50'].median():.3f}", flush=True)
    for cell in txt_cells:
        m = process_txt_cell(CAL/f'{cell}.zip', cell)
        m["soh"] = m["capacity_Ah"] / m["capacity_Ah"].dropna().iloc[0]
        frames.append(m)
        print(f"{cell}: {len(m)} rows | ica中位 {m['ica_main_peak'].median():.2f} | v_q50中位 {m['v_q50'].median():.3f}", flush=True)
    cal = pd.concat(frames, ignore_index=True)
    cal.to_csv(OUT / 'calce_full_v2.csv', index=False, encoding='utf-8-sig')
    print(f'UNIFIED: {OUT / "calce_full_v2.csv"} ({len(cal)} rows, {time.time()-t0:.0f}s)')

if __name__ == '__main__':
    main()
