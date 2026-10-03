# -*- coding: utf-8 -*-
"""结果复算（CX2 版）：只读 results_cx2/ 的结果文件，把扩充后论文的关键数字重算一遍。

    python code/checks/verify_results_cx2.py

口径与 verify_results.py 相同（纯标准库），期望值为投稿版（CX2 扩充）论文所报数值。
旧口径（results/）仍可用 verify_results.py 核对，保持 179/179。
"""
import csv, json, math, os, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results_cx2"
ROWS, BAD = [], []


def load(rel):
    return json.load(open(RES / rel, encoding="utf-8"))


def jl(rel):
    out = []
    for line in open(RES / rel, encoding="utf-8"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def check(label, expect, got, tol):
    ok = abs(expect - got) <= tol
    ROWS.append((label, expect, got, ok))
    if not ok:
        BAD.append(label)
    return ok


def betacf(a, b, x):
    MAXIT, EPS, FPMIN = 200, 3e-16, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        de = d * c
        h *= de
        if abs(de - 1.0) < EPS:
            break
    return h


def betai(a, b, x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    bt = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                  + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * betacf(a, b, x) / a
    return 1.0 - bt * betacf(b, a, 1.0 - x) / b


def paired_t_p(xs, ys):
    d = [x - y for x, y in zip(xs, ys)]
    n = len(d)
    m = st.mean(d)
    sd = st.stdev(d)
    t = abs(m) / (sd / math.sqrt(n))
    df = n - 1
    return t, betai(df / 2.0, 0.5, df / (df + t * t))


def mean_std(xs):
    return st.mean(xs), st.stdev(xs)


# ---------- 表 3：跨化学体系迁移（5 种子汇总，CX2） ----------
t = load("transfer/transfer_multiseed_v2.json")
tbl3 = {"CALCE_tcn": (0.1499, 0.0128, 0.0072, 0.0041, 0.0161, 0.0097, 2.23, 0.38),
        "CALCE_lstm": (0.1466, 0.0076, 0.0034, 0.0015, 0.0039, 0.0012, 1.17, 0.17),
        "NASA_tcn": (0.0987, 0.0079, 0.0129, 0.0039, 0.1879, 0.0679, 14.90, 4.52),
        "NASA_lstm": (0.0989, 0.0093, 0.0146, 0.0048, 0.0726, 0.0632, 6.74, 7.95)}
for k, v in tbl3.items():
    g = t[k]
    check("表 3 %s 零样本均值" % k, v[0], g["zero"]["mean"], 5e-5)
    check("表 3 %s 零样本标准差" % k, v[1], g["zero"]["std"], 5e-5)
    check("表 3 %s 微调均值" % k, v[2], g["ft"]["mean"], 5e-5)
    check("表 3 %s 微调标准差" % k, v[3], g["ft"]["std"], 5e-5)
    check("表 3 %s 目标域基线均值" % k, v[4], g["to"]["mean"], 5e-5)
    check("表 3 %s 目标域基线标准差" % k, v[5], g["to"]["std"], 5e-5)
    check("表 3 %s 增益比均值" % k, v[6], g["gain"]["mean"], 0.005)
    check("表 3 %s 增益比标准差" % k, v[7], g["gain"]["std"], 0.005)

# 4.3：NASA/LSTM 逐种子增益与中位数
_gains = [load("transfer/t3b_lstm_s%d.json" % s)["targets"]["NASA"]["target_only"]["rmse"] /
          load("transfer/t3b_lstm_s%d.json" % s)["targets"]["NASA"]["fine_tune"]["rmse"]
          for s in (42, 43, 44, 45, 46)]
check("4.3 NASA/LSTM 增益中位数", 3.3, st.median(_gains), 0.06)
check("4.3 CALCE/TCN 增益最大值", 2.80, max(
    load("transfer/t3b_tcn_s%d.json" % s)["targets"]["CALCE"]["target_only"]["rmse"] /
    load("transfer/t3b_tcn_s%d.json" % s)["targets"]["CALCE"]["fine_tune"]["rmse"]
    for s in (42, 43, 44, 45, 46)), 0.02)

# ---------- 表 4：原始协议漂移（CX2，5 种子） ----------
for tgt, mz, sdz in [("CALCE", 1337, 790), ("NASA", 779, 324)]:
    zs = [load("transfer/t3_soh_tcn_s%d.json" % s)["targets"][tgt]["zero_shot"]["rmse"] for s in (42, 43, 44, 45, 46)]
    check("表 4 %s 原始协议均值" % tgt, mz, st.mean(zs), 0.6)
    check("表 4 %s 原始协议标准差" % tgt, sdz, st.stdev(zs), 0.6)

# ---------- 表 5：保形双路由（CX2，逐种子 42-46） ----------
tbl5 = {"CALCE_tcn": (0.530, 0.116, 0.879, 0.138, 0.0307, 0.0068, 0.0035),
        "CALCE_lstm": (0.665, 0.156, 0.875, 0.122, 0.0082, 0.0041, 0.0019),
        "NASA_tcn": (0.206, 0.077, 0.946, 0.094, 0.0530, 0.0100, 0.0024),
        "NASA_lstm": (0.124, 0.115, 0.915, 0.157, 0.0592, 0.0108, 0.0017)}
allsrc, alltgt = [], []
for k, v in sorted(tbl5.items()):
    tgt, mod = k.split("_")
    src, tg, wt, rm = [], [], [], []
    for s in (42, 43, 44, 45, 46):
        o = load("conformal/t4_%s_s%d.json" % (mod, s))["targets"][tgt]
        src.append(o["source_calibrated"]["PICP"])
        tg.append(o["target_calibrated"]["PICP"])
        wt.append(o["target_calibrated"]["MPIW"])
        rm.append(o["point_rmse"])
    allsrc += src
    alltgt += tg
    ms, ss = mean_std(src)
    check("表 5 %s 源域校准 PICP 均值" % k, v[0], ms, 5e-4)
    check("表 5 %s 源域校准 PICP 标准差" % k, v[1], ss, 5e-4)
    check("表 5 %s 目标域校准 PICP 均值" % k, v[2], st.mean(tg), 5e-4)
    check("表 5 %s 目标域校准 PICP 标准差" % k, v[3], st.stdev(tg), 5e-4)
    check("表 5 %s 目标域校准 MPIW" % k, v[4], st.mean(wt), 5e-5)
    check("表 5 %s 点预测 RMSE" % k, v[5], st.mean(rm), 5e-5)
    check("表 5 %s 点预测 RMSE 标准差" % k, v[6], st.stdev(rm), 5e-5)
check("4.5 目标校准覆盖最低值", 0.637, min(alltgt), 5e-4)
check("4.5 目标校准达名义 0.90 的组数", 14, sum(1 for v in alltgt if v >= 0.90), 0)
check("4.5 源域校准覆盖最低值", 0.007, min(allsrc), 5e-4)
check("4.5 源域校准覆盖最高值(<0.90)", 0.892, max(allsrc), 5e-4)
check("4.5 源域全部欠覆盖", 1, 1 if all(v < 0.90 for v in allsrc) else 0, 0)

# ---------- 4.5：条件化收窄（CX2 协议 5/5/6、1/1/2） ----------
w = load("conformal/t4c_multiseed_summary.json")
check("4.5 CALCE Mondrian 覆盖率", 0.830, w["mondrian"]["CALCE"]["mondrian"]["PICP"]["mean"], 0.005)
check("4.5 CALCE Mondrian 单一基线", 0.855, w["mondrian"]["CALCE"]["single"]["PICP"]["mean"], 0.005)
check("4.5 NASA Mondrian 覆盖率", 0.635, w["mondrian"]["NASA"]["mondrian"]["PICP"]["mean"], 0.005)
check("4.5 NASA Mondrian 收窄比", 0.345, 1 - w["mondrian"]["NASA"]["mondrian"]["MPIW"] / w["mondrian"]["NASA"]["single"]["MPIW"], 0.01)
check("4.5 CALCE 加权覆盖率", 0.839, w["weighted"]["CALCE"]["weighted"]["PICP"]["mean"], 0.005)
check("4.5 CALCE 加权单一基线", 0.853, w["weighted"]["CALCE"]["single"]["PICP"]["mean"], 0.005)
check("4.5 NASA 加权=单一(退化)", 0.799, w["weighted"]["NASA"]["weighted"]["PICP"]["mean"], 0.005)

# ---------- 表 6：消融（CX2，配对 t） ----------
groups = {}
for feats in ("base7", "curve14"):
    for s in (42, 43, 44, 45, 46):
        o = load("ablation/t3e_%s_tcn_s%d.json" % (feats, s))["targets"]
        for tgt in ("CALCE", "NASA"):
            for mode, key in (("zero", "zero_shot"), ("ft", "fine_tune")):
                groups.setdefault((feats, tgt, mode), []).append(o[tgt][key]["rmse"])
EXP6 = {("CALCE", "zero"): (0.1499, 0.0128, 0.1602, 0.0089),
        ("CALCE", "ft"): (0.0072, 0.0041, 0.0114, 0.0051),
        ("NASA", "zero"): (0.0987, 0.0079, 0.0871, 0.0113),
        ("NASA", "ft"): (0.0129, 0.0039, 0.0268, 0.0143)}
for (tgt, mode), (e0, s0, e1, s1) in sorted(EXP6.items()):
    g0, g1 = groups[("base7", tgt, mode)], groups[("curve14", tgt, mode)]
    check("表 6 %s/%s 基础 7 维均值" % (tgt, mode), e0, st.mean(g0), 5e-5)
    check("表 6 %s/%s 基础 7 维标准差" % (tgt, mode), s0, st.stdev(g0), 5e-5)
    check("表 6 %s/%s 曲线增强 14 维均值" % (tgt, mode), e1, st.mean(g1), 5e-5)
    check("表 6 %s/%s 曲线增强 14 维标准差" % (tgt, mode), s1, st.stdev(g1), 5e-5)
    tv, pv = paired_t_p(g0, g1)
    if (tgt, mode) == ("CALCE", "ft"):
        check("4.7 CALCE 微调配对 t", 1.88, tv, 0.01)
        check("4.7 CALCE 微调 p 值", 0.133, pv, 0.001)
    if (tgt, mode) == ("CALCE", "zero"):
        check("4.7 CALCE 零样本配对 t", 1.88, tv, 0.01)
        check("4.7 CALCE 零样本 p 值", 0.133, pv, 0.001)
    if (tgt, mode) == ("NASA", "ft"):
        check("4.7 NASA 微调配对 t", 2.46, tv, 0.01)
        check("4.7 NASA 微调 p 值", 0.070, pv, 0.001)
    if (tgt, mode) == ("NASA", "zero"):
        check("4.7 NASA 零样本配对 t", 1.48, tv, 0.01)
        check("4.7 NASA 零样本 p 值", 0.212, pv, 0.001)

# ---------- 4.6：逐电芯诊断（CX2） ----------
for mod, agg_m, agg_sd, min_tgt in [("tcn", 0.61, 0.28, 0.40), ("lstm", 0.59, 0.23, 0.39)]:
    pc, agg = [], []
    for s in range(42, 47):
        o = load("conformal/t4d_per_cell_%s_s%d.json" % (mod, s))["targets"]["CALCE"]
        agg.append(o["cell_aggregated_conformal"]["median"]["PICP"])
        for cc in o["per_cell"].values():
            pc.append(cc["cov_tgt"])
    m_agg, s_agg = mean_std(agg)
    check("4.6 %s 聚合保形覆盖" % mod, agg_m, m_agg, 0.005)
    check("4.6 %s 聚合保形覆盖标准差" % mod, agg_sd, s_agg, 0.005)
    check("4.6 %s 逐电芯目标覆盖最低" % mod, min_tgt, min(pc), 0.005)

# ---------- 校准电芯数扫描（cal1-5 均值） ----------
sw = load("conformal/t4_split_sweep/sweep_summary.json")
SWEXP = {"tcn": [0.883, 0.874, 0.910, 0.898, 0.900],
         "lstm": [0.844, 0.875, 0.815, 0.854, 0.833]}
for m in ("tcn", "lstm"):
    for i, nc in enumerate((1, 2, 3, 4, 5)):
        check("校准电芯扫描 %s %d 颗覆盖率均值" % (m, nc), SWEXP[m][i], sw[m]["cal%d" % nc]["picp_tgt"]["mean"], 5e-4)

# ---------- 划分重抽抽样分布 ----------
rndp = RES / "conformal/i9_seeds/i9_summary.json"
if rndp.exists():
    rnd = json.load(open(rndp, encoding="utf-8"))
    RNDEXP = {"tcn": (50, 0.854, 0.121, 20), "lstm": (50, 0.856, 0.110, 21)}
    for m, v in rnd.items():
        e = RNDEXP[m]
        check("划分重抽 %s 次数" % m, e[0], v["n"], 0)
        check("划分重抽 %s 覆盖均值" % m, e[1], v["picp_tgt"]["mean"], 5e-4)
        check("划分重抽 %s 覆盖标准差" % m, e[2], v["picp_tgt"]["std"], 5e-4)
        check("划分重抽 %s 达标(>=0.90)次数" % m, e[3], round(v["frac_ge_090"] * v["n"]), 0)
        print("[划分重抽] %s n=%d 覆盖均值 %.4f std %.4f q05 %.4f q95 %.4f 达标比例 %.3f"
              % (m, v["n"], v["picp_tgt"]["mean"], v["picp_tgt"]["std"],
                 v["picp_tgt"]["q05"], v["picp_tgt"]["q95"], v["frac_ge_090"]))

# ---------- 输出 ----------
w1 = max(len(r0[0]) for r0 in ROWS)
print("%-*s  %-14s %-14s %s" % (w1, "论文位置", "论文数值", "本仓库重算", "结论"))
print("-" * (w1 + 42))
for label, expect, got, ok in ROWS:
    print("%-*s  %-14.4f %-14.4f %s" % (w1, label, expect, got, "一致" if ok else "不一致 <--"))
print("-" * (w1 + 42))
if BAD:
    print("有 %d/%d 项对不上：%s" % (len(BAD), len(ROWS), "; ".join(BAD[:5])))
    raise SystemExit(1)
print("全部 %d 项与论文数字一致。" % len(ROWS))
