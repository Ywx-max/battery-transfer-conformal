# -*- coding: utf-8 -*-
"""结果复算：只用本仓库 results/ 的结果文件，把论文里的关键数字重算一遍。

    python code/checks/verify_results.py

不需要原始数据集、不需要 GPU、不需要论文源文件，纯标准库，约 1 秒。
逐行输出「论文位置 | 论文数值 | 由本仓库数据重算 | 结论」；全部一致时返回码 0，
任何一项对不上都会给出具体定位。结果文件与论文表/图的对应关系见 results/README.md。
"""
import csv, json, math, statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "results"
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


def q_lower(xs, q):
    """下取整序统计量分位数（与论文表 2 的 lower 插值口径一致）"""
    s = sorted(xs)
    return s[min(int(q * (len(s) - 1)), len(s) - 1)]


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


def welch_t_p(xs, ys):
    """Welch 双样本 t 检验（论文表 6 显著性标注的口径）"""
    n1, n2 = len(xs), len(ys)
    m1, m2 = st.mean(xs), st.mean(ys)
    v1, v2 = st.variance(xs), st.variance(ys)
    t = abs(m1 - m2) / math.sqrt(v1 / n1 + v2 / n2)
    df = (v1 / n1 + v2 / n2) ** 2 / ((v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1))
    return t, betai(df / 2.0, 0.5, df / (df + t * t))


def mean_std(xs):
    return st.mean(xs), st.stdev(xs)



# ---------- 表 1：四骨干 5 折组交叉验证（3 种子） ----------
b = load("baselines/baseline_3seed_final.json")
for key, rmse, sd, mae in [("lstm", 129.81, 1.57, 75.47), ("gru", 126.27, 2.36, 73.22),
                           ("tcn", 124.20, 1.76, 77.35), ("transformer", 126.48, 1.09, 78.79)]:
    check("表 1 %s RMSE 均值" % key, rmse, b[key]["rmse_mean"], 0.005)
    check("表 1 %s RMSE 标准差" % key, sd, b[key]["rmse_std"], 0.005)
    check("表 1 %s MAE 均值" % key, mae, b[key]["mae_mean"], 0.005)

# 4.2 四骨干两两比较：仅 TCN 与 LSTM 的差异达到显著（论文 4.2 的口径）
_tv, _pv = welch_t_p(sorted(b["tcn"]["seeds"].values()), sorted(b["lstm"]["seeds"].values()))
check("4.2 TCN vs LSTM Welch t", 3.36, _tv, 0.01)
check("4.2 TCN vs LSTM p 值", 0.029, _pv, 0.001)

# 4.8 同种子两次独立执行的非确定性（两次重复结果随仓库提供）
_r1 = load("transfer/t3b_tcn_s42_repeat1.json")["targets"]["CALCE"]["zero_shot"]["rmse"]
_r2 = load("transfer/t3b_tcn_s42_repeat2.json")["targets"]["CALCE"]["zero_shot"]["rmse"]
check("4.8 同种子重复波动 %", 14.0, abs(_r1 - _r2) / ((_r1 + _r2) / 2) * 100, 0.05)

# ---------- 表 2：LOBO 逐电芯分布（119 折） ----------
for key, f, row in [("tcn", "baselines/lobo_tcn.jsonl", (18.15, 50.42, 80.90, 105.86, 92.47, 66.17)),
                    ("lstm", "baselines/lobo_lstm.jsonl", (26.86, 58.46, 83.08, 120.93, 97.07, 62.57)),
                    ("transformer", "baselines/lobo_transformer.jsonl", (27.57, 54.77, 77.69, 97.67, 90.73, 62.69))]:
    xs = [d["rmse"] for d in jl(f)]
    check("表 2 %s 折数" % key, 119, len(xs), 0)
    check("表 2 %s 最小" % key, row[0], min(xs), 0.005)
    check("表 2 %s Q1(lower)" % key, row[1], q_lower(xs, 0.25), 0.005)
    check("表 2 %s 中位数" % key, row[2], st.median(xs), 0.005)
    check("表 2 %s Q3(lower)" % key, row[3], q_lower(xs, 0.75), 0.005)
    check("表 2 %s 宏观均值" % key, row[4], st.mean(xs), 0.005)
    check("表 2 %s 宏观标准差" % key, row[5], st.stdev(xs), 0.005)

# 图 1：三个复现种子合并 357 折的分布（复现运行，与表 2 的单次运行口径不同）
pooled = []
for s in (42, 43, 44):
    pooled += [d["rmse"] for d in jl("baselines/lobo_tcn_s%d.jsonl" % s)]
check("图 1 合并折数", 357, len(pooled), 0)
check("图 1 合并口径中位数", 76.7, st.median(pooled), 0.05)

# ---------- 4.8 双模型互补性 + 集成 ----------
rows = list(csv.DictReader(open(RES / "baselines/lobo_3model_final.csv", encoding="utf-8-sig")))
a = [float(r["rmse_tcn"]) for r in rows]
c = [float(r["rmse_lstm"]) for r in rows]
d3 = [float(r["rmse_tf"]) for r in rows]
mx, my = st.mean(a), st.mean(c)
cov = sum((x - mx) * (y - my) for x, y in zip(a, c))
r = cov / math.sqrt(sum((x - mx) ** 2 for x in a) * sum((y - my) ** 2 for y in c))
check("4.8 TCN/LSTM 逐电芯相关 r", 0.699, r, 0.001)
check("表 2 双模型逐电芯最优集成", 79.02, st.mean([min(x, y) for x, y in zip(a, c)]), 0.005)
check("表 2 三模型逐电芯最优集成", 71.49, st.mean([min(x, y, z) for x, y, z in zip(a, c, d3)]), 0.005)

# ---------- 4.2 跨种子复核 ----------
m = load("baselines/lobo_final_multiseed.json")
check("4.2 TCN 跨种子宏观均值", 92.45, m["tcn"]["cross_mean"], 0.005)
check("4.2 TCN 跨种子标准差", 0.98, m["tcn"]["cross_std"], 0.005)
check("4.2 LSTM 跨种子宏观均值", 96.21, m["lstm"]["cross_mean"], 0.005)
check("4.2 Transformer 跨种子宏观均值", 93.42, m["transformer"]["cross_mean"], 0.005)
ens = m.get("ensemble") or m.get("best3") or {}
if ens:
    check("4.2 集成跨种子宏观均值", 70.52, ens["cross_mean"], 0.005)
    check("4.2 集成跨种子标准差", 0.27, ens["cross_std"], 0.005)

# ---------- 表 3：跨化学体系迁移（5 种子汇总） ----------
t = load("transfer/transfer_multiseed_v2.json")
tbl3 = {"CALCE_tcn": (0.1465, 0.0211, 0.0100, 0.0056, 0.0346, 0.0069, 4.29, 2.08),
        "CALCE_lstm": (0.1446, 0.0160, 0.0089, 0.0084, 0.0103, 0.0117, 1.11, 0.38),
        "NASA_tcn": (0.0975, 0.0060, 0.0141, 0.0057, 0.1888, 0.0681, 14.45, 5.94),
        "NASA_lstm": (0.0994, 0.0089, 0.0136, 0.0023, 0.0765, 0.0657, 5.93, 5.18)}
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

# 摘要：NASA 微调降至从头训练的 1/14.5（TCN）与 1/5.9（LSTM）；变异系数 86% 收窄至 17%
check("摘要 1/14.5 (TCN)", 14.5, t["NASA_tcn"]["gain"]["mean"], 0.06)
check("摘要 1/5.9 (LSTM)", 5.9, t["NASA_lstm"]["gain"]["mean"], 0.06)
check("摘要 变异系数 86%(从头训练)", 86, t["NASA_lstm"]["to"]["std"] / t["NASA_lstm"]["to"]["mean"] * 100, 1.0)
check("摘要 变异系数 17%(微调)", 17, t["NASA_lstm"]["ft"]["std"] / t["NASA_lstm"]["ft"]["mean"] * 100, 1.0)



# ---------- 表 4：双协议漂移分解（前两行按 4 种子 42-45 重算） ----------
for tgt, mz, sdz in [("CALCE", 1636, 1022), ("NASA", 695, 393)]:
    zs = [load("transfer/t3_soh_tcn_s%d.json" % s)["targets"][tgt]["zero_shot"]["rmse"] for s in (42, 43, 44, 45)]
    check("表 4 %s 原始协议均值(4 种子)" % tgt, mz, st.mean(zs), 0.6)
    check("表 4 %s 原始协议标准差(4 种子)" % tgt, sdz, st.stdev(zs), 0.6)

# ---------- 表 5：保形双路由（逐种子 42-46 重算） ----------
tbl5 = {"CALCE_tcn": (0.458, 0.108, 0.0052, 0.999, 0.002, 0.3785),
        "CALCE_lstm": (0.219, 0.210, 0.0027, 0.999, 0.002, 0.3630),
        "NASA_tcn": (0.121, 0.091, 0.0052, 1.000, 0.000, 0.3145),
        "NASA_lstm": (0.109, 0.096, 0.0027, 1.000, 0.000, 0.3062)}
allsrc, maxsd = [], 0.0
for k, v in sorted(tbl5.items()):
    tgt, mod = k.split("_")
    src, tg, ws, wt = [], [], [], []
    for s in (42, 43, 44, 45, 46):
        o = load("conformal/t4_%s_s%d.json" % (mod, s))["targets"][tgt]
        src.append(o["source_calibrated"]["PICP"])
        ws.append(o["source_calibrated"]["MPIW"])
        tg.append(o["target_calibrated"]["PICP"])
        wt.append(o["target_calibrated"]["MPIW"])
    allsrc += src
    ms, ss = mean_std(src)
    maxsd = max(maxsd, ss)
    check("表 5 %s 源域校准 PICP 均值" % k, v[0], ms, 5e-4)
    check("表 5 %s 源域校准 PICP 标准差" % k, v[1], ss, 5e-4)
    check("表 5 %s 源域校准 MPIW" % k, v[2], st.mean(ws), 5e-5)
    check("表 5 %s 目标域校准 PICP 均值" % k, v[3], st.mean(tg), 5e-4)
    check("表 5 %s 目标域校准 PICP 标准差" % k, v[4], st.stdev(tg), 5e-4)
    check("表 5 %s 目标域校准 MPIW" % k, v[5], st.mean(wt), 5e-5)
check("4.5 源域校准覆盖最低值", 0.00, min(allsrc), 5e-4)
check("4.5 源域校准覆盖最高值(两位小数)", 0.58, round(max(allsrc), 2), 1e-9)
check("4.5 源域校准覆盖总均值(20 组)", 0.227, st.mean(allsrc), 5e-4)
check("4.5 源域校准覆盖总标准差(20 组)", 0.190, st.stdev(allsrc), 5e-4)
check("4.5 源域校准最大标准差", 0.210, maxsd, 5e-4)

# ---------- 4.5 Mondrian / 加权条件化收窄 ----------
w = load("conformal/t4c_multiseed_summary.json")
check("4.5 NASA Mondrian 覆盖率", 0.64, w["mondrian"]["NASA"]["mondrian"]["PICP"]["mean"], 0.005)
check("4.5 NASA Mondrian 收窄比", 0.212, 1 - w["mondrian"]["NASA"]["mondrian"]["MPIW"] / w["mondrian"]["NASA"]["single"]["MPIW"], 0.005)
check("4.5 NASA 单一分位数基线覆盖率", 0.81, w["mondrian"]["NASA"]["single"]["PICP"]["mean"], 0.005)
check("4.5 CALCE 加权收窄比", 0.146, 1 - w["weighted"]["CALCE"]["weighted"]["MPIW"] / w["weighted"]["CALCE"]["single"]["MPIW"], 0.005)



# ---------- 表 6：特征丰富度消融（逐种子 42-46 重算 + Welch 检验） ----------
groups = {}
for feats in ("base7", "curve14"):
    for s in (42, 43, 44, 45, 46):
        o = load("ablation/t3e_%s_tcn_s%d.json" % (feats, s))["targets"]
        for tgt in ("CALCE", "NASA"):
            for mode, key in (("zero", "zero_shot"), ("ft", "fine_tune")):
                groups.setdefault((feats, tgt, mode), []).append(o[tgt][key]["rmse"])
EXP6 = {("CALCE", "zero"): (0.1488, 0.0151, 0.1761, 0.0206),
        ("CALCE", "ft"): (0.0106, 0.0061, 0.0226, 0.0080),
        ("NASA", "zero"): (0.1015, 0.0108, 0.0943, 0.0129),
        ("NASA", "ft"): (0.0162, 0.0100, 0.0287, 0.0157)}
for (tgt, mode), (e0, s0, e1, s1) in sorted(EXP6.items()):
    g0, g1 = groups[("base7", tgt, mode)], groups[("curve14", tgt, mode)]
    check("表 6 %s/%s 基础 7 维均值" % (tgt, mode), e0, st.mean(g0), 5e-5)
    check("表 6 %s/%s 基础 7 维标准差" % (tgt, mode), s0, st.stdev(g0), 5e-5)
    check("表 6 %s/%s 曲线增强 14 维均值" % (tgt, mode), e1, st.mean(g1), 5e-5)
    check("表 6 %s/%s 曲线增强 14 维标准差" % (tgt, mode), s1, st.stdev(g1), 5e-5)
    if (tgt, mode) == ("CALCE", "ft"):
        tv, pv = welch_t_p(g0, g1)
        check("4.6 CALCE 微调 Welch t", 2.66, tv, 0.01)
        check("4.6 CALCE 微调 p 值", 0.031, pv, 0.001)
    if (tgt, mode) == ("CALCE", "zero"):
        tv, pv = welch_t_p(g0, g1)
        check("4.6 CALCE 零样本 Welch t", 2.39, tv, 0.01)
        check("4.6 CALCE 零样本 p 值", 0.047, pv, 0.001)
    if (tgt, mode) == ("NASA", "ft"):
        tv, pv = welch_t_p(g0, g1)
        check("4.6 NASA 微调 Welch t", 1.49, tv, 0.01)
        check("4.6 NASA 微调 p 值", 0.180, pv, 0.001)

# ---------- 4.5 末：逐电芯诊断 ----------
for mod, pic, pic_sd, min_tgt, min_src in [("tcn", 0.990, 0.029, 0.97, 0.015),
                                           ("lstm", 0.998, 0.005, 0.99, 0.004)]:
    pc, ps, agg, mpiw, loop = [], [], [], [], []
    for s in (42, 43, 44, 45, 46):
        o = load("conformal/t4d_per_cell_%s_s%d.json" % (mod, s))["targets"]
        for tgt in ("CALCE", "NASA"):
            agg.append(o[tgt]["cell_aggregated_conformal"]["median"]["PICP"])
            for cc in o[tgt]["per_cell"].values():
                pc.append(cc["cov_tgt"])
                ps.append(cc["cov_src"])
            if tgt == "CALCE":
                mpiw.append(o[tgt]["cell_aggregated_conformal"]["median"]["MPIW"])
                loop.append(2 * o[tgt]["q_target"])
    m_agg, s_agg = mean_std(agg)
    check("4.5 末 %s 聚合保形覆盖率" % mod, pic, m_agg, 0.0005)
    check("4.5 末 %s 聚合保形覆盖标准差" % mod, pic_sd, s_agg, 0.0005)
    check("4.5 末 %s 逐电芯目标域覆盖最低" % mod, min_tgt, min(pc), 0.005)
    check("4.5 末 %s 逐电芯源域覆盖最低" % mod, min_src, min(ps), 0.005)
    check("4.5 末 %s CALCE 聚合/循环级宽度比" % mod, 0.47, st.mean(mpiw) / st.mean(loop), 0.01)

# ---------- 4.7 早期寿命预测 ----------
e = load("early_pred_summary.json")
check("4.7 log 空间 RMSE", 0.117, e["rmse_log"], 5e-4)
check("4.7 循环空间相对误差 %", 19.5, e["rel_err_pct"], 0.05)
check("4.7 有寿命标签电芯数", 119, e["n"], 0)

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

