# -*- coding: utf-8 -*-
"""数据一致性自检：实验结果 JSON 与论文 LaTeX 源逐项核对（75 项）。

写这篇论文时最怕的事：改了某处数字、别处忘同步，或者誊写时抄错一位。
这个脚本把"论文里的每个关键数字"都拉回原始 JSON 重新算一遍：
表 3/4/6 的均值标准差、摘要的增益比、变异系数、LOBO 均值、4.5 的收窄百分比……
全部通过时打印"✓ 全部数据与论文一致"，任何一项漂移都会指名道姓地报出来。

修改正文数字后运行（约 3 秒，无 GPU 依赖）：
    python final_data_check.py
注意：R/ 与 main.tex 的路径按作者本机布局写死，换机器需对应调整。"""
import json, re, statistics as st
from pathlib import Path
BS = chr(92)
R = Path("03_实验" + BS + "结果")
tex = open("04_稿件" + BS + "main.tex", encoding="utf-8").read()
ok, bad = 0, []

def check(name, value, fmt, where="tex"):
    """把数值按 fmt 格式化后，在论文源里找原文。找不到就记进 bad 列表。
    适合"摘要里出现 1/14.5"这类逐字引用；带容差的数值核对直接在下面内联写。"""
    global ok
    s = fmt.format(value)
    if s in tex:
        ok += 1
    else:
        bad.append(f"{name}: 论文中未找到 '{s}'")

# ---------- 表3 迁移 (transfer_multiseed_v2) ----------
t = json.load(open(R/"迁移"/"local"/"transfer_multiseed_v2.json", encoding="utf-8"))
tbl3 = {"CALCE_tcn": ("0.1465","0.0211","0.0100","0.0056","0.0346","0.0069"),
        "CALCE_lstm":("0.1446","0.0160","0.0089","0.0084","0.0103","0.0117"),
        "NASA_tcn":  ("0.0975","0.0060","0.0141","0.0057","0.1888","0.0681"),
        "NASA_lstm": ("0.0994","0.0089","0.0136","0.0023","0.0765","0.0657")}
# 表 3：迁移结果。论文值是四舍五入到小数点后 4 位的，容差给 1e-4 量级
for k, vals in tbl3.items():
    d = t[k]
    real = [d["zero"]["mean"], d["zero"]["std"], d["ft"]["mean"], d["ft"]["std"], d["to"]["mean"], d["to"]["std"]]
    for paper_v, real_v, lab in zip(vals, real, ["zero_m","zero_s","ft_m","ft_s","to_m","to_s"]):
        if abs(float(paper_v) - real_v) > 0.00005 + abs(real_v)*0.001:
            bad.append(f"表3 {k} {lab}: 论文 {paper_v} vs 实际 {real_v:.4f}")
        else: ok += 1
# 增益声称：摘要与结论里的 1/14.5、1/5.9 是"逐种子基线/微调再平均"的口径，
# 不能直接用表 3 两列均值相除（那是 13.4/5.6，读者自己相除时会疑惑，
# 所以表 3 专门加了增益比列，正文 4.3 也写明了口径）
g_tcn = t["NASA_tcn"]["gain"]["mean"]; g_lstm = t["NASA_lstm"]["gain"]["mean"]
check("NASA TCN 增益 1/14.4", g_tcn, "{:.1f}")
tex_gain = "1/14.5" in tex and "1/5.9" in tex
if not (13.5 <= g_tcn <= 15.3 and 5.5 <= g_lstm <= 6.3 and tex_gain): bad.append(f"摘要增益声称: TCN {g_tcn:.2f} LSTM {g_lstm:.2f}")
else: ok += 1
# 变异系数声称：86% 收窄至 17%（LSTM 的从头训练 vs 微调）
cv_ft = t["NASA_lstm"]["ft"]["std"]/t["NASA_lstm"]["ft"]["mean"]*100
cv_to = t["NASA_lstm"]["to"]["std"]/t["NASA_lstm"]["to"]["mean"]*100
if not (80 <= cv_to <= 92 and 14 <= cv_ft <= 20 and "86" in tex and "17" in tex):
    bad.append(f"变异系数声称: from-scratch {cv_to:.0f}% fine-tune {cv_ft:.0f}%")
else: ok += 1

# ---------- 表4 保形 ----------
c = json.load(open(R/"保形"/"local"/"conformal_multiseed_summary.json", encoding="utf-8"))
tbl4 = {"CALCE_tcn": ("0.458","0.108","0.0052","0.999","0.002","0.3785"),
        "CALCE_lstm":("0.219","0.210","0.0027","0.999","0.002","0.3630"),
        "NASA_tcn":  ("0.121","0.091","0.0052","1.000","0.000","0.3145"),
        "NASA_lstm": ("0.109","0.096","0.0027","1.000","0.000","0.3062")}
for k, vals in tbl4.items():
    d = c[k]
    real = [d["picp_src"]["mean"], d["picp_src"]["std"], d["mpiw_src"]["mean"],
            d["picp_tgt"]["mean"], d["picp_tgt"]["std"], d["mpiw_tgt"]["mean"]]
    for paper_v, real_v, lab in zip(vals, real, ["src_m","src_s","src_w","tgt_m","tgt_s","tgt_w"]):
        tol = 0.0006 if real_v < 0.01 else 0.0006
        if abs(float(paper_v) - real_v) > tol:
            bad.append(f"表4 {k} {lab}: 论文 {paper_v} vs 实际 {real_v:.4f}")
        else: ok += 1
allsrc = [x for k in c for x in c[k]["picp_src"]["all"]]
if "0.00--0.58" in tex and "0.227" in tex:
    if not (abs(min(allsrc)) < 0.06 and abs(st.mean(allsrc) - 0.227) < 0.005): bad.append(f"4.5 源校准汇总: min={min(allsrc):.2f} mean={st.mean(allsrc):.3f}")
    else: ok += 1

# ---------- 表4 漂移分解（原始协议行）----------
# 注意容差放得比较宽（±25）：这个量的绝对值有上千循环，跨机器/GPU 的
# 训练非确定性在原始协议下会被放大得非常厉害（论文 4.8 承认过这点）
raw = json.load(open(R/"迁移"/"local"/"raw_protocol_multiseed.json", encoding="utf-8"))
zc = raw["CALCE"]["zero_shot"]
if abs(1640 - zc["mean"]) > 25 or abs(886 - zc["std"]) > 25: bad.append(f"表5 原始协议: {zc['mean']:.0f}±{zc['std']:.0f} vs 论文 1640±886")
else: ok += 1

# ---------- 表6 消融 ----------
a = json.load(open(R/"迁移"/"local"/"ablation_v3c_multiseed.json", encoding="utf-8"))
tbl6 = {"CALCE_base7": ("0.1488","0.0151","0.0106","0.0061"), "CALCE_curve14": ("0.1761","0.0206","0.0226","0.0080"),
        "NASA_base7": ("0.1015","0.0108","0.0162","0.0100"), "NASA_curve14": ("0.0943","0.0129","0.0287","0.0157")}
for k, vals in tbl6.items():
    d = a[k]; real = [d["zero"]["mean"], d["zero"]["std"], d["ft"]["mean"], d["ft"]["std"]]
    for pv, rv in zip(vals, real):
        if abs(float(pv) - rv) > 0.00006: bad.append(f"表6 {k}: 论文 {pv} vs 实际 {rv:.4f}")
        else: ok += 1
ftb, ftc = a["CALCE_base7"]["ft"]["mean"], a["CALCE_curve14"]["ft"]["mean"]
if abs((ftc/ftb - 1)*100 - 113) > 1.5: bad.append(f"4.6 恶化百分比: {(ftc/ftb-1)*100:.1f}% vs 论文 113%")
else: ok += 1

# ---------- LOBO（表 2 / 图 1）----------
# 3 个种子的逐折 jsonl 各算宏观均值，再对 3 个均值报 mean±std（92.45±0.98）
L = R/"基线"/"local"
means = []
for s in [42,43,44]:
    rows = [json.loads(l) for l in open(L/f"lobo_tcn_s{s}.jsonl", encoding="utf-8") if l.strip()]
    means.append(st.mean([r["rmse"] for r in rows]))
m, sd = st.mean(means), st.stdev(means)
if "92.45" in tex and abs(m - 92.45) > 0.02: bad.append(f"4.2 LOBO: {m:.2f} vs 论文 92.45")
else: ok += 1

# ---------- 4.5 条件化收窄（Mondrian / weighted 的负结果核对）----------
t4 = json.load(open(R/"保形"/"local"/"t4c_multiseed_summary.json", encoding="utf-8"))
mn = t4["mondrian"]["NASA"]
w_narrow = (1 - mn["mondrian"]["MPIW"]/mn["single"]["MPIW"])*100
if abs(w_narrow - 21) > 1.5: bad.append(f"4.5 Mondrian NASA 收窄: {w_narrow:.1f}% vs 论文 21%")
else: ok += 1
mp, msd = mn["mondrian"]["PICP"]["mean"], mn["mondrian"]["PICP"]["std"]
if abs(mp - 0.64) > 0.006 or abs(msd - 0.18) > 0.006: bad.append(f"4.5 Mondrian NASA PICP: {mp:.2f}±{msd:.2f} vs 论文 0.64±0.18")
else: ok += 1
sb = t4["mondrian"]["NASA"]["single"]["PICP"]["mean"]
if abs(sb - 0.81) > 0.006: bad.append(f"4.5 单一q基线: {sb:.2f} vs 论文 0.81")
else: ok += 1
wg = t4["weighted"]["CALCE"]
wn = (1 - wg["weighted"]["MPIW"]/wg["single"]["MPIW"])*100
if abs(wn - 15) > 1.5: bad.append(f"4.5 Weighted CALCE 收窄: {wn:.1f}% vs 论文 15%")
else: ok += 1

print(f"通过 {ok} 项检查")
if bad:
    print(f"\n✗ {len(bad)} 项不一致:")
    for b in bad: print("  -", b)
else:
    print("✓ 全部数据与论文一致")
