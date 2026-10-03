# -*- coding: utf-8 -*-
"""按《储能科学与技术》绘图体例重绘论文全部 5 张图（英文图内文字版）。

数据源（全部随仓库分发，克隆后可直接运行）：
  results_cx2/  投稿版 16 电芯口径 —— 图 2/3/4/5
  results/      8 电芯历史口径 —— 仅图 1 的 LOBO 逐电芯数据（results_cx2 无 baselines）
输出：本目录 figures/ 下的 PNG(600 dpi) + PDF(矢量)，文件名与论文一致。

绘图体例：
  图宽 75 mm（单栏）；Arial 六号字（7.5 pt）；封闭图框；刻度朝向图内；不用背景网格；
  图内坐标轴标题/图例/图注全英文；分图用 (a)(b) 区分且置于各分图下方；
  坐标物理量与单位间用斜线；对数尺度误差棒仅向上展开（对称下须会被轴下限裁掉）。

跑一遍约 10 秒、无 GPU；改论文数据后重跑即可同步全部图。"""
import json, os, statistics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "Arial", "font.size": 7.5, "axes.linewidth": 0.6,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True,
    "ytick.right": True, "axes.grid": False, "legend.frameon": False,
    "legend.fontsize": 7, "axes.labelsize": 7.5, "xtick.labelsize": 7,
    "ytick.labelsize": 7,
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
R = os.path.join(ROOT, "results_cx2")   # 投稿版口径（CALCE 16 颗）
R8 = os.path.join(ROOT, "results")      # 8 电芯历史口径（仅图 1 使用）
OUT = os.path.join(HERE, "figures")
os.makedirs(OUT, exist_ok=True)
W = 2.953  # 75 mm


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved:", name)


def log_yerr(vals, mean_ref):
    """对数尺度的 +1 标准差（仅向上）。对称展开的下须会低于轴下限而被裁掉。"""
    lg = [np.log10(v) for v in vals]
    s = float(np.std(lg, ddof=1))
    lm = np.log10(mean_ref)
    return mean_ref, 0.0, 10 ** (lm + s) - mean_ref


# ---------- 图 1: LOBO 逐电芯 RMSE 分布（3 种子合并 357 折）----------
rmses = []
for _s in (42, 43, 44):
    with open(os.path.join(R8, "baselines", f"lobo_tcn_s{_s}.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rmses.append(json.loads(line)["rmse"])
assert len(rmses) == 357, len(rmses)
med = statistics.median(rmses)

fig, ax = plt.subplots(figsize=(W, 2.1))
ax.hist(rmses, bins=30, color="#3d7ab5", edgecolor="white", linewidth=0.3)
ax.axvline(med, color="#c0392b", linestyle="--", linewidth=0.8)
ax.text(med + 8, ax.get_ylim()[1] * 0.92, f"Median {med:.1f}", color="#c0392b", fontsize=7)
ax.set_xlabel("Per-cell RMSE / cycles")
ax.set_ylabel("Count")
ax.set_xlim(0, max(rmses) * 1.02)
fig.tight_layout()
save(fig, "lobo_distribution")


# ---------- 图 2: 跨化学体系迁移对比（表 3 的可视化）----------
d = json.load(open(os.path.join(R, "transfer", "transfer_multiseed_v2.json"), encoding="utf-8"))
T3B = os.path.join(R, "transfer")
KMAP = {"zero": "zero_shot", "ft": "fine_tune", "to": "target_only"}


def per_seed(dom, model, key):
    out = []
    for s in (42, 43, 44, 45, 46):
        j = json.load(open(os.path.join(T3B, f"t3b_{model}_s{s}.json"), encoding="utf-8"))
        out.append(j["targets"][dom][KMAP[key]]["rmse"])
    return out


groups = ["CALCE_tcn", "CALCE_lstm", "NASA_tcn", "NASA_lstm"]
# 图宽 75 mm 时 4 组横排标签会相互重叠，改为两行显示
glabels = ["CALCE\nTCN", "CALCE\nLSTM", "NASA\nTCN", "NASA\nLSTM"]
series = [("zero", "Zero-shot", "#3d7ab5"), ("ft", "Fine-tune", "#e8a33d"), ("to", "Target-only", "#7f7f7f")]
x = np.arange(len(groups)); w = 0.26
fig, ax = plt.subplots(figsize=(W, 2.1))
for i, (key, lab, color) in enumerate(series):
    st = [log_yerr(per_seed(*g.split("_"), key), d[g][key]["mean"]) for g in groups]
    means = [s[0] for s in st]
    yerr = [[s[1] for s in st], [s[2] for s in st]]
    ax.bar(x + (i - 1) * w, means, w, yerr=yerr, capsize=1.5, color=color, label=lab,
           error_kw=dict(linewidth=0.6))
ax.set_yscale("log")
ax.set_ylabel("RMSE (SOH, log scale)")
ax.set_xticks(x); ax.set_xticklabels(glabels)
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14), columnspacing=0.8, handlelength=1.2)
fig.tight_layout()
save(fig, "transfer_comparison")


# ---------- 图 3: 覆盖率-校准电芯数扫描 ----------
sw = json.load(open(os.path.join(R, "conformal", "t4_split_sweep", "sweep_summary.json"), encoding="utf-8"))
fig, ax = plt.subplots(figsize=(W, 2.0))
marks = {"tcn": ("o", "#3d7ab5", "TCN"), "lstm": ("s", "#e8a33d", "LSTM")}
for m in ("tcn", "lstm"):
    xs = sorted(int(k[3:]) for k in sw[m])
    ys = [sw[m][f"cal{nc}"]["picp_tgt"]["mean"] for nc in xs]
    es = [sw[m][f"cal{nc}"]["picp_tgt"]["std"] for nc in xs]
    mk, cl, lab = marks[m]
    ax.errorbar(xs, ys, yerr=es, marker=mk, markersize=3.5, color=cl, label=lab,
                linewidth=0.9, capsize=2, elinewidth=0.6)
ax.axhline(0.90, color="black", linestyle="--", linewidth=0.7)
ax.text(1.02, 1.048, "Nominal 0.90", fontsize=7, ha="left", va="top")
ax.set_xlabel("Number of calibration cells")
ax.set_ylabel("PICP (target-calibrated)")
ax.set_xticks([1, 2, 3, 4, 5])
ax.set_ylim(0.5, 1.05)
ax.legend(loc="lower right")
fig.tight_layout()
save(fig, "calibration_sweep")


# ---------- 图 4: 保形覆盖双路由对照（表 5 的可视化，分图）----------
d = json.load(open(os.path.join(R, "conformal", "conformal_multiseed_summary.json"), encoding="utf-8"))
fig, axes = plt.subplots(1, 2, figsize=(W, 1.9), sharey=True)
for _i_ax, (ax, domain) in enumerate(zip(axes, ["CALCE", "NASA"])):
    backs = ["tcn", "lstm"]
    x = np.arange(2); w = 0.34
    src_m = [d[f"{domain}_{b}"]["picp_src"]["mean"] for b in backs]
    src_s = [d[f"{domain}_{b}"]["picp_src"]["std"] for b in backs]
    tgt_m = [d[f"{domain}_{b}"]["picp_tgt"]["mean"] for b in backs]
    tgt_s = [d[f"{domain}_{b}"]["picp_tgt"]["std"] for b in backs]
    ax.bar(x - w / 2, src_m, w, yerr=src_s, capsize=1.5, color="#3d7ab5",
           label="Source-calibrated", error_kw=dict(linewidth=0.6))
    ax.bar(x + w / 2, tgt_m, w, yerr=tgt_s, capsize=1.5, color="#e8a33d",
           label="Target-calibrated", error_kw=dict(linewidth=0.6))
    rng = np.random.default_rng(0)
    for xi, vals in zip(x - w / 2, [d[f"{domain}_{b}"]["picp_src"].get("all", []) for b in backs]):
        if vals:
            ax.scatter(xi + rng.uniform(-0.06, 0.06, len(vals)), vals, s=3, color="#1f4e79",
                       zorder=3, linewidths=0)
    for xi, vals in zip(x + w / 2, [d[f"{domain}_{b}"]["picp_tgt"].get("all", []) for b in backs]):
        if vals:
            ax.scatter(xi + rng.uniform(-0.06, 0.06, len(vals)), vals, s=3, color="#a86a1a",
                       zorder=3, linewidths=0)
    ax.axhline(0.90, color="black", linestyle="--", linewidth=0.7)
    ax.text(0.04, 0.97, "Nominal 0.90", transform=ax.transAxes, fontsize=7, ha="left", va="top")
    ax.set_xticks(x); ax.set_xticklabels(["TCN", "LSTM"])
    ax.set_ylim(0, 1.12)
    # 《绘图体例》第 3 条：分图用（a）（b）区分，分图题置于各分图下方
    ax.text(0.5, -0.235, "(%s) Target: %s" % ("ab"[_i_ax], domain),
            transform=ax.transAxes, ha="center", va="top", fontsize=7)
axes[0].set_ylabel("PICP (empirical coverage)")
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.06))
fig.tight_layout(rect=(0, 0.07, 1, 0.94))
save(fig, "conformal_coverage")


# ---------- 图 5: 特征丰富度消融（表 6 的可视化，v3c 数据版本）----------
d = json.load(open(os.path.join(R, "ablation", "ablation_v3c_multiseed.json"), encoding="utf-8"))
T3E = os.path.join(R, "ablation")
KMAP2 = {"zero": "zero_shot", "ft": "fine_tune"}


def per_seed_abl(suffix, dom, task):
    out = []
    for s in (42, 43, 44, 45, 46):
        j = json.load(open(os.path.join(T3E, f"t3e_{suffix}_tcn_s{s}.json"), encoding="utf-8"))
        out.append(j["targets"][dom][KMAP2[task]]["rmse"])
    return out


groups4 = [("CALCE", "zero", "CALCE\nzero-shot"), ("CALCE", "ft", "CALCE\nfine-tune"),
           ("NASA", "zero", "NASA\nzero-shot"), ("NASA", "ft", "NASA\nfine-tune")]
x = np.arange(len(groups4)); w = 0.34
fig, ax = plt.subplots(figsize=(W, 2.1))
for i, (suffix, lab, color) in enumerate([("base7", "Base-7", "#3d7ab5"), ("curve14", "Curve-14", "#e8a33d")]):
    st = [log_yerr(per_seed_abl(suffix, dom, task), d[f"{dom}_{suffix}"][task]["mean"]) for dom, task, _ in groups4]
    means = [s[0] for s in st]
    yerr = [[s[1] for s in st], [s[2] for s in st]]
    ax.bar(x + (i - 0.5) * w, means, w, yerr=yerr, capsize=1.5, color=color, label=lab,
           error_kw=dict(linewidth=0.6))
ax.set_yscale("log")
ax.set_ylabel("RMSE (SOH, log scale)")
ax.set_xticks(x); ax.set_xticklabels([g[2] for g in groups4], fontsize=7)
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.13))
fig.tight_layout()
save(fig, "feature_ablation")

print("全部完成 ->", OUT)
