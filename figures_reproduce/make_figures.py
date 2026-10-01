# -*- coding: utf-8 -*-
"""按《储能科学与技术》绘图体例重绘论文全部 4 张图（英文图内文字版）。

早期版本的图是 matplotlib 默认样式的中文图，投稿前按期刊绘图体例统一重绘：
图宽 75mm（单栏）、Arial、封闭图框、刻度朝内、无背景网格、图内文字全英文。
体例的红线是"图内不得出现中文"，所以连图例里的中文也全部换成了英文标签。

输入：仓库内 results/ 下的实验结果 JSON（随仓库分发，克隆后直接可跑）；
输出：本目录 figures/ 下的 PNG(600dpi) + PDF(矢量) 各一份，文件名与论文一致。

跑一遍约 10 秒、无 GPU；改论文数据后重跑即可同步全部图。"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 7,
    "axes.linewidth": 0.6,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "axes.grid": False,
    "legend.frameon": False,
    "legend.fontsize": 6,
    "axes.labelsize": 7,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
})

# 两个数据目录改成你的本地路径（实验结果汇总 JSON + LOBO 逐折 jsonl）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根目录
RESULTS = os.path.join(ROOT, "results")  # 论文结果数据，随仓库分发
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

W = 2.953  # 75mm 换算成英寸（期刊单栏宽度），所有 figsize 都以它为基准

def save(fig, name):
    """PNG 600dpi（投稿系统要位图）+ PDF 矢量（排版首选）双格式落盘。"""
    fig.savefig(os.path.join(OUT, name + ".png"), dpi=600, bbox_inches="tight")
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved:", name)

# ---------- 图1: LOBO 逐电芯 RMSE 分布（3 种子合并 357 折）----------
# 表 2 报的是单次运行的分位数，图 1 画的是 3 个种子合并后的分布，
# 两者口径不同，图题里专门写了"合并口径中位数 76.7"，别混用
rmses = []
for s in (42, 43, 44):
    with open(os.path.join(RESULTS, "baselines", f"lobo_tcn_s{s}.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rmses.append(json.loads(line)["rmse"])
# 357 = 119 折 × 3 种子；数量对不上说明 jsonl 不全，宁可炸也别画出错的图
assert len(rmses) == 357, len(rmses)
import statistics  # 只在这里用一次，就地导入
med = statistics.median(rmses)  # 合并口径中位数（表 2 的 80.90 是单次运行口径）

fig, ax = plt.subplots(figsize=(W, 2.1))
ax.hist(rmses, bins=30, color="#3d7ab5", edgecolor="white", linewidth=0.3)
ax.axvline(med, color="#c0392b", linestyle="--", linewidth=0.8)
ax.text(med + 8, ax.get_ylim()[1] * 0.92, f"Median {med:.1f}", color="#c0392b", fontsize=6.5)
ax.set_xlabel("Per-cell RMSE (cycles)")
ax.set_ylabel("Count")
ax.set_xlim(0, max(rmses) * 1.02)
fig.tight_layout()
save(fig, "lobo_distribution")

# ---------- 图2: 跨化学体系迁移对比（表 3 的可视化）----------
# 用 log 轴：zero-shot 与微调差一个数量级，线性轴下微调的误差棒会被压成一条线
d = json.load(open(os.path.join(RESULTS, "transfer", "transfer_multiseed_v2.json"), encoding="utf-8"))
groups = ["CALCE_tcn", "CALCE_lstm", "NASA_tcn", "NASA_lstm"]
glabels = ["CALCE/TCN", "CALCE/LSTM", "NASA/TCN", "NASA/LSTM"]
series = [("zero", "Zero-shot", "#c0504d"), ("ft", "Fine-tune", "#70ad47"), ("to", "Target-only", "#7f7f7f")]
import numpy as np

T3B = os.path.join(RESULTS, "transfer")
KMAP = {"zero": "zero_shot", "ft": "fine_tune", "to": "target_only"}
def per_seed(dom, model, key):
    out = []
    for s in (42, 43, 44, 45, 46):
        j = json.load(open(os.path.join(T3B, f"t3b_{model}_s{s}.json"), encoding="utf-8"))
        out.append(j["targets"][dom][KMAP[key]]["rmse"])
    return out

def log_yerr(vals, mean_ref):
    """对数尺度的 ±1 标准差：柱高保持算术均值（与表内数字一致），
    上下须按 log10 空间的 std 对称展开——原始尺度对称 std 在 log 轴上
    会把下须拉到轴底（如 CALCE/LSTM 微调 mean-std≈0.0005）。"""
    lg = [np.log10(v) for v in vals]
    s = float(np.std(lg, ddof=1))
    lm = np.log10(mean_ref)
    lo = 10 ** (lm - s); hi = 10 ** (lm + s)
    return mean_ref, mean_ref - lo, hi - mean_ref

x = np.arange(len(groups)); w = 0.26
fig, ax = plt.subplots(figsize=(W, 2.1))
for i, (key, lab, color) in enumerate(series):
    st = [log_yerr(per_seed(*g.split("_"), key), d[g][key]["mean"]) for g in groups]
    for s, g in zip(st, groups):
        assert abs(s[0] - d[g][key]["mean"]) < 1e-6, g
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

# ---------- 图3: 保形覆盖双路由对照（表 5 的可视化）----------
# 0.90 名义线是全图参照系：源校准柱比它矮一截、目标校准柱顶着 1.0，
# 一张图把"崩塌与恢复"说清楚
d = json.load(open(os.path.join(RESULTS, "conformal", "conformal_multiseed_summary.json"), encoding="utf-8"))
fig, axes = plt.subplots(1, 2, figsize=(W, 1.9), sharey=True)
for ax, domain in zip(axes, ["CALCE", "NASA"]):
    backs = ["tcn", "lstm"]
    x = np.arange(2); w = 0.34
    src_m = [d[f"{domain}_{b}"]["picp_src"]["mean"] for b in backs]
    src_s = [d[f"{domain}_{b}"]["picp_src"]["std"] for b in backs]
    tgt_m = [d[f"{domain}_{b}"]["picp_tgt"]["mean"] for b in backs]
    tgt_s = [d[f"{domain}_{b}"]["picp_tgt"]["std"] for b in backs]
    ax.bar(x - w/2, src_m, w, yerr=src_s, capsize=1.5, color="#c0504d",
           label="Source-calibrated", error_kw=dict(linewidth=0.6))
    ax.bar(x + w/2, tgt_m, w, yerr=tgt_s, capsize=1.5, color="#70ad47",
           label="Target-calibrated", error_kw=dict(linewidth=0.6))
    ax.axhline(0.90, color="black", linestyle="--", linewidth=0.7)
    ax.text(1.35, 0.87, "Nominal 0.90", fontsize=6, ha="right", va="top")
    ax.set_xticks(x); ax.set_xticklabels(["TCN", "LSTM"])
    ax.set_ylim(0, 1.12)
    ax.set_title(f"Target: {domain}", fontsize=7)
axes[0].set_ylabel("PICP (empirical coverage)")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.06))
fig.tight_layout(rect=(0, 0, 1, 0.94))
save(fig, "conformal_coverage")

# ---------- 图4: 特征丰富度消融（表 6 的可视化，v3c 数据版本）----------
d = json.load(open(os.path.join(RESULTS, "ablation", "ablation_v3c_multiseed.json"), encoding="utf-8"))
groups = [("CALCE", "zero", "CALCE\nzero-shot"), ("CALCE", "ft", "CALCE\nfine-tune"),
          ("NASA", "zero", "NASA\nzero-shot"), ("NASA", "ft", "NASA\nfine-tune")]
x = np.arange(len(groups)); w = 0.34
fig, ax = plt.subplots(figsize=(W, 2.1))
T3E = os.path.join(RESULTS, "ablation")
def per_seed_abl(suffix, dom, task):
    kmap = {"zero": "zero_shot", "ft": "fine_tune"}
    out = []
    for s in (42, 43, 44, 45, 46):
        j = json.load(open(os.path.join(T3E, f"t3e_{suffix}_tcn_s{s}.json"), encoding="utf-8"))
        out.append(j["targets"][dom][kmap[task]]["rmse"])
    return out

for i, (suffix, lab, color) in enumerate([("base7", "Base-7", "#3d7ab5"), ("curve14", "Curve-14", "#e8a33d")]):
    st = [log_yerr(per_seed_abl(suffix, dom, task), d[f"{dom}_{suffix}"][task]["mean"]) for dom, task, _ in groups]
    for s, (dom, task, _) in zip(st, groups):
        assert abs(s[0] - d[f"{dom}_{suffix}"][task]["mean"]) < 1e-6, (dom, task)
    means = [s[0] for s in st]
    yerr = [[s[1] for s in st], [s[2] for s in st]]
    ax.bar(x + (i - 0.5) * w, means, w, yerr=yerr, capsize=1.5, color=color, label=lab,
           error_kw=dict(linewidth=0.6))
ax.set_yscale("log")
ax.set_ylabel("RMSE (SOH, log scale)")
ax.set_xticks(x); ax.set_xticklabels([g[2] for g in groups], fontsize=6)
ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.13))
fig.tight_layout()
save(fig, "feature_ablation")

print("全部完成 ->", OUT)
