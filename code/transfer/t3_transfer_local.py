# -*- coding: utf-8 -*-
"""T3 跨化学体系迁移主实验：MIT(LFP 源域) → CALCE / NASA(LCO 目标域)。

三种机制对照（论文表 3 的数据源）：
  zero-shot   源域模型直接上目标域，预期崩，崩法是论文贡献之一
  fine-tune   加载源域权重、小学习率微调，本方法
  target-only 同量目标域数据从头训练，回答"迁移到底值不值"

任务口径 --task：soh 为主口径（每个循环都有 SOH 标签，样本多）；
rul 只统计到达 EOL 的电芯（NASA 只有部分电芯观测到 EOL），样本少、仅辅助对照。
本脚本沿用源域统计量标准化（"原始协议"），目标域统计量标准化的对照见 t3b_std_local.py。

运行：python t3_transfer_local.py --model tcn --task soh --seed 42
论文口径：种子 42~46 各跑一次，报 5 种子均值±标准差。
输出：results/transfer/t3_<task>_<model>_s<seed>.json"""
import argparse, json, os, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

FEATS = ["capacity_Ah", "soh", "discharge_dur_s", "v_mean_V", "v_min_V",
         "ica_peak", "ica_peak_V", "charge_dur_s"]
WINDOW = 20
# 建模表路径（数据放 data/ 下即可，合并方法见 code/README.md）
DATA = "data/建模表_v3.csv"

def set_seed(seed):
    """固定随机源。GPU 卷积本身仍非确定性，同种子重跑的指标会有小幅浮动，
    这正是"同一配置重复运行 RMSE 波动可达 14%"（论文 4.8）的主要来源。"""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_windows_ds(df, dataset, task, window=WINDOW):
    """按数据集切滑窗。task=soh：每循环 y=SOH；task=rul：y=RUL，
    只保留观测到 EOL 的电芯（其余电芯没有寿命终点，RUL 标签无定义）。"""
    cells = {}
    sub = df[df["dataset"] == dataset]
    ycol = "soh" if task == "soh" else "rul"
    for bid, g in sub.sort_values("cycle").groupby("battery_id"):
        f = g[FEATS].astype(float).copy()
        f = f.interpolate(limit_direction="both").fillna(0.0).values
        cyc = g["cycle"].values
        y_all = g[ycol].astype(float).values
        ok = ~np.isnan(y_all)
        f, cyc, y_all = f[ok], cyc[ok], y_all[ok]
        X, y = [], []
        for i in range(window - 1, len(f)):
            if cyc[i] - cyc[i - window + 1] == window - 1:
                X.append(f[i - window + 1:i + 1]); y.append(y_all[i])
        if X:
            cells[bid] = (np.asarray(X, np.float32), np.asarray(y, np.float32))
    return cells

class CausalBlock(nn.Module):
    """膨胀因果卷积残差块（与 t2_train_local 完全同构。各脚本刻意自包含，
    单独拷一个文件也能跑，代价是这几段代码在多个脚本里重复）。"""
    def __init__(self, in_ch, out_ch, k, dil):
        super().__init__()
        self.pad = (k - 1) * dil
        self.conv = nn.Conv1d(in_ch, out_ch, k, dilation=dil, padding=self.pad)
        self.res = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x):
        y = self.conv(x)
        if self.pad:
            y = y[:, :, :-self.pad]
        return torch.relu(y + self.res(x))

class TCN(nn.Module):
    def __init__(self, input_dim, d=64, layers=4):
        super().__init__()
        ch = [input_dim] + [d] * layers
        self.blocks = nn.ModuleList([CausalBlock(ch[i], ch[i+1], 3, 2**i) for i in range(layers)])
        self.fc = nn.Linear(d, 1)
    def forward(self, x):
        h = x.transpose(1, 2)
        for blk in self.blocks:
            h = blk(h)
        return self.fc(h[:, :, -1]).squeeze(-1)

class RNNWrap(nn.Module):
    def __init__(self, input_dim, kind):
        super().__init__()
        self.r = (nn.LSTM if kind == "lstm" else nn.GRU)(input_dim, 128, 2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(128, 1)
    def forward(self, x):
        o, _ = self.r(x)
        return self.fc(o[:, -1]).squeeze(-1)

def new_model(name, input_dim):
    if name in ("lstm", "gru"):
        return RNNWrap(input_dim, name)
    if name == "tcn":
        return TCN(input_dim)
    raise ValueError(name)

def fit_model(model, Xtr, ytr, epochs, seed, device, Xva=None, yva=None, lr=1e-3):
    set_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(torch.tensor(Xtr, dtype=torch.float32),
                                        torch.tensor(ytr, dtype=torch.float32))
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    Xva = Xtr if Xva is None else Xva
    yva = ytr if yva is None else yva
    best, best_state, patience = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); lossf(model(xb), yb).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            va = lossf(model(torch.tensor(Xva, dtype=torch.float32).to(device)),
                       torch.tensor(yva, dtype=torch.float32).to(device)).item()
        if np.isfinite(va) and va < best - 1e-6:
            best, patience = va, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 10:
                break
    if best_state:
        model.load_state_dict(best_state)
    return model

def eval_model(model, X, y, device):
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()
    return {"rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
            "mae": float(np.mean(np.abs(pred - y))), "n": int(len(y))}

def concat_cells(cells, bids):
    X = np.concatenate([cells[b][0] for b in bids])
    y = np.concatenate([cells[b][1] for b in bids])
    return X, y, [b for b in bids for _ in range(len(cells[b][0]))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tcn", choices=["tcn", "lstm"])
    ap.add_argument("--task", default="soh", choices=["soh", "rul"])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--ft-epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/transfer")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"T3v2 device={device} model={args.model} task={args.task}", flush=True)
    df = pd.read_csv(DATA)
    src = build_windows_ds(df, "MIT", args.task)
    src_bids = sorted(src)
    # 源域内部切 90/10 训练/验证：种子只喂给 random.Random，划分跨运行可复现
    random.Random(args.seed).shuffle(src_bids)
    n_va = max(1, int(len(src_bids) * 0.1))
    Xtr, ytr, _ = concat_cells(src, src_bids[:-n_va])
    Xva, yva, _ = concat_cells(src, src_bids[-n_va:])
    # 源域 scaler。注意本脚本的目标域也用这个 scaler 标准化（"原始协议"）。
    # CALCE/NASA 特征量纲与 MIT 不同，这正是 zero-shot 崩到 0.1+ 量级的主要原因之一；
    # 逐数据集标准化的对照见 t3b_std_local.py，两者合起来 = 论文的漂移分解。
    sc = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[2]))
    Xtr = ((Xtr - sc.mean_) / (sc.scale_ + 1e-8)).astype(np.float32)
    Xva = ((Xva - sc.mean_) / (sc.scale_ + 1e-8)).astype(np.float32)
    print(f"源域 MIT: 训练电芯 {len(src_bids)-n_va}, 窗口 {len(Xtr)}", flush=True)
    model = new_model(args.model, Xtr.shape[2]).to(device)
    t0 = time.time()
    model = fit_model(model, Xtr, ytr, args.epochs, args.seed, device, Xva, yva)
    print(f"源域训练完成 {time.time()-t0:.0f}s", flush=True)

    results = {"model": args.model, "task": args.task, "seed": args.seed, "targets": {}}
    # soh 口径：CALCE 8 颗对半分微调/测试；NASA 4 颗取 2 颗微调。
    # rul 口径只有 NASA 支持且固定 1 颗微调（样本太少，不做自动划分）。
    targets = [("CALCE", None), ("NASA", None)] if args.task == "soh" else [("NASA", 1)]
    for tgt_name, n_ft_fixed in targets:
        tgt = build_windows_ds(df, tgt_name, args.task)
        tb = sorted(tgt)
        if len(tb) < 2:
            print(f"[{tgt_name}] 有效电芯不足({len(tb)}), 跳过", flush=True)
            continue
        n_ft = n_ft_fixed if n_ft_fixed else max(1, len(tb) // 2)
        random.Random(args.seed).shuffle(tb)
        ft_bids, te_bids = tb[:n_ft], tb[n_ft:]
        Xte, yte, _ = concat_cells(tgt, te_bids)
        Xte = ((Xte - sc.mean_) / (sc.scale_ + 1e-8)).astype(np.float32)
        r_zero = eval_model(model, Xte, yte, device)
        Xft, yft, _ = concat_cells(tgt, ft_bids)
        Xft = ((Xft - sc.mean_) / (sc.scale_ + 1e-8)).astype(np.float32)
        ft_model = new_model(args.model, Xtr.shape[2]).to(device)
        ft_model.load_state_dict(model.state_dict())
        # 微调学习率 3e-4 = 源域预训练(1e-3)的 1/3 量级：目标域只有几颗电芯，
        # 大学习率会把源域学到的退化知识冲掉，等于白白扔掉预训练
        ft_model = fit_model(ft_model, Xft, yft, args.ft_epochs, args.seed, device, lr=3e-4)
        r_ft = eval_model(ft_model, Xte, yte, device)
        # target-only 对照：同样的数据、同样的轮数，但从零初始化、全量学习率。
        # "预训练有没有用"只有跟它比才知道（这就是表 3 的"目标域基线"列）
        to_model = new_model(args.model, Xtr.shape[2]).to(device)
        to_model = fit_model(to_model, Xft, yft, args.ft_epochs, args.seed, device, lr=1e-3)
        r_to = eval_model(to_model, Xte, yte, device)
        results["targets"][tgt_name] = {"test_cells": te_bids, "ft_cells": ft_bids,
                                        "zero_shot": r_zero, "fine_tune": r_ft, "target_only": r_to}
        print(f"[{tgt_name}] zero-shot RMSE={r_zero['rmse']:.4f} | "
              f"fine-tune RMSE={r_ft['rmse']:.4f} | target-only RMSE={r_to['rmse']:.4f}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/t3_{args.task}_{args.model}_s{args.seed}.json", "w") as f:
        json.dump(results, f, indent=1)
    print("T3 DONE", flush=True)

if __name__ == "__main__":
    main()
