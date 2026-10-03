# -*- coding: utf-8 -*-
"""T3e = T3c 消融协议在建模表 v3c 上的复跑（数据版本对照之二，论文最终采用版）。

与 t3c_ablation_local.py 唯一的区别：DATA 指向建模表_v3c.csv，输出前缀 t3e_。
checks/final_data_check.py 里表 6 的数字核对用的就是 ablation_v3c 的结果，
即本脚本（经聚合后）的产出。协议细节见 t3c_ablation_local.py。"""
import argparse, json, os, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

FEATS_BASE7 = ["capacity_Ah", "soh", "discharge_dur_s", "v_mean_V", "v_min_V",
              "ica_peak", "ica_peak_V"]  # 剔除 charge_dur_s (MIT 缺失)
# 曲线增强: +定分数容量电压点(v_q10..v_q90) +ICA次峰 +半高宽
FEATS_CURVE14 = FEATS_BASE7 + ["v_q10", "v_q30", "v_q50", "v_q70", "v_q90",
                               "ica2_peak", "ica_fwhm"]
FEATS = FEATS_BASE7  # 运行时由 --feats 覆盖
WINDOW = 20
# 建模表路径（数据放 data/ 下即可，合并方法见 code/README.md）
DATA = "data/建模表_v3c.csv"

def set_seed(seed):
    """固定随机源（GPU 卷积仍非确定性，重跑有小幅浮动，见 t2_train_local 同名函数）。"""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_windows_ds(df, dataset, window=WINDOW):
    cells = {}
    sub = df[df["dataset"] == dataset]
    for bid, g in sub.sort_values("cycle").groupby("battery_id"):
        f = g[FEATS].astype(float).copy()
        f = f.interpolate(limit_direction="both").fillna(0.0).values
        cyc = g["cycle"].values
        y = g["soh"].astype(float).values
        ok = ~np.isnan(y)
        f, cyc, y = f[ok], cyc[ok], y[ok]
        X, yy = [], []
        for i in range(window - 1, len(f)):
            if cyc[i] - cyc[i - window + 1] == window - 1:
                X.append(f[i - window + 1:i + 1]); yy.append(y[i])
        if X:
            cells[bid] = (np.asarray(X, np.float32), np.asarray(yy, np.float32))
    return cells

class CausalBlock(nn.Module):
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

def std_with(sc, X):
    return ((X - sc.mean_) / (sc.scale_ + 1e-8)).astype(np.float32)

def concat_cells(cells, bids):
    X = np.concatenate([cells[b][0] for b in bids])
    y = np.concatenate([cells[b][1] for b in bids])
    return X, y, [b for b in bids for _ in range(len(cells[b][0]))]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tcn", choices=["tcn", "lstm"])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--ft-epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--feats", default="base7", choices=["base7", "curve14"])
    ap.add_argument("--out", default="results/ablation")
    ap.add_argument("--deterministic", action="store_true",
                    help="开启 cuDNN/PyTorch 确定性算法（诊断跨管线复现性用；默认关闭，"
                         "与论文发布结果的生产环境一致）")
    ap.add_argument("--data", default=DATA, help="建模表 csv 路径")
    ap.add_argument("--src-cache", default=None,
                    help="统一源模型缓存目录：命中则跳过源域预训练，未命中则训练后写入")
    args = ap.parse_args()
    # 特征集在运行时切换：base7 = 论文的"基础 7 维"，curve14 = "曲线增强 14 维"
    global FEATS
    FEATS = FEATS_BASE7 if args.feats == "base7" else FEATS_CURVE14
    if args.deterministic:
        # CUBLAS_WORKSPACE_CONFIG 必须在首个 CUDA 操作前设置
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("[deterministic] cuDNN/PyTorch deterministic algorithms ON", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"T3b(逐数据集标准化) device={device} model={args.model}", flush=True)
    df = pd.read_csv(args.data)
    src = build_windows_ds(df, "MIT")
    src_bids = sorted(src)
    random.Random(args.seed).shuffle(src_bids)
    n_va = max(1, int(len(src_bids) * 0.1))
    Xtr, ytr, _ = concat_cells(src, src_bids[:-n_va])
    Xva, yva, _ = concat_cells(src, src_bids[-n_va:])
    sc_src = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[2]))
    Xtr, Xva = std_with(sc_src, Xtr), std_with(sc_src, Xva)
    print(f"源域 MIT: 训练电芯 {len(src_bids)-n_va}, 窗口 {len(Xtr)}", flush=True)
    model = new_model(args.model, Xtr.shape[2]).to(device)
    import os as _os
    _cache_hit = None
    if args.src_cache:
        _os.makedirs(args.src_cache, exist_ok=True)
        # base7 源模型与其他脚本共用（统一缓存）；curve14 的输入维度不同，
        # 必须用带特征集后缀的独立缓存名，严禁回退到 7 维旧缓存
        if args.feats == "base7":
            _hit = _os.path.join(args.src_cache, f"src_{args.model}_s{args.seed}_ep{args.epochs}.pt")
            _legacy = _os.path.join(args.src_cache, f"t4d_src_{args.model}_s{args.seed}.pt")
            _cache_hit = _hit if _os.path.exists(_hit) else (_legacy if _os.path.exists(_legacy) else None)
        else:
            _hit = _os.path.join(args.src_cache, f"src_{args.model}_s{args.seed}_ep{args.epochs}_{args.feats}.pt")
            _cache_hit = _hit if _os.path.exists(_hit) else None
    if _cache_hit:
        model.load_state_dict(torch.load(_cache_hit, map_location=device, weights_only=True))
        print(f"[cache] 源模型 {_cache_hit}", flush=True)
    else:
        t0 = time.time()
        model = fit_model(model, Xtr, ytr, args.epochs, args.seed, device, Xva, yva)
        print(f"源域训练完成 {time.time()-t0:.0f}s", flush=True)
        if args.src_cache:
            _sfx = "" if args.feats == "base7" else f"_{args.feats}"
            _save = _os.path.join(args.src_cache, f"src_{args.model}_s{args.seed}_ep{args.epochs}{_sfx}.pt")
            torch.save(model.state_dict(), _save)
            print(f"[cache] 源模型已写入 {_save}", flush=True)
    results = {"model": args.model, "task": "soh", "protocol": "per-dataset-std", "feats": args.feats,
               "seed": args.seed, "targets": {}}
    for tgt_name in ["CALCE", "NASA"]:
        tgt = build_windows_ds(df, tgt_name)
        tb = sorted(tgt)
        n_ft = max(1, len(tb) // 2)
        random.Random(args.seed).shuffle(tb)
        ft_bids, te_bids = tb[:n_ft], tb[n_ft:]
        # 目标域 scaler: 全部目标窗口的无标签统计量(标准无监督DA口径)
        Xall_t, yall_t, _ = concat_cells(tgt, tb)
        sc_tgt = StandardScaler().fit(Xall_t.reshape(-1, Xall_t.shape[2]))
        Xte, yte, _ = concat_cells(tgt, te_bids)
        r_zero = eval_model(model, std_with(sc_tgt, Xte), yte, device)
        Xft, yft, _ = concat_cells(tgt, ft_bids)
        ft_model = new_model(args.model, Xtr.shape[2]).to(device)
        ft_model.load_state_dict(model.state_dict())
        # 微调 lr=3e-4（比源域小一个量级），与 t3b 主实验一致。
        # 消融只动特征集，其他变量全部锁死，差异才能归因到特征上
        ft_model = fit_model(ft_model, std_with(sc_tgt, Xft), yft,
                             args.ft_epochs, args.seed, device, lr=3e-4)
        r_ft = eval_model(ft_model, std_with(sc_tgt, Xte), yte, device)
        to_model = new_model(args.model, Xtr.shape[2]).to(device)
        to_model = fit_model(to_model, std_with(sc_tgt, Xft), yft,
                             args.ft_epochs, args.seed, device, lr=1e-3)
        r_to = eval_model(to_model, std_with(sc_tgt, Xte), yte, device)
        results["targets"][tgt_name] = {"test_cells": te_bids, "ft_cells": ft_bids,
                                        "zero_shot": r_zero, "fine_tune": r_ft, "target_only": r_to}
        print(f"[{tgt_name}] zero-shot RMSE={r_zero['rmse']:.4f} | "
              f"fine-tune RMSE={r_ft['rmse']:.4f} | target-only RMSE={r_to['rmse']:.4f}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/t3e_{args.feats}_{args.model}_s{args.seed}.json", "w") as f:
        json.dump(results, f, indent=1)
    print("T3B DONE", flush=True)

if __name__ == "__main__":
    main()
