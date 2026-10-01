# -*- coding: utf-8 -*-
"""T2 源域基线：四种骨干（LSTM/GRU/TCN/Transformer）在 MIT-Stanford 上的 RUL 预测。

这是训练侧的起点：先在这里比较四种骨干、选出迁移骨干（论文选了 TCN，精度与
Transformer 同档、训练便宜得多，理由见论文 4.2），再进入 ../transfer/ 的跨域迁移。

评估用 5 折"组"交叉验证：按电芯分组切折，同一颗电芯的窗口要么整颗进训练、
要么整颗进测试。相邻循环的窗口几乎一样，若按窗口随机切分，测试集会被训练集
"背过答案"，指标虚高得毫无意义。

运行：python t2_train_local.py --model tcn --smoke   # 单电芯过拟合自检
      python t2_train_local.py --model tcn --folds 5 # 正式 5 折
输入：建模表_v3.csv；输出：results/baselines/<model>_f<fold>_s<seed>.json（逐折 + 汇总）"""
import argparse, json, os, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

# 8 维循环级特征。charge_dur_s 只在 MIT/CALCE 有、NASA 缺失，所以这里用 8 维；
# 到 T3b 逐数据集标准化协议会剔除它降为 7 维（见 ../transfer/t3b_std_local.py）。
FEATS = ["capacity_Ah", "soh", "discharge_dur_s", "v_mean_V", "v_min_V",
         "ica_peak", "ica_peak_V", "charge_dur_s"]
WINDOW = 20
# 建模表路径（数据放 data/ 下即可，合并方法见 code/README.md）
DATA = "data/建模表_v3.csv"

def set_seed(seed):
    """固定三处随机源。注意这只保证划分/初始化/洗牌可复现：GPU 卷积本身非确定性，
    同配置重跑仍有零点几到几个百分点的浮动。论文报 5 种子统计就是为此。"""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_windows(df, window=WINDOW):
    # 只收循环号严格连续的窗口：休眠/补测造成的循环号跳变口不收，
    # 避免把"时间断裂"喂给模型；RUL 缺失（未观测到 EOL）的行一并剔除。
    cells = {}
    for bid, g in df.sort_values("cycle").groupby("battery_id"):
        f = g[FEATS].astype(float).copy()
        f = f.interpolate(limit_direction="both").fillna(0.0).values
        cyc = g["cycle"].values
        rul = g["rul"].astype(float).values
        ok = ~np.isnan(rul)
        f, cyc, rul = f[ok], cyc[ok], rul[ok]
        X, y = [], []
        for i in range(window - 1, len(f)):
            if cyc[i] - cyc[i - window + 1] == window - 1:
                X.append(f[i - window + 1:i + 1]); y.append(rul[i])
        if X:
            cells[bid] = (np.asarray(X, np.float32), np.asarray(y, np.float32))
    return cells

class CausalBlock(nn.Module):
    """膨胀因果卷积 + 残差。"因果"= 每步只看过去，不偷看未来；左 padding 补的零
    会混进序列尾部，forward 里切掉，否则窗口末端（真正拿去预测的位置）被污染。"""
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

class Transformer(nn.Module):
    """轻量 Transformer：3 层 / d=128 / 4 头，可学习位置编码。
    精度与 TCN 同档但训练慢数倍，这是最终选 TCN 做迁移骨干的原因之一。"""
    def __init__(self, input_dim, d=128, heads=4, layers=3):
        super().__init__()
        self.inp = nn.Linear(input_dim, d)
        self.pos = nn.Parameter(torch.randn(1, 512, d) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(d, heads, 256, dropout=0.1, batch_first=True)
        self.enc = nn.TransformerEncoder(enc_layer, layers)
        self.fc = nn.Linear(d, 1)
    def forward(self, x):
        h = self.inp(x) + self.pos[:, :x.shape[1]]
        return self.fc(self.enc(h)[:, -1]).squeeze(-1)

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
    if name == "transformer":
        return Transformer(input_dim)
    raise ValueError(name)

def train_one(model_name, Xtr, ytr, Xte, yte, epochs, seed, device, Xva=None, yva=None):
    """训练并评估一个折。早停盯验证折损失（patience=10），200 轮上限经常跑不满。"""
    set_seed(seed)
    model = new_model(model_name, Xtr.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.MSELoss()
    ds = torch.utils.data.TensorDataset(torch.tensor(Xtr, dtype=torch.float32), torch.tensor(ytr, dtype=torch.float32))
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=True)
    Xva = Xtr if Xva is None else Xva
    yva = ytr if yva is None else yva
    best, best_state, patience = float("inf"), None, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); lossf(model(xb), yb).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval()
        with torch.no_grad():
            va = lossf(model(torch.tensor(Xva, dtype=torch.float32).to(device)), torch.tensor(yva, dtype=torch.float32).to(device)).item()
        if va < best - 1e-6:
            best, patience = va, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 10:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(Xte, dtype=torch.float32).to(device)).cpu().numpy()
    rmse = float(np.sqrt(np.mean((pred - yte) ** 2)))
    mae = float(np.mean(np.abs(pred - yte)))
    return rmse, mae, ep + 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="lstm", choices=["lstm", "gru", "tcn", "transformer"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dataset", default="MIT")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="results/baselines")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={args.model} dataset={args.dataset}", flush=True)
    df = pd.read_csv(DATA)
    df = df[df["dataset"] == args.dataset]
    cells = build_windows(df)
    bids = sorted(cells)
    print(f"有效窗口电芯: {len(bids)}/{df['battery_id'].nunique()}", flush=True)
    if args.smoke:
        # 冒烟自检：拿一颗电芯自己训自己，RMSE 应压到接近 0；压不下去 = 管线有 bug。
        # 写/改训练代码时这是最便宜的"通不通"判据，任何改动后建议先跑一遍。
        bid = bids[0]
        X, y = cells[bid]
        n = len(X); tr_n = int(n * 0.8)
        sc = StandardScaler().fit(X[:tr_n].reshape(-1, X.shape[2]))
        Xt = (X - sc.mean_) / (sc.scale_ + 1e-8)
        rmse, mae, ep = train_one(args.model, Xt[:tr_n], y[:tr_n], Xt[:tr_n], y[:tr_n], 50, args.seed, device)
        print(f"[SMOKE 过拟合自检] {bid}: train RMSE={rmse:.2f} MAE={mae:.2f} epochs={ep}", flush=True)
        rmse, mae, ep = train_one(args.model, Xt[:tr_n], y[:tr_n], Xt[tr_n:], y[tr_n:], 100, args.seed, device)
        print(f"[SMOKE 保留集] {bid}: test RMSE={rmse:.2f} MAE={mae:.2f} epochs={ep}", flush=True)
        return
    X_all = np.concatenate([cells[b][0] for b in bids])
    y_all = np.concatenate([cells[b][1] for b in bids])
    g_all = np.concatenate([[b] * len(cells[b][0]) for b in bids])
    gkf = GroupKFold(n_splits=args.folds)
    res, t0 = [], time.time()
    # 组交叉验证正式跑批
    for k, (tri, tei) in enumerate(gkf.split(X_all, y_all, g_all)):
        # scaler 只用训练折拟合：拿全量拟合会把测试折的均值方差泄漏进来
        sc = StandardScaler().fit(X_all[tri].reshape(-1, X_all.shape[2]))
        # 训练/验证/测试三处一律用同一套统计量标准化。
        # 早期版本漏了训练集：训练在原始量纲、评测在标准化量纲，指标被整体抬高（2026-10 修订）
        Xtr = (X_all[tri] - sc.mean_) / (sc.scale_ + 1e-8)
        Xte = (X_all[tei] - sc.mean_) / (sc.scale_ + 1e-8)
        # 从训练折尾部切 12% 做早停验证（仍按窗口切：只用来决定何时停，不报指标）
        n_va = max(1, int(len(tri) * 0.12))
        rmse, mae, ep = train_one(args.model, Xtr[:-n_va], y_all[tri][:-n_va],
            Xte, y_all[tei],
            args.epochs, args.seed, device,
            Xtr[-n_va:], y_all[tri][-n_va:])
        res.append({"fold": k, "rmse": rmse, "mae": mae, "epochs": ep})
        print(f"fold{k}: RMSE={rmse:.2f} MAE={mae:.2f} ep={ep} ({time.time()-t0:.0f}s)", flush=True)
    rmses = [r["rmse"] for r in res]
    print(f"SUMMARY [{args.model}] {args.folds}fold RMSE {np.mean(rmses):.2f}+-{np.std(rmses):.2f}", flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/{args.model}_f{args.folds}_s{args.seed}.json", "w") as f:
        json.dump({"model": args.model, "dataset": args.dataset, "folds": res,
                   "rmse_mean": float(np.mean(rmses)), "rmse_std": float(np.std(rmses)),
                   "window": WINDOW, "seed": args.seed}, f, indent=1)

if __name__ == "__main__":
    main()
