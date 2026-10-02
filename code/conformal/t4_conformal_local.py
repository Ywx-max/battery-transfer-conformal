# -*- coding: utf-8 -*-
"""T4 保形区间双路由对照。论文最核心的一张表（表 5）出自这里。

同一颗微调后的模型、同一批测试窗口，只换校准分位数的来源：
  源域校准（朴素）  沿用源域验证残差算分位数 q_src，预期覆盖崩塌
  目标域校准（本文）用 1~2 颗目标电芯的循环级残差算 q_tgt，预期覆盖大幅回升

口径（2026-10 修订）：q_tgt 一律用**微调后（部署）模型**在校准电芯上的残差计算。
拆分保形的有限样本覆盖保证要求校准分数与测试分数出自同一个预测函数，因此
"用哪个模型评估，就用哪个模型校准"。旧版实现用微调前的源模型残差算 q_tgt、
却在微调后模型上评估覆盖率，分位数被系统性放大 10 倍以上，机械地把覆盖率
抬到 ≈1.00，该版本结果作废（旧 t4_*.json 已被本版重跑结果覆盖）。
修订后的诚实结论：源域校准 20/20 组配置欠覆盖（PICP 0.03~0.58）；目标域校准
把覆盖率拉回 0.6~0.9，但仍普遍低于名义 0.90——1~2 颗电芯的校准集换不来
完整的名义覆盖，这与 t4c 系列负结果一致。

口径提醒（论文 3.4 末有声明）：这里报告的是边际经验覆盖率。循环级残差存在
自相关，严格可交换性不满足，所以是经验证据而非有限样本保证。
逐电芯诊断与按电芯聚合变体见 t4d_per_cell_diag.py。

划分协议：CALCE (3 微调, 2 校准, 3 测试)、NASA (2, 1, 1)，划分随种子重新抽取。
运行：python t4_conformal_local.py --model tcn --seed 42（论文口径：42~46 各一次）
可选：--src-cache <dir> 指向含 t4d_src_<model>_s<seed>.pt 的目录，跳过源域预训练
      （缓存模型与 t4d 诊断共用，协议同本脚本；不提供则现场训练）。
输出：results/conformal/t4_<model>_s<seed>.json（含逐电芯残差向量）"""
import argparse, json, os, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

FEATS = ["capacity_Ah", "soh", "discharge_dur_s", "v_mean_V", "v_min_V",
         "ica_peak", "ica_peak_V"]
WINDOW = 20
# 建模表路径（数据放 data/ 下即可，合并方法见 code/README.md）
DATA = "data/建模表_v3.csv"
ALPHA = 0.10

def set_seed(seed):
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

def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy()

def std_with(sc, X):
    return ((X - sc.mean_) / (sc.scale_ + 1e-8)).astype(np.float32)

def concat_cells(cells, bids):
    X = np.concatenate([cells[b][0] for b in bids])
    y = np.concatenate([cells[b][1] for b in bids])
    return X, y, [b for b in bids for _ in range(len(cells[b][0]))]

def conformal_q(residuals, alpha=ALPHA):
    """split conformal 的经验分位数：取第 ceil((n+1)(1-alpha)) 个次序统计量。

    min(n-1, ...) 是工程兜底：n 很小（比如 NASA 校准集只有 1 颗电芯、电芯级
    得分的 n 只有 1~2）时，(n+1)(1-alpha) 会越过 n，此时索引被压到最大值，
    等价于取最差的校准样本。要清醒：这种情况下保形的有限样本保证本来就是
    空的，n 太小时区间只有经验意义。"""
    n = len(residuals)
    idx = min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)
    return float(np.sort(residuals)[idx])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tcn")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--ft-epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results/conformal")
    ap.add_argument("--src-cache", default=None,
                    help="含 t4d_src_<model>_s<seed>.pt 的目录；提供则跳过源域预训练")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"T4 conformal device={device} model={args.model} alpha={ALPHA}", flush=True)
    df = pd.read_csv(DATA)
    src = build_windows_ds(df, "MIT")
    src_bids = sorted(src)
    random.Random(args.seed).shuffle(src_bids)
    n_va = max(1, int(len(src_bids) * 0.1))
    Xtr, ytr, _ = concat_cells(src, src_bids[:-n_va])
    Xva, yva, _ = concat_cells(src, src_bids[-n_va:])
    sc_src = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[2]))
    Xtr, Xva = std_with(sc_src, Xtr), std_with(sc_src, Xva)
    model = new_model(args.model, Xtr.shape[2]).to(device)
    cache_p = None
    if args.src_cache:
        cache_p = os.path.join(args.src_cache, "t4d_src_%s_s%d.pt" % (args.model, args.seed))
        if os.path.exists(cache_p):
            model.load_state_dict(torch.load(cache_p, map_location=device, weights_only=True))
            print(f"[cache] load source model {cache_p}", flush=True)
        else:
            print(f"[cache] {cache_p} 不存在，改为现场训练", flush=True)
            cache_p = None
    if cache_p is None:
        t0 = time.time()
        model = fit_model(model, Xtr, ytr, args.epochs, args.seed, device, Xva, yva)
        print(f"源域训练完成 {time.time()-t0:.0f}s", flush=True)
    # 源域校准残差 (朴素路由用)
    # 朴素路由的分位数：源域验证集上的绝对残差。源域内模型拟合近乎完美，
    # q_src 非常小，这正是它到了目标域全面失守的伏笔
    src_cal_res = np.abs(predict(model, Xva, device) - yva)
    q_src = conformal_q(src_cal_res)
    print(f"源域校准分位数 q_src={q_src:.4f}", flush=True)

    results = {"model": args.model, "alpha": ALPHA, "q_src": q_src, "targets": {}}
    for tgt_name, split in [("CALCE", (3, 2, 3)), ("NASA", (2, 1, 1))]:
        tgt = build_windows_ds(df, tgt_name)
        tb = sorted(tgt)
        n_ft, n_cal, n_te = split
        if len(tb) < n_ft + n_cal + n_te:
            print(f"[{tgt_name}] 电芯不足, 跳过", flush=True)
            continue
        random.Random(args.seed).shuffle(tb)
        ft_b, cal_b, te_b = tb[:n_ft], tb[n_ft:n_ft+n_cal], tb[n_ft+n_cal:]
        Xall_t, _, _ = concat_cells(tgt, tb)
        sc_tgt = StandardScaler().fit(Xall_t.reshape(-1, Xall_t.shape[2]))
        Xft, yft, _ = concat_cells(tgt, ft_b)
        ft_model = new_model(args.model, Xtr.shape[2]).to(device)
        ft_model.load_state_dict(model.state_dict())
        ft_model = fit_model(ft_model, std_with(sc_tgt, Xft), yft,
                             args.ft_epochs, args.seed, device, lr=3e-4)
        Xcal, ycal, _ = concat_cells(tgt, cal_b)
        Xcal = std_with(sc_tgt, Xcal)
        # 目标域校准残差：用微调后的部署模型在校准电芯上取残差（校准与评估同源）。
        # 校准集 ft_b 与 cal_b 互斥，微调模型没见过校准电芯，不存在"过于熟悉"的问题
        cal_res = np.abs(predict(ft_model, Xcal, device) - ycal)
        q_tgt = conformal_q(cal_res)
        Xte, yte, _ = concat_cells(tgt, te_b)
        Xte_s = std_with(sc_tgt, Xte)
        pred = predict(ft_model, Xte_s, device)
        res_te = np.abs(pred - yte)
        # 目标域校准路由 (本方法)
        # 覆盖率按"汇集的所有测试窗口"计（边际口径）；MPIW = 区间平均全宽 = 2q
        cov_tgt = float(np.mean(res_te <= q_tgt)); w_tgt = 2 * q_tgt
        # 源域校准路由 (朴素对照)
        cov_src = float(np.mean(res_te <= q_src)); w_src = 2 * q_src
        # 逐电芯残差与覆盖率：校准电芯存部署模型残差向量，测试电芯另存逐窗口
        # 覆盖判定与 RMSE，后续换诊断指标不用重训
        idx0 = 0
        per_cell = {}
        for b in te_b:
            Xb, yb, _ = concat_cells(tgt, [b])
            res_b = res_te[idx0:idx0 + len(yb)]; idx0 += len(yb)
            per_cell[b] = {"n_windows": int(len(res_b)),
                           "residuals": [float(v) for v in res_b],
                           "cov_tgt": float(np.mean(res_b <= q_tgt)),
                           "cov_src": float(np.mean(res_b <= q_src)),
                           "rmse": float(np.sqrt(np.mean(res_b ** 2)))}
        cal_cells_res = {}
        off = 0
        for b in cal_b:
            Xb, yb, _ = concat_cells(tgt, [b])
            cal_cells_res[b] = [float(v) for v in cal_res[off:off + len(yb)]]
            off += len(yb)
        results["targets"][tgt_name] = {
            "split": {"ft": ft_b, "cal": cal_b, "te": te_b},
            "q_target": q_tgt, "q_src": q_src,
            "point_rmse": float(np.sqrt(np.mean((pred - yte) ** 2))),
            "target_calibrated": {"PICP": cov_tgt, "MPIW": w_tgt},
            "source_calibrated": {"PICP": cov_src, "MPIW": w_src},
            "per_cell": per_cell,
            "cal_cells_residuals": cal_cells_res}
        print(f"[{tgt_name}] 点预测RMSE={results['targets'][tgt_name]['point_rmse']:.4f} | "
              f"目标校准: PICP={cov_tgt:.2f} MPIW={w_tgt:.4f} | "
              f"源校准: PICP={cov_src:.2f} MPIW={w_src:.4f} (名义覆盖 {1-ALPHA:.2f})", flush=True)
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/t4_{args.model}_s{args.seed}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print("T4 DONE", flush=True)

if __name__ == "__main__":
    main()
