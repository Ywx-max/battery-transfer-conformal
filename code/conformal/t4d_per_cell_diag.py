# -*- coding: utf-8 -*-
"""T4d 保形覆盖的两项补充诊断（论文 4.5 末段的数据出处）。

背景：表 5 报的是"汇集所有测试窗口"的边际覆盖率。这个口径有两个疑点，
本脚本逐一诊断：

第一，边际覆盖不等于条件覆盖。边际覆盖 1.00 有没有可能是某几颗电芯撑起来的？
   把覆盖率拆到每颗测试电芯（cov_tgt / cov_src），看有没有谁被平均了。
   实测：目标校准下 CALCE 3 颗测试电芯跨 5 种子最低 0.97，没有系统性塌陷。

第二，循环级残差自相关，可交换性不成立。把校准得分聚到电芯级再走保形会怎样？
   每颗校准电芯的残差聚成中位数/p90/均值三种电芯级得分，再算分位数。
   实测：覆盖同样恢复（约 0.99~1.00）且区间更窄，但 1~2 颗校准电芯只给出
   1~2 个电芯级得分，分位数退化为最大值，保证是空的，结果只有经验意义。
   这个"能用但没资格"的边界本身就是结论（论文 4.5 末如实写了）。

复用 t4_conformal_local 的协议与随机次序：电芯划分与已存 t4_*.json 逐组
assert 校验一致，区别只是本脚本把逐电芯残差也存了下来（t4 只存汇总指标）。
源模型按 (model, seed) 缓存到 src_cache/，重跑免预训练。

用法：python t4d_per_cell_diag.py --model tcn --seed 42
输出：results/conformal/t4d_per_cell_{model}_s{seed}.json"""
import argparse, json, os, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

FEATS = ["capacity_Ah", "soh", "discharge_dur_s", "v_mean_V", "v_min_V",
         "ica_peak", "ica_peak_V"]
WINDOW = 20
DATA = "data/建模表_v3.csv"
ALPHA = 0.10
OUT = "results/conformal"
CACHE = os.path.join(OUT, "src_cache")
REF = "results/conformal"


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
    # 同 t4_conformal.conformal_q：ceil((n+1)(1-alpha)) 次序统计量，n 太小时兜底取最大
    n = len(residuals)
    idx = min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)
    return float(np.sort(residuals)[idx])


def get_source_model(model_name, seed, Xtr, ytr, Xva, yva, input_dim, device, epochs):
    """源模型按 (model, seed) 落盘缓存。诊断要跑 2 骨干 × 5 种子共 10 次预训练，
    不缓存的话每次调整诊断口径都要重练源域；缓存后增量分析只是秒级推理。"""
    os.makedirs(CACHE, exist_ok=True)
    p = os.path.join(CACHE, "t4d_src_%s_s%d.pt" % (model_name, seed))
    model = new_model(model_name, input_dim).to(device)
    if os.path.exists(p):
        model.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        print("[cache] load source model %s" % p, flush=True)
        return model
    t0 = time.time()
    model = fit_model(model, Xtr, ytr, epochs, seed, device, Xva, yva)
    torch.save(model.state_dict(), p)
    print("source pretrain done %ds -> %s" % (time.time()-t0, p), flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tcn")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--ft-epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("T4d per-cell diag device=%s model=%s seed=%d" % (device, args.model, args.seed), flush=True)
    df = pd.read_csv(DATA)

    src = build_windows_ds(df, "MIT")
    src_bids = sorted(src)
    random.Random(args.seed).shuffle(src_bids)
    n_va = max(1, int(len(src_bids) * 0.1))
    Xtr, ytr, _ = concat_cells(src, src_bids[:-n_va])
    Xva, yva, _ = concat_cells(src, src_bids[-n_va:])
    sc_src = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[2]))
    Xtr, Xva = std_with(sc_src, Xtr), std_with(sc_src, Xva)
    model = get_source_model(args.model, args.seed, Xtr, ytr, Xva, yva,
                             Xtr.shape[2], device, args.epochs)
    src_cal_res = np.abs(predict(model, Xva, device) - yva)
    q_src = conformal_q(src_cal_res)
    print("q_src=%.4f" % q_src, flush=True)

    # 一致性校验：诊断运行的电芯划分必须与论文表 5 用的 t4_*.json 完全一致，
    # 否则逐电芯结果没法和主表对上。划分由种子唯一决定，理论上必然一致，assert 兜底
    ref_p = os.path.join(REF, "t4_%s_s%d.json" % (args.model, args.seed))
    ref = json.load(open(ref_p, encoding="utf-8")) if os.path.exists(ref_p) else None
    if ref is not None and abs(ref["q_src"] - q_src) > 1e-3:
        print("[warn] q_src differs from saved t4: new %.4f vs old %.4f (GPU nondeterminism, expected)"
              % (q_src, ref["q_src"]), flush=True)

    results = {"model": args.model, "alpha": ALPHA, "seed": args.seed, "q_src": q_src,
               "targets": {}}
    for tgt_name, split in [("CALCE", (3, 2, 3)), ("NASA", (2, 1, 1))]:
        tgt = build_windows_ds(df, tgt_name)
        tb = sorted(tgt)
        n_ft, n_cal, n_te = split
        random.Random(args.seed).shuffle(tb)
        ft_b, cal_b, te_b = tb[:n_ft], tb[n_ft:n_ft+n_cal], tb[n_ft+n_cal:]
        if ref is not None:
            ref_split = ref["targets"][tgt_name]["split"]
            assert ft_b == ref_split["ft"] and cal_b == ref_split["cal"] and te_b == ref_split["te"], \
                "cell split mismatch vs saved t4!"
        Xall_t, _, _ = concat_cells(tgt, tb)
        sc_tgt = StandardScaler().fit(Xall_t.reshape(-1, Xall_t.shape[2]))
        cal_cells = {}
        for b in cal_b:
            Xc, yc, _ = concat_cells(tgt, [b])
            pred_c = predict(model, std_with(sc_tgt, Xc), device)
            cal_cells[b] = (np.abs(pred_c - yc)).tolist()
        cal_res = np.concatenate([np.asarray(v) for v in cal_cells.values()])
        q_tgt = conformal_q(cal_res)

        Xft, yft, _ = concat_cells(tgt, ft_b)
        ft_model = new_model(args.model, Xtr.shape[2]).to(device)
        ft_model.load_state_dict(model.state_dict())
        ft_model = fit_model(ft_model, std_with(sc_tgt, Xft), yft,
                             args.ft_epochs, args.seed, device, lr=3e-4)

        te_cells = {}
        for b in te_b:
            Xb, yb, _ = concat_cells(tgt, [b])
            pred_b = predict(ft_model, std_with(sc_tgt, Xb), device)
            te_cells[b] = {"pred": pred_b.tolist(), "y": yb.tolist(),
                           "res": np.abs(pred_b - yb).tolist()}

        # 逐电芯覆盖率：两条校准路由各算一遍；res 里存了逐循环残差，
        # 之后想换诊断指标不用重训
        per_cell = {}
        for b, d in te_cells.items():
            res = np.asarray(d["res"])
            per_cell[b] = {"n_windows": int(len(res)),
                           "cov_tgt": float(np.mean(res <= q_tgt)),
                           "cov_src": float(np.mean(res <= q_src)),
                           "rmse": float(np.sqrt(np.mean(res ** 2)))}
        # 按电芯聚合的保形变体：三种聚合粒度各试一遍。median 最稳
        # （对电芯内误差尾部的离群循环不敏感），p90 相当于块内 90 分位，mean 居中。
        # 注意 q_cell 的 n 只有电芯数（1~2），模块 docstring 里说的"退化"就指这里
        agg_variants = {}
        for agg_name, agg_fn in [("median", lambda r: float(np.median(r))),
                                 ("p90", lambda r: float(np.quantile(r, 0.90))),
                                 ("mean", lambda r: float(np.mean(r)))]:
            cell_scores = [agg_fn(np.asarray(v)) for v in cal_cells.values()]
            q_cell = conformal_q(np.asarray(cell_scores))
            cov_cell = float(np.mean(np.concatenate(
                [np.asarray(d["res"]) for d in te_cells.values()]) <= q_cell))
            agg_variants[agg_name] = {"q_cell": q_cell, "PICP": cov_cell,
                                      "MPIW": 2 * q_cell,
                                      "n_cal_cells": len(cell_scores)}
        te_res_all = np.concatenate([np.asarray(d["res"]) for d in te_cells.values()])
        tgt_res = {
            "split": {"ft": ft_b, "cal": cal_b, "te": te_b},
            "q_target": q_tgt, "q_src": q_src,
            "pooled_cov_tgt": float(np.mean(te_res_all <= q_tgt)),
            "pooled_cov_src": float(np.mean(te_res_all <= q_src)),
            "per_cell": per_cell,
            "cal_cells_residual_summary": {b: {"n": len(v), "median": float(np.median(v)),
                                               "p90": float(np.quantile(v, 0.9)),
                                               "max": float(np.max(v))}
                                           for b, v in cal_cells.items()},
            "cell_aggregated_conformal": agg_variants}
        results["targets"][tgt_name] = tgt_res
        print("[%s] pooled_tgt=%.3f pooled_src=%.3f | per-cell cov_tgt=%s | cell-agg PICP median=%.3f"
              % (tgt_name, tgt_res["pooled_cov_tgt"], tgt_res["pooled_cov_src"],
                 {b: round(v["cov_tgt"], 3) for b, v in per_cell.items()},
                 agg_variants["median"]["PICP"]), flush=True)

    os.makedirs(OUT, exist_ok=True)
    out_p = os.path.join(OUT, "t4d_per_cell_%s_s%d.json" % (args.model, args.seed))
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)
    print("T4d DONE -> %s" % out_p, flush=True)


if __name__ == "__main__":
    main()
