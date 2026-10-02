# -*- coding: utf-8 -*-
"""LOBO 留一电芯全量验证（MIT 124 颗电芯 = 119 折；b1c0~b1c4 共 5 颗无 EOL 标签，
不参与 RUL 评估，并非窗口不足被过滤——124 颗均有足够窗口）。

比 5 折组交叉验证细一层的评估：每次留一整颗电芯做测试，其余 123 颗训练，
给出逐电芯误差分布而不是单个均值。论文表 2 的分布列、图 1 的直方图都出自这里。

支持断点续跑：每折结果立即追加进 jsonl 一行，中断后重启会自动跳过已完成折。
119 折在本机 4060 上约 2~3 小时/模型，不值得因为断电重跑。

运行：python t2_lobo_local.py --model tcn --seed 42
输出：results/baselines/lobo_<model>_s<seed>.jsonl（每折一行：te_cell/rmse/mae/epochs）"""
import argparse, json, os, time
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
# 模型、滑窗、训练循环全部复用 t2_train_local，本脚本只负责"怎么切折"和"怎么断点续跑"
from t2_train_local import build_windows, new_model, train_one, DATA

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="tcn", choices=["tcn", "lstm", "gru", "transformer"])
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--out", default="results/baselines")
args = ap.parse_args()

OUT = args.out
os.makedirs(OUT, exist_ok=True)
JSONL = f"{OUT}/lobo_{args.model}_s{args.seed}.jsonl"
EPOCHS = args.epochs
MODEL = args.model

df = pd.read_csv(DATA)
df = df[df["dataset"] == "MIT"]
cells = build_windows(df)
bids = sorted(cells)
X_all = np.concatenate([cells[b][0] for b in bids])
y_all = np.concatenate([cells[b][1] for b in bids])
g_all = np.concatenate([[b] * len(cells[b][0]) for b in bids])

# 断点续跑：读旧 jsonl 里已完成的测试电芯，重启时跳过
done = set()
if os.path.exists(JSONL):
    with open(JSONL) as f:
        for line in f:
            try:
                done.add(json.loads(line)["te_cell"])
            except Exception:
                pass

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(OUT, exist_ok=True)
print(f"LOBO device={device} model={MODEL} seed={args.seed} 电芯={len(bids)} 已完成={len(done)} 剩余={len(bids)-len(done)}", flush=True)
t0 = time.time()
cnt = 0
for i, te_b in enumerate(bids):
    if te_b in done:
        continue
    # 留一电芯：训练集 = 其余全部电芯的窗口（同一颗电芯的窗口绝不分家）
    tri_mask = g_all != te_b
    Xtr, ytr = X_all[tri_mask], y_all[tri_mask]
    sc = StandardScaler().fit(Xtr.reshape(-1, Xtr.shape[2]))
    # 尾部 8% 做早停验证；LOBO 折数多，验证集小一点能省不少训练时间
    n_va = max(1, int(len(Xtr) * 0.08))
    Xa = ((Xtr - sc.mean_) / (sc.scale_ + 1e-8)).astype(np.float32)
    Xte = ((X_all[~tri_mask] - sc.mean_) / (sc.scale_ + 1e-8)).astype(np.float32)
    rmse, mae, ep = train_one(MODEL, Xa[:-n_va], ytr[:-n_va], Xte, y_all[~tri_mask],
                              EPOCHS, args.seed, device, Xa[-n_va:], ytr[-n_va:])
    # 每折立刻落盘（append 模式），断点续跑和中途看结果都靠它
    rec = {"te_cell": te_b, "rmse": rmse, "mae": mae, "epochs": ep, "n_test": int((~tri_mask).sum())}
    with open(JSONL, "a") as f:
        f.write(json.dumps(rec) + "\n")
    cnt += 1
    if cnt % 10 == 0 or i == len(bids) - 1:
        el = time.time() - t0
        eta = el / max(1, cnt) * (len(bids) - len(done) - cnt)
        print(f"[{i+1}/{len(bids)}] 最近 {te_b}: RMSE={rmse:.1f} | 已用 {el/60:.0f}min ETA {eta/60:.0f}min", flush=True)
print("LOBO DONE", flush=True)
