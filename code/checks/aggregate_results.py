# -*- coding: utf-8 -*-
"""聚合脚本：从 results/ 下的逐种子原始文件重新生成论文引用的全部多种子汇总文件。

背景（2026-10）：本仓库此前发布了 10 个汇总 JSON/CSV，但没有任何脚本生成它们，
其中 3 个还存在与逐种子文件对不上的问题（transfer_multiseed_v2.json 的 TCN 行、
ablation_multiseed_v5.json 的 base7 行、lobo_final_multiseed.json 中 LSTM /
Transformer / 集成的"各 2 次重复"聚合）。本脚本补齐聚合层，逐种子文件是唯一输入：

  baselines/baseline_3seed_final.json   <- {model}_f5_s42-44.json（5 折组 CV）
  baselines/lobo_final_multiseed.json   <- lobo_{tcn_s42,tcn_s43,tcn_s44}.jsonl（3 次重复）
                                           + lobo_tcn/lstm/transformer.jsonl（原运行）
  baselines/lobo_3model_final.csv       <- 上述三个原运行 jsonl 按电芯对齐
  transfer/transfer_multiseed_v2.json   <- t3b_{tcn,lstm}_s42-46.json（表 3）
  transfer/raw_protocol_multiseed.json  <- t3_soh_tcn_s42-46.json（表 4 第 1 行口径）
  ablation/ablation_multiseed.json      <- t3c_base7_tcn_s42-46.json（v3 数据版本）
  ablation/ablation_multiseed_v5.json   <- t3d_{base7,curve14}_tcn_s42-46.json（v5）
  ablation/ablation_v3c_multiseed.json  <- t3e_{base7,curve14}_tcn_s42-46.json（v3c，论文采用）
  conformal/conformal_multiseed_summary.json <- t4_{tcn,lstm}_s42-46.json（表 5）
  conformal/t4c_multiseed_summary.json  <- t4c_{mondrian,weighted}_s42-46.json（4.5 扩展）

口径说明：
  * 全部 std 为样本标准差（ddof=1），与论文表题一致。
    （旧版 baseline_3seed_final.json 的 rmse_std 误用总体标准差 ddof=0，已订正。）
  * lobo_final_multiseed.json 中 lstm / transformer / ensemble 的"各 2 次重复"
    聚合无逐折原始文件支撑，已从汇总中撤下，只保留有 jsonl 支撑的原运行与
    TCN 的 3 次重复（对应论文正文相应复现表述的修订）。
  * t4c_mondrian_*.json 为 2026-10 之前的原始运行（该脚本无缺陷，未重跑）；
    t4c_weighted_*.json 为修复测试标签泄漏与加权分位数 off-by-one 后的重跑结果。

运行：python code/checks/aggregate_results.py（在仓库根目录执行）
"""
import csv
import json
import glob
import os
import numpy as np

import argparse

R = "results"  # 可由 --results 覆盖（cx2 模式指向 results_cx2）


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    print("wrote", path)


def ms(vals):
    v = np.asarray(vals, dtype=float)
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1))}


def agg_t3b():
    """表 3：逐数据集标准化协议，t3b_{model}_s{seed}.json。"""
    out = {}
    for model in ("tcn", "lstm"):
        for ds in ("CALCE", "NASA"):
            zero, ft, to, gain = [], [], [], []
            for s in range(42, 47):
                t = load(f"{R}/transfer/t3b_{model}_s{s}.json")["targets"][ds]
                zero.append(t["zero_shot"]["rmse"])
                ft.append(t["fine_tune"]["rmse"])
                to.append(t["target_only"]["rmse"])
                gain.append(t["target_only"]["rmse"] / t["fine_tune"]["rmse"])
            out[f"{ds}_{model}"] = {"zero": ms(zero), "ft": ms(ft),
                                    "to": ms(to), "gain": ms(gain)}
    dump(f"{R}/transfer/transfer_multiseed_v2.json", out)


def agg_t3soh():
    """表 4 第 1 行口径：原始协议（源域统计量标准化），t3_soh_tcn_s{seed}.json。"""
    out = {}
    for ds in ("CALCE", "NASA"):
        zero, ft = [], []
        for s in range(42, 47):
            t = load(f"{R}/transfer/t3_soh_tcn_s{s}.json")["targets"][ds]
            zero.append(t["zero_shot"]["rmse"])
            ft.append(t["fine_tune"]["rmse"])
        out[ds] = {"n_seeds": 5, "seeds": [42, 43, 44, 45, 46],
                   "zero_shot": ms(zero), "fine_tune": ms(ft)}
    dump(f"{R}/transfer/raw_protocol_multiseed.json", out)


def agg_t3x(dirname, stem, out_name, variants):
    """消融三兄弟（t3c/t3d/t3e）共用结构：{targets ds} x {variants 集}。"""
    out = {}
    for variant in variants:
        for ds in ("CALCE", "NASA"):
            zero, ft, to = [], [], []
            for s in range(42, 47):
                p = f"{R}/{dirname}/{stem}_{variant}_tcn_s{s}.json"
                t = load(p)["targets"][ds]
                zero.append(t["zero_shot"]["rmse"])
                ft.append(t["fine_tune"]["rmse"])
                to.append(t["target_only"]["rmse"])
            out[f"{ds}_{variant}"] = {"n": 5, "zero": ms(zero), "ft": ms(ft), "to": ms(to)}
    dump(f"{R}/{dirname}/{out_name}", out)


def agg_f5():
    """表 1：5 折组 CV，3 种子。std 为种子间样本标准差（ddof=1）。"""
    out = {}
    for model in ("lstm", "gru", "tcn", "transformer"):
        seed_rmse, seed_mae = [], []
        for s in range(42, 45):
            d = load(f"{R}/baselines/{model}_f5_s{s}.json")
            folds = d["folds"]
            # 每种子的 RMSE = 5 折 RMSE 的算术平均（与旧版汇总及论文表 1 口径一致）
            seed_rmse.append(np.mean([f["rmse"] for f in folds]))
            seed_mae.append(np.mean([f["mae"] for f in folds]))
        seed_rmse, seed_mae = np.asarray(seed_rmse), np.asarray(seed_mae)
        out[model] = {"rmse_mean": float(seed_rmse.mean()),
                      "rmse_std": float(seed_rmse.std(ddof=1)),
                      "seeds": {str(s): float(v) for s, v in zip((42, 43, 44), seed_rmse)},
                      "mae_mean": float(seed_mae.mean())}
    dump(f"{R}/baselines/baseline_3seed_final.json", out)


def lobo_stats(recs):
    """一次 LOBO 运行（jsonl）的逐电芯 RMSE 分布统计，分位数用 lower 插值口径。"""
    r = np.sort(np.asarray([x["rmse"] for x in recs]))
    q = lambda p: float(np.quantile(r, p, method="lower"))
    return {"min": float(r[0]), "q1": q(0.25), "median": q(0.50), "q3": q(0.75),
            "mean": float(r.mean()), "std": float(r.std(ddof=0))}


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def agg_lobo():
    """表 2 / 图 1：TCN 3 次重复 + 三骨干原运行。lstm/transformer 的
    "各 2 次重复"聚合因逐折文件缺失而撤下（见模块 docstring）。"""
    out = {}
    run_files = sorted(glob.glob(f"{R}/baselines/lobo_tcn_s4[2-4].jsonl"))
    seeds = [int(x.split("_s")[-1].split(".")[0]) for x in run_files]
    per_run = {str(s): lobo_stats(read_jsonl(p)) for s, p in zip(seeds, run_files)}
    means = np.asarray([per_run[str(s)]["mean"] for s in seeds])
    medians = np.asarray([per_run[str(s)]["median"] for s in seeds])
    out["tcn"] = {"n_runs": len(seeds), "seeds": seeds, "per_run": per_run,
                  "cross_mean": float(means.mean()), "cross_std": float(means.std(ddof=1)),
                  "cross_median": float(medians.mean()), "median_std": float(medians.std(ddof=1))}
    for model in ("lstm", "transformer"):
        st = lobo_stats(read_jsonl(f"{R}/baselines/lobo_{model}.jsonl"))
        out[model] = {"n_runs": 1, "seeds": [42], "per_run": {"42": st},
                      "cross_mean": st["mean"], "cross_std": None,
                      "cross_median": st["median"], "median_std": None}
    dump(f"{R}/baselines/lobo_final_multiseed.json", out)


def agg_lobo_csv():
    """表 2 集成行：三骨干原运行按电芯对齐（lobo_3model_final.csv 的再生成）。"""
    models = ("tcn", "lstm", "transformer")
    by_cell = {}
    for m in models:
        for rec in read_jsonl(f"{R}/baselines/lobo_{m}.jsonl"):
            by_cell.setdefault(rec["te_cell"], {})[m] = rec["rmse"]
    rows = []
    for cell in sorted(by_cell):
        d = by_cell[cell]
        if len(d) == len(models):
            rows.append({"te_cell": cell, "rmse_tcn": d["tcn"],
                         "rmse_lstm": d["lstm"], "rmse_tf": d["transformer"]})
    p = f"{R}/baselines/lobo_3model_final.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["te_cell", "rmse_tcn", "rmse_lstm", "rmse_tf"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", p)
    # 集成（逐电芯取三模型最优，需测试真值，为不可部署的上界参照）
    ens = float(np.mean([min(r["rmse_tcn"], r["rmse_lstm"], r["rmse_tf"]) for r in rows]))
    print(f"逐电芯三模型择优（oracle 上界）均值 = {ens:.2f}")


def agg_conformal():
    """表 5：t4_{model}_s{seed}.json（2026-10 重跑：校准与评估同源）。"""
    out = {}
    for model in ("tcn", "lstm"):
        for ds in ("CALCE", "NASA"):
            pt, ps, wt, ws, rm = [], [], [], [], []
            for s in range(42, 47):
                t = load(f"{R}/conformal/t4_{model}_s{s}.json")["targets"][ds]
                pt.append(t["target_calibrated"]["PICP"])
                ps.append(t["source_calibrated"]["PICP"])
                wt.append(t["target_calibrated"]["MPIW"])
                ws.append(t["source_calibrated"]["MPIW"])
                rm.append(t["point_rmse"])
            out[f"{ds}_{model}"] = {
                "seeds": [42, 43, 44, 45, 46],
                "picp_tgt": {**ms(pt), "all": [float(v) for v in pt]},
                "picp_src": {**ms(ps), "all": [float(v) for v in ps]},
                "mpiw_tgt": ms(wt), "mpiw_src": ms(ws), "rmse": ms(rm)}
    dump(f"{R}/conformal/conformal_multiseed_summary.json", out)


def agg_t4c():
    """4.5 扩展：Mondrian（原始运行）+ Weighted（修复后重跑）。"""
    out = {}
    for variant in ("mondrian", "weighted"):
        out[variant] = {}
        for ds in ("CALCE", "NASA"):
            single_p, single_w, var_p, var_w, cond_p, cond_w = [], [], [], [], [], []
            for s in range(42, 47):
                t = load(f"{R}/conformal/t4c_{variant}_s{s}.json")["targets"][ds]
                single_p.append(t["single"]["PICP"]); single_w.append(t["single"]["MPIW"])
                var_p.append(t[variant]["PICP"]); var_w.append(t[variant]["MPIW"])
                cond_p.append(t[variant]["PICP"]); cond_w.append(t[variant]["MPIW"])
            out[variant][ds] = {
                "n_seeds": 5, "seeds": [42, 43, 44, 45, 46],
                "single": {"PICP": ms(single_p), "MPIW": float(np.mean(single_w))},
                variant: {"PICP": ms(var_p), "MPIW": float(np.mean(var_w))},
                "point_rmse": float(np.mean([
                    load(f"{R}/conformal/t4c_{variant}_s{s}.json")["targets"][ds]["point_rmse"]
                    for s in range(42, 47)])),
                "cond_picp_all": [float(v) for v in cond_p],
                "cond_mpiw_all": [float(v) for v in cond_w]}
    dump(f"{R}/conformal/t4c_multiseed_summary.json", out)


def agg_sweep():
    """I7 覆盖率-校准电芯数扫描汇总（cx2）。"""
    out = {}
    for model in ("tcn", "lstm"):
        for nc in (1, 2, 3, 4, 5):
            files = sorted(glob.glob(f"{R}/conformal/t4_split_sweep/t4_{model}_s4[2-6]_cal{nc}.json"))
            for tgt in ("CALCE",):
                pt, pw, ps = [], [], []
                for f in files:
                    d = load(f)
                    if tgt not in d["targets"]:
                        continue
                    t = d["targets"][tgt]
                    pt.append(t["target_calibrated"]["PICP"])
                    pw.append(t["target_calibrated"]["MPIW"])
                    ps.append(t["source_calibrated"]["PICP"])
                if not pt:
                    continue
                v = np.asarray(pt)
                out.setdefault(model, {})[f"cal{nc}"] = {
                    "split_ft_cal_te": [16 - 7 - nc, nc, 7],
                    "picp_tgt": {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                                 "all": [float(x) for x in pt]},
                    "mpiw_tgt_mean": float(np.mean(pw)),
                    "picp_src_mean": float(np.mean(ps))}
    dump(f"{R}/conformal/t4_split_sweep/sweep_summary.json", out)


def agg_i9():
    """I9 统计效力：50 次划分重抽的抽样分布（cx2）。"""
    out = {}
    for model in ("tcn", "lstm"):
        pt, pw, rm = [], [], []
        for s in range(1001, 1051):
            pth = f"{R}/conformal/i9_seeds/t4_{model}_s{s}.json"
            if not os.path.exists(pth):
                continue
            d = load(pth)
            t = d["targets"]["CALCE"]
            pt.append(t["target_calibrated"]["PICP"])
            pw.append(t["target_calibrated"]["MPIW"])
            rm.append(t["point_rmse"])
        v = np.asarray(pt)
        q = lambda p: float(np.quantile(v, p))
        out[model] = {"n": len(v),
                      "picp_tgt": {"mean": float(v.mean()), "std": float(v.std(ddof=1)),
                                   "q05": q(0.05), "q50": q(0.50), "q95": q(0.95),
                                   "min": float(v.min()), "max": float(v.max()),
                                   "all": [float(x) for x in v]},
                      "frac_ge_090": float(np.mean(v >= 0.90)),
                      "mpiw_mean": float(np.mean(pw)), "rmse_mean": float(np.mean(rm))}
    dump(f"{R}/conformal/i9_seeds/i9_summary.json", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results", help="结果根目录（cx2 模式传 results_cx2）")
    args = ap.parse_args()
    R = args.results
    agg_t3b()
    agg_t3soh()
    agg_t3x("ablation", "t3c", "ablation_multiseed.json", ["base7"])
    agg_t3x("ablation", "t3d", "ablation_multiseed_v5.json", ["base7", "curve14"])
    agg_t3x("ablation", "t3e", "ablation_v3c_multiseed.json", ["base7", "curve14"])
    agg_f5()
    agg_lobo()
    agg_lobo_csv()
    agg_conformal()
    agg_t4c()
    if R != "results":
        agg_sweep()
        agg_i9()
    print("AGGREGATE DONE")
