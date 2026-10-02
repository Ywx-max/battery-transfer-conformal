# -*- coding: utf-8 -*-
"""T4c-Weighted：协变量偏移下的加权保形（Tibshirani et al., 2019；论文 4.5 第二个扩展对照）。

思路：如果目标域里校准块的误差规律和测试块不一样，校准样本就不该一视同仁。
按"这块校准数据有多像测试块"加权，再加权算分位数。密度比 w(x)=p_test(x)/p_cal(x)
在**目标域内部的校准块与测试块之间**估计（不是源域 vs 目标域），x 取块均值的
模型预测 SOH——部署时测试块只有预测值、没有真值，权重只能用预测侧协变量
（2026-10 修订：旧版误用测试真值 SOH 估密度比，构成标签泄漏，已改）。

实测（多种子）：NASA 上校准/测试块的预测 SOH 分布几乎重合，权重归一化后退化为
均匀权重，结果与单一分位数几乎一样；CALCE 上宽度收窄但覆盖率也下降。和
Mondrian 一样是小校准集上以覆盖换宽度的负结果。为什么收窄这么难，见论文 4.8 的讨论。

运行：python t4c_weighted_local.py --seed 42（可选 --src-cache <dir> 复用源模型缓存）
输出：results/conformal/t4c_weighted_s<seed>.json"""
import argparse, json, os, random, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
OUT = "results/conformal"
FEATS=['capacity_Ah','soh','discharge_dur_s','v_mean_V','v_min_V','ica_peak','ica_peak_V']
W=20; DATA="data/建模表_v3.csv"; ALPHA=0.10; NB=3

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def build_windows(df, W=20):
    cells={}
    for bid,g in df.sort_values('cycle').groupby('battery_id'):
        f=g[FEATS].astype(float).copy()
        f=f.interpolate(limit_direction='both').fillna(0.0).values
        cyc=g['cycle'].values; y=g['soh'].astype(float).values
        ok=~np.isnan(y); f,cyc,y=f[ok],cyc[ok],y[ok]
        X,yy=[],[]
        for i in range(W-1,len(f)):
            if cyc[i]-cyc[i-W+1]==W-1:
                X.append(f[i-W+1:i+1]); yy.append(y[i])
        if X: cells[bid]=(np.asarray(X,np.float32),np.asarray(yy,np.float32))
    return cells

class CB(nn.Module):
    def __init__(s,ic,oc,k,d):
        super().__init__(); s.p=(k-1)*d
        s.conv=nn.Conv1d(ic,oc,k,dilation=d,padding=s.p)
        s.res=nn.Conv1d(ic,oc,1) if ic!=oc else nn.Identity()
    def forward(s,x):
        y=s.conv(x)
        if s.p: y=y[:,:,:-s.p]
        return torch.relu(y+s.res(x))

class TCN(nn.Module):
    def __init__(s,ic,d=64,L=4):
        super().__init__(); ch=[ic]+[d]*L
        s.blocks=nn.ModuleList([CB(ch[i],ch[i+1],3,2**i) for i in range(L)])
        s.fc=nn.Linear(d,1)
    def forward(s,x):
        h=x.transpose(1,2)
        for b in s.blocks: h=b(h)
        return s.fc(h[:,:,-1]).squeeze(-1)

def fit(m,X,y,ep,s,d,Xv=None,yv=None,lr=1e-3):
    set_seed(s); opt=torch.optim.Adam(m.parameters(),lr=lr)
    lf=nn.MSELoss(); ds=torch.utils.data.TensorDataset(torch.tensor(X,dtype=torch.float32),torch.tensor(y,dtype=torch.float32))
    dl=torch.utils.data.DataLoader(ds,batch_size=256,shuffle=True)
    Xv=X if Xv is None else Xv; yv=y if yv is None else yv
    best,bst,pat=float('inf'),None,0
    for e in range(ep):
        m.train()
        for xb,yb in dl:
            xb,yb=xb.to(d),yb.to(d); opt.zero_grad(); lf(m(xb),yb).backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
        m.eval()
        with torch.no_grad(): va=lf(m(torch.tensor(Xv,dtype=torch.float32).to(d)),torch.tensor(yv,dtype=torch.float32).to(d)).item()
        if np.isfinite(va) and va<best-1e-6: best,pat=va,0; bst={k:v.clone() for k,v in m.state_dict().items()}
        else: pat+=1
        if pat>=10: break
    if bst: m.load_state_dict(bst)
    return m

def pred(m,X,d):
    m.eval()
    with torch.no_grad(): return m(torch.tensor(X,dtype=torch.float32).to(d)).cpu().numpy()

def sw(sc,X): return ((X-sc.mean_)/(sc.scale_+1e-8)).astype(np.float32)
def cc(cells,bids):
    X=np.concatenate([cells[b][0] for b in bids]); y=np.concatenate([cells[b][1] for b in bids])
    return X,y
def cq(res,a):
    # 单一分位数基线（与 t4_conformal.conformal_q 同式），给 weighted 当对照
    n=len(res); idx=min(n-1,int(np.ceil((n+1)*(1-a)))-1)
    return float(np.sort(res)[idx])
def chunk_mean(arr, w=20):
    n=len(arr)//w
    return np.array([np.mean(arr[i*w:(i+1)*w]) for i in range(n)])
def gaussian_w(cal_soh, te_soh):
    # 密度比 w(x) = p_target(x) / p_cal(x)，x 取块均值 SOH，两个分布各用
    # 高斯近似。末尾除以均值做归一化：让权重只表达"相对重要性"，量级稳定
    # （+1e-8 防 std=0；NASA 校准块少、SOH 几乎恒定时真的会触发）
    mu_s,std_s=np.mean(cal_soh),np.std(cal_soh)+1e-8
    mu_t,std_t=np.mean(te_soh),np.std(te_soh)+1e-8
    w=(std_s/std_t)*np.exp(-0.5*((cal_soh-mu_t)/std_t)**2+0.5*((cal_soh-mu_s)/std_s)**2)
    return w/np.mean(w)
def weighted_cq(res,wts,a):
    # 加权分位数：按残差升序累加权重，取累计权重首次达到 (1-alpha)·(n+1)/n·总权重
    # 处的残差。均匀权重下该式退化为 ceil((n+1)(1-alpha)) 次序统计量，与 cq() 同口径
    # （2026-10 修订：旧版阈值漏乘 (n+1)/n，等价于普通 (1-alpha) 分位数，
    # 均匀权重下比标准保形分位数小一档，区间系统性偏窄）
    order=np.argsort(res); sr=res[order]; sw2=wts[order]
    cw=np.cumsum(sw2); tw=cw[-1]
    n=len(res)
    threshold=(1-a)*(n+1)/n*tw
    idx=np.searchsorted(cw,threshold)
    idx=min(idx,len(sr)-1)
    return float(sr[idx])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--src-cache", default=None,
                    help="含 t4d_src_<model>_s<seed>.pt 的目录；提供则跳过源域预训练")
    args = ap.parse_args()
    SEED = args.seed
    device='cuda' if torch.cuda.is_available() else 'cpu'
    df=pd.read_csv(DATA)
    src=build_windows(df[df['dataset']=='MIT'])
    sb=sorted(src); random.Random(SEED).shuffle(sb)
    nv=max(1,int(len(sb)*0.1)); Xtr,ytr=cc(src,sb[:-nv]); Xva,yva=cc(src,sb[-nv:])
    sc=StandardScaler().fit(Xtr.reshape(-1,Xtr.shape[2]))
    model=TCN(Xtr.shape[2]).to(device)
    cache_p = os.path.join(args.src_cache, "t4d_src_tcn_s%d.pt" % SEED) if args.src_cache else None
    if cache_p and os.path.exists(cache_p):
        model.load_state_dict(torch.load(cache_p, map_location=device, weights_only=True))
        print('[cache] load source model %s' % cache_p, flush=True)
    else:
        t0=time.time()
        model=fit(model,sw(sc,Xtr),ytr,120,SEED,device,sw(sc,Xva),yva)
        print('SRC done %ds' % (time.time()-t0),flush=True)
    results={'targets':{}}
    for tg in ['CALCE','NASA']:
        tgt=build_windows(df[df['dataset']==tg])
        tbg=sorted(tgt); random.Random(SEED).shuffle(tbg)
        # 与 t4c_mondrian 完全相同的三分协议（CALCE 2/2/4、NASA 1/1/2），
        # 两个扩展对照之间才可比
        nf=max(1,len(tbg)//3)
        ft_b,cal_b,te_b=tbg[:nf],tbg[nf:nf*2],tbg[nf*2:]
        Xa,ya=cc(tgt,tbg); sct=StandardScaler().fit(Xa.reshape(-1,Xa.shape[2]))
        Xft,yft=cc(tgt,ft_b)
        ftm=TCN(Xtr.shape[2]).to(device); ftm.load_state_dict(model.state_dict())
        ftm=fit(ftm,sw(sct,Xft),yft,60,SEED,device,lr=3e-4)
        Xcal,ycal=cc(tgt,cal_b); pcal=pred(ftm,sw(sct,Xcal),device)
        cres=np.abs(pcal-ycal)
        cres_c=chunk_mean(cres,W); n_c=len(cres_c)
        cal_soh_c=np.array([np.mean(ycal[i*W:(i+1)*W]) for i in range(n_c)])
        q_single=cq(cres_c,ALPHA)
        Xte,yte=cc(tgt,te_b); pte=pred(ftm,sw(sct,Xte),device)
        te_res=np.abs(pte-yte)
        te_res_c=chunk_mean(te_res,W); n_t=len(te_res_c)
        # 权重用的测试侧"SOH"取模型预测的块均值（部署时可得），不用测试真值：
        # 密度比在目标域内部的校准块与测试块之间估计，属于直推式设定
        te_pred_c=chunk_mean(pte,W)
        picp_single=float(np.mean(te_res_c<=q_single)); mpiw_single=2*q_single
        # 权重只喂给校准块；测试块不需要权重（覆盖评估就是普通的"落在区间内吗"）
        w=gaussian_w(cal_soh_c, te_pred_c)
        q_weighted=weighted_cq(cres_c, w, ALPHA)
        picp_weighted=float(np.mean(te_res_c<=q_weighted)); mpiw_weighted=2*q_weighted
        rmse=float(np.sqrt(np.mean((pte-yte)**2)))
        results['targets'][tg]={
            'single':{'PICP':picp_single,'MPIW':mpiw_single,'q':q_single},
            'weighted':{'PICP':picp_weighted,'MPIW':mpiw_weighted,'q':q_weighted},
            'point_rmse':rmse}
        print(f'[{tg}] RMSE={rmse:.4f} single_PICP={picp_single:.2f} MPIW={mpiw_single:.4f} weighted_PICP={picp_weighted:.2f} MPIW={mpiw_weighted:.4f}',flush=True)
    os.makedirs(OUT,exist_ok=True)
    json.dump(results,open(OUT+f'/t4c_weighted_s{SEED}.json','w'),indent=1)
    print('T4C DONE',flush=True)
if __name__=='__main__': main()
