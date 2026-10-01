# -*- coding: utf-8 -*-
"""T4c-Mondrian：按 SOH 分箱的条件化保形（论文 4.5 的"扩展"对照之一）。

想法很自然：SOH 高低段的误差不一样，按 SOH 把校准窗口分成 3 箱、各箱单独算
保形分位数，区间应该更贴身。实测结论（多种子后）是个诚实的负结果：
NASA 上宽度收窄 ~21% 但覆盖率掉到 0.64±0.18；CALCE 上宽度和覆盖都没实质改善。
单次运行里偶尔看到的"Mondrian 修正过覆盖"，多种子平均后站不住。

实现要点：残差与 SOH 都先按每 W=20 个连续循环聚成一块再分箱，逐窗口的
SOH 抖动太大会把箱子搅浑；这也是标题里"修复长度对齐"的由来。

运行：python t4c_mondrian_local.py --seed 42
输出：results/conformal/t4c_mondrian_s<seed>.json（single = 单一分位数基线对照）"""
import json, os, random, time
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
import sys
SEED = int(sys.argv[sys.argv.index('--seed')+1]) if '--seed' in sys.argv else 42
OUT = "results/conformal"
FEATS=['capacity_Ah','soh','discharge_dur_s','v_mean_V','v_min_V','ica_peak','ica_peak_V']
# NB=3：按 SOH 分 3 箱（分位点 1/3、2/3 处切）。校准电芯只有 1~2 颗时每箱样本极少，
# 这是 Mondrian 在小校准集上失灵的直接原因
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
class CausalBlock(nn.Module):
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
        s.blocks=nn.ModuleList([CausalBlock(ch[i],ch[i+1],3,2**i) for i in range(L)])
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
    # 与 t4_conformal.conformal_q 同一个公式；min(n-1,...) 是 n 太小时的兜底
    n=len(res); idx=min(n-1,int(np.ceil((n+1)*(1-a)))-1)
    return float(np.sort(res)[idx])
def chunk_res(res, W=20):
    # 每 W 个连续循环取均值 = 一个"块"。尾部不足 W 的零头直接丢掉，
    # 换来校准/测试两侧块数天然对齐（早期版本的长度对齐 bug 就出在没丢零头）
    n=len(res)//W
    return np.array([np.mean(res[i*W:(i+1)*W]) for i in range(n)])
def main():
    device='cuda' if torch.cuda.is_available() else 'cpu'
    df=pd.read_csv(DATA)
    src=build_windows(df[df['dataset']=='MIT'])
    sb=sorted(src); random.Random(SEED).shuffle(sb)
    nv=max(1,int(len(sb)*0.1)); Xtr,ytr=cc(src,sb[:-nv]); Xva,yva=cc(src,sb[-nv:])
    sc=StandardScaler().fit(Xtr.reshape(-1,Xtr.shape[2]))
    model=TCN(Xtr.shape[2]).to(device)
    model=fit(model,sw(sc,Xtr),ytr,120,SEED,device,sw(sc,Xva),yva)
    print('SRC done',flush=True)
    results={'targets':{}}
    for tg in ['CALCE','NASA']:
        tgt=build_windows(df[df['dataset']==tg])
        tbg=sorted(tgt); random.Random(SEED).shuffle(tbg)
        # 三分协议：微调/校准/测试按 1/3 切（与表 5 的硬编码划分不同，论文 4.5
        # 扩展段声明的"不同电芯划分协议"就是这里）。实际切法：CALCE 8 颗 = 2/2/4，
        # NASA 4 颗 = 1/1/2；两侧共用同一个随机洗牌序列，划分随种子可复现
        nf=max(1,len(tbg)//3)
        ft_b,cal_b,te_b=tbg[:nf],tbg[nf:nf*2],tbg[nf*2:]
        Xa,ya=cc(tgt,tbg); sct=StandardScaler().fit(Xa.reshape(-1,Xa.shape[2]))
        Xft,yft=cc(tgt,ft_b)
        ftm=TCN(Xtr.shape[2]).to(device); ftm.load_state_dict(model.state_dict())
        ftm=fit(ftm,sw(sct,Xft),yft,60,SEED,device,lr=3e-4)
        Xcal,ycal=cc(tgt,cal_b); pcal=pred(ftm,sw(sct,Xcal),device)
        cres=np.abs(pcal-ycal)
        cres_c=chunk_res(cres,W); n_c=len(cres_c)
        cal_soh_c=np.array([np.mean(ycal[i*W:(i+1)*W]) for i in range(n_c)])
        q1=cq(cres_c,ALPHA)
        Xte,yte=cc(tgt,te_b); pte=pred(ftm,sw(sct,Xte),device)
        tres=np.abs(pte-yte)
        tres_c=chunk_res(tres,W); n_t=len(tres_c)
        te_soh_c=np.array([np.mean(yte[i*W:(i+1)*W]) for i in range(n_t)])
        picp_s=float(np.mean(tres_c<=q1)); mpiw_s=2*q1
        # 分箱保形主体：按块均值 SOH 切 3 箱，每箱独立算分位数
        be=np.quantile(cal_soh_c,[1/NB,2/NB])
        pl,wl=[],[]
        # cm.sum()<2 的箱子直接放弃：两三个块算出来的分位数毫无统计意义
        for b in range(NB):
            if b==0: cm=cal_soh_c<=be[0]; tm=te_soh_c<=be[0]
            elif b==NB-1: cm=cal_soh_c>be[-1]; tm=te_soh_c>be[-1]
            else: cm=(cal_soh_c>be[b-1])&(cal_soh_c<=be[b]); tm=(te_soh_c>be[b-1])&(te_soh_c<=be[b])
            if cm.sum()<2: continue
            qb=cq(cres_c[cm],ALPHA)
            if tm.sum()>0: pl.append(float(np.mean(tres_c[tm]<=qb))); wl.append(2*qb)
        picp_m=float(np.mean(pl)) if pl else float('nan')
        mpiw_m=float(np.mean(wl)) if wl else float('nan')
        rmse=float(np.sqrt(np.mean((pte-yte)**2)))
        results['targets'][tg]={
            'single':{'PICP':picp_s,'MPIW':mpiw_s},
            'mondrian':{'PICP':picp_m,'MPIW':mpiw_m},
            'point_rmse':rmse}
        print(f'[{tg}] RMSE={rmse:.4f} single_PICP={picp_s:.2f} MPIW={mpiw_s:.4f} Mondrian_PICP={picp_m:.2f} MPIW={mpiw_m:.4f}',flush=True)
    os.makedirs(OUT,exist_ok=True)
    json.dump(results,open(OUT+f'/t4c_mondrian_s{SEED}.json','w'),indent=1)
    print('T4C DONE',flush=True)
if __name__=='__main__': main()