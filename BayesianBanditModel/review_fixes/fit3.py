"""Fitter v3: implements the adversarial-review fixes.
 1 per-trap sister flag (SL pairs share mother state; NL pairs share only the channel)
 2 pipeline parity: simulated output gets the repo's per-lineage per-variable |z|>3 -> NaN filter and tau quantised to the imaging interval
 3 real per-lineage lengths
 4 channel variances of Delta and f are free parameters (no hard-coded ratios)
 5 multi-start Nelder-Mead
 7 partial first cycles (start_time==0) removed before anything is computed
 9 k estimated by instrumental variables (mother's x_tau), OLS kept for comparison
 model ladder (frozen-parameter subsets) and lineage-split cross-validation are in run_ladder()."""
import numpy as np, pandas as pd, json, os, sys, warnings
from scipy.optimize import minimize
from scipy.stats import ks_2samp, t as student_t
warnings.filterwarnings('ignore')
HERE=os.path.dirname(os.path.abspath(__file__))
DATA_DIR=os.path.abspath(os.path.join(HERE,'..','..','Datasets'))   # this file lives in BayesianBanditModel/review_fixes/
V=['fold_growth','division_ratio','added_length','generationtime','length_birth','length_final','growth_rate','efold']
LOGV={'added_length','generationtime','length_birth','length_final','growth_rate','efold'}
DATAV=V[:-1]
EXPS={'Pooled_SM':'SM','Maryam_LongTraces':'MM Maryam','MG1655_inLB_LongTraces':'MM MG1655','lambda_LB':'MM λ',
 'MC4100_25C (Tanouchi 2015)':'Tan25','MC4100_27C (Tanouchi 2015)':'Tan27','MC4100_37C (Tanouchi 2015)':'Tan37',
 '20090129_E_coli_MG1655_(CGSC_6300)_Wang2010':'Wang 0129','20090131_E_coli_MG1655_(CGSC_6300)_Wang2010':'Wang 0131',
 '20090210_E_coli_MG1655_(CGSC_6300)_Wang2010':'Wang 0210','20090512_E_coli_MG1655_(CGSC_6300)_Wang2010':'Wang 0512',
 '20090525_E_coli_MG1655_(CGSC_6300)_Wang2010':'Wang 0525','20090702_E_coli_MG1655_(CGSC_6300)_Wang2010':'Wang 0702',
 '20090922_E_coli_MG1655_lexA3_Wang2010':'lexA3 0922','20090923_E_coli_MG1655_lexA3_Wang2010':'lexA3 0923','20090930_E_coli_MG1655_lexA3_Wang2010':'lexA3 0930'}

# ---------------- data ----------------
def load(ds, raw=False):
    """Use the repo's UNFILTERED physical_units.csv, drop partial first cycles (start_time==0),
    re-index generations, then apply the repo's outlier filter ourselves (so that model and data share one pipeline)."""
    d=pd.read_csv(f'{DATA_DIR}/{ds}/ProcessedData/physical_units.csv')
    if 'trap_ID' not in d: d['trap_ID']=d.lineage_ID; d['trace']='A'; d['dataset']='MM'
    d['lin']=d.lineage_ID.astype(str)+'|'+d.trap_ID.astype(str)+'|'+d.trace
    d=d[d.start_time>0].copy()                                    # fix 7: cycle already underway when filming began
    d['generation']=d.groupby('lin').cumcount()
    d.loc[d.generation==0,'division_ratio']=np.nan                 # first complete cycle has no observed mother
    d['efold']=np.exp(d.fold_growth)
    if not raw: d=apply_filter(d)
    return d

def apply_filter(d):
    'repo filter: per lineage, per variable, |z|>3 -> NaN'
    z=d.groupby('lin')[DATAV].transform(lambda s:(s-s.mean())/s.std())
    for v in DATAV: d.loc[z[v].abs()>3, v]=np.nan
    d['efold']=np.exp(d.fold_growth); return d

def step_size(ds):
    'imaging interval from the tau resolution of the raw processed file'
    t=np.sort(pd.read_csv(f'{DATA_DIR}/{ds}/ProcessedData/physical_units.csv').generationtime.dropna().unique()); dt=np.diff(t)
    dt=dt[dt>1e-4]; return float(np.round(np.min(dt),4))

def to_matrix(d):
    lins=d.lin.unique(); idx={l:i for i,l in enumerate(lins)}
    G=int(d.groupby('lin').size().max()); M={v:np.full((len(lins),G),np.nan) for v in V}
    li=d.lin.map(idx).values; gi=d.generation.values.astype(int)
    for v in V: M[v][li,gi]=d[v].values
    meta=d.groupby('lin').first()[['trap_ID','trace','dataset']].loc[lins]
    L=d.groupby('lin').size().loc[lins].values                      # fix 3: real lineage lengths
    sis=(meta.dataset.values=='SL')                                  # fix 1: per-lineage sister flag
    return M, meta.trap_ID.values, meta.trace.values, L, sis

def filter_matrix(M):
    'the same |z|>3 filter applied to a simulated (n_lin x G) matrix set'
    out={}
    for v in DATAV:
        X=M[v]; mu=np.nanmean(X,axis=1,keepdims=True); sd=np.nanstd(X,axis=1,keepdims=True); sd[sd==0]=np.inf
        out[v]=np.where(np.abs((X-mu)/sd)>3, np.nan, X)
    out['efold']=np.exp(out['fold_growth']); return out

def measured_inputs(d):
    'mu_a, Delta_bar, <f>, x0_bar, k (OLS and IV)'
    tc=d.copy(); tc[DATAV]=d.groupby('lin')[DATAV].transform(lambda s:s-s.mean())
    mxt=tc.groupby('lin').length_final.shift(1)                    # mother's final length (independent measurement)
    m=tc.added_length.notna()&tc.length_birth.notna(); k_ols=float(np.polyfit(tc.length_birth[m],tc.added_length[m],1)[0])
    m2=m&mxt.notna(); k_iv=float(np.cov(tc.added_length[m2],mxt[m2])[0,1]/np.cov(tc.length_birth[m2],mxt[m2])[0,1])   # fix 9
    return dict(mu_a=float(np.exp(np.nanmean(np.log(d.growth_rate)))), mu_d=float(np.nanmean(d.added_length)), f0=float(np.nanmean(d.division_ratio)),
                x0bar=float(np.nanmean(d.length_birth)), k=k_iv, k_ols=k_ols)

# ---------------- statistics ----------------
QS=[0.05,0.25,0.5,0.75,0.95]
def zref(M):
    ref={}
    for v in V:
        x=M[v][~np.isnan(M[v])]; x=np.log(x[x>0]) if v in LOGV else x; ref[v]=(x.mean(),x.std())
    return ref

def stats(M, trap, trace, sis, ref):
    out={}; tc={}
    for v in V:
        X=M[v]; tc[v]=X-np.nanmean(X,axis=1,keepdims=True)
        out[f'cv_{v}']=np.nanstd(X)/np.nanmean(X); out[f'fracL_{v}']=1-np.nanvar(tc[v])/np.nanvar(X)
        for k in (1,2,3,5,10):
            a=tc[v][:,:-k]; b=tc[v][:,k:]; m=~np.isnan(a)&~np.isnan(b); out[f'acf{k}_{v}']=np.corrcoef(a[m],b[m])[0,1] if m.sum()>30 else np.nan
        x=X[~np.isnan(X)]; x=np.log(x[x>0]) if v in LOGV else x
        qv=np.quantile((x-ref[v][0])/ref[v][1],QS) if len(x)>10 else [np.nan]*5
        for q,val in zip(QS,qv): out[f'zq_{v}_{q}']=val
    for i,a in enumerate(V):
        for b in V[i+1:]:
            m=~np.isnan(tc[a])&~np.isnan(tc[b]); out[f'corr_{a}_{b}']=np.corrcoef(tc[a][m],tc[b][m])[0,1]
    for a in ('length_final','added_length','length_birth'):
        for b in ('fold_growth','length_birth','generationtime'):
            x=tc[a][:,:-1]; y=tc[b][:,1:]; m=~np.isnan(x)&~np.isnan(y); out[f'cross_{a}_{b}']=np.corrcoef(x[m],y[m])[0,1]
    x=M['length_birth']; y=M['length_final']; m=~np.isnan(x)&~np.isnan(y); out['adder_slope']=np.polyfit(x[m],y[m],1)[0] if m.sum()>10 else np.nan
    rm=np.nanmean(M['division_ratio']*M['efold'],axis=1); out['cv_lineage_r']=np.nanstd(rm)/np.nanmean(rm)
    # pair statistics, separately for sister (SL) and neighbour (NL) traps  (the sister test)
    A=np.where(trace=='A')[0]; B=np.where(trace=='B')[0]; ta={t:i for t,i in zip(trap[A],A)}
    for grp,flag in (('SL',True),('NL',False)):
        pairs=[(ta[t],j) for t,j in zip(trap[B],B) if t in ta and sis[j]==flag]
        if len(pairs)<10: continue
        ia,ib=map(list,zip(*pairs))
        for v in ('growth_rate','generationtime','length_birth','added_length','fold_growth'):
            ma=np.nanmean(M[v][ia],axis=1); mb=np.nanmean(M[v][ib],axis=1); m=~np.isnan(ma)&~np.isnan(mb); out[f'pairTA{grp}_{v}']=np.corrcoef(ma[m],mb[m])[0,1]
            g0a=M[v][ia,0]; g0b=M[v][ib,0]; m=~np.isnan(g0a)&~np.isnan(g0b); out[f'pairG0{grp}_{v}']=np.corrcoef(g0a[m],g0b[m])[0,1] if m.sum()>10 else np.nan
    return out

def is_fit(k):  return k.split('_')[0] in ('cv','fracL','acf1','acf2','corr','zq') or k.startswith('pairTA')
def is_hold(k): return k.split('_')[0] in ('acf3','acf5','acf10','cross') or k.startswith('pairG0') or k in ('adder_slope','cv_lineage_r')

# ---------------- model ----------------
PNAMES=['s_env_a','s_env_d','s_env_f','s_inh_a','rho_slow','rho_fast','s_fast_a','s_d','s_lin_d','s_f','s_lin_f','s_meas','log_nu_d','log_nu_a','rho_f']
P0=np.array([0.08,0.10,0.012,0.06,0.97,0.4,0.11,0.21,0.05,0.045,0.02,0.03,np.log(4),np.log(5),0.1])
LO=np.array([0,0,0,0,0.5,-0.3,0.01,0.03,0,0.01,0,0.005,np.log(2.2),np.log(2.2),-0.3])
HI=np.array([0.3,0.3,0.06,0.3,0.999,0.9,0.4,0.6,0.3,0.12,0.08,0.12,np.log(200),np.log(200),0.7])
GAUSS=np.log(200)   # "no heavy tails"

def simulate(theta, fx, trap, trace, L, sis, step, seed=0):
    p=dict(zip(PNAMES,theta)); rng=np.random.default_rng(seed); n_lin=len(L); gens=int(L.max())
    mu_a,mu_d,f0,k,x0bar=fx['mu_a'],fx['mu_d'],fx['f0'],fx['k'],fx['x0bar']; nu_d,nu_a=np.exp(p['log_nu_d']),np.exp(p['log_nu_a'])
    traps,tinv=np.unique(trap,return_inverse=True); nT=len(traps)
    env_a=rng.normal(0,p['s_env_a'],nT)[tinv]; env_d=rng.normal(0,p['s_env_d'],nT)[tinv]; env_f=rng.normal(0,p['s_env_f'],nT)[tinv]   # fix 4
    lin_f=rng.normal(0,p['s_lin_f'],n_lin); lin_d=rng.normal(0,p['s_lin_d'],n_lin)
    # inherited states: sister lineages share the mother's draw (same trap), neighbours get independent draws   (fix 1)
    tr_sl=rng.normal(0,p['s_inh_a'],nT)[tinv]; tr_fa=rng.normal(0,p['s_fast_a'],nT)[tinv]
    sl_a=np.where(sis,tr_sl,rng.normal(0,p['s_inh_a'],n_lin)); fa=np.where(sis,tr_fa,rng.normal(0,p['s_fast_a'],n_lin))
    fstate=np.zeros(n_lin)
    mother=x0bar/f0*np.exp(rng.normal(0,0.12,nT))[tinv]
    fA=np.clip(rng.normal(f0,p['s_f'],nT),0.3,0.7)[tinv]
    fprev=np.where(sis, np.where(trace=='A',fA,1-fA), np.clip(rng.normal(f0+env_f,p['s_f'],n_lin),0.3,0.7))
    x0=np.where(sis, fprev*mother, fprev*mother*np.exp(rng.normal(0,0.1,n_lin)))
    out={v:np.full((n_lin,gens),np.nan) for v in V}
    q_s,q_f=np.sqrt(1-p['rho_slow']**2),np.sqrt(1-p['rho_fast']**2)
    for g in range(gens):
        sl_a=p['rho_slow']*sl_a+rng.normal(0,p['s_inh_a']*q_s,n_lin)
        fa=p['rho_fast']*fa+student_t.rvs(nu_a,size=n_lin,random_state=rng)*np.sqrt((nu_a-2)/nu_a)*p['s_fast_a']*q_f
        alpha=mu_a*np.exp(env_a+sl_a+fa)
        noise=student_t.rvs(nu_d,size=n_lin,random_state=rng)*p['s_d']*np.sqrt((nu_d-2)/nu_d)
        delta=np.maximum((mu_d*np.exp(env_d+lin_d)+k*(x0-x0bar))*np.exp(noise),0.15*mu_d)
        xt=x0+delta; tau=np.log(xt/x0)/alpha
        tau_obs=np.maximum(np.round(tau/step),1)*step                                   # fix 2: frame quantisation
        x0m=x0*np.exp(rng.normal(0,p['s_meas'],n_lin)); xtm=xt*np.exp(rng.normal(0,p['s_meas'],n_lin))
        alive=g<L
        out['fold_growth'][alive,g]=np.log(xtm/x0m)[alive]; out['efold'][alive,g]=(xtm/x0m)[alive]; out['added_length'][alive,g]=(xtm-x0m)[alive]
        out['generationtime'][alive,g]=tau_obs[alive]; out['length_birth'][alive,g]=x0m[alive]; out['length_final'][alive,g]=xtm[alive]
        out['growth_rate'][alive,g]=(np.log(xtm/x0m)/tau_obs)[alive]; out['division_ratio'][alive,g]=fprev[alive] if g>0 else np.nan
        fstate=p['rho_f']*fstate+rng.normal(0,p['s_f']*np.sqrt(1-p['rho_f']**2),n_lin)
        f=np.clip(f0+env_f+lin_f+fstate,0.25,0.75); x0=f*xt; fprev=f
    return filter_matrix(out)                                                            # fix 2: same outlier filter as the data

# ---------------- fitting ----------------
LADDER={ # model ladder: parameters frozen (name -> value); everything else free
 'M0 iid Gaussian adder': {'s_env_a':0,'s_env_d':0,'s_env_f':0,'s_inh_a':0,'rho_slow':0.9,'rho_fast':0,'s_lin_d':0,'s_lin_f':0,'log_nu_d':GAUSS,'log_nu_a':GAUSS,'rho_f':0},
 'M1 +heavy tails':        {'s_env_a':0,'s_env_d':0,'s_env_f':0,'s_inh_a':0,'rho_slow':0.9,'rho_fast':0,'s_lin_d':0,'s_lin_f':0,'rho_f':0},
 'M2 +fast α state':       {'s_env_a':0,'s_env_d':0,'s_env_f':0,'s_inh_a':0,'rho_slow':0.9,'s_lin_d':0,'s_lin_f':0,'rho_f':0},
 'M3 +channel/lineage':    {'s_inh_a':0,'rho_slow':0.9},
 'M4 full':                {},
}
def prepare(ds, lineage_subset=None):
    d=load(ds); 
    if lineage_subset is not None: d=d[d.lin.isin(lineage_subset)]
    M,trap,trace,L,sis=to_matrix(d); fx=measured_inputs(d); step=step_size(ds); ref=zref(M); S=stats(M,trap,trace,sis,ref)
    return dict(d=d,M=M,trap=trap,trace=trace,L=L,sis=sis,fx=fx,step=step,ref=ref,S=S)

def fit(prep, frozen={}, starts=5, evals=500, polish=400, seed=0, verbose=False):
    S=prep['S']; keys=[k for k in S if is_fit(k) and S[k]==S[k]]; tgt=np.array([S[k] for k in keys])
    w=np.array([3.0 if k.startswith('cv') else (0.5 if k.startswith('zq') else 1.0) for k in keys])
    free=[i for i,p in enumerate(PNAMES) if p not in frozen]
    base=P0.copy()
    for p,v in frozen.items(): base[PNAMES.index(p)]=v
    def unpack(x): th=base.copy(); th[free]=x; return np.clip(th,LO,HI)
    sim_args=(prep['fx'],prep['trap'],prep['trace'],prep['L'],prep['sis'],prep['step'])
    def loss(x):
        s=stats(simulate(unpack(x),*sim_args,seed=0),prep['trap'],prep['trace'],prep['sis'],prep['ref'])
        r=np.array([s.get(k,np.nan) for k in keys])-tgt; r[np.isnan(r)]=1.0; return float(np.sum(w*r**2))   # nan = penalty
    rng=np.random.default_rng(seed); results=[]
    for st in range(starts):                                                      # fix 5: multi-start
        x0=base[free] if st==0 else np.clip(base[free]*np.exp(rng.normal(0,0.5,len(free)))+rng.normal(0,0.05,len(free)),LO[free],HI[free])
        r=minimize(loss,x0,method='Nelder-Mead',options={'maxfev':evals,'xatol':1e-3,'fatol':1e-4}); results.append((r.fun,r.x))
        if verbose: print('  start',st,'loss',round(r.fun,3),flush=True)
    results.sort(key=lambda t:t[0]); best=minimize(loss,results[0][1],method='Nelder-Mead',options={'maxfev':polish,'xatol':1e-3,'fatol':1e-4})
    th=unpack(best.x)
    return dict(theta=th, loss=float(best.fun), start_losses=[float(r[0]) for r in results], n_free=len(free),
                start_thetas=[unpack(r[1]).tolist() for r in results], keys=keys)

def evaluate(prep, theta, seeds=(7,)):
    'statistics of the fitted model with fresh seeds; returns fit/held-out pairs, z-quantile KS, physical means'
    S=prep['S']; sims=[stats(simulate(theta,prep['fx'],prep['trap'],prep['trace'],prep['L'],prep['sis'],prep['step'],seed=sd),prep['trap'],prep['trace'],prep['sis'],prep['ref']) for sd in seeds]
    s={k:float(np.nanmean([x.get(k,np.nan) for x in sims])) for k in S}
    out={'fit':{k:(float(S[k]),s[k]) for k in S if is_fit(k)}, 'holdout':{k:(float(S[k]),s[k]) for k in S if is_hold(k)}}
    sim=simulate(theta,prep['fx'],prep['trap'],prep['trace'],prep['L'],prep['sis'],prep['step'],seed=seeds[0]); M=prep['M']; ref=prep['ref']; ks={}
    for v in V:
        x=M[v][~np.isnan(M[v])]; y=sim[v][~np.isnan(sim[v])]
        if v in LOGV: x=np.log(x[x>0]); y=np.log(y[y>0])
        ks[v]=float(ks_2samp((x-ref[v][0])/ref[v][1],(y-ref[v][0])/ref[v][1]).statistic)
    out['ks']=ks; out['physical']={v:(float(np.nanmean(M[v])),float(np.nanmean(sim[v]))) for v in V}
    out['lineage_means_model']={v:np.nanmean(sim[v],axis=1).tolist() for v in ('growth_rate','generationtime','fold_growth','division_ratio','length_birth')}
    return out

def rmse(pairs): a=np.array([p for p in pairs.values() if not any(np.isnan(p))]); return float(np.sqrt(np.mean((a[:,0]-a[:,1])**2)))

def run_fit(ds, out_path, starts=5, evals=500, polish=400):
    res=json.load(open(out_path)) if os.path.exists(out_path) else {}
    prep=prepare(ds); f=fit(prep,starts=starts,evals=evals,polish=polish,verbose=True); ev=evaluate(prep,f['theta'],seeds=(7,8,9))
    r=dict(label=EXPS[ds],fixed=prep['fx'],step=prep['step'],n_lin=int(len(prep['L'])),params=dict(zip(PNAMES,map(float,f['theta']))),loss=f['loss'],
           start_losses=f['start_losses'],start_thetas=f['start_thetas'],n_free=f['n_free'],**ev)
    r['fit_rmse']=rmse(r['fit']); r['hold_rmse']=rmse(r['holdout']); res[ds]=r; json.dump(res,open(out_path,'w'),indent=1)
    print(f"{EXPS[ds]:11s} loss {f['loss']:.3f} starts {np.round(f['start_losses'],2).tolist()} fitRMSE {r['fit_rmse']:.3f} holdRMSE {r['hold_rmse']:.3f} ks_med {np.median(list(ev['ks'].values())):.3f} k_iv {prep['fx']['k']:+.2f} k_ols {prep['fx']['k_ols']:+.2f}",flush=True)

def run_ladder(ds, out_path, starts=2, evals=500, polish=300):
    res=json.load(open(out_path)) if os.path.exists(out_path) else {}
    prep=prepare(ds); res.setdefault(ds,{'label':EXPS[ds],'ladder':{},'cv':{}})
    for name,frozen in LADDER.items():
        f=fit(prep,frozen=frozen,starts=starts,evals=evals,polish=polish); ev=evaluate(prep,f['theta'],seeds=(7,8))
        res[ds]['ladder'][name]=dict(n_free=f['n_free'],loss=f['loss'],fit_rmse=rmse(ev['fit']),hold_rmse=rmse(ev['holdout']),ks_med=float(np.median(list(ev['ks'].values()))),ks_max=float(max(ev['ks'].values())))
        print(f"{EXPS[ds]:11s} {name:24s} free {f['n_free']:2d} loss {f['loss']:.3f} fit {rmse(ev['fit']):.3f} hold {rmse(ev['holdout']):.3f} ks {np.median(list(ev['ks'].values())):.3f}",flush=True)
        json.dump(res,open(out_path,'w'),indent=1)
    # lineage-split cross-validation (full model): fit on half the lineages, score all statistics on the other half
    rng=np.random.default_rng(1); lins=prep['d'].lin.unique(); traps=prep['d'].groupby('lin').trap_ID.first()
    utr=np.unique(traps.values); rng.shuffle(utr); half=set(utr[:len(utr)//2])
    train=[l for l in lins if traps[l] in half]; test=[l for l in lins if traps[l] not in half]
    ptr=prepare(ds,train); pte=prepare(ds,test); f=fit(ptr,starts=starts,evals=evals,polish=polish); ev=evaluate(pte,f['theta'],seeds=(7,8))
    allp={**ev['fit'],**ev['holdout']}
    res[ds]['cv']=dict(n_train=len(train),n_test=len(test),all_rmse=rmse(allp),fit_rmse=rmse(ev['fit']),hold_rmse=rmse(ev['holdout']),ks_med=float(np.median(list(ev['ks'].values()))))
    print(f"{EXPS[ds]:11s} lineage-split CV: trained on {len(train)} lineages, tested on {len(test)}: rmse all {rmse(allp):.3f} fit-type {rmse(ev['fit']):.3f} held-type {rmse(ev['holdout']):.3f}",flush=True)
    json.dump(res,open(out_path,'w'),indent=1)

if __name__=='__main__':
    mode=sys.argv[1]; dss=sys.argv[2:]
    if mode=='fit':
        for ds in dss: run_fit(ds, os.path.join(HERE,'fit3_results.json'))
    elif mode=='ladder':
        for ds in dss: run_ladder(ds, os.path.join(HERE,'ladder_results.json'))
