"""Fit the hierarchical Thompson-sampling adder to each experiment.
Fitting is done on scale-free / normalized statistics (CV, lineage-centred correlations, autocorrelations,
lineage variance shares). Validation uses held-out statistics in z-score units and in physical units."""
import numpy as np, pandas as pd, json, os, sys, warnings
from scipy.optimize import minimize
from scipy.stats import ks_2samp, t as student_t
warnings.filterwarnings('ignore')
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Datasets'))
V=['fold_growth','division_ratio','added_length','generationtime','length_birth','length_final','growth_rate','efold']
SYM={'fold_growth':'φ','division_ratio':'f','added_length':'Δ','generationtime':'τ','length_birth':'x0','length_final':'xτ','growth_rate':'α','efold':'e^φ'}
LOGV={'added_length','generationtime','length_birth','length_final','growth_rate','efold'}  # z-scored in log

EXPS={'Pooled_SM':'SM','Maryam_LongTraces':'MM Maryam','MG1655_inLB_LongTraces':'MM MG1655','lambda_LB':'MM λ',
 'MC4100_25C (Tanouchi 2015)':'Tan25','MC4100_27C (Tanouchi 2015)':'Tan27','MC4100_37C (Tanouchi 2015)':'Tan37',
 '20090129_E_coli_MG1655_(CGSC_6300)_Wang2010':'Wang 0129','20090131_E_coli_MG1655_(CGSC_6300)_Wang2010':'Wang 0131',
 '20090210_E_coli_MG1655_(CGSC_6300)_Wang2010':'Wang 0210','20090512_E_coli_MG1655_(CGSC_6300)_Wang2010':'Wang 0512',
 '20090525_E_coli_MG1655_(CGSC_6300)_Wang2010':'Wang 0525','20090702_E_coli_MG1655_(CGSC_6300)_Wang2010':'Wang 0702',
 '20090922_E_coli_MG1655_lexA3_Wang2010':'lexA3 0922','20090923_E_coli_MG1655_lexA3_Wang2010':'lexA3 0923','20090930_E_coli_MG1655_lexA3_Wang2010':'lexA3 0930'}

# ---------------- data ----------------
def load(ds):
    d=pd.read_csv(f'{ds}/ProcessedData/z_score_under_3/physical_units_without_outliers.csv')
    if 'trap_ID' not in d: d['trap_ID']=d.lineage_ID; d['trace']='A'; d['dataset']='MM'
    d['lin']=d.lineage_ID.astype(str)+'|'+d.trap_ID.astype(str)+'|'+d.trace
    d['efold']=np.exp(d.fold_growth)
    return d

def to_matrix(d):
    """return dict var -> (n_lin, maxgen) matrix with nan padding, plus trap/trace arrays"""
    lins=d.lin.unique(); idx={l:i for i,l in enumerate(lins)}
    G=int(d.groupby('lin').size().max()); M={v:np.full((len(lins),G),np.nan) for v in V}
    li=d.lin.map(idx).values; gi=d.generation.values.astype(int)
    for v in V: M[v][li,gi]=d[v].values
    meta=d.groupby('lin').first()[['trap_ID','trace']].loc[lins]
    return M, meta.trap_ID.values, meta.trace.values

# ---------------- statistics (all scale-free) ----------------
QS=[0.05,0.25,0.5,0.75,0.95]
def zref(M):
    ref={}
    for v in V:
        x=M[v][~np.isnan(M[v])]
        if v in LOGV: x=np.log(x[x>0])
        ref[v]=(x.mean(),x.std())
    return ref
def stats(M, trap=None, trace=None, ref=None):
    out={}
    if ref is not None:
        for v in V:
            x=M[v][~np.isnan(M[v])]
            if v in LOGV: x=np.log(x[x>0])
            z=(x-ref[v][0])/ref[v][1]
            for q,val in zip(QS,np.quantile(z,QS)): out[f'zq_{v}_{q}']=val
    tc={}
    for v in V:
        X=M[v]; mu=np.nanmean(X,axis=1,keepdims=True); tc[v]=X-mu
        out[f'cv_{v}']=np.nanstd(X)/np.nanmean(X)
        out[f'fracL_{v}']=1-np.nanvar(tc[v])/np.nanvar(X)
        for k in (1,2,3):
            a=tc[v][:,:-k]; b=tc[v][:,k:]; m=~np.isnan(a)&~np.isnan(b)
            out[f'acf{k}_{v}']=np.corrcoef(a[m],b[m])[0,1] if m.sum()>30 else 0.0
    for i,a in enumerate(V):
        for b in V[i+1:]:
            m=~np.isnan(tc[a])&~np.isnan(tc[b]); out[f'corr_{a}_{b}']=np.corrcoef(tc[a][m],tc[b][m])[0,1]
    # cross generation: mother var -> daughter var (held out from fitting)
    for a in ('length_final','added_length','length_birth'):
        for b in ('fold_growth','length_birth','generationtime'):
            x=tc[a][:,:-1]; y=tc[b][:,1:]; m=~np.isnan(x)&~np.isnan(y); out[f'cross_{a}_{b}']=np.corrcoef(x[m],y[m])[0,1]
    # adder slope and CV of lineage mean of r (held out)
    x=M['length_birth']; y=M['length_final']; m=~np.isnan(x)&~np.isnan(y); out['adder_slope']=np.polyfit(x[m],y[m],1)[0]
    r=M['division_ratio']*M['efold']; rm=np.nanmean(r,axis=1); out['cv_lineage_r']=np.nanstd(rm)/np.nanmean(rm)
    # sister / pair statistics
    if trap is not None:
        A=np.where(trace=='A')[0]; B=np.where(trace=='B')[0]
        ta={t:i for t,i in zip(trap[A],A)}; pairs=[(ta[t],j) for t,j in zip(trap[B],B) if t in ta]
        if len(pairs)>10:
            ia,ib=zip(*pairs)
            for v in ('growth_rate','generationtime','length_birth','added_length','fold_growth'):
                ma=np.nanmean(M[v][list(ia)],axis=1); mb=np.nanmean(M[v][list(ib)],axis=1); m=~np.isnan(ma)&~np.isnan(mb)
                out[f'pairTA_{v}']=np.corrcoef(ma[m],mb[m])[0,1]
                g0a=M[v][list(ia),0]; g0b=M[v][list(ib),0]; m=~np.isnan(g0a)&~np.isnan(g0b)
                out[f'pairG0_{v}']=np.corrcoef(g0a[m],g0b[m])[0,1] if m.sum()>10 else np.nan
    return out

# ---------------- model (vectorized over lineages) ----------------
PNAMES=['s_env_a','s_inh_a','rho_slow','rho_fast','s_fast_a','s_d','s_lin_d','s_f','s_lin_f','s_meas','log_nu_d','s_fast_d','rho_fast_d','log_nu_a','rho_f']
P0=np.array([0.08,0.06,0.97,0.4,0.11,0.21,0.08,0.045,0.02,0.03,np.log(4),0.0,0.3,np.log(5),0.1])
LO=np.array([0,0,0.5,-0.3,0.01,0.03,0,0.01,0,0.005,np.log(2.2),0,0,np.log(2.2),-0.3])
HI=np.array([0.3,0.3,0.999,0.9,0.4,0.6,0.3,0.12,0.08,0.12,np.log(60),0.3,0.9,np.log(60),0.7])

def simulate(theta, fixed, n_lin, gens, trap, trace, seed=0):
    p=dict(zip(PNAMES,theta)); rng=np.random.default_rng(seed)
    mu_a,mu_d,f0,k=fixed['mu_a'],fixed['mu_d'],fixed['f0'],fixed['k']; nu=np.exp(p['log_nu_d']); nua=np.exp(p['log_nu_a']); fstate=np.zeros(n_lin)
    # trap-level offsets shared by both lineages of a trap
    traps,tinv=np.unique(trap,return_inverse=True); nT=len(traps)
    env_a=rng.normal(0,p['s_env_a'],nT)[tinv]; env_d=rng.normal(0,p['s_env_a']*1.2,nT)[tinv]; env_f=rng.normal(0,p['s_f']*0.3,nT)[tinv]
    lin_f=rng.normal(0,p['s_lin_f'],n_lin); lin_d=rng.normal(0,p['s_lin_d'],n_lin)
    # inherited states at birth: sisters (trace B with same trap, SL) share mother's state; handled by giving both the trap-level draw
    sis=fixed.get('sisters',False)
    sl_a=rng.normal(0,p['s_inh_a'],nT)[tinv] if sis else rng.normal(0,p['s_inh_a'],n_lin)
    fa=rng.normal(0,p['s_fast_a'],nT)[tinv] if sis else rng.normal(0,p['s_fast_a'],n_lin)
    fd=np.zeros(n_lin)
    x0bar=mu_d*f0/(1-f0)
    mother=x0bar/f0*np.exp(rng.normal(0,0.12,nT))[tinv]
    fprev=np.clip(rng.normal(f0+env_f,p['s_f'],n_lin),0.3,0.7)
    if sis:  # sisters: A gets f, B gets 1-f of the same mother
        fA=np.clip(rng.normal(f0,p['s_f'],nT),0.3,0.7)[tinv]; fprev=np.where(trace=='A',fA,1-fA)
    x0=fprev*mother
    out={v:np.full((n_lin,gens),np.nan) for v in V}
    q_s=np.sqrt(1-p['rho_slow']**2); q_f=np.sqrt(1-p['rho_fast']**2); q_fd=np.sqrt(1-p['rho_fast_d']**2)
    for g in range(gens):
        sl_a=p['rho_slow']*sl_a+rng.normal(0,p['s_inh_a']*q_s,n_lin)
        fa=p['rho_fast']*fa+student_t.rvs(nua,size=n_lin,random_state=rng)*np.sqrt((nua-2)/nua)*p['s_fast_a']*q_f
        fd=p['rho_fast_d']*fd+rng.normal(0,p['s_fast_d']*q_fd,n_lin)
        alpha=mu_a*np.exp(env_a+sl_a+fa)
        noise_d=student_t.rvs(nu,size=n_lin,random_state=rng)*p['s_d']*np.sqrt((nu-2)/nu)
        delta=(mu_d*np.exp(env_d+lin_d+fd)+k*(x0-x0bar))*np.exp(noise_d)
        delta=np.maximum(delta,0.15*mu_d)
        xt=x0+delta; phi=np.log(xt/x0); tau=phi/alpha
        x0m=x0*np.exp(rng.normal(0,p['s_meas'],n_lin)); xtm=xt*np.exp(rng.normal(0,p['s_meas'],n_lin))
        out['fold_growth'][:,g]=np.log(xtm/x0m); out['efold'][:,g]=xtm/x0m; out['added_length'][:,g]=xtm-x0m
        out['generationtime'][:,g]=tau; out['length_birth'][:,g]=x0m; out['length_final'][:,g]=xtm
        out['growth_rate'][:,g]=np.log(xtm/x0m)/tau; out['division_ratio'][:,g]=fprev if g>0 else np.nan
        fstate=p['rho_f']*fstate+rng.normal(0,p['s_f']*np.sqrt(1-p['rho_f']**2),n_lin); f=np.clip(f0+env_f+lin_f+fstate,0.25,0.75); x0=f*xt; fprev=f
    return out

FIT_KEYS=lambda keys:[k for k in keys if k.split('_')[0] in ('cv','fracL','acf1','acf2','corr','zq') or k.startswith('pairTA')]
HOLD_KEYS=lambda keys:[k for k in keys if k.startswith('acf3') or k.startswith('cross') or k.startswith('pairG0') or k in ('adder_slope','cv_lineage_r')]

def fit_experiment(ds, iters=1200):
    d=load(ds); M,trap,trace=to_matrix(d); n_lin,G=M['fold_growth'].shape
    gens=int(np.nanmedian(np.sum(~np.isnan(M['fold_growth']),axis=1)))
    # fixed, directly measured quantities
    mu_a=float(np.exp(np.nanmean(np.log(M['growth_rate'])))); mu_d=float(np.nanmean(M['added_length'])); f0=float(np.nanmean(M['division_ratio']))
    tcD=M['added_length']-np.nanmean(M['added_length'],axis=1,keepdims=True); tcX=M['length_birth']-np.nanmean(M['length_birth'],axis=1,keepdims=True)
    m=~np.isnan(tcD)&~np.isnan(tcX); k=float(np.polyfit(tcX[m],tcD[m],1)[0])
    sisters = ds=='Pooled_SM'
    fixed=dict(mu_a=mu_a,mu_d=mu_d,f0=f0,k=k,sisters=sisters)
    ref=zref(M); S=stats(M,trap,trace,ref); keys=FIT_KEYS(S); hold=HOLD_KEYS(S)
    tgt=np.array([S[q] for q in keys]); w=np.array([3.0 if q.startswith('cv') else (0.5 if q.startswith('zq') else 1.0) for q in keys])
    def loss(th):
        th=np.clip(th,LO,HI)
        sim=simulate(th,fixed,n_lin,gens,trap,trace,seed=0); s=stats(sim,trap,trace,ref)
        pred=np.array([s.get(q,np.nan) for q in keys]); r=(pred-tgt); r[np.isnan(r)]=0
        return float(np.sum(w*r**2))
    best=minimize(loss,P0,method='Nelder-Mead',options={'maxfev':iters,'xatol':1e-3,'fatol':1e-4})
    best=minimize(loss,best.x,method='Nelder-Mead',options={'maxfev':iters//2,'xatol':1e-3,'fatol':1e-4})
    th=np.clip(best.x,LO,HI)
    # validation: new seed (out-of-sample noise), held-out statistics, z-score distributions, physical units
    sim=simulate(th,fixed,n_lin,gens,trap,trace,seed=7); s=stats(sim,trap,trace,ref)
    val={'fit_loss':float(best.fun),'nfev':int(best.nfev),'fixed':{k_:(float(v) if not isinstance(v,bool) else v) for k_,v in fixed.items()},
         'params':dict(zip(PNAMES,map(float,th))),'n_lin':int(n_lin),'gens':int(gens),'label':EXPS[ds]}
    val['fit']={q:(float(S[q]),float(s.get(q,np.nan))) for q in keys}
    val['holdout']={q:(float(S[q]),float(s.get(q,np.nan))) for q in hold}
    # z-score distribution comparison (normalized units): per variable, z = (x - mean)/sd using DATA's mean/sd, log for log-variables
    zs={}
    for v in V:
        x=M[v][~np.isnan(M[v])]; y=sim[v][~np.isnan(sim[v])]
        if v in LOGV: x=np.log(x[x>0]); y=np.log(y[y>0])
        mu,sd=x.mean(),x.std(); zx=(x-mu)/sd; zy=(y-mu)/sd
        qs=[0.05,0.25,0.5,0.75,0.95]
        zs[v]={'ks':float(ks_2samp(zx,zy).statistic),'q_data':np.quantile(zx,qs).round(3).tolist(),'q_model':np.quantile(zy,qs).round(3).tolist(),
               'mean_model_z':float(zy.mean()),'sd_model_z':float(zy.std())}
    val['zdist']=zs
    # physical units: means of raw variables, CV of lineage means
    val['physical']={v:{'mean_data':float(np.nanmean(M[v])),'mean_model':float(np.nanmean(sim[v])),
                       'cvTA_data':float(np.nanstd(np.nanmean(M[v],axis=1))/np.nanmean(M[v])),'cvTA_model':float(np.nanstd(np.nanmean(sim[v],axis=1))/np.nanmean(sim[v]))} for v in V}
    return val

if __name__=='__main__':
    which=sys.argv[1:] or list(EXPS)
    res=json.load(open('fit_results.json')) if os.path.exists('fit_results.json') else {}
    for ds in which:
        v=fit_experiment(ds); res[ds]=v
        fit=v['fit']; rmse=np.sqrt(np.mean([(a-b)**2 for a,b in fit.values()]))
        hrmse=np.sqrt(np.mean([(a-b)**2 for a,b in v['holdout'].values()]))
        print(f"{v['label']:11s} loss {v['fit_loss']:.3f} nfev {v['nfev']} fitRMSE {rmse:.3f} holdoutRMSE {hrmse:.3f} ks(max) {max(z['ks'] for z in v['zdist'].values()):.3f}", flush=True)
        json.dump(res,open('fit_results.json','w'),indent=1)
