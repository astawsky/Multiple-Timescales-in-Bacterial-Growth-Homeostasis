"""Cross-lab transfer test.
'Rule' parameters (per-generation posterior widths, heavy tails, fast memory) are frozen at the median of the fits
from the TRAINING labs. On each TEST experiment only the environment/lineage structure and measurement noise are refitted.
Everything else (measured inputs mu_a, delta_bar, <f>, k) is read from the test data as before."""
import numpy as np, json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fit_bandit_model import *   # load, to_matrix, zref, stats, simulate, PNAMES, LO, HI, FIT_KEYS, HOLD_KEYS, EXPS, V, LOGV
from scipy.optimize import minimize
from scipy.stats import ks_2samp

RULE = ['rho_fast','s_fast_a','s_d','log_nu_d','log_nu_a','s_f','rho_f','s_fast_d','rho_fast_d']   # frozen across labs
FREE = [p for p in PNAMES if p not in RULE]                                                       # s_env_a, s_inh_a, rho_slow, s_lin_d, s_lin_f, s_meas
THIS_LAB = ['Pooled_SM','Maryam_LongTraces','MG1655_inLB_LongTraces','lambda_LB']
OTHERS   = [d for d in EXPS if d not in THIS_LAB]
full = json.load(open('fit_results.json'))

def rule_from(train):
    return {p: float(np.median([full[d]['params'][p] for d in train])) for p in RULE}

def transfer_fit(ds, rule, iters=400):
    d=load(ds); M,trap,trace=to_matrix(d); n_lin,G=M['fold_growth'].shape
    gens=int(np.nanmedian(np.sum(~np.isnan(M['fold_growth']),axis=1)))
    fixed=dict(full[ds]['fixed']); fixed['sisters']= ds=='Pooled_SM'
    ref=zref(M); S=stats(M,trap,trace,ref); keys=FIT_KEYS(S); hold=HOLD_KEYS(S)
    tgt=np.array([S[q] for q in keys]); w=np.array([3.0 if q.startswith('cv') else (0.5 if q.startswith('zq') else 1.0) for q in keys])
    base=np.array([rule[p] if p in rule else full[ds]['params'][p] for p in PNAMES])   # start free params at the full-fit values
    fi=[PNAMES.index(p) for p in FREE]
    def unpack(x):
        th=base.copy(); th[fi]=x; return np.clip(th,LO,HI)
    def loss(x):
        s=stats(simulate(unpack(x),fixed,n_lin,gens,trap,trace,seed=0),trap,trace,ref)
        r=np.array([s.get(q,np.nan) for q in keys])-tgt; r[np.isnan(r)]=0; return float(np.sum(w*r**2))
    x0=base[fi]; best=minimize(loss,x0,method='Nelder-Mead',options={'maxfev':iters,'xatol':1e-3,'fatol':1e-4})
    th=unpack(best.x)
    sim=simulate(th,fixed,n_lin,gens,trap,trace,seed=7); s=stats(sim,trap,trace,ref)
    f=np.array([(S[q],s[q]) for q in keys if s.get(q,np.nan)==s.get(q,np.nan)]); h=np.array([(S[q],s[q]) for q in hold if s.get(q,np.nan)==s.get(q,np.nan)])
    ks={}
    for v in V:
        x=M[v][~np.isnan(M[v])]; y=sim[v][~np.isnan(sim[v])]
        if v in LOGV: x=np.log(x[x>0]); y=np.log(y[y>0])
        zx=(x-ref[v][0])/ref[v][1]; zy=(y-ref[v][0])/ref[v][1]; ks[v]=float(ks_2samp(zx,zy).statistic)
    return {'fit_rmse':float(np.sqrt(np.mean((f[:,0]-f[:,1])**2))),'hold_rmse':float(np.sqrt(np.mean((h[:,0]-h[:,1])**2))),
            'hold_pairs':h.tolist(),'ks_med':float(np.median(list(ks.values()))),'ks_max':float(max(ks.values())),'params':dict(zip(PNAMES,map(float,th))),'label':EXPS[ds]}

def full_scores(ds):
    r=full[ds]; f=np.array(list(r['fit'].values())); h=np.array([v for v in r['holdout'].values() if not any(np.isnan(v))]); ks=[z['ks'] for z in r['zdist'].values()]
    return {'fit_rmse':float(np.sqrt(np.mean((f[:,0]-f[:,1])**2))),'hold_rmse':float(np.sqrt(np.mean((h[:,0]-h[:,1])**2))),'ks_med':float(np.median(ks)),'ks_max':float(max(ks))}

if __name__=='__main__':
    out=json.load(open('transfer_results.json')) if os.path.exists('transfer_results.json') else {}
    direction=sys.argv[1]; tests=sys.argv[2:]
    train, name = (OTHERS,'others_to_thislab') if direction=='A' else (THIS_LAB,'thislab_to_others')
    rule=rule_from(train); out.setdefault(name,{'rule':rule,'train':[EXPS[d] for d in train],'tests':{}})
    for ds in tests:
        t=transfer_fit(ds,rule); t['full']=full_scores(ds); out[name]['tests'][ds]=t
        print(f"{t['label']:11s} transfer fit {t['fit_rmse']:.3f} hold {t['hold_rmse']:.3f} ks {t['ks_med']:.2f}/{t['ks_max']:.2f} | full fit {t['full']['fit_rmse']:.3f} hold {t['full']['hold_rmse']:.3f} ks {t['full']['ks_med']:.2f}/{t['full']['ks_max']:.2f}",flush=True)
        json.dump(out,open('transfer_results.json','w'),indent=1)
