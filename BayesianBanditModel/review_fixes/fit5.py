"""All 16 experiments with fixes 1, 2 and 4 (no timing state).
 1 measured measurement noise (meas_noise.json), OLS k
 2 CMA-ES on a seed-averaged loss, two independent restarts per group
 4 one parameter vector per lab group (lab x strain); experiment means and k stay per experiment."""
import numpy as np, json, os, sys, time
from scipy.stats import ks_2samp
import fit4
from fit4 import setup, cma_fit, evaluate, simulate, P, LO, HI, EXPS, V, LOGV
HERE=os.path.dirname(os.path.abspath(__file__)); OUT=os.path.join(HERE,'fit5_results.json')
GROUPS={
 'This lab (SM + MM)': ['Pooled_SM','Maryam_LongTraces','MG1655_inLB_LongTraces','lambda_LB'],
 'Tanouchi 2015': ['MC4100_25C (Tanouchi 2015)','MC4100_27C (Tanouchi 2015)','MC4100_37C (Tanouchi 2015)'],
 'Wang 2010 MG1655': ['20090129_E_coli_MG1655_(CGSC_6300)_Wang2010','20090131_E_coli_MG1655_(CGSC_6300)_Wang2010','20090210_E_coli_MG1655_(CGSC_6300)_Wang2010','20090512_E_coli_MG1655_(CGSC_6300)_Wang2010','20090525_E_coli_MG1655_(CGSC_6300)_Wang2010','20090702_E_coli_MG1655_(CGSC_6300)_Wang2010'],
 'Wang 2010 lexA3': ['20090922_E_coli_MG1655_lexA3_Wang2010','20090923_E_coli_MG1655_lexA3_Wang2010','20090930_E_coli_MG1655_lexA3_Wang2010'],
}
def full_eval(th,p):
    ev=evaluate(th,p,timing=False)
    sim=simulate(th,p['fx'],p['trap'],p['trace'],p['L'],p['sis'],p['step'],p['s_meas'],seed=7,timing=False); M=p['M']; ref=p['ref']; ks={}
    for v in V:
        x=M[v][~np.isnan(M[v])]; y=sim[v][~np.isnan(sim[v])]
        if v in LOGV: x=np.log(x[x>0]); y=np.log(y[y>0])
        ks[v]=float(ks_2samp((x-ref[v][0])/ref[v][1],(y-ref[v][0])/ref[v][1]).statistic)
    ev['ks']=ks; ev['physical']={v:(float(np.nanmean(M[v])),float(np.nanmean(sim[v]))) for v in V}
    ev['cv']={v:(float(p['S'][f'cv_{v}']),None) for v in V}
    return ev
if __name__=='__main__':
    R=json.load(open(OUT)) if os.path.exists(OUT) else {}
    rng=np.random.default_rng(7)
    for g in sys.argv[1:] or list(GROUPS):
        names=GROUPS[g]; preps=[setup(d) for d in names]; t=time.time(); runs=[]
        for r in range(2):
            st=None if r==0 else rng.uniform(0.1,0.9,len(P)-2)
            th,f=cma_fit(preps,timing=False,maxfev=500,seed=50+r,start=st); runs.append((f,th))
            print(f"{g} run {r}: loss {f:.3f} ({time.time()-t:.0f}s)",flush=True)
        runs.sort(key=lambda x:x[0]); th=runs[0][1]
        R[g]={'experiments':[EXPS[d] for d in names],'run_losses':[float(x[0]) for x in runs],
              'run_params':[dict(zip(P,map(float,x[1]))) for x in runs],'params':dict(zip(P,map(float,th))),
              'per_experiment':{}}
        for d,p in zip(names,preps):
            ev=full_eval(th,p); ev['fixed']={k:float(v) for k,v in p['fx'].items()}; ev['s_meas']=p['s_meas']; ev['step']=p['step']
            R[g]['per_experiment'][d]={'label':EXPS[d],**ev}
            print(f"   {EXPS[d]:11s} fit {ev['fit_rmse']:.3f} hold {ev['hold_rmse']:.3f} ks_med {np.median(list(ev['ks'].values())):.3f}",flush=True)
        json.dump(R,open(OUT,'w'),indent=1)
