"""Remaining fixes, one at a time.
Step 1 (meas_noise.py): s_meas is measured from raw images and fixed; k reverts to OLS (consistent once noise is small).
Step 2: CMA-ES on a seed-averaged loss replaces single-seed Nelder-Mead; convergence is checked with independent restarts.
Step 3: an inherited timing state c: x_tau = (x0 + Delta) * exp(c), c_n = rho_t c_{n-1} + noise; sisters share c at birth.
        Fitted on sisters' first-cycle correlations; judged on the decay over cycles 2-3 and on neighbours (held out).
Step 4: within-lab pooling: one parameter vector per lab group, experiment means and k still per experiment."""
import numpy as np, json, os, sys, time, cma, warnings
from scipy.stats import t as student_t
import fit3
from fit3 import prepare, V, EXPS, filter_matrix, rmse, LOGV
warnings.filterwarnings('ignore')
H=os.path.dirname(os.path.abspath(__file__)); MEAS=json.load(open(f'{H}/meas_noise.json'))

P=['s_env_a','s_env_d','s_env_f','s_inh_a','rho_slow','rho_fast','s_fast_a','s_d','s_lin_d','s_f','s_lin_f','log_nu_d','log_nu_a','rho_f','s_t','rho_t']
LO=np.array([0,0,0,0,0.5,-0.3,0.01,0.03,0,0.01,0,np.log(2.2),np.log(2.2),-0.3,0,-0.3])
HI=np.array([0.3,0.3,0.06,0.3,0.999,0.9,0.4,0.6,0.3,0.12,0.08,np.log(200),np.log(200),0.7,0.15,0.9])
P0=np.array([0.08,0.10,0.012,0.06,0.97,0.4,0.11,0.21,0.05,0.045,0.02,np.log(4),np.log(5),0.1,0.0,0.3])

def simulate(th, fx, trap, trace, L, sis, step, s_meas, seed=0, timing=True):
    p=dict(zip(P,th)); rng=np.random.default_rng(seed); n=len(L); G=int(L.max())
    mu_a,mu_d,f0,k,x0bar=fx['mu_a'],fx['mu_d'],fx['f0'],fx['k_ols'],fx['x0bar']; nu_d,nu_a=np.exp(p['log_nu_d']),np.exp(p['log_nu_a'])
    traps,ti=np.unique(trap,return_inverse=True); nT=len(traps)
    env_a=rng.normal(0,p['s_env_a'],nT)[ti]; env_d=rng.normal(0,p['s_env_d'],nT)[ti]; env_f=rng.normal(0,p['s_env_f'],nT)[ti]
    lin_f=rng.normal(0,p['s_lin_f'],n); lin_d=rng.normal(0,p['s_lin_d'],n)
    sh=lambda s:np.where(sis, rng.normal(0,s,nT)[ti], rng.normal(0,s,n))          # shared by sisters, independent otherwise
    sl_a=sh(p['s_inh_a']); fa=sh(p['s_fast_a']); c=sh(p['s_t']) if timing else np.zeros(n)
    fstate=np.zeros(n); mother=x0bar/f0*np.exp(rng.normal(0,0.12,nT))[ti]
    fA=np.clip(rng.normal(f0,p['s_f'],nT),0.3,0.7)[ti]
    fprev=np.where(sis,np.where(trace=='A',fA,1-fA),np.clip(rng.normal(f0+env_f,p['s_f'],n),0.3,0.7))
    x0=np.where(sis,fprev*mother,fprev*mother*np.exp(rng.normal(0,0.1,n)))
    out={v:np.full((n,G),np.nan) for v in V}
    qs,qf,qt=np.sqrt(1-p['rho_slow']**2),np.sqrt(1-p['rho_fast']**2),np.sqrt(1-p['rho_t']**2)
    for g in range(G):
        sl_a=p['rho_slow']*sl_a+rng.normal(0,p['s_inh_a']*qs,n)
        fa=p['rho_fast']*fa+student_t.rvs(nu_a,size=n,random_state=rng)*np.sqrt((nu_a-2)/nu_a)*p['s_fast_a']*qf
        if timing and g>0: c=p['rho_t']*c+rng.normal(0,p['s_t']*qt,n)
        alpha=mu_a*np.exp(env_a+sl_a+fa)
        noise=student_t.rvs(nu_d,size=n,random_state=rng)*p['s_d']*np.sqrt((nu_d-2)/nu_d)
        delta=np.maximum((mu_d*np.exp(env_d+lin_d)+k*(x0-x0bar))*np.exp(noise),0.15*mu_d)
        xt=np.maximum((x0+delta)*np.exp(c),1.05*x0)
        tau=np.maximum(np.round(np.log(xt/x0)/alpha/step),1)*step
        x0m=x0*np.exp(rng.normal(0,s_meas,n)); xtm=xt*np.exp(rng.normal(0,s_meas,n)); a=g<L
        out['fold_growth'][a,g]=np.log(xtm/x0m)[a]; out['efold'][a,g]=(xtm/x0m)[a]; out['added_length'][a,g]=(xtm-x0m)[a]
        out['generationtime'][a,g]=tau[a]; out['length_birth'][a,g]=x0m[a]; out['length_final'][a,g]=xtm[a]
        out['growth_rate'][a,g]=(np.log(xtm/x0m)/tau)[a]; out['division_ratio'][a,g]=fprev[a] if g>0 else np.nan
        fstate=p['rho_f']*fstate+rng.normal(0,p['s_f']*np.sqrt(1-p['rho_f']**2),n)
        f=np.clip(f0+env_f+lin_f+fstate,0.25,0.75); x0=f*xt; fprev=f
    return filter_matrix(out)

def stats(M,trap,trace,sis,ref):
    out=fit3.stats(M,trap,trace,sis,ref)
    A=np.where(trace=='A')[0]; B=np.where(trace=='B')[0]; ta={t:i for t,i in zip(trap[A],A)}
    for grp,flag in (('SL',True),('NL',False)):
        pr=[(ta[t],j) for t,j in zip(trap[B],B) if t in ta and sis[j]==flag]
        if len(pr)<10: continue
        ia,ib=map(list,zip(*pr))
        for gen in (1,2):
            for v in ('growth_rate','generationtime','length_birth','added_length','fold_growth'):
                a=M[v][ia,gen]; b=M[v][ib,gen]; m=~np.isnan(a)&~np.isnan(b); out[f'pairG{gen}{grp}_{v}']=np.corrcoef(a[m],b[m])[0,1] if m.sum()>10 else np.nan
    return out
is_fit =lambda k: fit3.is_fit(k) or k.startswith('pairG0SL')
is_hold=lambda k: (fit3.is_hold(k) and not k.startswith('pairG0SL')) or k.startswith('pairG1') or k.startswith('pairG2')

def setup(ds):
    p=prepare(ds); p['s_meas']=MEAS[ds]['sigma_endpoint']; p['S']=stats(p['M'],p['trap'],p['trace'],p['sis'],p['ref'])
    p['keys']=[k for k in p['S'] if is_fit(k) and p['S'][k]==p['S'][k]]; p['tgt']=np.array([p['S'][k] for k in p['keys']])
    p['w']=np.array([3.0 if k.startswith('cv') else (0.5 if k.startswith('zq') else 1.0) for k in p['keys']]); return p

def loss_one(th,p,timing,seeds=(0,1)):
    tot=0
    for sd in seeds:
        s=stats(simulate(th,p['fx'],p['trap'],p['trace'],p['L'],p['sis'],p['step'],p['s_meas'],seed=sd,timing=timing),p['trap'],p['trace'],p['sis'],p['ref'])
        r=np.array([s.get(k,np.nan) for k in p['keys']])-p['tgt']; r[np.isnan(r)]=1.0; tot+=np.sum(p['w']*r**2)
    return float(tot/len(seeds))

def cma_fit(preps, timing=True, maxfev=700, seed=0, start=None):
    'one CMA-ES run on the summed loss of all experiments in preps (pooling when len>1)'
    free=[i for i,n in enumerate(P) if timing or n not in ('s_t','rho_t')]
    base=P0.copy(); 
    if not timing: base[P.index('s_t')]=0
    to_th=lambda u:(lambda th:(th.__setitem__(free,LO[free]+np.clip(u,0,1)*(HI[free]-LO[free])),th)[1])(base.copy())
    u0=(P0[free]-LO[free])/(HI[free]-LO[free]) if start is None else start
    es=cma.CMAEvolutionStrategy(list(u0),0.25,{'bounds':[0,1],'maxfevals':maxfev,'seed':seed+1,'verbose':-9,'popsize':10})
    es.optimize(lambda u: sum(loss_one(to_th(np.array(u)),p,timing) for p in preps))
    return to_th(np.array(es.result.xbest)), float(es.result.fbest)

def evaluate(th,p,timing=True):
    sims=[stats(simulate(th,p['fx'],p['trap'],p['trace'],p['L'],p['sis'],p['step'],p['s_meas'],seed=sd,timing=timing),p['trap'],p['trace'],p['sis'],p['ref']) for sd in (7,8,9)]
    s={k:float(np.nanmean([x.get(k,np.nan) for x in sims])) for k in p['S']}
    fit={k:(float(p['S'][k]),s[k]) for k in p['S'] if is_fit(k)}; hold={k:(float(p['S'][k]),s[k]) for k in p['S'] if is_hold(k)}
    return dict(fit_rmse=rmse(fit),hold_rmse=rmse(hold),pairs={k:v for k,v in {**fit,**hold}.items() if k.startswith('pair')},params=dict(zip(P,map(float,th))))

def log(msg): print(msg,flush=True)

if __name__=='__main__':
    step=sys.argv[1]; R=json.load(open(f'{H}/fit4_results.json')) if os.path.exists(f'{H}/fit4_results.json') else {}
    save=lambda: json.dump(R,open(f'{H}/fit4_results.json','w'),indent=1)
    rng=np.random.default_rng(42)
    if step=='2':   # optimizer convergence, no timing state, sister machine
        p=setup('Pooled_SM'); runs=[]
        for r in range(3):
            st=None if r==0 else rng.uniform(0.1,0.9,len(P)-2)
            th,f=cma_fit([p],timing=False,maxfev=600,seed=r,start=st); ev=evaluate(th,p,timing=False); runs.append(dict(loss=f,**ev))
            log(f"step2 SM CMA run {r}: loss {f:.3f} fit {ev['fit_rmse']:.3f} hold {ev['hold_rmse']:.3f}  rho_fast {ev['params']['rho_fast']:.2f} s_d {ev['params']['s_d']:.2f} nu_d {np.exp(ev['params']['log_nu_d']):.1f} s_inh_a {ev['params']['s_inh_a']:.3f}")
        R['step2']=runs; save()
    if step=='3':   # timing state, sister machine; 2 runs with and without
        p=setup('Pooled_SM'); R['step3']={}
        for timing in (False,True):
            th,f=cma_fit([p],timing=timing,maxfev=700,seed=10); ev=evaluate(th,p,timing=timing); R['step3']['timing' if timing else 'none']=dict(loss=f,**ev); save()
            pr=ev['pairs']; g=lambda k:f"{pr[k][0]:.2f}/{pr[k][1]:.2f}" if k in pr else '—'
            log(f"step3 timing={timing}: loss {f:.3f} fit {ev['fit_rmse']:.3f} hold {ev['hold_rmse']:.3f} | tau SL g0 {g('pairG0SL_generationtime')} g1 {g('pairG1SL_generationtime')} g2 {g('pairG2SL_generationtime')} NL g0 {g('pairG0NL_generationtime')} | s_t {ev['params']['s_t']:.3f} rho_t {ev['params']['rho_t']:.2f}")
    if step=='4':   # pooling within a lab: Tanouchi group, separate vs pooled, 2 runs each
        names=['MC4100_25C (Tanouchi 2015)','MC4100_27C (Tanouchi 2015)','MC4100_37C (Tanouchi 2015)']; preps=[setup(d) for d in names]
        STRUCT=['s_env_a','s_env_d','s_lin_d','s_lin_f','s_inh_a','rho_slow']; R['step4']={'separate':{},'pooled':[]}
        for d,p in zip(names,preps):
            R['step4']['separate'][d]=[]
            for r in range(2):
                st=None if r==0 else rng.uniform(0.1,0.9,len(P))
                th,f=cma_fit([p],maxfev=500,seed=20+r,start=st); ev=evaluate(th,p); R['step4']['separate'][d].append(dict(loss=f,**ev)); save()
                log(f"step4 separate {EXPS[d]} run {r}: loss {f:.3f} hold {ev['hold_rmse']:.3f} struct "+' '.join(f"{q}={ev['params'][q]:.3f}" for q in STRUCT))
        for r in range(2):
            st=None if r==0 else rng.uniform(0.1,0.9,len(P))
            th,f=cma_fit(preps,maxfev=600,seed=30+r,start=st); evs={EXPS[d]:evaluate(th,p) for d,p in zip(names,preps)}
            R['step4']['pooled'].append(dict(loss=f,evals={k:{'fit_rmse':v['fit_rmse'],'hold_rmse':v['hold_rmse']} for k,v in evs.items()},params=dict(zip(P,map(float,th))))); save()
            log(f"step4 pooled run {r}: loss {f:.3f} hold "+' '.join(f"{k} {v['hold_rmse']:.3f}" for k,v in evs.items())+" struct "+' '.join(f"{q}={th[P.index(q)]:.3f}" for q in STRUCT))
    if step=='wang':   # full new model on Wang 0525, 2 runs (convergence check on a long-lineage experiment)
        p=setup('20090525_E_coli_MG1655_(CGSC_6300)_Wang2010'); R['wang']=[]
        for r in range(2):
            st=None if r==0 else rng.uniform(0.1,0.9,len(P))
            th,f=cma_fit([p],maxfev=500,seed=40+r,start=st); ev=evaluate(th,p); R['wang'].append(dict(loss=f,**ev)); save()
            log(f"wang run {r}: loss {f:.3f} fit {ev['fit_rmse']:.3f} hold {ev['hold_rmse']:.3f} rho_fast {ev['params']['rho_fast']:.2f} s_d {ev['params']['s_d']:.2f} nu_d {np.exp(ev['params']['log_nu_d']):.1f} s_t {ev['params']['s_t']:.3f}")
