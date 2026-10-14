"""Step 1 of the remaining fixes: measure the length-measurement noise instead of fitting it.
Within each cycle ln(length) vs time is a straight line. The scatter of the raw points around that line is the per-image
noise sigma_pt; the fitted end points x0 and x_tau inherit an error sigma_pt*sqrt(1/n + (t_end - t_mean)^2/Sxx).
That endpoint error is what the model's s_meas stands for, so it is fixed at this value per experiment."""
import numpy as np, pandas as pd, json, os, sys
from fit3 import EXPS, DATA_DIR
out={}
for ds,lab in EXPS.items():
    raw=pd.read_csv(f'{DATA_DIR}/{ds}/raw_data_all_in_one.csv'); pu=pd.read_csv(f'{DATA_DIR}/{ds}/ProcessedData/physical_units.csv')
    raw=raw[raw.length>0]
    key=[c for c in ('dataset','trap_ID','trace','lineage_ID') if c in raw.columns and c in pu.columns]
    for c in key:
        if c in ('trap_ID','lineage_ID'): raw[c]=raw[c].astype(int); pu=pu[pu[c].notna()]; pu[c]=pu[c].astype(int)
    pts=[]; ends=[]
    g=raw.groupby(key)
    for kv,c in pu.dropna(subset=['start_time','end_time']).sample(min(3000,len(pu)),random_state=0).iterrows():
        k=tuple(c[x] for x in key)
        try: r=g.get_group(k)
        except KeyError: continue
        seg=r[(r.time>=c.start_time-1e-6)&(r.time<=c.end_time+1e-6)]
        if len(seg)<5: continue
        t=seg.time.values; y=np.log(seg.length.values); b=np.polyfit(t,y,1); res=y-np.polyval(b,t)
        s=np.sqrt(np.sum(res**2)/(len(t)-2)); Sxx=np.sum((t-t.mean())**2)
        pts.append(s); ends.append(s*np.sqrt(1/len(t)+(t[-1]-t.mean())**2/Sxx))
    out[ds]=dict(label=lab,n_cycles=len(pts),sigma_point=float(np.median(pts)),sigma_endpoint=float(np.median(ends)))
    print(f"{lab:11s} cycles {len(pts):5d}  per-image noise {np.median(pts):.3f}  endpoint error (s_meas) {np.median(ends):.3f}",flush=True)
json.dump(out,open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'meas_noise.json'),'w'),indent=1)
