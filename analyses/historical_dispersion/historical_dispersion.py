from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import chi2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from g_metrology.data import load_yaml, active_covariance_from_dataset

h = load_yaml(ROOT / 'data' / 'codata_2018_g.yaml')
y = h.y
u = h.u
year = h.years
N = len(y)
S = active_covariance_from_dataset(h)

def gls_fit(idx, trend=False):
    idx=np.array(idx,int); yy=y[idx]; SS=S[np.ix_(idx,idx)]; W=np.linalg.inv(SS)
    x=(year[idx]-2000)/10.0
    X=np.c_[np.ones(len(idx)),x] if trend else np.ones((len(idx),1))
    beta=np.linalg.solve(X.T@W@X, X.T@W@yy)
    se_beta = np.sqrt(np.diag(np.linalg.inv(X.T @ W @ X)))
    r=yy-X@beta; q=float(r@W@r); dof=int(len(idx)-X.shape[1]); p=float(chi2.sf(q,dof))
    lam=float(np.sqrt(q/len(idx)))
    birge=float(np.sqrt(q/dof))
    return {'beta_si': beta.tolist(), 'beta_se_si': se_beta.tolist(), 'q': q, 'dof': dof, 'p': p, 'lambda_mle': lam, 'birge_ratio': birge}

subsets={
 'all':range(N),
 'exclude_BIPM14_only':[i for i in range(N) if i != 10],
}
fits={k:gls_fit(v) for k,v in subsets.items()}

result={'fits':fits,'parameter_conventions':{
    'lambda_mle':'Gaussian proportional covariance-scale maximum-likelihood estimate: sqrt(Q/N).',
    'birge_ratio':'Metrology consistency/expansion diagnostic: sqrt(Q/nu), with nu=N-M adjusted degrees of freedom.'
}}
out=ROOT/'results'/'machine_readable'/'historical_dispersion.json'
out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,indent=2))
