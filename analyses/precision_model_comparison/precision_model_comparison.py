from __future__ import annotations
import json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from scipy.optimize import minimize
ACTIVE_COVARIANCE_K = 1.0  # Source metadata retain the CODATA expansion factor; the active fit uses k=1.
from g_metrology.data import covariance_diagnostics
from g_metrology.data import load_yaml,active_covariance_from_dataset
from g_metrology.standardization import StandardizationContext
OUT=ROOT/'results'/'machine_readable'; OUT.mkdir(parents=True,exist_ok=True)
D=load_yaml(ROOT/'data/codata_2018_g.yaml'); covariance_diagnostics(active_covariance_from_dataset(D), name='CODATA-2018 active covariance'); ctx=StandardizationContext.from_data(D.y); y=ctx.transform(D.y); S=ctx.covariance_to_std(active_covariance_from_dataset(D)); n=len(y)
# Fit Gaussian covariance models in standardized units. delta is additive SD floor in same units as y.
def fit(model):
    if model=='prop':
        def unpack(z): return np.exp(z[0]),0.0,1.0
        x0=[0.5]
    elif model=='floor':
        def unpack(z): return 0.0,np.exp(z[0]),1.0
        x0=[0.5]
    elif model=='both':
        def unpack(z): return np.exp(z[0]),np.exp(z[1]),1.0
        x0=[0.0,-1.0]
    elif model=='power':
        # covariance uses diag(u^(2p)) scaled by a^2, with S transformed back to nominal diagonal only; keep correlation matrix.
        u=np.sqrt(np.diag(S))
        R=S/np.outer(u,u)
        def unpack(z): return np.exp(z[0]),0.0,z[1]
        x0=[0.0,1.0]
    def nll(z):
        a,delta,p=unpack(z)
        if model=='prop': C=a*a*S
        elif model=='floor': C=S+delta*delta*np.eye(n)
        elif model=='both': C=a*a*S+delta*delta*np.eye(n)
        else:
            # geometric power model: relative covariance preserves reported correlation matrix.
            u=np.sqrt(np.diag(S)); u0=float(np.median(u)); up=(u/u0)**p; C=np.diag(up)@R@np.diag(up)*((a*u0)**2)
        # free mean MLE
        one=np.ones(n); Ci=np.linalg.inv(C); mu=(one@Ci@y)/(one@Ci@one)
        d=y-mu; sign,ld=np.linalg.slogdet(C)
        if sign<=0:return 1e99
        return 0.5*(n*np.log(2*np.pi)+ld+d@Ci@d)
    r=minimize(nll,x0,method='Nelder-Mead',options={'maxiter':30000,'xatol':1e-9,'fatol':1e-9})
    k=len(r.x)+1; ll=-r.fun; bic=-2*ll+k*np.log(n); return {'model':model,'params':r.x.tolist(),'nll':float(r.fun),'loglik':float(ll),'k':k,'bic':float(bic)}
res=[fit(m) for m in ['prop','floor','both','power']]
# Formal power family: sigma_i(p)=a*u0*(u_i/u0)^p, where u0=median(u);
# this is dimensionally invariant and p=1 nests the proportional covariance model.
# BIC weights: w_i is proportional to exp(-0.5 * Delta BIC).
bics=np.array([r['bic'] for r in res]); w=np.exp(-0.5*(bics-bics.min())); w=w/w.sum()
for r,wi in zip(res,w): r['bic_weight']=float(wi)
# Parametric bootstrap for the boundary comparison prop vs both.
# Approximate LR boundary calibration under the proportional model with 500 parametric bootstrap replicates,
# preserving the reported covariance. The count is deliberately fixed and recorded in the output.
prop=next(r for r in res if r['model']=='prop'); a=np.exp(prop['params'][0]); one=np.ones(n); C=a*a*S; covariance_diagnostics(C, name='proportional covariance'); Ci=np.linalg.inv(C); mu=one@Ci@y/(one@Ci@one); rng=np.random.default_rng(20260908); M=500; lr=[]
for _ in range(M):
    yy=rng.multivariate_normal(np.ones(n)*mu,C)
    # Refit both covariance models on each bootstrap sample.
    def ll_for(model, z):
        if model=='prop': Cx=np.exp(2*z[0])*S
        else: Cx=np.exp(2*z[0])*S+np.exp(2*z[1])*np.eye(n)
        Cix=np.linalg.inv(Cx); mux=one@Cix@yy/(one@Cix@one); dd=yy-mux; sd,ld=np.linalg.slogdet(Cx); return -0.5*(n*np.log(2*np.pi)+ld+dd@Cix@dd)
    rp=minimize(lambda z:-ll_for('prop',z),[np.log(a)],method='Nelder-Mead'); rb=minimize(lambda z:-ll_for('both',z),[np.log(a),-3],method='Nelder-Mead'); lr.append(2*(ll_for('both',rb.x)-ll_for('prop',rp.x)))
lr=np.array(lr); observed=2*((next(r['loglik'] for r in res if r['model']=='both'))-(next(r['loglik'] for r in res if r['model']=='prop')))

for r in res:
    if r['model']=='prop': r['scale_lambda']=float(np.exp(r['params'][0]))
    elif r['model']=='floor': r['delta_std']=float(np.exp(r['params'][0]))
    elif r['model']=='both': r['scale_lambda']=float(np.exp(r['params'][0])); r['delta_std']=float(np.exp(r['params'][1]))
    elif r['model']=='power': r['scale_a']=float(np.exp(r['params'][0])); r['power_p']=float(r['params'][1])
out={'models':res,'bic_delta':{r['model']:float(r['bic']-bics.min()) for r in res},'observed_LR_both_vs_prop':float(observed),'bootstrap_LR_95':float(np.quantile(lr,.95)),'bootstrap_LR_99':float(np.quantile(lr,.99)),'bootstrap_p_add1':float((1+np.sum(lr>=observed))/(M+1)),'bootstrap_replicates':M,'bootstrap_seed':20260908,'bic_derived_relative_weights':{r['model']:r['bic_weight'] for r in res}}
(OUT/'precision_model_compare.json').write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
