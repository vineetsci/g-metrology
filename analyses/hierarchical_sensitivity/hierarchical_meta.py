from __future__ import annotations
import json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
from scipy.optimize import minimize
from numba import njit
ACTIVE_COVARIANCE_K = 1.0  # source uncertainties retain CODATA expansion; analysis intentionally resets to k=1
from g_metrology.data import load_yaml,covariance_from_dataset
from g_metrology.config import DEFAULT_PRIORS
from g_metrology.standardization import StandardizationContext
DATA=ROOT/'data/codata_2018_g.yaml'; OUT=ROOT/'results'/'machine_readable'; OUT.mkdir(parents=True,exist_ok=True)
@njit
def _logpost_numba(z,y,S,tau_beta,mu_sd):
    mu=z[0]; tau=np.exp(z[1]); lam=np.exp(z[2]); n=y.shape[0]
    if tau<=0 or lam<=0:return -1e300
    C=np.empty((n,n)); t2=tau*tau; l2=lam*lam
    for i in range(n):
      for j in range(n): C[i,j]=l2*S[i,j] + (t2 if i==j else 0.0)
    sign,ld=np.linalg.slogdet(C)
    if sign<=0:return -1e300
    d=y-mu; sol=np.linalg.solve(C,d); q=d@sol
    return -0.5*(mu*mu/(mu_sd*mu_sd)+np.log(mu_sd*mu_sd*2*np.pi)+ld+q) + np.log(2.0/(np.pi*tau_beta*(1+(tau/tau_beta)**2)))+z[1] + (-0.5*z[2]*z[2]-0.5*np.log(2*np.pi))+z[2]
def logpost(z,y,S,tau_beta=1.0,mu_sd=float(DEFAULT_PRIORS.mu_sd)): return float(_logpost_numba(np.asarray(z),np.asarray(y),np.asarray(S),tau_beta,mu_sd))
def map_est(y,S,tau_beta=1.0,mu_sd=float(DEFAULT_PRIORS.mu_sd)):
 r=minimize(lambda z:-logpost(z,y,S,tau_beta,mu_sd),[np.mean(y),np.log(.5),0],method='Nelder-Mead',options={'maxiter':8000,'xatol':1e-7}); return r.x
def mh(y,S,seed=20260908,chains=4,n_draws=4000,burn=2000,tau_beta=1.0,mu_sd=float(DEFAULT_PRIORS.mu_sd)):
 _=logpost(np.zeros(3),y,S,tau_beta,mu_sd); mode=map_est(y,S,tau_beta,mu_sd); out=[]
 for c in range(chains):
  rng=np.random.default_rng(seed+7919*c); x=mode+rng.normal(0,.03,3); cur=logpost(x,y,S,tau_beta,mu_sd); propcov=np.eye(3)*.03**2; hist=[]; arr=np.empty((n_draws,3)); acc=0
  for it in range(burn+n_draws):
   prop=x+rng.multivariate_normal(np.zeros(3),propcov); val=logpost(prop,y,S,tau_beta,mu_sd)
   if np.log(rng.random())<val-cur: x=prop; cur=val; acc+=1
   if it<burn:
    hist.append(x.copy())
    if (it+1)%200==0:
     C=np.cov(np.asarray(hist).T)
     if np.all(np.isfinite(C)): propcov=(2.38**2/3)*(C+1e-5*np.eye(3))
   else: arr[it-burn]=x
  out.append(arr)
 return np.stack(out)
def posterior_bias_sample(theta,y,S):
 out=[]; sd=[]
 for z in theta:
  mu=z[0]; tau=np.exp(z[1]); lam=np.exp(z[2]); R=lam*lam*S; Ri=np.linalg.inv(R); V=np.linalg.inv(Ri+np.eye(len(y))/(tau*tau)); m=V@(Ri@(y-mu)); out.append(m); sd.append(np.sqrt(np.diag(V)))
 return np.asarray(out),np.asarray(sd)
D=load_yaml(DATA); ctx=StandardizationContext.from_data(D.y); y=ctx.transform(D.y); S=ctx.covariance_to_std(covariance_from_dataset(D,ACTIVE_COVARIANCE_K))
chains=mh(y,S); raw=chains.reshape(-1,3); tau=np.exp(raw[:,1]); lam=np.exp(raw[:,2]); mu=raw[:,0]
rng=np.random.default_rng(3); sub=raw[rng.choice(len(raw),size=min(2500,len(raw)),replace=False)]; bm,bs=posterior_bias_sample(sub,y,S)
def rh(ch):
 out={}
 for j,nm in enumerate(['mu','log_tau','log_lambda']):
  z=ch[:,:,j]; m=z.shape[1]//2; z=np.concatenate([z[:,:m],z[:,m:2*m]],axis=0); W=z.var(1,ddof=1).mean(); B=m*z.mean(1).var(ddof=1); out[nm]=float(np.sqrt(((m-1)/m*W+B/m)/W))
 return out
rhats=rh(chains)
res={'model':'Bayesian hierarchical Gaussian random-effects sensitivity analysis','equation':'y_i = mu + b_i + e_i; b_i~N(0,tau^2); e~N(0,lambda^2 S_reported)','n':16,'posterior_draws':len(raw),'rhat':rhats,'convergence_flags':{'rhat_le_1.01':{k: bool(v <= 1.01) for k,v in rhats.items()},'rhat_le_1.05':{k: bool(v <= 1.05) for k,v in rhats.items()},'any_above_1.01':any(v > 1.01 for v in rhats.values()),'note':'R-hat above 1.01 indicates a mild convergence warning for that parameter; above 1.05 is a stronger warning.'},'mu_si_median':float(np.median(mu)*ctx.scale_si+ctx.mean_si),'mu_si_ci95':[float(np.quantile(mu,.025)*ctx.scale_si+ctx.mean_si),float(np.quantile(mu,.975)*ctx.scale_si+ctx.mean_si)],'tau_si_median':float(np.median(tau)*ctx.scale_si),'tau_si_ci95':[float(np.quantile(tau,.025)*ctx.scale_si),float(np.quantile(tau,.975)*ctx.scale_si)],'lambda_median':float(np.median(lam)),'lambda_ci95':[float(np.quantile(lam,.025)),float(np.quantile(lam,.975))],'shrinkage':{}}
mu_med=np.median(mu)*ctx.scale_si+ctx.mean_si
for i,label in enumerate(D.labels):
 obs=D.y[i]-mu_med; bmed=np.median(bm[:,i])*ctx.scale_si; blo=np.quantile(bm[:,i],.025)*ctx.scale_si; bhi=np.quantile(bm[:,i],.975)*ctx.scale_si; sdmed=np.median(bs[:,i])*ctx.scale_si
 res['shrinkage'][label]={'observed_minus_mu_si':float(obs),'posterior_bias_median_si':float(bmed),'posterior_bias_ci95_si':[float(blo),float(bhi)],'shrinkage_fraction':float(abs(bmed)/(abs(obs)+1e-30)),'bias_over_posterior_sd':float(bmed/sdmed)}

sens={}
for beta in [0.5,1.0,2.0]:
    ch,_=mh(y,S,seed=20260908+int(beta*100),chains=2,n_draws=1200,burn=600,tau_beta=beta)
    rr=ch.reshape(-1,3); tt=np.exp(rr[:,1]); ll=np.exp(rr[:,2])
    sens[str(beta)]={'tau_si_median':float(np.median(tt)*ctx.scale_si),'tau_si_ci95':[float(np.quantile(tt,.025)*ctx.scale_si),float(np.quantile(tt,.975)*ctx.scale_si)],'lambda_median':float(np.median(ll))}
res['prior_sensitivity_tau_beta']=sens
(OUT/'hierarchical_meta_results.json').write_text(json.dumps(res,indent=2)); print(json.dumps(res,indent=2))
