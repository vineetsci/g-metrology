from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
import math
from numba import njit
from .inference import PosteriorDraws

@njit
def _log_mvt(y, mu, cov, nu):
    n=y.shape[0]
    diff=y-mu
    sign, logdet=np.linalg.slogdet(cov)
    if sign<=0: return -1e300
    sol=np.linalg.solve(cov,diff)
    q=0.0
    for i in range(n): q += diff[i]*sol[i]
    return math.lgamma((nu+n)/2)-math.lgamma(nu/2)-0.5*(n*np.log(nu*np.pi)+logdet)-(nu+n)/2*np.log1p(q/nu)

@njit
def _lp(z,y,cov,mu_sd,tau_beta,llm,lls,nu_rate,fixed_mu):
    k=0
    mu=0.0
    if fixed_mu<1e98:
        mu=fixed_mu
    else:
        mu=z[0]; k=1
        lp=-0.5*(mu/mu_sd)**2-np.log(mu_sd*np.sqrt(2*np.pi))
    lt=z[k]; ll=z[k+1]; lnu=z[k+2]
    tau=np.exp(lt); lam=np.exp(ll); nu=2.0+np.exp(lnu)
    if tau<=0 or lam<=0: return -1e300
    # Half-Cauchy prior with transform Jacobian.
    lp += np.log(2.0/(np.pi*tau_beta*(1+(tau/tau_beta)**2))) + lt
    # Lognormal density in lambda times Jacobian d lambda/d log lambda = lambda: the 1/lambda terms cancel.
    lp += -0.5*((ll-llm)/lls)**2-np.log(lls*np.sqrt(2*np.pi))
    # Exponential prior on nu-2 plus Jacobian.
    nu2=np.exp(lnu)
    lp += np.log(nu_rate)-nu_rate*nu2+lnu
    S=np.empty_like(cov)
    for i in range(cov.shape[0]):
        for j in range(cov.shape[1]):
            S[i,j]=lam*lam*cov[i,j]
            if i==j: S[i,j]+=tau*tau
    mean=np.empty_like(y)
    for i in range(y.shape[0]): mean[i]=mu
    return lp+_log_mvt(y,mean,S,nu)


def _log_prior_and_lp(z,y,cov,priors,fixed_mu):
    return float(_lp(z,y,cov,priors.mu_sd,priors.tau_beta,priors.log_lambda_mu,priors.log_lambda_sd,priors.nu_minus2_rate,1e99 if fixed_mu is None else fixed_mu))

def _map(y,cov,priors,fixed_mu):
    d=4 if fixed_mu is None else 3
    z=np.zeros(d)
    if fixed_mu is None: z[0]=np.mean(y)
    k=1 if fixed_mu is None else 0; z[k]=np.log(.5); z[k+1]=0.; z[k+2]=np.log(18.)
    def f(x):
        v=_log_prior_and_lp(x,y,cov,priors,fixed_mu); return -v if np.isfinite(v) else 1e100
    r=minimize(f,z,method='Nelder-Mead',options={'maxiter':30000,'xatol':1e-8,'fatol':1e-8})
    return r

def metropolis(y,cov,priors,fixed_mu=None,seed=0,n_chains=4,n_draws=5000,burn=2500,return_logpost=True):
    # Compile JIT path once.
    _= _lp(np.zeros(4 if fixed_mu is None else 3), np.asarray(y,float), np.asarray(cov,float), priors.mu_sd,priors.tau_beta,priors.log_lambda_mu,priors.log_lambda_sd,priors.nu_minus2_rate,1e99 if fixed_mu is None else fixed_mu)
    mode=_map(np.asarray(y,float),np.asarray(cov,float),priors,fixed_mu).x; d=len(mode)
    arr=np.empty((n_chains,n_draws,d)); accepts=[]
    for c in range(n_chains):
        rng=np.random.default_rng(seed+7919*c); x=mode+rng.normal(0,.03,d); cur=_log_prior_and_lp(x,y,cov,priors,fixed_mu); pcov=np.eye(d)*.03**2; hist=[]; acc=0
        for it in range(burn+n_draws):
            prop=x+rng.multivariate_normal(np.zeros(d),pcov); val=_log_prior_and_lp(prop,y,cov,priors,fixed_mu)
            if np.log(rng.random())<val-cur: x=prop; cur=val; acc+=1
            if it<burn:
                hist.append(x.copy())
                if (it+1)%100==0 and len(hist)>=100:
                    C=np.cov(np.asarray(hist).T)
                    if np.all(np.isfinite(C)): pcov=(2.38**2/d)*(C+1e-5*np.eye(d))
            else: arr[c,it-burn]=x
        accepts.append(acc/(burn+n_draws))
    flat=arr.reshape(-1,d); k=0; sm={}; ch={}
    if fixed_mu is None: sm['mu']=flat[:,0]; ch['mu']=arr[:,:,0]; k=1
    sm['tau']=np.exp(flat[:,k]); ch['tau']=np.exp(arr[:,:,k]);
    sm['lambda']=np.exp(flat[:,k+1]); ch['lambda']=np.exp(arr[:,:,k+1]);
    sm['nu']=2+np.exp(flat[:,k+2]); ch['nu']=2+np.exp(arr[:,:,k+2]);
    if return_logpost:
        lps=np.array([_log_prior_and_lp(z,y,cov,priors,fixed_mu) for z in flat])
    else:
        lps=np.empty(0, dtype=float)
    return PosteriorDraws(sm,lps,float(np.mean(accepts)),ch)