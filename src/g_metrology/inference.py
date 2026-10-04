from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from .models import log_posterior

@dataclass
class PosteriorDraws:
    samples: dict[str,np.ndarray]
    logpost: np.ndarray
    acceptance: float
    chains: dict[str,np.ndarray]


def pack(mu,tau,lam,nu,fixed_mu=None):
    return np.array(([mu] if fixed_mu is None else [])+[np.log(tau),np.log(lam),np.log(nu-2)],float)

def unpack(z,fixed_mu=None):
    k=0
    mu=float(z[k]) if fixed_mu is None else float(fixed_mu); k += 0 if fixed_mu is not None else 1
    tau=float(np.exp(z[k])); lam=float(np.exp(z[k+1])); nu=2+float(np.exp(z[k+2]));
    return mu,tau,lam,nu

def map_fit(y,cov,priors,fixed_mu=None):
    y=np.asarray(y,float)
    x0=pack(float(np.mean(y)),.5,1.,20.,fixed_mu)
    def fun(z):
        mu,tau,lam,nu=unpack(z,fixed_mu)
        v=log_posterior(type('P',(),dict(mu=mu,tau=tau,lam=lam,nu=nu)),y,cov,priors,fixed_mu)
        return -v if np.isfinite(v) else 1e100
    r=minimize(fun,x0,method='Nelder-Mead',options={'maxiter':20000,'xatol':1e-9,'fatol':1e-8})
    if not np.isfinite(-r.fun): raise RuntimeError('MAP failed')
    return r

def propose_cov(samples, dim):
    if len(samples)<dim+5: return np.eye(dim)*0.03**2
    C=np.cov(np.asarray(samples).T)
    if np.ndim(C)==0: C=np.array([[float(C)]])
    return (2.38**2/max(dim,1))*(C+1e-6*np.eye(dim))

def _rhat(ch):
    a=np.asarray(ch,float)
    if a.ndim!=2 or a.shape[1]<20: return np.nan
    half=a.shape[1]//2; z=np.concatenate([a[:,:half],a[:,half:2*half]],axis=0); m=z.mean(1); W=z.var(1,ddof=1).mean(); B=half*m.var(ddof=1)
    return float(np.sqrt(((half-1)/half*W+B/half)/W)) if W>0 else np.nan

def rhat_table(p):
    return {k:_rhat(v) for k,v in p.chains.items()}
