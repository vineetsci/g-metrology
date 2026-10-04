from __future__ import annotations
import numpy as np
from scipy.optimize import minimize
from scipy.stats import multivariate_normal, multivariate_t, t

SCALE = 1e-14


def _fit_generic(y, cov, *, fixed_mu=None, fit_lambda=True, fit_delta=True, kind='normal', nu=30.0):
    y=np.asarray(y,float); C=np.asarray(cov,float); n=len(y)
    ys=y/SCALE; Cs=C/SCALE**2
    z0=[]
    if fixed_mu is None: z0.append(np.mean(ys))
    if fit_lambda: z0.append(0.0)
    if fit_delta: z0.append(np.log(0.5))

    def unpack(z):
        k=0
        if fixed_mu is None:
            mu=float(z[k]); k+=1
        else:
            mu=float(fixed_mu/SCALE)
        lam=float(np.exp(z[k])) if fit_lambda else 1.0
        if fit_lambda: k+=1
        delta=float(np.exp(z[k])) if fit_delta else 0.0
        return mu,lam,delta

    def loglik(z):
        mu,lam,delta=unpack(z)
        S=lam**2*Cs+delta**2*np.eye(n)
        if kind=='normal': return float(multivariate_normal.logpdf(ys, mean=np.full(n,mu), cov=S))
        return float(multivariate_t.logpdf(ys, loc=np.full(n,mu), shape=S, df=nu))

    if not z0:
        x=np.array([])
        success=True
        message='analytic fixed-noise-free model'
    else:
        r=minimize(lambda z:-loglik(z), z0, method='Nelder-Mead', options={'maxiter':50000,'xatol':1e-11,'fatol':1e-11})
        x=r.x; success=bool(r.success); message=str(r.message)
    mu,lam,delta=unpack(x)
    ll=loglik(x)
    kpar=len(z0)
    aic=-2*ll+2*kpar
    aicc=aic+2*kpar*(kpar+1)/(n-kpar-1) if n-kpar-1>0 else np.nan
    return {'mu_si':mu*SCALE,'lambda':lam,'delta_si':delta*SCALE,'loglik':ll,'aic':aic,'aicc':aicc,'n_params':kpar,'success':success,'message':message}


def fit_normal(y,cov,**kwargs):
    return _fit_generic(y,cov,kind='normal',**kwargs)


def fit_student(y,cov,nu=30.0,**kwargs):
    return _fit_generic(y,cov,kind='student',nu=nu,**kwargs)
