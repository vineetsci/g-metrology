from __future__ import annotations
import numpy as np
from scipy.special import logsumexp
from .canonical import conditional_mvt, total_scale, mvt_logpdf

def conditional_holdout_draws(posterior,y,cov,train_idx,hold_idx,fixed_mu=None,method_codes=None):
    y=np.asarray(y,float); n=len(y); train_idx=np.asarray(train_idx,int); hold_idx=np.asarray(hold_idx,int)
    logs=[]
    for s in range(len(posterior.samples['tau'])):
        mu=float(fixed_mu if fixed_mu is not None else posterior.samples['mu'][s])
        if method_codes is not None and 'method_b' in posterior.samples:
            mf=np.full(n,mu)+posterior.samples['method_b'][s,method_codes]
        else: mf=np.full(n,mu)
        S=total_scale(cov,float(posterior.samples['tau'][s]),float(posterior.samples['lambda'][s])); nu=float(posterior.samples['nu'][s])
        A=S[np.ix_(train_idx,train_idx)]; B=S[np.ix_(train_idx,hold_idx)]; C=S[np.ix_(hold_idx,hold_idx)]
        loc,scale,df=conditional_mvt(y[train_idx],mf[train_idx],mf[hold_idx],A,C,B,nu)
        logs.append(mvt_logpdf(y[hold_idx],loc,scale,df))
    return np.asarray(logs)

def lppd(logs): return float(logsumexp(logs)-np.log(len(logs)))
