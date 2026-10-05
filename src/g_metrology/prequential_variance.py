from __future__ import annotations
import numpy as np
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
from .data import load_yaml, active_covariance_from_dataset

SCALE = 1e-14

def _fit_mode(y, cov, *, fixed_mu=None, kind="lambda", prior_scale=1.0):
    y = np.asarray(y, float)/SCALE; C = np.asarray(cov, float)/SCALE**2; n = len(y)
    mu0 = float(np.mean(y)) if fixed_mu is None else float(fixed_mu/SCALE)
    zparts = []; labels = []
    if fixed_mu is None:
        zparts.append(mu0); labels.append("mu")
    if kind in ("lambda", "both"):
        zparts.append(0.0); labels.append("log_lambda")
    if kind in ("dark", "both"):
        zparts.append(np.log(0.5)); labels.append("log_delta")
    z0 = np.asarray(zparts, float)
    def unpack(z):
        k = 0
        mu = mu0 if fixed_mu is not None else float(z[k]); k += (fixed_mu is None)
        lam = float(np.exp(z[k])) if kind in ("lambda", "both") else 1.0
        if kind in ("lambda", "both"): k += 1
        delta = float(np.exp(z[k])) if kind in ("dark", "both") else 0.0
        return mu, lam, delta, z
    def objective(z):
        mu, lam, delta, _ = unpack(z)
        S = lam**2*C + delta**2*np.eye(n)
        ll = multivariate_normal.logpdf(y, mean=np.full(n, mu), cov=S)
        lp = 0.0
        if kind in ("lambda", "both"):
            idx = labels.index("log_lambda"); lp += -0.5*(z[idx]/prior_scale)**2-np.log(prior_scale*np.sqrt(2*np.pi))
        if kind in ("dark", "both"):
            idx = labels.index("log_delta"); lp += -0.5*(z[idx]/prior_scale)**2-np.log(prior_scale*np.sqrt(2*np.pi))
        return -(ll + lp)
    r = minimize(objective, z0, method="Nelder-Mead", options={"maxiter":30000,"xatol":1e-10,"fatol":1e-10})
    mu, lam, delta, _ = unpack(r.x)
    return mu*SCALE, lam, delta*SCALE, float(-objective(r.x))

def _conditional_normal_score(y, C, train, hold, mu, lam, delta):
    y = np.asarray(y)/SCALE; C = np.asarray(C)/SCALE**2
    n = len(y); S = lam**2*C + delta**2*np.eye(n)
    A = S[np.ix_(train,train)]; B = S[np.ix_(train,hold)]; D = S[np.ix_(hold,hold)]
    yt = y[train]; mm = np.full(len(train), mu/SCALE); mh = np.full(len(hold), mu/SCALE)
    cond_mean = mh + B.T @ np.linalg.solve(A, yt-mm)
    cond_cov = D - B.T @ np.linalg.solve(A,B)
    return float(multivariate_normal.logpdf(y[hold], mean=cond_mean, cov=cond_cov))

def run_prequential_variance(data_path, *, start_train=7, codata=6.67430e-11):
    ds = load_yaml(data_path); y = ds.y; C = active_covariance_from_dataset(ds); years = ds.years.copy(); labels = ds.labels
    years = np.array([2014 if lab in ('UCI-14','BIPM-14') else int(yr) for lab,yr in zip(labels,years)])
    order = np.argsort(years, kind='stable'); y = y[order]; C = C[np.ix_(order,order)]; years = years[order]; labels = [labels[i] for i in order]
    rows = []
    models = [('free_lambda',None,'lambda'),('free_dark',None,'dark'),('free_both',None,'both')]
    for k in range(start_train, len(y)):
        tr = np.arange(k); ho = np.array([k]); row = {'train_end_year':int(years[k-1]),'holdout_year':int(years[k]),'holdout_label':labels[k]}
        for name, _, kind in models:
            mu,lam,delta,ll = _fit_mode(y[tr], C[np.ix_(tr,tr)], fixed_mu=None, kind=kind)
            row[name] = _conditional_normal_score(y,C,tr,ho,mu,lam,delta)
            row[name+'_lambda'] = lam; row[name+'_delta'] = delta
        for kind in ('lambda','dark','both'):
            mu,lam,delta,ll = _fit_mode(y[tr], C[np.ix_(tr,tr)], fixed_mu=codata, kind=kind)
            row[f'codata_{kind}'] = _conditional_normal_score(y,C,tr,ho,mu,lam,delta)
        rows.append(row)
    out = {'start_train':start_train,'inference_type':'plug-in conditional Gaussian prequential sensitivity','not_posterior_integrated':True,'rows':rows}
    for key in ('free_lambda','free_dark','free_both'):
        out[key+'_sum'] = float(sum(r[key] for r in rows))
    out['lambda_minus_dark'] = out['free_lambda_sum']-out['free_dark_sum']
    out['lambda_minus_both'] = out['free_lambda_sum']-out['free_both_sum']
    out['dark_minus_both'] = out['free_dark_sum']-out['free_both_sum']
    for kind in ('lambda','dark','both'):
        out[f'codata_{kind}_sum'] = float(sum(r[f'codata_{kind}'] for r in rows))
    return out
