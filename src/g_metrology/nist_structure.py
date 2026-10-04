from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

@dataclass(frozen=True)
class CovarianceStructureFit:
    name: str
    mu_si: float
    components_si: dict[str, float]
    loglik: float
    aic: float
    success: bool


def _fit(y, C, Zs: dict[str, np.ndarray], name: str):
    y = np.asarray(y, float) / 1e-14
    C = np.asarray(C, float) / 1e-28
    n = len(y)
    mats = {k: np.asarray(z, float) @ np.asarray(z, float).T for k, z in Zs.items()}

    def unpack(x):
        vals = {k: float(np.exp(x[i])) for i, k in enumerate(mats)}
        return vals

    def profiled(x):
        vals = unpack(x)
        S = C.copy()
        for k, v in vals.items():
            S += v*v*mats[k]
        inv1 = np.linalg.solve(S, np.ones(n))
        invy = np.linalg.solve(S, y)
        mu = float(np.sum(invy) / np.sum(inv1))
        r = y - mu
        sign, ld = np.linalg.slogdet(S)
        if sign <= 0:
            return 1e100, mu, S
        ll = -0.5 * (n*np.log(2*np.pi) + ld + r @ np.linalg.solve(S, r))
        return -ll, mu, S

    starts = [np.log(np.array([0.5]*len(mats))), np.log(np.array([1.0]*len(mats))), np.log(np.array([2.0]*len(mats)))]
    best = None
    for x0 in starts:
        r = minimize(lambda x: profiled(x)[0], x0, method='L-BFGS-B', bounds=[(-12, 5)]*len(mats), options={'maxiter': 50000, 'ftol':1e-12, 'gtol':1e-9})
        if best is None or r.fun < best.fun:
            best = r
    negll, mu, _ = profiled(best.x)
    vals = unpack(best.x)
    k = 1 + len(mats)
    ll = -negll
    return CovarianceStructureFit(name, mu*1e-14, {k: v*1e-14 for k,v in vals.items()}, float(ll), float(-2*ll+2*k), bool(best.success))


def fit_nist_structures(y, C):
    n=4
    Z_common=np.ones((n,1))
    Z_material=np.array([[1],[1],[0],[0]],float)
    Z_mode=np.array([[1],[0],[1],[0]],float)
    Z_id=np.eye(n)
    return [
        _fit(y,C,{'delta':Z_id},'independent_dark'),
        _fit(y,C,{'material':Z_material},'material_effect'),
        _fit(y,C,{'mode':Z_mode},'mode_effect'),
        _fit(y,C,{'material':Z_material,'mode':Z_mode},'material_plus_mode'),
        _fit(y,C,{'material':Z_material,'mode':Z_mode,'residual':Z_id},'material_mode_residual'),
        _fit(y,C,{'common':Z_common},'common_mean_random_effect'),
    ]
