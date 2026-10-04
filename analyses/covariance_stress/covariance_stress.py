from __future__ import annotations
"""Locked covariance/error stress grid for the historical proportional-scale model.

Two perturbation axes are varied independently and then jointly:
  k_u: common multiplier on all quoted historical standard uncertainties;
  alpha: correlation-strength multiplier, with R' = I + alpha(R-I).
The same perturbed covariance is used to refit the historical lambda-only model.
A separate NIST quoted-uncertainty scale check is reported at alpha=1, holding
historical covariance at its nominal form.
"""
from pathlib import Path
import json, math, sys
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import chi2

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from g_metrology.data import load_yaml, active_covariance_from_dataset

D=load_yaml(ROOT/'data/codata_2018_g.yaml'); N=load_yaml(ROOT/'data/nist2026_configurations.yaml')
SCALE=1e-11
y=D.y/SCALE; yn=N.y/SCALE
S=active_covariance_from_dataset(D)/(SCALE*SCALE); SN=active_covariance_from_dataset(N)/(SCALE*SCALE)

def perturbed_cov(C,k,alpha):
    sd=np.sqrt(np.diag(C)); R=C/np.outer(sd,sd); R=np.asarray(R,float); np.fill_diagonal(R,1.0)
    Rp=np.eye(len(R))+alpha*(R-np.eye(len(R)))
    return (k*k)*np.outer(sd,sd)*Rp

def fit_lambda(C):
    one = np.ones(len(y))

    def obj(logl):
        lam = math.exp(float(logl))
        V = lam * lam * C
        sign, ld = np.linalg.slogdet(V)
        if sign <= 0:
            return 1e100
        Vi = np.linalg.inv(V)
        mu = float(one @ Vi @ y / (one @ Vi @ one))
        r = y - mu * one
        return .5 * (ld + r @ Vi @ r)

    r = minimize_scalar(
        obj,
        bounds=(-3, 3),
        method='bounded',
        options={'xatol': 1e-12},
    )

    lam = math.exp(float(r.x))

    # Nominal, pre-inflation GLS quantities.
    Cinv = np.linalg.inv(C)
    a = float(one @ Cinv @ one)
    mu = float(one @ Cinv @ y / a)
    rvec = y - mu * one
    q_nominal = float(rvec @ Cinv @ rvec)
    birge = math.sqrt(q_nominal / (len(y) - 1))

    return lam, mu, q_nominal, birge, a

ks=(0.8,0.9,1.0,1.1,1.2)
alphas=(0.0,0.5,1.0,1.25,1.5)
rows=[]
for k in ks:
  for a in alphas:
    C = perturbed_cov(S, k, a)
    lam, mu, q_nominal, birge, a_mean = fit_lambda(C)

    # Exact GLS variance of the historical mean under the perturbed
    # reported covariance C is 1/(1^T C^{-1} 1). Under the proportional
    # scale model it is multiplied by lambda^2.
    V = lam * lam * SN + (lam * lam / a_mean) * np.ones((4, 4))

    d = yn - mu
    Q = float(d @ np.linalg.solve(V, d))
    p = float(chi2.sf(Q, 4))

    rows.append({
        'u_scale': k,
        'corr_scale': a,
        'lambda_mle': lam,
        'mu_scaled': mu,
        'q_nominal': q_nominal,
        'historical_birge_nominal': birge,
        'nist_Q_gaussian': Q,
        'nist_tail_chi2': p,
    })


# NIST quoted-error scale sensitivity, nominal historical covariance.
lam0, mu0, _, _, a0 = fit_lambda(S)
nist_rows = []

for kn in ks:
    Cn = perturbed_cov(SN, kn, 1.0)
    V = lam0 * lam0 * Cn + (lam0 * lam0 / a0) * np.ones((4, 4))

    d = yn - mu0
    Q = float(d @ np.linalg.solve(V, d))
    p = float(chi2.sf(Q, 4))

    nist_rows.append({
        'u_scale': kn,
        'nist_Q_gaussian': Q,
        'nist_tail_chi2': p,
    })

out={'description':'Full 5x5 historical quoted-error/correlation stress grid plus 5-point NIST quoted-error scale sensitivity.',
     'historical_u_scale_grid':list(ks),'correlation_scale_grid':list(alphas),
     'historical_grid':rows,
     'nist_u_scale_sensitivity_nominal_history':nist_rows,
     'nominal_reference':{'lambda_mle':lam0,'mu_scaled':mu0},
     'notes':[
       'Correlation scaling is R_prime = I + alpha*(R-I); alpha=0 removes reported off-diagonal correlations and alpha=1 is nominal.',
       'The historical grid tests whether the learned proportional scale is qualitatively stable to plausible covariance perturbations.',
       'The stress-grid NIST Q values use the exact historical GLS mean-variance term under the proportional Gaussian model and a plug-in chi-square reference; they are robustness screens, not the primary Student-t posterior predictive statistic.'
     ]}
( ROOT/'results'/'machine_readable'/'covariance_stress.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
