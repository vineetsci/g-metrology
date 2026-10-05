from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import halfcauchy, lognorm, expon, norm
from .canonical import total_scale, mvt_logpdf
from .config import Priors, DEFAULT_PRIORS
DEFAULT_PYMC_PRIORS = DEFAULT_PRIORS

def log_prior(mu,tau,lam,nu,priors: Priors, fixed_mu=None):
    lp=0.0
    if fixed_mu is None: lp += norm.logpdf(mu,0,priors.mu_sd)
    if tau<=0 or lam<=0 or nu<=2: return -np.inf
    lp += halfcauchy.logpdf(tau,scale=priors.tau_beta) + np.log(tau)
    lp += lognorm.logpdf(lam,s=priors.log_lambda_sd,scale=np.exp(priors.log_lambda_mu)) + np.log(lam)
    nu2=nu-2
    lp += expon.logpdf(nu2,scale=1/priors.nu_minus2_rate)+np.log(nu2)
    return float(lp)

def log_posterior(params, y, cov, priors, fixed_mu=None):
    mu=params.mu if fixed_mu is None else fixed_mu
    lp=log_prior(mu,params.tau,params.lam,params.nu,priors,fixed_mu)
    if not np.isfinite(lp): return -np.inf
    return lp + mvt_logpdf(y,np.full(len(y),mu),total_scale(cov,params.tau,params.lam),params.nu)


# PyMC builders are kept here as the canonical model API so external analysis
# scripts can import the same model specification used by the reference backend.
def build_student_t_hbma(G_std, scaled_cov, std_ctx):
    import pymc as pm
    import pytensor.tensor as pt
    y = np.asarray(G_std, dtype=float)
    S = np.asarray(scaled_cov, dtype=float)
    if S.shape != (y.size, y.size):
        raise ValueError(f"scaled_cov must have shape {(y.size, y.size)}, got {S.shape}")
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=DEFAULT_PYMC_PRIORS.mu_sd)
        tau = pm.HalfCauchy("tau", beta=DEFAULT_PYMC_PRIORS.tau_beta)
        lam = pm.Lognormal("lambda", mu=DEFAULT_PYMC_PRIORS.log_lambda_mu, sigma=DEFAULT_PYMC_PRIORS.log_lambda_sd)
        nu = pm.Deterministic("nu", 2.0 + pm.Exponential("nu_minus2", lam=DEFAULT_PYMC_PRIORS.nu_minus2_rate))
        scale = pm.Deterministic("Sigma", lam**2 * pt.as_tensor_variable(S) + tau**2 * pt.eye(y.size))
        pm.MvStudentT("obs", nu=nu, mu=pt.ones(y.size) * mu, scale=scale, observed=y)
        pm.Deterministic("mu_g", mu * std_ctx.scale_si + std_ctx.mean_si)
        pm.Deterministic("tau_g", tau * std_ctx.scale_si)
    return model

def build_student_t_fixed_mu(G_std, scaled_cov, std_ctx, mu_fixed_std):
    import pymc as pm
    import pytensor.tensor as pt
    y = np.asarray(G_std, dtype=float)
    S = np.asarray(scaled_cov, dtype=float)
    mu_fixed_std = float(mu_fixed_std)
    if S.shape != (y.size, y.size):
        raise ValueError(f"scaled_cov must have shape {(y.size, y.size)}, got {S.shape}")
    with pm.Model() as model:
        tau = pm.HalfCauchy("tau", beta=DEFAULT_PYMC_PRIORS.tau_beta)
        lam = pm.Lognormal("lambda", mu=DEFAULT_PYMC_PRIORS.log_lambda_mu, sigma=DEFAULT_PYMC_PRIORS.log_lambda_sd)
        nu = pm.Deterministic("nu", 2.0 + pm.Exponential("nu_minus2", lam=DEFAULT_PYMC_PRIORS.nu_minus2_rate))
        scale = pm.Deterministic("Sigma", lam**2 * pt.as_tensor_variable(S) + tau**2 * pt.eye(y.size))
        pm.MvStudentT("obs", nu=nu, mu=pt.ones(y.size) * mu_fixed_std, scale=scale, observed=y)
        mu_fixed_si = mu_fixed_std * std_ctx.scale_si + std_ctx.mean_si
        pm.Deterministic("mu_g", pt.as_tensor_variable(mu_fixed_si, dtype="float64"))
        pm.Deterministic("tau_g", tau * std_ctx.scale_si)
    return model
