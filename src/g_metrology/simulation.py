from __future__ import annotations
import numpy as np
from .canonical import total_scale

def draw_prior(rng,priors):
    return {'mu':float(rng.normal(0,priors.mu_sd)), 'tau':float(abs(rng.standard_cauchy())*priors.tau_beta), 'lambda':float(np.exp(rng.normal(priors.log_lambda_mu,priors.log_lambda_sd))), 'nu':float(2+rng.exponential(1/priors.nu_minus2_rate))}

def sample_mvt(rng,mean,scale,nu):
    w=rng.gamma(shape=nu/2,scale=2/nu)
    L=np.linalg.cholesky(scale)
    return np.asarray(mean)+L@rng.normal(size=len(mean))/np.sqrt(w)

def generate_prior_predictive(rng,cov,priors):
    p=draw_prior(rng,priors); y=sample_mvt(rng,np.full(len(cov),p['mu']),total_scale(cov,p['tau'],p['lambda']),p['nu']); p['y']=y; return p
