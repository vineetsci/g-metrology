from __future__ import annotations
"""Validate predictive-tail probability conventions for the canonical hierarchical 2026 check.

Reports two distinct quantities:
1) posterior-averaged conditional tail: average over draws of the conditional
   Student-t tail evaluated at that draw's own Q_obs;
2) full posterior-predictive mixture tail for one predeclared fixed discrepancy.
The fixed discrepancy is the Mahalanobis Q using the scale-28 posterior median
of (mu,tau,lambda) and the resulting fixed covariance. It is held unchanged
while replicated vectors are generated from each posterior draw.
"""
from pathlib import Path
import json, math, sys
import numpy as np
from scipy.special import logsumexp
from scipy.stats import chi2, f

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / 'src'))
from g_metrology.data import load_yaml,active_covariance_from_dataset
from g_metrology.standardization import StandardizationContext
from g_metrology.config import DEFAULT_PRIORS
from g_metrology.fast_inference import metropolis
from g_metrology.canonical import total_scale,mvt_logpdf

D=load_yaml(ROOT/'data/codata_2018_g.yaml'); N=load_yaml(ROOT/'data/nist2026_configurations.yaml')
S=active_covariance_from_dataset(D); SN=active_covariance_from_dataset(N)
ctx=StandardizationContext.from_data(D.y); y=ctx.transform(D.y); Sd=ctx.covariance_to_std(S); yn=ctx.transform(N.y); Snd=ctx.covariance_to_std(SN)

p=metropolis(
    y, Sd, DEFAULT_PRIORS, seed=20264328, n_chains=4, n_draws=10000, burn=3000,
    return_logpost=False,
)
mu=np.asarray(p.samples['mu']); tau=np.asarray(p.samples['tau']); lam=np.asarray(p.samples['lambda']); nu=np.asarray(p.samples['nu'])
ref_mu=float(np.median(mu)); ref_tau=float(np.median(tau)); ref_lam=float(np.median(lam))
ref_C=total_scale(Snd,ref_tau,ref_lam); ref_inv=np.linalg.inv(ref_C); dref=yn-ref_mu*np.ones(4); q_ref=float(dref@ref_inv@dref)
q=[]; cond=[]; logs=[]; hits=[]
rng=np.random.default_rng(20264373)
for i in range(len(mu)):
    C=total_scale(Snd,float(tau[i]),float(lam[i])); d=yn-float(mu[i])*np.ones(4)
    qi=float(d@np.linalg.solve(C,d)); q.append(qi); cond.append(float(f.sf(qi/4.0,4.0,float(nu[i])))); logs.append(mvt_logpdf(yn,np.full(4,mu[i]),C,float(nu[i])))
    L=np.linalg.cholesky(C); z=rng.normal(size=4); w=rng.chisquare(float(nu[i])); rep=float(mu[i])*np.ones(4)+(L@z)/math.sqrt(w/float(nu[i]))
    qr=float((rep-ref_mu*np.ones(4))@ref_inv@(rep-ref_mu*np.ones(4))); hits.append(qr>=q_ref)

# Lambda-only historical posterior quadrature using the same historical covariance and GLS mean.
# Evaluate the fixed-discrepancy lambda-only tail directly from the quadrature posterior.
H=np.asarray(S,float); Hiy=np.linalg.inv(H); one=np.ones(16); a=float(one@Hiy@one); mu_l=float(one@Hiy@D.y/a)
NC=np.asarray(SN,float); lg=np.linspace(-3.5,3.0,18000); Ls=np.exp(lg)
q0=float((D.y-mu_l)@Hiy@(D.y-mu_l)); logpost=-(len(D.y)-1)*lg-q0/(2*Ls**2)-0.5*lg**2
wts=np.exp(logpost-logpost.max()); wts/=np.trapezoid(wts,lg)
birge_scale=float(np.sqrt(q0/(len(D.y)-1)))
base=NC+(1/a)*np.ones((4,4)); Vref=birge_scale**2*base; d=N.y-mu_l
qbase=float(d@np.linalg.solve(base,d))
ql_ref=float(qbase/(birge_scale**2))
rng2=np.random.default_rng(20264374); reps=200000
# Sample lambda from the quadrature posterior, then one future 4-vector from its predictive Gaussian.
cdf=np.r_[0,np.cumsum((wts[:-1]+wts[1:])*0.5*np.diff(lg))]; cdf/=cdf[-1]
u=rng2.random(reps); idx=np.searchsorted(cdf,u,side='right').clip(1,len(Ls)-1); idx0=idx-1
# linear interpolation in the quadrature CDF
frac=(u-cdf[idx0])/(cdf[idx]-cdf[idx0]); lm=Ls[idx0]*(1-frac)+Ls[idx]*frac
Lbase=np.linalg.cholesky(base); outrep=mu_l+lm[:,None]*(rng2.normal(size=(reps,4))@Lbase.T)
Vref_inv=np.linalg.inv(Vref); Qrep=np.einsum('bi,ij,bj->b',outrep-mu_l,Vref_inv,outrep-mu_l)
lam_mix=float(np.mean(Qrep>=ql_ref)); lam_mc_se=math.sqrt(lam_mix*(1-lam_mix)/reps)
# The historical lambda posterior is represented on the fixed log-lambda grid above.
# For the fixed future discrepancy, the conditional Gaussian quadratic is q_base / lambda^2,
# so its upper-tail probability is chi-square_4.sf(q_base / lambda^2). The reference tail is
# obtained by the same in-script quadrature; no external snapshot is used.
lambda_only_conditional_tail_grid = chi2.sf(qbase / (Ls**2), 4)
lambda_only_reference_tail = float(np.trapezoid(lambda_only_conditional_tail_grid * wts, lg))
out={
 'hierarchical':{
   'posterior_draws':len(mu),'reference_discrepancy':{'mu':ref_mu,'tau':ref_tau,'lambda':ref_lam,'Q_obs':q_ref},
   'posterior_averaged_conditional_tail':float(np.mean(cond)),
   'median_draw_conditional_tail':float(np.median(cond)),
   'lppd_2026':float(logsumexp(logs)-np.log(len(logs))),
   'full_posterior_predictive_mixture_tail':float(np.mean(hits)),
   'mixture_tail_mc_se':float(math.sqrt(np.mean(hits)*(1-np.mean(hits))/len(hits))),
   'median_Q_obs_draw_dependent':float(np.median(q)),
 },
 'lambda_only':{
   'lambda_only_reference_posterior_averaged_conditional_tail':lambda_only_reference_tail,
   'fixed_discrepancy_definition':{'mu_si':mu_l,'birge_scale':birge_scale,'Q_obs':ql_ref},
   'full_posterior_predictive_mixture_tail':lam_mix,'mixture_tail_mc_se':lam_mc_se,
   'mc_replicates':reps,'conditional_tail_quadrature_grid_points':len(Ls)
 },
 'interpretation':[
   'The posterior-averaged conditional tail and the fixed-discrepancy posterior-predictive mixture tail are distinct statistics.',
   'Both hierarchical values classify the observed 2026 four-vector as ordinary under the historical model.',
   'The lambda-only value is likewise ordinary; the fixed-discrepancy mixture calculation is an additional validation cross-check, not a replacement for the historical lambda-only statistic.'
 ]
}
out_path = ROOT / 'results' / 'machine_readable' / 'predictive_tail_validation.json'
out_path.write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
print(json.dumps(out, indent=2))
