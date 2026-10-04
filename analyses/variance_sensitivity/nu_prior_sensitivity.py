from __future__ import annotations
"""Production posterior sensitivity to the Student-t nu prior scale.

Varies only nu-2 ~ Exponential(rate=1/scale).  The historical data, covariance,
mu/tau/lambda priors, sampler, and seeds are otherwise held fixed.  For each
prior scale we report posterior summaries, the posterior-averaged conditional
tail for the observed 2026 vector, and a fixed-discrepancy posterior-predictive
mixture tail.  The latter uses one common discrepancy definition frozen from
the scale-28 posterior median, so prior-scale comparisons do not change the
statistic being tested.
"""
from pathlib import Path
import json, math, sys, time
import numpy as np
from scipy.stats import f

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT / 'src'))
from g_metrology.data import load_yaml, active_covariance_from_dataset
from g_metrology.standardization import StandardizationContext
from g_metrology.config import Priors
from g_metrology.fast_inference import metropolis
from g_metrology.canonical import total_scale

OUT=ROOT/'results'/'machine_readable'; OUT.mkdir(parents=True,exist_ok=True)
D=load_yaml(ROOT/'data/codata_2018_g.yaml'); N=load_yaml(ROOT/'data/nist2026_configurations.yaml')
S=active_covariance_from_dataset(D); SN=active_covariance_from_dataset(N)
ctx=StandardizationContext.from_data(D.y); y=ctx.transform(D.y); Sd=ctx.covariance_to_std(S)
yn=ctx.transform(N.y); Snd=ctx.covariance_to_std(SN)

SCALES=(10.0,28.0,60.0)
DRAW=10000; BURN=3000; CHAINS=4
SEEDS={10.0:20264310,28.0:20264328,60.0:20264360}

def posterior_scale(scale:float):
    pri= P=Priors(nu_minus2_rate=1.0/scale)
    return metropolis(y,Sd,pri,seed=SEEDS[scale],n_chains=CHAINS,n_draws=DRAW,burn=BURN), pri

def summaries(p):
    out={}
    for k,v in p.samples.items():
        a=np.asarray(v,float)
        out[k]={"mean":float(np.mean(a)),"sd":float(np.std(a,ddof=1)),
                "median":float(np.median(a)),"q025":float(np.quantile(a,.025)),"q975":float(np.quantile(a,.975))}
    return out

def rhat(p):
    out={}
    for k,a in p.chains.items():
        a=np.asarray(a,float); n=a.shape[1]//2; z=np.concatenate([a[:,:n],a[:,n:2*n]],axis=0)
        m=z.mean(axis=1); W=z.var(axis=1,ddof=1).mean(); B=n*m.var(ddof=1)
        out[k]=float(np.sqrt(((n-1)/n*W+B/n)/W))
    return out

posts={}; t0=time.time()
for sc in SCALES:
    posts[sc], _ = posterior_scale(sc)
base=posts[28.0]
ref_mu=float(np.median(base.samples['mu'])); ref_tau=float(np.median(base.samples['tau'])); ref_lam=float(np.median(base.samples['lambda']))
ref_C=total_scale(Snd,ref_tau,ref_lam); ref_inv=np.linalg.inv(ref_C)
ref_d=yn-np.ones(4)*ref_mu
q_ref=float(ref_d@ref_inv@ref_d)
rows=[]
rng=np.random.default_rng(20264377)
for sc in SCALES:
    p=posts[sc]
    tau=np.asarray(p.samples['tau']); lam=np.asarray(p.samples['lambda']); nu=np.asarray(p.samples['nu']); mu=np.asarray(p.samples['mu'])
    q=[]; cond=[]; mix_hits=[]; lps=[]
    # One replicated vector per posterior draw is an unbiased Monte Carlo estimator
    # of the posterior-predictive mixture probability for the fixed discrepancy.
    for i in range(len(tau)):
        C=total_scale(Snd,float(tau[i]),float(lam[i])); d=yn-np.ones(4)*float(mu[i])
        qi=float(d@np.linalg.solve(C,d)); q.append(qi); cond.append(float(f.sf(qi/4.0,4.0,float(nu[i]))))
        # multivariate Student-t: mu + L z / sqrt(w/nu)
        L=np.linalg.cholesky(C); z=rng.normal(size=4); w=rng.chisquare(float(nu[i])); yrep=float(mu[i])*np.ones(4)+L@z/math.sqrt(w/float(nu[i]))
        qrep=float((yrep-ref_mu*np.ones(4))@ref_inv@(yrep-ref_mu*np.ones(4))); mix_hits.append(qrep>=q_ref)
    rows.append({
      'nu_prior_scale':sc,'nu_prior_rate':1/sc,'draws':len(tau),
      'posterior':summaries(p),'rhat':rhat(p),
      'observed_2026_discrepancy_draw_dependent':{'median_Q':float(np.median(q)),'mean_Q':float(np.mean(q)),
          'posterior_averaged_conditional_tail':float(np.mean(cond)),'median_conditional_tail':float(np.median(cond))},
      'fixed_discrepancy_reference':{'reference_prior_scale':28.0,'mu':ref_mu,'tau':ref_tau,'lambda':ref_lam,
          'Q_obs':q_ref,'posterior_predictive_mixture_tail_mc':float(np.mean(mix_hits)),
          'mc_se':float(math.sqrt(np.mean(mix_hits)*(1-np.mean(mix_hits))/len(mix_hits)))},
    })

out={'description':'Full posterior sensitivity to the corrected Student-t nu prior. Only nu prior scale varies.',
     'model':'nu-2 ~ Exponential(rate=1/scale)','scales':list(SCALES),
     'sampler':{'backend':'Numba Metropolis','chains':CHAINS,'draws_per_chain':DRAW,'burn':BURN,'seeds':SEEDS},
     'reference_discrepancy':{'frozen_from_scale':28.0,'Q_obs':q_ref,'mu':ref_mu,'tau':ref_tau,'lambda':ref_lam},
     'cases':rows,'runtime_seconds':time.time()-t0}
(OUT/'nu_prior_sensitivity.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
