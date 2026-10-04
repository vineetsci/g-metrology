from __future__ import annotations
import numpy as np
from .prediction import conditional_holdout_draws,lppd

def score_temporal(posterior,y,cov,train_idx,hold_idx,fixed_mu=None,method_codes=None):
    logs=conditional_holdout_draws(posterior,y,cov,train_idx,hold_idx,fixed_mu=fixed_mu,method_codes=method_codes)
    return {'lppd':lppd(logs),'logpdf_draws':logs}

def paired_delta(logs_a,logs_b):
    # Posterior-draw pairing gives a useful uncertainty descriptor, not a frequentist p-value.
    d=np.asarray(logs_a)-np.asarray(logs_b)
    return {'mean_draw_difference':float(d.mean()),'sd_draw_difference':float(d.std(ddof=1)),'q025':float(np.quantile(d,.025)),'q975':float(np.quantile(d,.975))}
