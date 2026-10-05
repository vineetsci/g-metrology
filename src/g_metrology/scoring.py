from __future__ import annotations
from .prediction import conditional_holdout_draws,lppd

def score_temporal(posterior,y,cov,train_idx,hold_idx,fixed_mu=None,method_codes=None):
    logs=conditional_holdout_draws(posterior,y,cov,train_idx,hold_idx,fixed_mu=fixed_mu,method_codes=method_codes)
    return {'lppd':lppd(logs),'logpdf_draws':logs}