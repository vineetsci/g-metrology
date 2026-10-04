from __future__ import annotations
import numpy as np
from .data import active_covariance_from_dataset
from .nist_repro import NIST_FINAL_G, NIST_FINAL_U
from .standardization import StandardizationContext
from .config import DEFAULT_PRIORS
from .fast_inference import metropolis
from .scoring import score_temporal


def assemble_world_dataset(d2018, g2026=None, u2026=None):
    if g2026 is None:
        g2026 = NIST_FINAL_G
    if u2026 is None:
        u2026 = NIST_FINAL_U
    y = np.r_[d2018.y, g2026]
    C = np.zeros((len(y), len(y)))
    C[:len(d2018.measurements), :len(d2018.measurements)] = active_covariance_from_dataset(d2018)
    C[-1, -1] = u2026**2
    years = [m.year for m in d2018.measurements] + [2026]
    return y, C, np.asarray(years, int), d2018.labels + ["NIST-BIPM-2026"]


def prequential_world(d2018, *, start_train=7, draws=1400, burn=900, chains=2, seed=20260903):
    y, C, years, labels = assemble_world_dataset(d2018)
    order = np.argsort(years)
    y = y[order]; C = C[np.ix_(order, order)]; years = years[order]; labels = [labels[i] for i in order]
    rows = []
    for k in range(start_train, len(y)):
        tr = np.arange(k); ho = np.array([k])
        ctx = StandardizationContext.from_data(y[tr]); ys = ctx.transform(y); Ss = ctx.covariance_to_std(C)
        p_free = metropolis(ys[tr], Ss[np.ix_(tr, tr)], DEFAULT_PRIORS, fixed_mu=None, seed=seed + 17*k, n_chains=chains, n_draws=draws, burn=burn)
        gmean = float((np.ones(k) @ np.linalg.solve(Ss[np.ix_(tr, tr)], ys[tr])) / (np.ones(k) @ np.linalg.solve(Ss[np.ix_(tr, tr)], np.ones(k))))
        p_gls = metropolis(ys[tr], Ss[np.ix_(tr, tr)], DEFAULT_PRIORS, fixed_mu=gmean, seed=seed + 47*k, n_chains=chains, n_draws=draws, burn=burn)
        sc_free = score_temporal(p_free, ys, Ss, tr, ho)['lppd']
        sc_gls = score_temporal(p_gls, ys, Ss, tr, ho, fixed_mu=gmean)['lppd']
        rows.append({
            'train_end_year': int(years[k-1]), 'holdout_year': int(years[k]), 'holdout_label': labels[k],
            'free': float(sc_free), 'train_gls': float(sc_gls),
            'train_gls_minus_free': float(sc_gls-sc_free),
        })
    arr = np.array([r['train_gls_minus_free'] for r in rows])
    return {
        'start_train': start_train, 'draws': draws, 'burn': burn, 'chains': chains,
        'n_predictions': len(rows), 'rows': rows,
        'cumulative_train_gls_minus_free': float(arr.sum()),
        'mean_train_gls_minus_free': float(arr.mean()),
        'sd_train_gls_minus_free': float(arr.std(ddof=1)),
    }
