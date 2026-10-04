from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import chi2, f

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from g_metrology.data import load_yaml, active_covariance_from_dataset
from g_metrology.nist_repro import NIST_FINAL_G, NIST_FINAL_U

h = load_yaml(ROOT / 'data/codata_2018_g.yaml')
n = load_yaml(ROOT / 'data/nist2026_configurations.yaml')
y = h.y
u = h.u
year = h.years
N = len(y)
S = active_covariance_from_dataset(h)
yn = n.y
Sn = active_covariance_from_dataset(n)
nist_dim = len(yn)

def gls_fit(idx):
    idx = np.array(idx, int)
    yy = y[idx]
    SS = S[np.ix_(idx, idx)]
    W = np.linalg.inv(SS)
    one = np.ones(len(idx))
    a = float(one @ W @ one)
    mu = float((one @ W @ yy) / a)
    r = yy - mu
    q = float(r @ W @ r)
    dof = len(idx) - 1
    return {
        'beta_si': [mu],
        'q': q,
        'dof': int(dof),
        'p': float(chi2.sf(q, dof)),
        'lambda_mle': float(np.sqrt(q / len(idx))),
        'birge_ratio': float(np.sqrt(q / dof)),
        'mean_precision': a,
    }

def prospective_metrics(idx):
    fit = gls_fit(idx)
    mu_h = fit['beta_si'][0]
    lam = fit['lambda_mle']
    a_hist = fit['mean_precision']

    n_train = len(idx)
    nu_hist = fit['dof']

    # Baseline predictive covariance:
    # reported NIST covariance + uncertainty in the historical GLS mean.
    V0 = Sn + (1 / a_hist) * np.ones((nist_dim, nist_dim))
    d = yn - mu_h
    q_no = float(d @ np.linalg.solve(V0, d))

    # ------------------------------------------------------------
    # Primary predictive convention:
    # uniformly scale both the historical-mean variance and the
    # reported NIST covariance by lambda^2.
    #
    # Algebraically:
    #   Q_uniform = Q0 / lambda^2
    # ------------------------------------------------------------
    q_uniform = q_no / lam**2
    p_uniform_chi2 = float(chi2.sf(q_uniform, nist_dim))

    # Exact finite-sample reference under the stated independent
    # Gaussian common-scale model:
    #
    #   (nu / (N*m)) Q_uniform ~ F(m, nu)
    #
    # where N=n_train, nu=nu_hist, and m=nist_dim.
    f_stat = (nu_hist / (n_train * nist_dim)) * q_uniform
    p_uniform_f = float(f.sf(f_stat, nist_dim, nu_hist))

    # ------------------------------------------------------------
    # Mean-only sensitivity:
    # scale the historical-mean variance, but leave the published
    # NIST covariance unchanged.
    #
    # There is no corresponding simple exact F reference law for
    # this mixed-scale covariance, so this is reported with the
    # plug-in chi-square reference only.
    # ------------------------------------------------------------
    V_mean_only = Sn + (lam**2 / a_hist) * np.ones((nist_dim, nist_dim))
    q_mean_only = float(d @ np.linalg.solve(V_mean_only, d))
    p_mean_only_chi2 = float(chi2.sf(q_mean_only, nist_dim))

    Gfinal = NIST_FINAL_G
    ufinal = NIST_FINAL_U
    Shist_mu = lam * np.sqrt(1 / a_hist)
    z_final = (Gfinal - mu_h) / np.sqrt(ufinal**2 + Shist_mu**2)

    return {
        'historical_mean_SI': mu_h,
        'lambda_mle': lam,
        'birge_ratio': fit['birge_ratio'],

        'Q_without_inflation': q_no,
        'p_without_inflation': float(chi2.sf(q_no, nist_dim)),

        'Q_with_uniform_inflation': q_uniform,
        'p_with_uniform_inflation_chi2': p_uniform_chi2,
        'p_with_uniform_inflation_finite_sample': p_uniform_f,
        'finite_sample_F_statistic': f_stat,
        'finite_sample_F_degrees_of_freedom': [nist_dim, nu_hist],

        'Q_with_mean_only_inflation': q_mean_only,
        'p_with_mean_only_inflation_chi2': p_mean_only_chi2,

        'final_NIST_G_SI': Gfinal,
        'final_NIST_u_SI': ufinal,
        'historical_mean_se_after_uniform_inflation_SI': Shist_mu,
        'final_NIST_vs_historical_z': float(z_final),

        'training_count': int(n_train),
        'training_dof': int(nu_hist),
        'training_indices': [int(i) for i in idx],
    }

primary_indices = [i for i in range(N) if i != 10]
all_indices = list(range(N))
result = {
    'analysis': 'Prospective NIST prediction',
    'input_sources': ['data/codata_2018_g.yaml','data/nist2026_configurations.yaml'],
    'primary_configuration': 'exclude_BIPM14_only',
    'primary_exclusion': {
        'measurement_label': 'BIPM-14',
        'reason': 'The primary prospective test excludes the historical BIPM-14 determination because the later NIST configuration is lineage-related to the BIPM apparatus. This reduces direct historical overlap in the reference distribution; it does not make the NIST result an orthogonally independent replication.',
    },
    'parameter_conventions': {
    'lambda_mle': 'Gaussian proportional covariance-scale maximum-likelihood estimate: sqrt(Q/N).',
    'birge_ratio': 'Metrology consistency/expansion diagnostic: sqrt(Q/nu), with nu=N-M adjusted degrees of freedom.',
    'uniform_predictive_convention': 'Apply the fitted historical lambda multiplicatively to both the historical-mean variance and the reported NIST covariance.',
    'mean_only_predictive_convention': 'Apply the fitted historical lambda only to the historical-mean variance; leave the reported NIST covariance unchanged. This is a sensitivity analysis with plug-in chi-square reference only.',
    'uniform_finite_sample_reference': 'Under independent Gaussian common-scale assumptions, (nu/(N*m))*Q_uniform follows F(m,nu), where N is the historical training count, nu=N-1, and m is the NIST vector dimension.',
    },
    'configurations': {
        'exclude_BIPM14_only': prospective_metrics(primary_indices),
        'all_history': prospective_metrics(all_indices),
    },
}
out = ROOT / 'results' / 'machine_readable' / 'prospective_nist.json'
out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
