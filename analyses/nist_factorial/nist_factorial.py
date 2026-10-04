from __future__ import annotations
import json
from pathlib import Path
import sys
import numpy as np
from scipy.stats import chi2, norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from g_metrology.nist_repro import NIST_VALUES, reported_covariance
OUT = ROOT / 'results' / 'machine_readable'
OUT.mkdir(exist_ok=True)

# Row order: Cu-servo, Cu-free, Sa-servo, Sa-free.
# Design coding (rows: Cu-servo, Cu-free, Sa-servo, Sa-free):
#   intercept : Cu-free mean
#   servo     : Cu_servo - Cu_free (servo effect at Cu material only)
#   sapphire  : Sa_free - Cu_free (sapphire effect at free mode only)
#   interaction: Sa_servo - Sa_free - Cu_servo + Cu_free (difference of mode contrasts)
# This 0/1 coding is not the symmetric +/-1 factorial coding; the interaction is the
# same up to sign, and main-effect coefficients are conditional rather than marginal.

X = np.column_stack([
    np.ones(4),
    np.array([1, 0, 1, 0.], float),
    np.array([0, 0, 1, 1.], float),
    np.array([0, 0, 0, 1.], float),
])

C = reported_covariance()
Ci = np.linalg.inv(C)
beta = np.linalg.solve(X.T @ Ci @ X, X.T @ Ci @ NIST_VALUES)
cov_beta = np.linalg.inv(X.T @ Ci @ X)
se = np.sqrt(np.diag(cov_beta))

# The interaction is the unique 2x2 deviation from an additive mode+material model.
L_int = np.array([-1, 1, 1, -1.], float)
interaction = float(L_int @ NIST_VALUES)
interaction_se = float(np.sqrt(L_int @ C @ L_int))
z_int = interaction / interaction_se
p_int = 2 * norm.sf(abs(z_int))

# Additive model vs intercept-only LR under known covariance.
X0 = np.ones((4, 1))
b0 = np.linalg.solve(X0.T @ Ci @ X0, X0.T @ Ci @ NIST_VALUES)
r0 = NIST_VALUES - X0 @ b0
X_add = X[:, :3]
beta_add = np.linalg.solve(
    X_add.T @ Ci @ X_add,
    X_add.T @ Ci @ NIST_VALUES,
)
r1 = NIST_VALUES - X_add @ beta_add

q0 = float(r0 @ Ci @ r0)
q1 = float(r1 @ Ci @ r1)
lr_additive = q0 - q1

# Plain GLS mean and its standard error under the reported covariance.
mu_gls = float(np.ones(4) @ Ci @ NIST_VALUES / (np.ones(4) @ Ci @ np.ones(4)))
se_gls = float(np.sqrt(1.0 / (np.ones(4) @ Ci @ np.ones(4))))

out = {
  'input_source': 'data/nist2026_configurations.yaml',
  'estimates_si': beta.tolist(),
  'standard_errors_si': se.tolist(),
  'interaction_contrast_sapphire_difference_of_differences_si': interaction,
  'interaction_se_si': interaction_se,
  'interaction_z': z_int,
  'interaction_p_two_sided': p_int,
  'intercept_only_q': q0,
  'additive_q': q1,
  'additive_lr_2df': lr_additive,
  'additive_p_chi2_2df': float(chi2.sf(lr_additive, 2)),
  'gls_mean_si': mu_gls,
  'gls_mean_se_si': se_gls,
  'interpretation': 'The interaction tests whether the material contrast differs between servo and free modes. A large interaction p-value supports an additive description but cannot establish a material mechanism.'
}
(OUT/'nist_factorial_results.json').write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
