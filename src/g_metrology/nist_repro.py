from __future__ import annotations

import numpy as np
from scipy.stats import chi2
from .data import active_covariance_from_dataset, load_yaml

NIST_DATA_PATH = __import__('pathlib').Path(__file__).resolve().parents[2] / 'data' / 'nist2026_configurations.yaml'
NIST_DATASET = load_yaml(NIST_DATA_PATH)
NIST_VALUES = NIST_DATASET.y
NIST_U_REL_PPM = np.array([m.u_relative * 1e6 for m in NIST_DATASET.measurements], float)
NIST_DARK = np.array([m.dark_u_si for m in NIST_DATASET.measurements], float)
NIST_EFFECT = np.array([m.config_effect_si for m in NIST_DATASET.measurements], float)
NIST_LABELS = tuple(NIST_DATASET.labels)
NIST_CORR = np.eye(len(NIST_VALUES))
for i, j, r in NIST_DATASET.correlations:
    NIST_CORR[i, j] = NIST_CORR[j, i] = r

AUX = NIST_DATASET.auxiliary or {}
TABLE14_AUTO = AUX['table14_autocollimator_support_ppm']
TABLE15_TORQUE = AUX['table15_torque_nNm']
TABLE17_BUDGET = AUX['table17_uncertainty_budget_ppm']
TABLE19_CONSENSUS = AUX['table19_bayesian_consensus']
NIST_FINAL_G = float(TABLE19_CONSENSUS['final_consensus']['G_SI'])
NIST_FINAL_U = float(TABLE19_CONSENSUS['final_consensus']['u_SI'])
NIST_THERMAL = AUX['thermal_statement']


def reported_covariance() -> np.ndarray:
    """Published covariance reconstructed from each reported G_i, its u_r, and published correlations."""
    return active_covariance_from_dataset(NIST_DATASET)


def _gls_mean(y: np.ndarray, C: np.ndarray) -> tuple[float, float, float]:
    one = np.ones(len(y))
    Ci1 = np.linalg.solve(C, one)
    Ciy = np.linalg.solve(C, y)
    mu = float((one @ Ciy) / (one @ Ci1))
    q = float((y - mu) @ np.linalg.solve(C, y - mu))
    se = float(1.0 / np.sqrt(one @ Ci1))
    return mu, se, q


def raw_internal_inconsistency() -> dict:
    C = reported_covariance()
    mu, se, q = _gls_mean(NIST_VALUES, C)
    return {'weighted_mean': mu, 'mean_se': se, 'chi2': q, 'dof': 3, 'birge_ratio': float(np.sqrt(q / 3))}


def published_dark_uncertainty_check() -> dict:
    C = reported_covariance()
    corrected = NIST_VALUES - NIST_EFFECT
    total = C + np.diag(NIST_DARK**2)
    mu, se, q = _gls_mean(corrected, total)
    return {
        'corrected_gls_mean': mu, 'mean_se': se, 'chi2': q, 'dof': 3,
        'birge_ratio': float(np.sqrt(q / 3)),
        'residual_1e-15': ((corrected - mu) / 1e-15).tolist(),
        'note': 'Descriptive check only; not an independent validation of the published Bayesian posterior.'
    }
