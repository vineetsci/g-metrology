from __future__ import annotations

import numpy as np
from .data import active_covariance_from_dataset, load_yaml

NIST_DATA_PATH = __import__('pathlib').Path(__file__).resolve().parents[2] / 'data' / 'nist2026_configurations.yaml'
NIST_DATASET = load_yaml(NIST_DATA_PATH)
NIST_VALUES = NIST_DATASET.y
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