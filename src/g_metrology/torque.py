from __future__ import annotations
import numpy as np
from scipy.stats import chi2
from .nist_repro import TABLE15_TORQUE

TABLE15_CONFIG_ORDER = ("Cu0", "Cu120", "Cu240", "sapphire")
_KEYS = ("copper_0", "copper_120", "copper_240", "sapphire")
_FREE_NNM = np.array([TABLE15_TORQUE[k]["free"] for k in _KEYS], dtype=float)
_SERVO_NNM = np.array([TABLE15_TORQUE[k]["servo"] for k in _KEYS], dtype=float)
_UFREE_NNM = np.array([TABLE15_TORQUE[k]["free_u_typeA"] for k in _KEYS], dtype=float)
_USERVO_NNM = np.array([TABLE15_TORQUE[k]["servo_u_typeA"] for k in _KEYS], dtype=float)

TABLE15_DELTA_PNM = (_FREE_NNM - _SERVO_NNM) * 1e3
TABLE15_TYPEA_PNM = np.sqrt(_UFREE_NNM**2 + _USERVO_NNM**2) * 1e3
TABLE15_SIGNAL_PNM = _FREE_NNM * 1e3


def common_torque_fit(delta_pNm=TABLE15_DELTA_PNM, u_pNm=TABLE15_TYPEA_PNM):
    delta = np.asarray(delta_pNm, dtype=float)
    u = np.asarray(u_pNm, dtype=float)
    if delta.shape != u.shape or delta.ndim != 1:
        raise ValueError("delta and u must be one-dimensional arrays of equal length")
    if np.any(u <= 0):
        raise ValueError("all uncertainties must be positive")
    w = 1.0 / u**2
    common = float(np.sum(w * delta) / np.sum(w))
    se = float(np.sqrt(1.0 / np.sum(w)))
    q = float(np.sum(w * (delta - common)**2))
    p = float(chi2.sf(q, len(delta) - 1))
    return common, se, q, p


def fractional_torque_fit(
    delta_pNm=TABLE15_DELTA_PNM,
    u_pNm=TABLE15_TYPEA_PNM,
    signal_pNm=TABLE15_SIGNAL_PNM,
):
    """Fit d_i = f * N_free_i by one-parameter weighted least squares.

    The predictor N_free_i is treated as fixed, and the four differences are
    treated as independent with combined Type-A uncertainties. This is a
    descriptive comparison, not a causal model or likelihood-ratio test.
    """
    delta = np.asarray(delta_pNm, dtype=float)
    u = np.asarray(u_pNm, dtype=float)
    signal = np.asarray(signal_pNm, dtype=float)
    if delta.ndim != 1 or u.ndim != 1 or signal.ndim != 1:
        raise ValueError("delta, u, and signal must be one-dimensional arrays")
    if delta.shape != u.shape or delta.shape != signal.shape:
        raise ValueError("delta, u, and signal must be one-dimensional arrays of equal length")
    if not (np.all(np.isfinite(delta)) and np.all(np.isfinite(u)) and np.all(np.isfinite(signal))):
        raise ValueError("delta, u, and signal must contain only finite values")
    if np.any(u <= 0):
        raise ValueError("all uncertainties must be positive")

    w = 1.0 / u**2
    denominator = float(np.sum(w * signal**2))
    if not np.isfinite(denominator) or denominator <= 0:
        raise ValueError("weighted design denominator must be finite and nonzero")

    f_hat = float(np.sum(w * signal * delta) / denominator)
    se_f = float(np.sqrt(1.0 / denominator))
    chi2_value = float(np.sum(w * (delta - f_hat * signal)**2))
    dof = len(delta) - 1
    p_value = float(chi2.sf(chi2_value, dof))
    return f_hat, se_f, chi2_value, p_value


def table15_reordered(order):
    index = {name: i for i, name in enumerate(TABLE15_CONFIG_ORDER)}
    idx = [index[name] for name in order]
    return TABLE15_DELTA_PNM[idx].copy(), TABLE15_TYPEA_PNM[idx].copy()


TABLE15_SAPPHIRE_FIRST_ORDER = ("sapphire", "Cu0", "Cu120", "Cu240")
