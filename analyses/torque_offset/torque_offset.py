from __future__ import annotations
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from g_metrology.torque import (
    common_torque_fit,
    fractional_torque_fit,
    TABLE15_DELTA_PNM,
    TABLE15_TYPEA_PNM,
    TABLE15_SIGNAL_PNM,
    TABLE15_CONFIG_ORDER,
)

estimate, se, q, p = common_torque_fit()
f_hat, se_f, q_f, p_f = fractional_torque_fit()
result = {
    'source': 'Schlamminger et al. 2026, Table 15',
    'input_source': 'data/nist2026_configurations.yaml',
    'configuration_order': list(TABLE15_CONFIG_ORDER),
    'delta_definition': 'd_i = N_free_i - N_servo_i',
    'predictor_definition': 'x_i = N_free_i',
    'torque_unit': 'pN m',
    'delta_torque_pNm': TABLE15_DELTA_PNM.tolist(),
    'u_typeA_pNm': TABLE15_TYPEA_PNM.tolist(),
    'signal_pNm': TABLE15_SIGNAL_PNM.tolist(),
    # Legacy top-level absolute-offset result keys retained unchanged.
    'common_offset_pNm': estimate,
    'standard_error_pNm': se,
    'chi2': q,
    'dof': len(TABLE15_DELTA_PNM) - 1,
    'p_value': p,
    'absolute_offset_model': {
        'model_equation': 'd_i = Delta_N + epsilon_i',
        'parameter': 'Delta_N',
        'estimate_pNm': estimate,
        'standard_error_pNm': se,
        'chi2': q,
        'dof': len(TABLE15_DELTA_PNM) - 1,
        'p_value_chi2_upper_tail': p,
    },
    'fractional_scale_model': {
        'model_equation': 'd_i = f * N_free_i + epsilon_i',
        'predictor': 'N_free_i',
        'predictor_unit': 'pN m',
        'parameter': 'f',
        'estimate_dimensionless': f_hat,
        'standard_error_dimensionless': se_f,
        'estimate_ppm': 1e6 * f_hat,
        'standard_error_ppm': 1e6 * se_f,
        'chi2': q_f,
        'dof': len(TABLE15_DELTA_PNM) - 1,
        'p_value_chi2_upper_tail': p_f,
    },
    'model_comparison': {
        'delta_chi2_fractional_minus_absolute': q_f - q,
        'comparison_note': 'Both are one-parameter, non-nested descriptive weighted least-squares models fit to the same four observations. Delta chi-square is reported as a goodness-of-fit contrast, not as a chi-square likelihood-ratio test or calibrated p-value.',
        'uncertainty_note': 'Uses combined independent Type-A uncertainties for d_i, treats N_free_i as fixed, and does not supply a full covariance model for the four Table-15 differences.',
    },
    'assumptions': [
        'Differences use free minus servo.',
        'Uncertainties are combined Type-A values in quadrature.',
        'Observations are treated as independent for this analysis.',
        'The predictor N_free_i is treated as fixed.',
        'The fractional-scale result does not identify a physical mechanism.',
        'Delta chi-square is not a nested likelihood-ratio-test statistic.',
    ],
}
out = ROOT / 'results' / 'machine_readable' / 'torque_offset.json'
out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
print(json.dumps(result, indent=2))
