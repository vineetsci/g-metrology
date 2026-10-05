#!/usr/bin/env python3
"""Generate the Gaussian prospective scalar predictive band.

The calculation is deterministic and uses the historical Gaussian proportional
covariance-scale fit.  It is conditioned on the quoted k=1 standard uncertainty
u of one future independent scalar G determination.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from math import sqrt
from pathlib import Path

from scipy.stats import f

ROOT = Path(__file__).resolve().parents[2]

PROSPECTIVE_PATH = ROOT / "results" / "machine_readable" / "prospective_nist.json"
HISTORICAL_PATH = ROOT / "results" / "machine_readable" / "historical_dispersion.json"
OUTPUT_PATH = ROOT / "results" / "machine_readable" / "gaussian_predictive_band.json"

U_PPM_GRID = (20.0, 40.0, 60.0)
REFERENCE_G_SI = 6.67430e-11
GAUSSIAN_COVERAGE_FACTOR = 1.96
COVERAGE_PROBABILITY = 0.95

CONFIGS = (
    ("exclude_BIPM14_only", "exclude_BIPM14_only"),
    ("all_history", "all"),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return value or None
    except Exception:
        return None


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required machine-readable input not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _close(a: float, b: float, *, rtol: float = 1e-12, atol: float = 0.0) -> bool:
    scale = max(abs(a), abs(b), 1.0)
    return abs(a - b) <= max(rtol * scale, atol)


def _ppm_relative_to_reference(value_si: float) -> float:
    return float((value_si - REFERENCE_G_SI) / REFERENCE_G_SI * 1e6)


def predictive_intervals(
    *,
    mu_hat_si: float,
    lambda_mle: float,
    historical_mean_se_reported_si: float,
    training_count: int,
    training_dof: int,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    if not all(
        isinstance(v, (int, float)) and v == v and abs(v) != float("inf")
        for v in (mu_hat_si, lambda_mle, historical_mean_se_reported_si)
    ):
        raise ValueError("Non-finite Gaussian predictive input")
    if lambda_mle <= 0.0 or historical_mean_se_reported_si <= 0.0:
        raise ValueError("lambda_mle and historical_mean_se_reported_si must be positive")
    if training_count <= 1 or training_dof <= 0:
        raise ValueError("Invalid training count/degrees of freedom")
    if training_dof != training_count - 1:
        raise ValueError("Gaussian scalar predictive protocol requires nu = N - 1")

    f_quantile = float(f.ppf(COVERAGE_PROBABILITY, 1, training_dof))
    n_over_nu = float(training_count / training_dof)
    coverage_factor_f = float(sqrt(n_over_nu * f_quantile))
    variance_reported = float(historical_mean_se_reported_si ** 2)
    center_ppm = _ppm_relative_to_reference(mu_hat_si)

    intervals: list[dict[str, float]] = []
    for u_ppm in U_PPM_GRID:
        u_si = float(u_ppm * 1e-6 * REFERENCE_G_SI)
        sigma_pred_si = float(lambda_mle * sqrt(u_si ** 2 + variance_reported))
        half_width_si = float(coverage_factor_f * sigma_pred_si)
        half_width_gaussian_si = float(GAUSSIAN_COVERAGE_FACTOR * sigma_pred_si)

        lower_si = float(mu_hat_si - half_width_si)
        upper_si = float(mu_hat_si + half_width_si)
        lower_gaussian_si = float(mu_hat_si - half_width_gaussian_si)
        upper_gaussian_si = float(mu_hat_si + half_width_gaussian_si)

        intervals.append({
            "u_ppm": float(u_ppm),
            "u_si": u_si,
            "mu_hat_si": float(mu_hat_si),
            "sigma_pred_si": sigma_pred_si,
            "lower_si": lower_si,
            "upper_si": upper_si,
            "half_width_si": half_width_si,
            "half_width_ppm": float(half_width_si / REFERENCE_G_SI * 1e6),
            "center_ppm_relative_to_reference": center_ppm,
            "lower_ppm_relative_to_reference": _ppm_relative_to_reference(lower_si),
            "upper_ppm_relative_to_reference": _ppm_relative_to_reference(upper_si),
            "coverage_factor_used": coverage_factor_f,
            "coverage_factor_gaussian_diagnostic": GAUSSIAN_COVERAGE_FACTOR,
            "gaussian_diagnostic_lower_si": lower_gaussian_si,
            "gaussian_diagnostic_upper_si": upper_gaussian_si,
            "gaussian_diagnostic_lower_ppm_relative_to_reference": _ppm_relative_to_reference(lower_gaussian_si),
            "gaussian_diagnostic_upper_ppm_relative_to_reference": _ppm_relative_to_reference(upper_gaussian_si),
            "gaussian_diagnostic_half_width_si": half_width_gaussian_si,
            "gaussian_diagnostic_half_width_ppm": float(half_width_gaussian_si / REFERENCE_G_SI * 1e6),
        })

    return {
        "reference_g_si": REFERENCE_G_SI,
        "lambda_mle": float(lambda_mle),
        "historical_mean_se_reported_si": float(historical_mean_se_reported_si),
        "historical_mean_variance_reported_si2": variance_reported,
        "historical_mean_se_after_uniform_inflation_si": float(lambda_mle * historical_mean_se_reported_si),
        "training_count": int(training_count),
        "training_dof": int(training_dof),
        "F_quantile": f_quantile,
        "N_over_nu": n_over_nu,
        "coverage_factor_finite_sample": coverage_factor_f,
        "coverage_factor_gaussian_diagnostic": GAUSSIAN_COVERAGE_FACTOR,
        "center_ppm_relative_to_reference": center_ppm,
    }, intervals


def main() -> None:
    prospective = load_json(PROSPECTIVE_PATH)
    historical = load_json(HISTORICAL_PATH)

    results: dict[str, dict] = {}
    for config_name, historical_key in CONFIGS:
        p = prospective["configurations"][config_name]
        h = historical["fits"][historical_key]

        mu_hat = float(p["historical_mean_SI"])
        lambda_mle = float(p["lambda_mle"])
        se_reported = float(h["beta_se_si"][0])
        training_count = int(p["training_count"])
        training_dof = int(p["training_dof"])

        if not _close(mu_hat, float(h["beta_si"][0]), rtol=1e-12, atol=1e-30):
            raise ValueError(f"Historical mean mismatch for {config_name}")
        if not _close(lambda_mle, float(h["lambda_mle"]), rtol=1e-12, atol=1e-14):
            raise ValueError(f"lambda_mle mismatch for {config_name}")
        inflated_se = float(lambda_mle * se_reported)
        if not _close(
            inflated_se,
            float(p["historical_mean_se_after_uniform_inflation_SI"]),
            rtol=1e-12,
            atol=1e-29,
        ):
            raise ValueError(f"Inflated historical SE mismatch for {config_name}")

        meta, intervals = predictive_intervals(
            mu_hat_si=mu_hat,
            lambda_mle=lambda_mle,
            historical_mean_se_reported_si=se_reported,
            training_count=training_count,
            training_dof=training_dof,
        )
        if any(b["half_width_si"] <= a["half_width_si"] for a, b in zip(intervals, intervals[1:])):
            raise ValueError(f"Predictive half-width is not strictly increasing for {config_name}")

        results[config_name] = {
            **meta,
            "intervals": intervals,
        }

    output = {
        "analysis": "Gaussian prospective scalar predictive band",
        "protocol": {
            "name": "gaussian_proportional_common_scale_predictive_band",
            "status": "primary Gaussian predictive model",
            "analysis_script": "analyses/gaussian_predictive_band/gaussian_predictive_band.py",
            "prospective_input": "results/machine_readable/prospective_nist.json",
            "historical_input": "results/machine_readable/historical_dispersion.json",
            "prospective_input_sha256": sha256_file(PROSPECTIVE_PATH),
            "historical_input_sha256": sha256_file(HISTORICAL_PATH),
            "repository_git_commit": git_commit(),
            "creation_timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "model": {
            "family": "Gaussian proportional common-scale covariance model",
            "historical_covariance": "Sigma_hist = lambda^2 S_reported",
            "future_variance": "Var(y_fut | u, lambda) = lambda^2 u^2",
            "predictive_variance": "lambda^2 [u^2 + Var(mu_hat)_reported]",
            "lambda_estimator": "profile MLE sqrt(Q/N)",
            "future_experiment": "one independent scalar G determination",
            "future_u_conditioning": "The interval is conditional on the future determination reporting quoted k=1 standard uncertainty u.",
            "finite_sample_reference": "Under the declared independent Gaussian common-scale model, ((y_fut-mu_hat)^2 / (lambda_hat^2*(u^2+Var_rep))) * (N/nu) follows F(1,nu), with nu=N-1.",
            "finite_sample_coverage": "Exact under the declared model and the assumption that the same multiplicative lambda scale applies to the future quoted uncertainty.",
            "gaussian_diagnostic": "The 1.96 factor is a known-scale/Gaussian plug-in diagnostic; it is not the finite-sample calibration used for the primary interval.",
            "k": 1.0,
            "reference_g_si": REFERENCE_G_SI,
        },
        "configurations": results,
        "archived_student_t_sensitivity": {
            "artifact": "results/machine_readable/archived_student_t_predictive_band.json",
            "status": "frozen sensitivity analysis; not merged with or refit to the Gaussian model",
            "training_dataset": "all 16 historical CODATA-2018 determinations",
            "comparison_caveat": "The Student-t sensitivity artifact uses the complete 16-point archive, whereas the primary Gaussian configuration excludes BIPM-14; the widths are therefore not a same-training-set model comparison.",
        },
        "limitations": [
            "The band predicts a future scalar determination conditional on its quoted k=1 standard uncertainty u; the three stored u values are illustrative points on the closed-form curve.",
            "The finite-sample F calibration is exact only under the declared Sigma=lambda^2*S_reported common-scale Gaussian model and the stated independence assumptions.",
            "The future quoted uncertainty is treated as a fixed input to the predictive calculation.",
            "The reference G value is used only for ppm conversion and is fixed at 6.67430e-11 SI.",
            "The archived Student-t predictive-band artifact is not the primary model and is not refit by this analysis.",
        ],
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    for config_name, result in results.items():
        print(f"[{config_name}] center={result['center_ppm_relative_to_reference']:.6f} ppm")
        for item in result["intervals"]:
            print(
                f"  u={item['u_ppm']:.1f} ppm: "
                f"[{item['lower_ppm_relative_to_reference']:.6f}, "
                f"{item['upper_ppm_relative_to_reference']:.6f}] ppm; "
                f"half-width={item['half_width_ppm']:.6f} ppm"
            )


if __name__ == "__main__":
    main()
