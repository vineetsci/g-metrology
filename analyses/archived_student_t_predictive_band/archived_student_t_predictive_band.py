#!/usr/bin/env python3
"""Generate the archived historical scalar posterior-predictive G band.

This is a separate archival provenance analysis. It uses the already-saved,
frozen free-mean NUTS posterior from the historical CODATA-2018 archive and does
not fit on or otherwise use the 2026 NIST data.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy import __version__ as scipy_version

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from g_metrology import __version__
from g_metrology.config import DEFAULT_PRIORS, SEED
from g_metrology.data import load_yaml, active_covariance_from_dataset
from g_metrology.standardization import StandardizationContext

TRAINING_DATA_PATH = ROOT / "data" / "codata_2018_g.yaml"
POSTERIOR_PATH = ROOT / "results" / "nuts_reference" / "free.nc"
NUTS_SCRIPT_PATH = ROOT / "analyses" / "sampler_validation" / "run_nuts.py"
OUTPUT_PATH = ROOT / "results" / "machine_readable" / "archived_student_t_predictive_band.json"

U_PPM_GRID = (20.0, 40.0, 60.0)
REFERENCE_G_SI = 6.67430e-11
PREDICTIVE_SEED = SEED
PREDICTIVE_DRAWS_PER_POSTERIOR = 8
EXPECTED_CHAINS = 4
EXPECTED_DRAWS_PER_CHAIN = 4000
EXPECTED_TUNE = 2000
EXPECTED_TARGET_ACCEPT = 0.95


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit() -> str | None:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return value or None
    except Exception:
        return None


def _read_posterior(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Frozen posterior reference not found: {path}")
    with h5py.File(path, "r") as root:
        posterior = root["posterior"]
        required = ("mu", "tau", "lambda", "nu")
        for name in required:
            if name not in posterior:
                raise ValueError(f"Frozen posterior is missing posterior/{name}")
        mu = np.asarray(posterior["mu"][...], dtype=float)
        tau = np.asarray(posterior["tau"][...], dtype=float)
        lam = np.asarray(posterior["lambda"][...], dtype=float)
        nu = np.asarray(posterior["nu"][...], dtype=float)
        if not (mu.shape == tau.shape == lam.shape == nu.shape):
            raise ValueError("Frozen posterior parameter arrays have unequal shapes")
        if mu.ndim != 2:
            raise ValueError(f"Expected posterior arrays with (chain, draw) shape; got {mu.shape}")
        chain = np.arange(mu.shape[0], dtype=int)[:, None] * np.ones_like(mu, dtype=int)
        draw = np.arange(mu.shape[1], dtype=int)[None, :] * np.ones_like(mu, dtype=int)
    return mu, tau, lam, nu, chain, draw


def _one_predictive_sample(mu: float, tau: float, lam: float, nu: float,
                           u_std: float, seed: int, chain: int, draw: int,
                           replicate: int) -> float:
    if not all(np.isfinite(v) for v in (mu, tau, lam, nu, u_std)):
        raise ValueError("Non-finite predictive parameter or uncertainty")
    if tau <= 0.0 or lam <= 0.0 or nu <= 2.0 or u_std <= 0.0:
        raise ValueError("Invalid predictive parameter or uncertainty")
    # The scalar marginal implied by the canonical multivariate Student-t model
    # has scale/shape sqrt(lambda^2*u_std^2 + tau^2), not ordinary covariance.
    scale_std = float(np.sqrt((lam * u_std) ** 2 + tau ** 2))
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(chain), int(draw), int(replicate)]))
    return float(mu + scale_std * rng.standard_t(df=nu))


def predictive_draws(mu: np.ndarray, tau: np.ndarray, lam: np.ndarray, nu: np.ndarray,
                     chain: np.ndarray, draw: np.ndarray, u_std: float,
                     seed: int = PREDICTIVE_SEED,
                     draws_per_posterior: int = PREDICTIVE_DRAWS_PER_POSTERIOR) -> np.ndarray:
    if draws_per_posterior <= 0:
        raise ValueError("draws_per_posterior must be positive")
    arrays = [np.asarray(a) for a in (mu, tau, lam, nu, chain, draw)]
    if not all(a.shape == arrays[0].shape for a in arrays[1:]):
        raise ValueError("Posterior parameter and coordinate arrays must have equal shapes")
    values: list[float] = []
    for index in np.ndindex(mu.shape):
        c = int(chain[index])
        d = int(draw[index])
        for replicate in range(draws_per_posterior):
            values.append(_one_predictive_sample(
                float(mu[index]), float(tau[index]), float(lam[index]), float(nu[index]),
                float(u_std), seed, c, d, replicate,
            ))
    return np.asarray(values, dtype=float)


def quantiles_for_uncertainty(mu: np.ndarray, tau: np.ndarray, lam: np.ndarray, nu: np.ndarray,
                              chain: np.ndarray, draw: np.ndarray, ctx: StandardizationContext,
                              u_ppm: float) -> dict[str, float]:
    if not np.isfinite(u_ppm) or u_ppm <= 0.0:
        raise ValueError("u_ppm must be finite and positive")
    u_si = float(u_ppm * 1e-6 * REFERENCE_G_SI)
    u_std = float(u_si / ctx.scale_si)
    y_std = predictive_draws(mu, tau, lam, nu, chain, draw, u_std)
    if not np.all(np.isfinite(y_std)):
        raise ValueError("Predictive draws are not all finite")
    y_si = ctx.inverse_transform(y_std)
    q025, q50, q975 = np.quantile(y_si, [0.025, 0.50, 0.975])
    if not (np.isfinite(q025) and np.isfinite(q50) and np.isfinite(q975)):
        raise ValueError("Predictive quantiles are not finite")
    if not (q025 < q50 < q975):
        raise ValueError("Predictive quantiles are not strictly ordered")
    half_width = float((q975 - q025) / 2.0)
    return {
        "u_ppm": float(u_ppm),
        "u_si": u_si,
        "predictive_center_si": float(q50),
        "predictive_q025_si": float(q025),
        "predictive_q975_si": float(q975),
        "half_width_si": half_width,
        "predictive_center_ppm_relative_to_reference": float((q50 - REFERENCE_G_SI) / REFERENCE_G_SI * 1e6),
        "q025_ppm_relative_to_reference": float((q025 - REFERENCE_G_SI) / REFERENCE_G_SI * 1e6),
        "q975_ppm_relative_to_reference": float((q975 - REFERENCE_G_SI) / REFERENCE_G_SI * 1e6),
        "half_width_ppm": float(half_width / REFERENCE_G_SI * 1e6),
    }


def main() -> None:
    dataset = load_yaml(TRAINING_DATA_PATH)
    if len(dataset.measurements) != 16:
        raise ValueError(f"Registered band requires the frozen 16-point archive; found {len(dataset.measurements)}")
    covariance = active_covariance_from_dataset(dataset)
    if covariance.shape != (16, 16):
        raise ValueError("Unexpected historical covariance shape")
    ctx = StandardizationContext.from_data(dataset.y)
    mu, tau, lam, nu, chain, draw = _read_posterior(POSTERIOR_PATH)
    if mu.shape != (EXPECTED_CHAINS, EXPECTED_DRAWS_PER_CHAIN):
        raise ValueError(
            f"Unexpected frozen posterior shape {mu.shape}; expected "
            f"({EXPECTED_CHAINS}, {EXPECTED_DRAWS_PER_CHAIN})"
        )
    if not (np.all(np.isfinite(mu)) and np.all(np.isfinite(tau)) and
            np.all(np.isfinite(lam)) and np.all(np.isfinite(nu))):
        raise ValueError("Frozen posterior contains non-finite values")

    intervals = [
        quantiles_for_uncertainty(mu, tau, lam, nu, chain, draw, ctx, u_ppm)
        for u_ppm in U_PPM_GRID
    ]
    half_widths = [item["half_width_si"] for item in intervals]
    if any(b <= a for a, b in zip(half_widths, half_widths[1:])):
        raise ValueError("Predictive half-width is not strictly increasing over the frozen uncertainty grid")

    convergence = None
    diagnostics_path = ROOT / "results" / "nuts_reference" / "nuts_diagnostics.json"
    if diagnostics_path.exists():
        try:
            diag = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            convergence = diag.get("models", {}).get("free", {}).get("checks")
        except Exception:
            convergence = None

    priors = {
        "mu_sd_standardized": DEFAULT_PRIORS.mu_sd,
        "tau_beta_standardized": DEFAULT_PRIORS.tau_beta,
        "log_lambda_mu": DEFAULT_PRIORS.log_lambda_mu,
        "log_lambda_sd": DEFAULT_PRIORS.log_lambda_sd,
        "nu_minus2_rate": DEFAULT_PRIORS.nu_minus2_rate,
        "nu_minus2_scale": 1.0 / DEFAULT_PRIORS.nu_minus2_rate,
    }
    output = {
        "protocol": {
            "name": "archived_student_t_scalar_G_predictive_band",
            "status": "archived historical Student-t predictive-band artifact",
            "package_version": __version__,
            "analysis_script": "analyses/archived_student_t_predictive_band/archived_student_t_predictive_band.py",
            "training_data": "data/codata_2018_g.yaml",
            "training_dataset_label": dataset.name,
            "training_count": len(dataset.measurements),
            "training_data_sha256": sha256_file(TRAINING_DATA_PATH),
            "posterior_reference": "results/nuts_reference/free.nc",
            "posterior_reference_sha256": sha256_file(POSTERIOR_PATH),
            "sampler_script": "analyses/sampler_validation/run_nuts.py",
            "sampler_script_sha256": sha256_file(NUTS_SCRIPT_PATH),
            "repository_git_commit": git_commit(),
            "creation_timestamp_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
            "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "frozen_definition": {
            "model": "canonical_hierarchical_multivariate_Student_t",
            "model_equation": "y ~ MvStudentT(nu, mu*1, Sigma), Sigma = lambda^2*S_reported + tau^2*I",
            "future_scalar_scale_equation": "scale_new = sqrt(lambda^2*u_std^2 + tau^2)",
            "future_experiment": "one independent scalar G determination",
            "active_covariance_convention": "k=1.0; reported CODATA-2018 standard uncertainties are not pre-expanded in active covariance",
            "priors": priors,
            "standardization": {
                "mean_si": ctx.mean_si,
                "scale_si": ctx.scale_si,
                "definition": "historical data are standardized with the arithmetic mean and population standard deviation used by StandardizationContext",
            },
            "future_uncertainty_conversion": "u_si = u_ppm * 1e-6 * reference_g_si",
            "reference_g_si": REFERENCE_G_SI,
            "reference_g_description": "project's frozen CODATA reference used for ppm conversion",
            "quantiles": "empirical 0.025, 0.50, and 0.975 quantiles of deterministic seeded posterior-predictive draws",
            "posterior_draws_used": int(mu.size),
            "predictive_draws_per_posterior_draw": PREDICTIVE_DRAWS_PER_POSTERIOR,
            "predictive_draws_per_uncertainty": int(mu.size * PREDICTIVE_DRAWS_PER_POSTERIOR),
            "predictive_seed": PREDICTIVE_SEED,
            "no_nist_2026_used": True,
        },
        "sampler": {
            "algorithm": "PyMC NUTS",
            "chains": EXPECTED_CHAINS,
            "draws_per_chain": EXPECTED_DRAWS_PER_CHAIN,
            "tune_per_chain": EXPECTED_TUNE,
            "target_accept": EXPECTED_TARGET_ACCEPT,
            "seed": SEED,
            "native_progressbar": False,
            "reference_posterior_is_previously_saved": True,
            "convergence_checks": convergence,
        },
        "software_versions": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy": package_version("numpy"),
            "scipy": scipy_version,
            "pymc": package_version("pymc"),
            "arviz": package_version("arviz"),
            "pytensor": package_version("pytensor"),
            "xarray": package_version("xarray"),
            "h5netcdf": package_version("h5netcdf"),
        },
        "future_intervals": intervals,
        "limitations": [
            "The band is frozen from the 16 historical CODATA-2018 determinations and must be confronted with a future observation before any refit that includes it.",
            "The target is one future independent scalar G determination; it is not an unchanged band for a correlated multi-configuration apparatus vector.",
            "The quoted future uncertainty u is treated as a fixed input to the scalar predictive scale.",
            "No NIST-2026 observation is used to fit, tune, choose, or validate this registered band.",
            "The protocol is not a confidence interval for G and is not a CODATA recommendation.",
            "A later model update must create a new dated/versioned predictive band; it must not overwrite this artifact.",
        ],
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH.relative_to(ROOT)}")
    for item in intervals:
        print(
            f"u={item['u_ppm']:.1f} ppm: "
            f"[{item['q025_ppm_relative_to_reference']:.6f}, "
            f"{item['q975_ppm_relative_to_reference']:.6f}] ppm relative to reference; "
            f"center={item['predictive_center_ppm_relative_to_reference']:.6f} ppm; "
            f"half-width={item['half_width_ppm']:.6f} ppm"
        )


if __name__ == "__main__":
    main()
