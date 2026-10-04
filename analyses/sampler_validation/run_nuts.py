#!/usr/bin/env python3
"""Run the optional PyMC/NUTS cross-check of the canonical historical model."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))

from g_metrology.data import load_yaml, active_covariance_from_dataset
from g_metrology.models import build_student_t_hbma, build_student_t_fixed_mu
from g_metrology.standardization import StandardizationContext

CODATA = 6.67430e-11


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the optional PyMC/NUTS cross-check.")
    parser.add_argument("--draws", type=int, default=4000)
    parser.add_argument("--tune", type=int, default=2000)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--cores", type=int, default=4)
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if args.draws <= 0 or args.tune < 0 or args.chains <= 0 or args.cores <= 0:
        raise SystemExit("draws, chains, and cores must be positive; tune must be non-negative")

    try:
        import pymc as pm
    except Exception as exc:
        raise RuntimeError(
            "PyMC is required for the NUTS cross-check. Install the optional Bayesian "
            "dependencies described in the repository documentation."
        ) from exc

    data = load_yaml(ROOT / "data" / "codata_2018_g.yaml")
    covariance = active_covariance_from_dataset(data)
    context = StandardizationContext.from_data(data.y)
    y = context.transform(data.y)
    scaled_covariance = context.covariance_to_std(covariance)
    codata_std = float(context.transform([CODATA])[0])

    output = ROOT / "results" / "nuts_reference"
    output.mkdir(parents=True, exist_ok=True)
    available_cores = os.cpu_count() or 1
    cores = min(args.cores, available_cores)

    records = {}
    cases = [("free", None), ("codata", codata_std)]
    for name, fixed_mu in cases:
        model = (
            build_student_t_hbma(y, scaled_covariance, context)
            if fixed_mu is None
            else build_student_t_fixed_mu(y, scaled_covariance, context, fixed_mu)
        )
        with model:
            idata = pm.sample(
                draws=args.draws,
                tune=args.tune,
                chains=args.chains,
                cores=cores,
                target_accept=args.target_accept,
                random_seed=args.seed,
                return_inferencedata=True,
                progressbar=False,
            )
            # Preserve pointwise log-likelihoods in the frozen NUTS reference artifacts for future LOO/predictive diagnostics.
            idata = pm.compute_log_likelihood(idata, model=model, progressbar=False)
        path = output / f"{name}.nc"
        idata.to_netcdf(path)
        records[name] = {
            "file": f"results/nuts_reference/{name}.nc",
            "summary_keys": list(idata.posterior.data_vars),
        }

    (output / "run_manifest.json").write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
