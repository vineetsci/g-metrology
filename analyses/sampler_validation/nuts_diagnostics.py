from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import arviz as az

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results" / "nuts_reference"
OUT = RESULTS / "nuts_diagnostics.json"

FILES = {
    "free": RESULTS / "free.nc",
    "codata": RESULTS / "codata.nc",
}

EXPECTED_VARS = {
    "free": ["mu", "tau", "lambda", "nu_minus2", "nu"],
    "codata": ["tau", "lambda", "nu_minus2", "nu"],
}


def scalar(x: Any) -> float:
    arr = np.asarray(x)
    if arr.size != 1:
        raise ValueError(f"Expected scalar, got shape {arr.shape}")
    return float(arr.reshape(-1)[0])


def chain_values(da) -> np.ndarray:
    return np.asarray(da.values, dtype=float)


def main() -> None:
    report: dict[str, Any] = {
        "purpose": (
            "NUTS diagnostics completion from already-saved PyMC/NUTS NetCDF outputs: "
            "rank-normalized R-hat, bulk/tail ESS, BFMI, divergences, "
            "energy, tree depth, acceptance rate, step size, and chain summaries."
        ),
        "files": {},
        "models": {},
        "initialization_robustness": {
            "status": "not_separately_tested",
            "disposition": "non_blocking",
            "reason": (
                "The saved NUTS run provides four-chain convergence, but starting states "
                "are not retained for this purpose, so a deliberate multi-start sensitivity experiment "
                "cannot be reconstructed retrospectively. This is supplementary sampler "
                "robustness testing, not a requirement for the present scientific conclusion."
            ),
        },
    }

    for name, path in FILES.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing NUTS output: {path}")

        idata = az.from_netcdf(path)
        posterior = idata.posterior
        stats = idata.sample_stats

        expected_vars = EXPECTED_VARS[name]
        posterior_vars = [v for v in expected_vars if v in posterior.data_vars]
        missing_vars = [v for v in expected_vars if v not in posterior.data_vars]
        if missing_vars:
            raise ValueError(f"NUTS posterior for {name} is missing expected variables: {missing_vars}")

        # Rank-normalized R-hat and bulk/tail ESS.
        rhat_ds = az.rhat(idata, var_names=posterior_vars, method="rank")
        ess_bulk_ds = az.ess(idata, var_names=posterior_vars, method="bulk")
        ess_tail_ds = az.ess(idata, var_names=posterior_vars, method="tail")

        variables: dict[str, Any] = {}
        for var in posterior_vars:
            x = chain_values(posterior[var])
            # x is [chain, draw] for scalar parameters.
            chain_means = np.mean(x, axis=1)
            chain_sds = np.std(x, axis=1, ddof=1)
            pooled = x.reshape(-1)

            variables[var] = {
                "mean": float(np.mean(pooled)),
                "sd": float(np.std(pooled, ddof=1)),
                "r_hat_rank": scalar(rhat_ds[var]),
                "ess_bulk": scalar(ess_bulk_ds[var]),
                "ess_tail": scalar(ess_tail_ds[var]),
                "chain_means": chain_means.tolist(),
                "chain_sds": chain_sds.tolist(),
                "chain_mean_range": float(np.max(chain_means) - np.min(chain_means)),
            }

        # Sample-stat diagnostics. PyMC names are stable for the saved NUTS data,
        # but report only fields that are actually present.
        sample_stats_names = sorted(str(v) for v in stats.data_vars)

        diag: dict[str, Any] = {"available_sample_stats": sample_stats_names}

        if "diverging" in stats:
            div = np.asarray(stats["diverging"].values, dtype=bool)
            diag["divergences_total"] = int(div.sum())
            diag["divergences_by_chain"] = div.sum(axis=1).astype(int).tolist()

        if "energy" in stats:
            energy = np.asarray(stats["energy"].values, dtype=float)
            diag["energy_mean_by_chain"] = np.mean(energy, axis=1).tolist()
            diag["energy_sd_by_chain"] = np.std(energy, axis=1, ddof=1).tolist()
            # Compute E-BFMI directly from each chain's energy draws.
            # This avoids dependence on ArviZ-specific DataTree return types.
            # E-BFMI = mean((E_t - E_{t-1})^2) / var(E).
            delta_e = np.diff(energy, axis=1)
            energy_var = np.var(energy, axis=1, ddof=1)
            bfmi = np.mean(delta_e**2, axis=1) / energy_var
            diag["bfmi_by_chain"] = bfmi.tolist()
            diag["bfmi_min"] = float(np.min(bfmi))

        if "tree_depth" in stats:
            td = np.asarray(stats["tree_depth"].values, dtype=int)
            diag["tree_depth_max"] = int(td.max())
            diag["tree_depth_mean_by_chain"] = np.mean(td, axis=1).tolist()
            max_count = []
            for row in td:
                m = row.max()
                max_count.append(float(np.mean(row == m)))
            diag["tree_depth_max_fraction_by_chain"] = max_count

        if "acceptance_rate" in stats:
            acc = np.asarray(stats["acceptance_rate"].values, dtype=float)
            diag["acceptance_rate_mean_by_chain"] = np.mean(acc, axis=1).tolist()

        if "step_size" in stats:
            step = np.asarray(stats["step_size"].values, dtype=float)
            # Step size is effectively constant after tuning; retain final values.
            diag["final_step_size_by_chain"] = step[:, -1].tolist()

        if "n_steps" in stats:
            ns = np.asarray(stats["n_steps"].values, dtype=float)
            diag["n_steps_mean_by_chain"] = np.mean(ns, axis=1).tolist()
            diag["n_steps_max_by_chain"] = np.max(ns, axis=1).tolist()

        # Simple explicit assessment rules for this cross-check.
        rhat_values = [v["r_hat_rank"] for v in variables.values()]
        ess_values = [min(v["ess_bulk"], v["ess_tail"]) for v in variables.values()]

        checks = {
            "rhat_all_le_1.01": bool(all(v <= 1.01 for v in rhat_values)),
            "ess_all_ge_1000": bool(all(v >= 1000 for v in ess_values)),
        }
        if "divergences_total" in diag:
            checks["zero_divergences"] = diag["divergences_total"] == 0
        if "bfmi_min" in diag:
            checks["bfmi_all_gt_0.3"] = bool(diag["bfmi_min"] > 0.3)

        report["files"][name] = f"results/nuts_reference/{path.name}"
        report["models"][name] = {
            "chains": int(posterior.sizes["chain"]),
            "draws_per_chain": int(posterior.sizes["draw"]),
            "variables": variables,
            "sample_stats": diag,
            "checks": checks,
        }

    report["overall_interpretation"] = (
        "The report closes the NUTS sampler-diagnostics cross-check if all model checks pass. "
        "A deliberate multi-start initialization experiment was not performed and is treated as "
        "non-blocking supplementary robustness work rather than an unresolved defect."
    )

    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved: {OUT}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
