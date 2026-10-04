#!/usr/bin/env python3
"""Run simulation-based calibration for the hierarchical Student-t model."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from numba import njit
from scipy.stats import chisquare

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from g_metrology.config import DEFAULT_PRIORS, SEED
from g_metrology.data import active_covariance_from_dataset, load_yaml
from g_metrology.fast_inference import _lp, _map
from g_metrology.simulation import generate_prior_predictive
from g_metrology.standardization import StandardizationContext
from g_metrology.terminal_ui import SBCProgress


DEFAULT_SBC_CONFIG = {"n_sims": 1000, "n_draws": 4000, "burn": 2000, "n_chains": 4}
SBC_PARAMETERS = ("mu", "tau", "lambda", "nu")
RANK_DRAWS = 500
RANK_BINS = 20


@njit(cache=True)
def _logpost_eigen(z, yt, one_t, eigvals, mu_sd, tau_beta, llm, lls, nu_rate):
    """Exact transformed log posterior in the eigenbasis of the standardized covariance."""
    mu = z[0]
    lt = z[1]
    ll = z[2]
    lnu = z[3]
    tau = np.exp(lt)
    lam = np.exp(ll)
    nu2 = np.exp(lnu)
    nu = 2.0 + nu2

    if tau <= 0.0 or lam <= 0.0 or nu <= 2.0:
        return -1.0e300

    lp = -0.5 * (mu / mu_sd) ** 2 - np.log(mu_sd * np.sqrt(2.0 * np.pi))
    lp += np.log(2.0 / (np.pi * tau_beta * (1.0 + (tau / tau_beta) ** 2))) + lt
    lp += -0.5 * ((ll - llm) / lls) ** 2 - np.log(lls * np.sqrt(2.0 * np.pi))
    lp += np.log(nu_rate) - nu_rate * nu2 + lnu

    n = yt.shape[0]
    logdet = 0.0
    q = 0.0
    for i in range(n):
        d = lam * lam * eigvals[i] + tau * tau
        if d <= 0.0 or not np.isfinite(d):
            return -1.0e300
        diff = yt[i] - mu * one_t[i]
        logdet += np.log(d)
        q += diff * diff / d

    return lp + (
        math.lgamma((nu + n) / 2.0)
        - math.lgamma(nu / 2.0)
        - 0.5 * (n * np.log(nu * np.pi) + logdet)
        - (nu + n) / 2.0 * np.log1p(q / nu)
    )


@njit(cache=True)
def _covariance_from_history(hist, n_hist, dim):
    mean = np.zeros(dim)
    for i in range(n_hist):
        for j in range(dim):
            mean[j] += hist[i, j]
    mean /= n_hist
    cov = np.zeros((dim, dim))
    denom = max(n_hist - 1, 1)
    for i in range(n_hist):
        for a in range(dim):
            da = hist[i, a] - mean[a]
            for b in range(dim):
                cov[a, b] += da * (hist[i, b] - mean[b])
    cov /= denom
    return cov


@njit(cache=True)
def _mh_sbc_chains(y_t, one_t, eigvals, mode, mu_sd, tau_beta, llm, lls, nu_rate, seed, n_chains, n_draws, burn):
    """Random-walk Metropolis kernel matching the existing fast_inference adaptation."""
    dim = 4
    out = np.empty((n_chains, n_draws, dim), dtype=np.float64)
    accepts = np.empty(n_chains, dtype=np.float64)

    for c in range(n_chains):
        np.random.seed(seed + 7919 * c)
        x = mode + np.random.normal(0.0, 0.03, dim)
        cur = _logpost_eigen(x, y_t, one_t, eigvals, mu_sd, tau_beta, llm, lls, nu_rate)
        pcov = np.eye(dim) * (0.03 ** 2)
        L = np.linalg.cholesky(pcov)
        hist = np.empty((max(burn, 1), dim), dtype=np.float64)
        acc = 0
        total = burn + n_draws

        for it in range(total):
            prop = x + L @ np.random.normal(0.0, 1.0, dim)
            val = _logpost_eigen(prop, y_t, one_t, eigvals, mu_sd, tau_beta, llm, lls, nu_rate)
            if np.log(np.random.random()) < val - cur:
                x = prop
                cur = val
                acc += 1

            if it < burn:
                hist[it, :] = x
                if (it + 1) % 100 == 0 and it + 1 >= 100:
                    C = _covariance_from_history(hist, it + 1, dim)
                    pcov = (2.38 ** 2 / dim) * C
                    for j in range(dim):
                        pcov[j, j] += (2.38 ** 2 / dim) * 1.0e-5
                    L = np.linalg.cholesky(pcov)
            else:
                out[c, it - burn, :] = x

        accepts[c] = acc / total

    return out, float(np.mean(accepts))


def _rhat_basic(chains):
    a = np.asarray(chains, dtype=float)
    m, n = a.shape
    if m < 2 or n < 20:
        return float("nan")
    half = n // 2
    z = np.concatenate([a[:, :half], a[:, half : 2 * half]], axis=0)
    means = z.mean(axis=1)
    W = z.var(axis=1, ddof=1).mean()
    if W <= 0:
        return float("nan")
    B = half * means.var(ddof=1)
    return float(np.sqrt(((half - 1) / half * W + B / half) / W))


def _ess_bulk_basic(chains):
    """ESS using the standard positive-pair autocorrelation truncation on each chain."""
    x = np.asarray(chains, dtype=float)
    m, n = x.shape
    if m < 1 or n < 20:
        return float("nan")
    total = 0.0
    for c in range(m):
        y = x[c] - np.mean(x[c])
        var = np.dot(y, y) / n
        if not np.isfinite(var) or var <= 0:
            return float("nan")
        size = 1 << (2 * n - 1).bit_length()
        fy = np.fft.rfft(y, size)
        ac = np.fft.irfft(fy * np.conjugate(fy), size)[:n]
        ac = ac / np.arange(n, 0, -1)
        rho = ac / ac[0]
        tau = 1.0
        k = 1
        while k + 1 < n:
            pair = rho[k] + rho[k + 1]
            if not np.isfinite(pair) or pair <= 0:
                break
            tau += 2.0 * pair
            k += 2
        total += n / max(tau, 1.0)
    return float(total)


def _rank_histogram(rank_fractions):
    x = np.asarray(rank_fractions, dtype=float)
    edges = np.linspace(0.0, 1.0, RANK_BINS + 1)
    counts, _ = np.histogram(np.clip(x, 0.0, 1.0), bins=edges)
    expected = np.full(RANK_BINS, len(x) / RANK_BINS, dtype=float)
    stat, p = chisquare(counts, f_exp=expected)
    return {"bins": RANK_BINS, "counts": counts.astype(int).tolist(), "chi2": float(stat), "p_value": float(p)}


def _wilson_interval(successes, total):
    z = 1.959963984540054
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / denom
    return float(center - half), float(center + half)


def _one_simulation(task):
    index, cov, eigvals, eigvecs, priors, seed, n_draws, burn, n_chains = task
    rng = np.random.default_rng(seed)

    # The SBC model is defined in the same standardized coordinates as the
    # canonical Student-t analysis: y and S are dimensionless, while the priors
    # for mu and tau use those standardized coordinates.
    sim = generate_prior_predictive(rng, cov, priors)

    mode = _map(sim["y"], cov, priors, None).x.astype(float)
    y_t = eigvecs.T @ np.asarray(sim["y"], dtype=float)
    one_t = eigvecs.T @ np.ones(len(cov), dtype=float)
    samples_z, acceptance = _mh_sbc_chains(
        y_t,
        one_t,
        eigvals,
        mode,
        priors.mu_sd,
        priors.tau_beta,
        priors.log_lambda_mu,
        priors.log_lambda_sd,
        priors.nu_minus2_rate,
        seed + 1000 + index,
        n_chains,
        n_draws,
        burn,
    )

    samples = {
        "mu": samples_z[:, :, 0],
        "tau": np.exp(samples_z[:, :, 1]),
        "lambda": np.exp(samples_z[:, :, 2]),
        "nu": 2.0 + np.exp(samples_z[:, :, 3]),
    }
    per_chain = RANK_DRAWS // n_chains
    if n_draws % per_chain != 0:
        raise ValueError("Declared draw count is incompatible with the balanced SBC rank sample")
    stride = n_draws // per_chain

    result = {"index": index, "acceptance": acceptance, "parameters": {}, "stride": stride}
    for name in SBC_PARAMETERS:
        chains = samples[name]
        rank_chains = chains[:, ::stride]
        ranked = rank_chains.reshape(-1)
        truth = float(sim[name])
        result["parameters"][name] = {
            "rank_fraction": float(np.count_nonzero(ranked < truth) / RANK_DRAWS),
            "raw_rank_fraction": float(np.count_nonzero(chains.reshape(-1) < truth) / chains.size),
            "covered95": bool(np.quantile(ranked, 0.025) <= truth <= np.quantile(ranked, 0.975)),
            "ess_bulk": _ess_bulk_basic(chains),
            "ess_bulk_rank": _ess_bulk_basic(rank_chains),
            "rhat": _rhat_basic(chains),
            "rhat_rank": _rhat_basic(rank_chains),
        }
    return result


def _aggregate(records, cfg, workers, assessment, purpose):
    params = {}
    for name in SBC_PARAMETERS:
        rows = [r["parameters"][name] for r in records]
        ranks = np.asarray([x["rank_fraction"] for x in rows])
        raw = np.asarray([x["raw_rank_fraction"] for x in rows])
        covered = np.asarray([x["covered95"] for x in rows], dtype=bool)
        ess = np.asarray([x["ess_bulk"] for x in rows], dtype=float)
        ess_rank = np.asarray([x["ess_bulk_rank"] for x in rows], dtype=float)
        rhat = np.asarray([x["rhat"] for x in rows], dtype=float)
        rhat_rank = np.asarray([x["rhat_rank"] for x in rows], dtype=float)
        hist = _rank_histogram(ranks)
        n_cov = int(covered.sum())
        ci_lo, ci_hi = _wilson_interval(n_cov, len(covered))
        m = RANK_DRAWS
        mcse = np.sqrt((m + 2.0) / (12.0 * m * len(records)))
        params[name] = {
            "mean_rank_fraction": float(ranks.mean()),
            "mean_rank_mcse_under_uniform": float(mcse),
            "mean_raw_rank_fraction": float(raw.mean()),
            "coverage95": float(covered.mean()),
            "coverage95_wilson_95pct": [ci_lo, ci_hi],
            "rank_fractions": ranks.tolist(),
            "rank_histogram": hist,
            "ess_bulk_raw_median": float(np.nanmedian(ess)),
            "ess_bulk_raw_p05": float(np.nanpercentile(ess, 5)),
            "ess_bulk_raw_min": float(np.nanmin(ess)),
            "fraction_ess_bulk_raw_below_rank_draws": float(np.mean(ess < RANK_DRAWS)),
            "ess_bulk_rank_median": float(np.nanmedian(ess_rank)),
            "ess_bulk_rank_p05": float(np.nanpercentile(ess_rank, 5)),
            "ess_bulk_rank_min": float(np.nanmin(ess_rank)),
            "rhat_raw_median": float(np.nanmedian(rhat)),
            "rhat_raw_p95": float(np.nanpercentile(rhat, 95)),
            "rhat_raw_max": float(np.nanmax(rhat)),
            "fraction_rhat_raw_above_1p01": float(np.mean(rhat > 1.01)),
            "rhat_rank_median": float(np.nanmedian(rhat_rank)),
            "rhat_rank_p95": float(np.nanpercentile(rhat_rank, 95)),
            "rhat_rank_max": float(np.nanmax(rhat_rank)),
        }

    mu = params["mu"]
    return {
        "n_sims": len(records),
        "run_type": "complete",
        "n_draws": cfg["n_draws"],
        "burn": cfg["burn"],
        "n_chains": cfg["n_chains"],
        "n_workers": workers,
        "rank_draws": RANK_DRAWS,
        "rank_bins": RANK_BINS,
        "rank_thinning": int(records[0]["stride"]),
        "parameter_names": list(SBC_PARAMETERS),
        # Kept for compatibility with existing consumers; this explicitly names
        # the parameter represented by the legacy top-level vector.
        "rank_fractions": mu["rank_fractions"],
        "rank_fractions_parameter": "mu",
        "mean_rank_fraction": mu["mean_rank_fraction"],
        "coverage95": mu["coverage95"],
        "parameters": params,
        "acceptance_rate_mean": float(np.mean([r["acceptance"] for r in records])),
        "assessment": assessment,
        "purpose": purpose,
    }


def _verify_target(covariance, eigvals, eigvecs, priors):
    """Check the optimized eigenbasis log posterior against the packaged implementation."""
    rng = np.random.default_rng(SEED)
    # Synthetic y is deliberately dimensionless: this check uses the same
    # standardized-unit convention as the canonical Student-t analysis.
    y = rng.normal(size=len(covariance))
    yt = eigvecs.T @ y
    one_t = eigvecs.T @ np.ones(len(covariance))
    max_diff = 0.0
    for _ in range(20):
        z = np.array([
            rng.normal(0, 1),
            rng.normal(-0.3, 0.7),
            rng.normal(0, 0.7),
            rng.normal(3, 0.7),
        ])
        old = float(_lp(
            z,
            y,
            covariance,
            priors.mu_sd,
            priors.tau_beta,
            priors.log_lambda_mu,
            priors.log_lambda_sd,
            priors.nu_minus2_rate,
            1e99,
        ))
        new = float(_logpost_eigen(
            z,
            yt,
            one_t,
            eigvals,
            priors.mu_sd,
            priors.tau_beta,
            priors.log_lambda_mu,
            priors.log_lambda_sd,
            priors.nu_minus2_rate,
        ))
        max_diff = max(max_diff, abs(old - new))
    if max_diff > 1e-11:
        raise RuntimeError(
            f"SBC optimized log-posterior failed equivalence check: max difference {max_diff:.3e}"
        )
    return max_diff


def main():
    ap = argparse.ArgumentParser(description="Simulation-based calibration.")
    ap.add_argument("--sims", type=int, default=None)
    ap.add_argument("--draws", type=int, default=None)
    ap.add_argument("--burn", type=int, default=None)
    ap.add_argument("--chains", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--output-name", default="sbc_1000.json")
    ap.add_argument("--checkpoint-sims", type=int, default=None)
    ap.add_argument("--checkpoint-output-name", default=None)
    args = ap.parse_args()

    if args.checkpoint_sims is not None and args.checkpoint_output_name is None:
        ap.error("--checkpoint-sims requires --checkpoint-output-name")

    cfg = DEFAULT_SBC_CONFIG.copy()
    for key, value in (
        ("n_sims", args.sims),
        ("n_draws", args.draws),
        ("burn", args.burn),
        ("n_chains", args.chains),
    ):
        if value is not None:
            cfg[key] = value
    if cfg["n_sims"] < 1 or cfg["n_draws"] < 1 or cfg["burn"] < 0 or cfg["n_chains"] < 1:
        ap.error("sims/draws/chains must be positive and burn must be non-negative")
    if RANK_DRAWS % cfg["n_chains"] != 0:
        ap.error(f"The declared {RANK_DRAWS:,} SBC rank draws must divide evenly across chains")
    per_chain = RANK_DRAWS // cfg["n_chains"]
    if cfg["n_draws"] < per_chain or cfg["n_draws"] % per_chain != 0:
        ap.error("draws is incompatible with the balanced SBC rank sample")
    if args.checkpoint_sims is not None and not (1 <= args.checkpoint_sims <= cfg["n_sims"]):
        ap.error("checkpoint-sims must lie between 1 and sims")

    data = load_yaml(ROOT / "data" / "codata_2018_g.yaml")
    covariance_si = active_covariance_from_dataset(data)
    standardization = StandardizationContext.from_data(data.y)
    covariance = standardization.covariance_to_std(covariance_si)
    eigvals, eigvecs = np.linalg.eigh(np.asarray(covariance, dtype=float))
    priors = DEFAULT_PRIORS
    target_diff = _verify_target(covariance, eigvals, eigvecs, priors)

    workers = min(max(1, args.workers), os.cpu_count() or 1, cfg["n_sims"])
    print(
        f"SBC: {cfg['n_sims']:,} simulations; {cfg['n_chains']} chains; "
        f"{cfg['n_draws']:,} draws/chain; {cfg['burn']:,} burn-in; {workers} worker(s).",
        flush=True,
    )

    output_dir = ROOT / "results" / "machine_readable"
    output = output_dir / args.output_name
    progress = SBCProgress(cfg["n_sims"])
    progress.update(0, cfg["n_sims"])

    tasks = [
        (
            i,
            np.asarray(covariance),
            eigvals,
            eigvecs,
            priors,
            SEED + i,
            cfg["n_draws"],
            cfg["burn"],
            cfg["n_chains"],
        )
        for i in range(cfg["n_sims"])
    ]
    records = [None] * cfg["n_sims"]

    def write_checkpoint(done):
        selected = records[:done]
        checkpoint = _aggregate(
            selected,
            cfg,
            workers,
            "execution_check_only",
            "First checkpoint of a single larger SBC run; not used as a calibration assessment.",
        )
        checkpoint["checkpoint_sims"] = done
        checkpoint["checkpoint_of_n_sims"] = cfg["n_sims"]
        checkpoint["checkpoint_rule"] = (
            "First checkpoint_sims simulations from the same deterministic run; "
            "no separate SBC job is required for the Archive checkpoint."
        )
        checkpoint["target_equivalence_max_abs_logpost_diff"] = target_diff
        checkpoint["standardization"] = {
            "mean_si": standardization.mean_si,
            "scale_si": standardization.scale_si,
            "definition": "Historical data are standardized using the arithmetic mean and population standard deviation used by StandardizationContext; the SBC generative model is sampled directly in those standardized coordinates.",
        }
        (output_dir / args.checkpoint_output_name).write_text(
            json.dumps(checkpoint, indent=2) + "\n", encoding="utf-8"
        )

    def consume(items):
        for rec in items:
            records[int(rec["index"])] = rec
            done = int(rec["index"]) + 1
            progress.update(done, cfg["n_sims"])
            if args.checkpoint_sims is not None and done == args.checkpoint_sims:
                write_checkpoint(done)

    if workers == 1:
        consume(_one_simulation(t) for t in tasks)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            consume(pool.map(_one_simulation, tasks, chunksize=1))

    assessment = "calibration_assessment" if cfg["n_sims"] >= 1000 else "execution_check_only"
    purpose = (
        "Full simulation-based calibration of all four scalar parameters under the declared generative model. "
        "Each simulation uses the exact packaged Student-t posterior target in the canonical standardized coordinates, "
        f"the declared Metropolis chain configuration, and a balanced {RANK_DRAWS:,}-draw rank sample obtained by deterministic thinning; "
        "ESS and split-R-hat are reported as sampler diagnostics."
        if cfg["n_sims"] >= 1000
        else
        "Reduced execution check of all four scalar SBC parameters; the reduced simulation count is not a standalone calibration assessment."
    )
    result = _aggregate(records, cfg, workers, assessment, purpose)
    result["target_equivalence_max_abs_logpost_diff"] = target_diff
    result["standardization"] = {
        "mean_si": standardization.mean_si,
        "scale_si": standardization.scale_si,
        "definition": "Historical data are standardized using the arithmetic mean and population standard deviation used by StandardizationContext; the SBC generative model is sampled directly in those standardized coordinates.",
    }
    if args.checkpoint_sims is not None:
        result["checkpoint_sims"] = args.checkpoint_sims
        result["checkpoint_output_name"] = args.checkpoint_output_name
        result["checkpoint_note"] = "Checkpoint is written during this same run from the first checkpoint_sims simulations."
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
