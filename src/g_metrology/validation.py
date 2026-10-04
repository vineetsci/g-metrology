from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import os

import numpy as np

from .fast_inference import metropolis
from .simulation import generate_prior_predictive


def _sbc_one_simulation(args):
    index, cov, priors_dict, base_seed, n_draws, burn, n_chains, return_logpost = args
    from .config import Priors
    priors = Priors(**priors_dict)
    rng = np.random.default_rng(base_seed)
    sim = generate_prior_predictive(rng, cov, priors)
    post = metropolis(
        sim["y"],
        cov,
        priors,
        seed=base_seed + 1000 + index,
        n_chains=n_chains,
        n_draws=n_draws,
        burn=burn,
        return_logpost=return_logpost,
    )
    mu = post.samples["mu"]
    rank_fraction = float(np.mean(mu < sim["mu"]))
    covered = bool(np.quantile(mu, 0.025) <= sim["mu"] <= np.quantile(mu, 0.975))
    return index, rank_fraction, covered


def run_sbc(
    cov,
    priors,
    n_sims=50,
    seed=1,
    n_draws=600,
    burn=400,
    n_chains=2,
    n_workers=1,
    progress=None,
    return_logpost=False,
    checkpoint_sims=None,
    checkpoint=None,
):
    """Run simulation-based calibration.

    Each simulated dataset is independent. The default sequential path is kept
    for deterministic, low-overhead use; independent simulation tasks may be
    distributed across processes with ``n_workers``. SBC only consumes posterior
    samples here, so storing post-hoc log-posterior values is disabled by default.
    """
    if n_sims < 1:
        raise ValueError("n_sims must be at least 1")
    if n_workers < 1:
        raise ValueError("n_workers must be at least 1")
    if checkpoint_sims is not None and (checkpoint_sims < 1 or checkpoint_sims > n_sims):
        raise ValueError("checkpoint_sims must lie between 1 and n_sims")
    if checkpoint_sims is not None and checkpoint is None:
        raise ValueError("checkpoint callback is required when checkpoint_sims is set")

    priors_dict = asdict(priors)
    tasks = [
        (i, np.asarray(cov, float), priors_dict, seed + i, n_draws, burn, n_chains, return_logpost)
        for i in range(n_sims)
    ]

    rank_fractions = [None] * n_sims
    coverage = [None] * n_sims

    def consume(items):
        for i, rank_fraction, covered in items:
            rank_fractions[i] = rank_fraction
            coverage[i] = float(covered)
            done = i + 1
            if progress is not None:
                progress(done, n_sims)
            if checkpoint is not None and checkpoint_sims is not None and done == checkpoint_sims:
                checkpoint(
                    done,
                    list(rank_fractions[:done]),
                    list(coverage[:done]),
                )

    if n_workers == 1:
        consume(_sbc_one_simulation(task) for task in tasks)
    else:
        max_workers = min(int(n_workers), os.cpu_count() or 1, n_sims)
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            consume(pool.map(_sbc_one_simulation, tasks, chunksize=1))

    return {
        "n_sims": n_sims,
        "n_draws": n_draws,
        "burn": burn,
        "n_chains": n_chains,
        "n_workers": min(int(n_workers), os.cpu_count() or 1, n_sims),
        "mean_rank_fraction": float(np.mean(rank_fractions)),
        "coverage95": float(np.mean(coverage)),
        "rank_fractions": rank_fractions,
    }
