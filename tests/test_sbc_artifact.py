from __future__ import annotations

import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "machine_readable"

EXPECTED_PARAMETERS = ["mu", "tau", "lambda", "nu"]


def _load_sbc(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def _check_sbc_artifact(data: dict, expected_sims: int) -> None:
    assert data["n_sims"] == expected_sims
    assert data["n_draws"] == 4000
    assert data["burn"] == 2000
    assert data["n_chains"] == 4

    assert data["rank_draws"] == 500
    assert data["rank_bins"] == 20
    assert data["rank_thinning"] == 32

    assert data["parameter_names"] == EXPECTED_PARAMETERS
    assert data["rank_fractions_parameter"] == "mu"

    assert len(data["rank_fractions"]) == expected_sims
    assert all(
        math.isfinite(x) and 0.0 <= x <= 1.0
        for x in data["rank_fractions"]
    )

    assert len(data["parameters"]) == 4

    for name in EXPECTED_PARAMETERS:
        assert name in data["parameters"]

        block = data["parameters"][name]
        ranks = block["rank_fractions"]

        assert len(ranks) == expected_sims
        assert all(
            math.isfinite(x) and 0.0 <= x <= 1.0
            for x in ranks
        )

        for key in (
            "mean_rank_fraction",
            "mean_rank_mcse_under_uniform",
            "mean_raw_rank_fraction",
            "coverage95",
            "ess_bulk_raw_median",
            "ess_bulk_raw_p05",
            "ess_bulk_raw_min",
            "ess_bulk_rank_median",
            "ess_bulk_rank_p05",
            "ess_bulk_rank_min",
            "rhat_raw_median",
            "rhat_raw_p95",
            "rhat_raw_max",
            "rhat_rank_median",
            "rhat_rank_p95",
            "rhat_rank_max",
        ):
            assert math.isfinite(block[key])

        assert 0.0 <= block["coverage95"] <= 1.0


def test_sbc_200_artifact_structure():
    _check_sbc_artifact(_load_sbc("sbc_200.json"), 200)


def test_sbc_1000_artifact_structure():
    _check_sbc_artifact(_load_sbc("sbc_1000.json"), 1000)