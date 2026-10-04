from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from g_metrology.data import load_yaml
from g_metrology.standardization import StandardizationContext
from analyses.archived_student_t_predictive_band.archived_student_t_predictive_band import (
    EXPECTED_CHAINS,
    EXPECTED_DRAWS_PER_CHAIN,
    OUTPUT_PATH,
    PREDICTIVE_DRAWS_PER_POSTERIOR,
    REFERENCE_G_SI,
    TRAINING_DATA_PATH,
    U_PPM_GRID,
    _read_posterior,
    predictive_draws,
    quantiles_for_uncertainty,
)


def test_archived_student_t_analysis_script_and_artifact_are_aligned():
    data = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    script = ROOT / data["protocol"]["analysis_script"]
    assert script.exists()
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    assert data["protocol"]["analysis_script_sha256"] == digest
    assert data["protocol"]["status"] == "archived historical Student-t predictive-band artifact"
    assert data["protocol"]["historical_generation_package_version"] == "0.9.5rc5"
    assert data["archive_status"]["numerical_result_unchanged"] is True


def test_archived_student_t_intervals_are_finite_ordered_and_width_increases():
    data = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    intervals = data["future_intervals"]
    assert [x["u_ppm"] for x in intervals] == list(U_PPM_GRID)
    for item in intervals:
        assert np.isfinite(item["predictive_q025_si"])
        assert np.isfinite(item["predictive_center_si"])
        assert np.isfinite(item["predictive_q975_si"])
        assert item["predictive_q025_si"] < item["predictive_center_si"] < item["predictive_q975_si"]
        assert item["half_width_si"] > 0
    widths = [x["half_width_si"] for x in intervals]
    assert widths[0] < widths[1] < widths[2]


def test_ppm_to_si_conversion_uses_frozen_reference():
    data = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    for item in data["future_intervals"]:
        assert np.isclose(item["u_si"], item["u_ppm"] * 1e-6 * REFERENCE_G_SI, rtol=0, atol=1e-30)
    assert data["frozen_definition"]["reference_g_si"] == REFERENCE_G_SI


def test_predictive_generation_is_invariant_to_draw_order():
    mu, tau, lam, nu, chain, draw = _read_posterior(ROOT / "results" / "nuts_reference" / "free.nc")
    ctx = StandardizationContext.from_data(load_yaml(TRAINING_DATA_PATH).y)
    u_std = (20.0e-6 * REFERENCE_G_SI) / ctx.scale_si

    # The production calculation uses every frozen posterior draw. This regression test
    # checks the same deterministic draw-addressing invariant on a representative
    # subset to avoid repeatedly constructing hundreds of thousands of RNG objects.
    n_per_chain = min(64, mu.shape[1])
    subset = (mu[:, :n_per_chain], tau[:, :n_per_chain], lam[:, :n_per_chain],
              nu[:, :n_per_chain], chain[:, :n_per_chain], draw[:, :n_per_chain])
    base = predictive_draws(*subset, u_std=u_std)
    rng = np.random.default_rng(271828)
    flat = [a.reshape(-1) for a in subset]
    perm = rng.permutation(len(flat[0]))
    shuffled = [a[perm].reshape(mu[:, :n_per_chain].shape) for a in flat]
    reordered = predictive_draws(*shuffled, u_std=u_std)
    assert np.allclose(np.sort(base), np.sort(reordered), rtol=0, atol=0)


def test_archived_student_t_training_has_no_nist_data():
    data = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    blob = json.dumps(data).lower()
    assert "nist-2026" in blob  # explicit statement that it was not used
    assert data["frozen_definition"]["no_nist_2026_used"] is True
    assert data["protocol"]["training_dataset_label"] == "CODATA-2018 G"


def test_archived_student_t_output_contains_exact_frozen_sampler_metadata():
    data = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    sampler = data["sampler"]
    assert sampler["algorithm"] == "PyMC NUTS"
    assert sampler["chains"] == EXPECTED_CHAINS
    assert sampler["draws_per_chain"] == EXPECTED_DRAWS_PER_CHAIN
    assert sampler["tune_per_chain"] == 2000
    assert sampler["target_accept"] == 0.95
    assert sampler["seed"] == 20260903
    assert sampler["native_progressbar"] is False
    assert data["frozen_definition"]["predictive_draws_per_posterior_draw"] == PREDICTIVE_DRAWS_PER_POSTERIOR
    assert data["frozen_definition"]["predictive_draws_per_uncertainty"] == EXPECTED_CHAINS * EXPECTED_DRAWS_PER_CHAIN * PREDICTIVE_DRAWS_PER_POSTERIOR


def test_u_grid_validation_rejects_invalid_uncertainty():
    mu = np.array([[0.0]])
    tau = np.array([[1.0]])
    lam = np.array([[1.0]])
    nu = np.array([[10.0]])
    chain = np.array([[0]])
    draw = np.array([[0]])
    ctx = StandardizationContext(0.0, 1.0)
    with pytest.raises(ValueError):
        quantiles_for_uncertainty(mu, tau, lam, nu, chain, draw, ctx, 0.0)


def test_analysis_script_hash_recorded():
    data = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    script = ROOT / data["protocol"]["analysis_script"]
    digest = hashlib.sha256(script.read_bytes()).hexdigest()
    assert data["protocol"]["analysis_script_sha256"] == digest
