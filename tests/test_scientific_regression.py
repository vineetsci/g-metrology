from __future__ import annotations
import hashlib
import json
import math
import subprocess
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
R = ROOT / "results" / "machine_readable"


# Regression tests use hard-coded accepted anchors for established scientific outputs
# and independent recomputation for derived quantities. Floating-point tolerances are
# chosen according to the numerical operation and quantity being tested; exact equality
# is reserved for discrete/integrity checks.


def load(name: str):
    return json.loads((R / name).read_text())


def test_historical_accepted_anchors():
    d = load("historical_dispersion.json")
    assert set(d["fits"]) == {"all", "exclude_BIPM14_only"}
    assert "trend_sensitivity" not in d
    f = d["fits"]["all"]
    assert math.isclose(f["q"], 195.67875422297615, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(f["lambda_mle"], 3.497130557891142, rel_tol=0, abs_tol=1e-14)
    b = d["fits"]["exclude_BIPM14_only"]
    assert math.isclose(b["lambda_mle"], 2.967469835321905, rel_tol=1e-7, abs_tol=1e-14)

def test_prospective_accepted_anchors():
    d = load("prospective_nist.json")
    pc = d["configurations"]
    primary = pc["exclude_BIPM14_only"]
    allhist = pc["all_history"]

    assert set(pc) == {"exclude_BIPM14_only", "all_history"}

    assert d["primary_configuration"] == "exclude_BIPM14_only"

    assert primary["training_count"] == 15
    assert primary["training_dof"] == 14
    assert primary["training_indices"] == [i for i in range(16) if i != 10]

    assert math.isclose(
        primary["lambda_mle"],
        2.967469835321905,
        rel_tol=1e-7,
        abs_tol=1e-14,
    )
    assert math.isclose(
        primary["Q_without_inflation"],
        42.62463368656711,
        rel_tol=1e-7,
        abs_tol=1e-12,
    )
    assert math.isclose(
        primary["Q_with_uniform_inflation"],
        4.840475582897763,
        rel_tol=1e-7,
        abs_tol=1e-12,
    )
    assert math.isclose(
        primary["p_with_uniform_inflation_chi2"],
        0.3040607648643238,
        rel_tol=1e-7,
        abs_tol=1e-14,
    )
    assert math.isclose(
        primary["p_with_uniform_inflation_finite_sample"],
        0.3822864894701691,
        rel_tol=1e-7,
        abs_tol=1e-14,
    )
    assert primary["finite_sample_F_degrees_of_freedom"] == [4, 14]
    assert math.isclose(
        primary["finite_sample_F_statistic"],
        (14.0 / (15.0 * 4.0)) * primary["Q_with_uniform_inflation"],
        rel_tol=0,
        abs_tol=1e-15,
    )

    assert math.isclose(
        primary["Q_with_mean_only_inflation"],
        36.37814668304636,
        rel_tol=1e-7,
        abs_tol=1e-12,
    )
    assert math.isclose(
        primary["p_with_mean_only_inflation_chi2"],
        2.419022008278885e-07,
        rel_tol=1e-7,
        abs_tol=1e-15,
    )

    assert allhist["training_count"] == 16
    assert allhist["training_dof"] == 15
    assert math.isclose(
        allhist["lambda_mle"],
        3.497130557891142,
        rel_tol=0,
        abs_tol=1e-14,
    )
    assert math.isclose(
        allhist["Q_with_uniform_inflation"],
        3.8570594645839007,
        rel_tol=1e-7,
        abs_tol=1e-12,
    )
    assert math.isclose(
        allhist["p_with_uniform_inflation_chi2"],
        0.4256962424009135,
        rel_tol=1e-7,
        abs_tol=1e-14,
    )
    assert math.isclose(
        allhist["p_with_uniform_inflation_finite_sample"],
        0.4863055977698166,
        rel_tol=1e-7,
        abs_tol=1e-14,
    )
    assert math.isclose(
        allhist["Q_with_mean_only_inflation"],
        37.678645413955245,
        rel_tol=1e-7,
        abs_tol=1e-12,
    )
    assert math.isclose(
        allhist["p_with_mean_only_inflation_chi2"],
        1.305307842382087e-07,
        rel_tol=1e-7,
        abs_tol=1e-15,
    )

def test_prospective_predictive_conventions_are_independently_recomputed():
    from scipy.stats import chi2, f
    from g_metrology.data import load_yaml, active_covariance_from_dataset

    h = load_yaml(ROOT / "data" / "codata_2018_g.yaml")
    n = load_yaml(ROOT / "data" / "nist2026_configurations.yaml")

    y = h.y
    S = active_covariance_from_dataset(h)
    yn = n.y
    Sn = active_covariance_from_dataset(n)

    idx = np.array([i for i in range(len(y)) if i != 10], dtype=int)

    SS = S[np.ix_(idx, idx)]
    W = np.linalg.inv(SS)
    one = np.ones(len(idx))
    a = float(one @ W @ one)
    mu = float(one @ W @ y[idx] / a)

    q_hist = float((y[idx] - mu) @ W @ (y[idx] - mu))
    lam = math.sqrt(q_hist / len(idx))

    d = yn - mu

    V0 = Sn + (1 / a) * np.ones((4, 4))
    q_uniform = float(
        d @ np.linalg.solve(V0, d)
    ) / lam**2

    V_mean_only = Sn + (lam**2 / a) * np.ones((4, 4))
    q_mean_only = float(
        d @ np.linalg.solve(V_mean_only, d)
    )

    out = load("prospective_nist.json")["configurations"]["exclude_BIPM14_only"]

    assert math.isclose(
        out["Q_with_uniform_inflation"],
        q_uniform,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        out["Q_with_mean_only_inflation"],
        q_mean_only,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        out["p_with_mean_only_inflation_chi2"],
        chi2.sf(q_mean_only, 4),
        rel_tol=0,
        abs_tol=1e-15,
    )
    assert math.isclose(
        out["p_with_uniform_inflation_finite_sample"],
        f.sf((14 / (15 * 4)) * q_uniform, 4, 14),
        rel_tol=0,
        abs_tol=1e-15,
    )

def test_four_point_torque_accepted_anchors():
    d = load("torque_offset.json")
    assert math.isclose(d["common_offset_pNm"], 1.9183544303807258, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(d["standard_error_pNm"], 0.21513169354983533, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(d["chi2"], 1.0928919182127068, rel_tol=0, abs_tol=1e-14)
    assert d["dof"] == 3
    assert math.isclose(d["p_value"], 0.7787902741569382, rel_tol=0, abs_tol=1e-15)


def test_fractional_torque_fit_matches_independent_recompute():
    from scipy.stats import chi2
    from g_metrology.torque import (
        fractional_torque_fit,
        TABLE15_DELTA_PNM,
        TABLE15_TYPEA_PNM,
        TABLE15_SIGNAL_PNM,
    )
    delta = np.asarray(TABLE15_DELTA_PNM, dtype=float)
    u = np.asarray(TABLE15_TYPEA_PNM, dtype=float)
    signal = np.asarray(TABLE15_SIGNAL_PNM, dtype=float)
    w = 1.0 / u**2
    denom = np.sum(w * signal**2)
    f = np.sum(w * signal * delta) / denom
    se = np.sqrt(1.0 / denom)
    q = np.sum(w * (delta - f * signal)**2)
    p = chi2.sf(q, 3)
    actual = fractional_torque_fit()
    assert np.allclose(actual, (f, se, q, p), rtol=0, atol=1e-14)


def test_fractional_torque_fit_is_permutation_invariant():
    from g_metrology.torque import fractional_torque_fit, TABLE15_DELTA_PNM, TABLE15_TYPEA_PNM, TABLE15_SIGNAL_PNM
    perm = np.array([2, 0, 3, 1])
    base = fractional_torque_fit()
    permuted = fractional_torque_fit(TABLE15_DELTA_PNM[perm], TABLE15_TYPEA_PNM[perm], TABLE15_SIGNAL_PNM[perm])
    assert np.allclose(base, permuted, rtol=0, atol=1e-14)


def test_fractional_torque_fit_rejects_invalid_inputs():
    from g_metrology.torque import fractional_torque_fit
    with np.testing.assert_raises(ValueError):
        fractional_torque_fit([1, 2], [1, 2, 3], [1, 2, 3])
    with np.testing.assert_raises(ValueError):
        fractional_torque_fit([1, 2], [1, 0], [1, 2])
    with np.testing.assert_raises(ValueError):
        fractional_torque_fit([1, 2], [1, 1], [0, 0])
    with np.testing.assert_raises(ValueError):
        fractional_torque_fit([1, np.nan], [1, 1], [1, 2])


def test_fractional_torque_fit_result_json_agrees_with_functions():
    from g_metrology.torque import common_torque_fit, fractional_torque_fit
    d = load("torque_offset.json")
    abs_fit = common_torque_fit()
    frac_fit = fractional_torque_fit()
    assert np.allclose(
        abs_fit,
        (d["absolute_offset_model"]["estimate_pNm"],
         d["absolute_offset_model"]["standard_error_pNm"],
         d["absolute_offset_model"]["chi2"],
         d["absolute_offset_model"]["p_value_chi2_upper_tail"]),
        rtol=0, atol=1e-14)
    assert np.allclose(
        frac_fit,
        (d["fractional_scale_model"]["estimate_dimensionless"],
         d["fractional_scale_model"]["standard_error_dimensionless"],
         d["fractional_scale_model"]["chi2"],
         d["fractional_scale_model"]["p_value_chi2_upper_tail"]),
        rtol=0, atol=1e-14)
    assert d["fractional_scale_model"]["dof"] == 3
    assert math.isclose(d["fractional_scale_model"]["chi2"], 12.133873827209998, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(d["fractional_scale_model"]["estimate_ppm"], 67.47729268844982, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(
        d["model_comparison"]["delta_chi2_fractional_minus_absolute"],
        d["fractional_scale_model"]["chi2"] - d["absolute_offset_model"]["chi2"],
        rel_tol=0, abs_tol=1e-14)

def test_nist_mechanism_discriminator_predictions_are_explicit_and_reproducible():
    from g_metrology.torque import common_torque_fit, fractional_torque_fit

    d = load("nist_mechanism_constraints.json")
    pred = d["discriminator_predictions"]

    expected_levels = [14.0, 31.0, 62.0]

    assert pred["gravitational_torque_levels_nNm"] == expected_levels
    assert len(pred["absolute_offset_predicted_ppm"]) == 3
    assert len(pred["fractional_scale_predicted_ppm"]) == 3

    common, _, _, _ = common_torque_fit()
    f_hat, _, _, _ = fractional_torque_fit()

    levels = np.asarray(expected_levels, dtype=float)
    expected_absolute_ppm = common * 1000.0 / levels
    expected_fractional_ppm = np.full(3, 1e6 * f_hat)

    assert np.allclose(
        pred["absolute_offset_predicted_ppm"],
        expected_absolute_ppm,
        rtol=0,
        atol=1e-12,
    )
    assert np.allclose(
        pred["fractional_scale_predicted_ppm"],
        expected_fractional_ppm,
        rtol=0,
        atol=1e-12,
    )

def test_gaussian_predictive_band_anchors():
    from scipy.stats import f

    d = load("gaussian_predictive_band.json")
    assert d["protocol"]["analysis_script"] == "analyses/gaussian_predictive_band/gaussian_predictive_band.py"
    assert d["model"]["reference_g_si"] == 6.67430e-11
    assert d["model"]["k"] == 1.0
    assert "conditional on the future determination reporting quoted k=1 standard uncertainty u" in d["model"]["future_u_conditioning"]

    primary = d["configurations"]["exclude_BIPM14_only"]
    assert primary["training_count"] == 15
    assert primary["training_dof"] == 14
    assert math.isclose(primary["lambda_mle"], 2.967469835321905, rel_tol=0, abs_tol=1e-14)
    prospective = load("prospective_nist.json")
    primary_mean_si = float(prospective["configurations"]["exclude_BIPM14_only"]["historical_mean_SI"])
    expected_primary_center_ppm = (primary_mean_si - 6.67430e-11) / 6.67430e-11 * 1e6
    assert math.isclose(
        primary["center_ppm_relative_to_reference"],
        expected_primary_center_ppm,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert math.isclose(primary["F_quantile"], f.ppf(0.95, 1, 14), rel_tol=0, abs_tol=1e-15)
    assert math.isclose(primary["coverage_factor_finite_sample"], math.sqrt((15.0 / 14.0) * f.ppf(0.95, 1, 14)), rel_tol=0, abs_tol=1e-15)

    expected_half_widths = {20.0: 137.2681259598919, 40.0: 266.3160846721925, 60.0: 397.1487393988586}
    intervals = primary["intervals"]
    assert [x["u_ppm"] for x in intervals] == [20.0, 40.0, 60.0]
    for item in intervals:
        assert math.isclose(item["half_width_ppm"], expected_half_widths[item["u_ppm"]], rel_tol=1e-10, abs_tol=1e-10)
        assert item["lower_si"] < item["mu_hat_si"] < item["upper_si"]
    assert intervals[0]["half_width_si"] < intervals[1]["half_width_si"] < intervals[2]["half_width_si"]

    allhist = d["configurations"]["all_history"]
    assert allhist["training_count"] == 16
    assert allhist["training_dof"] == 15
    allhist_mean_si = float(prospective["configurations"]["all_history"]["historical_mean_SI"])
    expected_allhist_center_ppm = (allhist_mean_si - 6.67430e-11) / 6.67430e-11 * 1e6
    assert math.isclose(
        allhist["center_ppm_relative_to_reference"],
        expected_allhist_center_ppm,
        rel_tol=0,
        abs_tol=1e-12,
    )
    assert math.isclose(allhist["coverage_factor_finite_sample"], math.sqrt((16.0 / 15.0) * f.ppf(0.95, 1, 15)), rel_tol=0, abs_tol=1e-15)


def test_gaussian_predictive_band_recomputes_from_historical_inputs():
    d = load("gaussian_predictive_band.json")
    h = load("historical_dispersion.json")
    p = load("prospective_nist.json")
    for name, hname in (("exclude_BIPM14_only", "exclude_BIPM14_only"), ("all_history", "all")):
        out = d["configurations"][name]
        hist = h["fits"][hname]
        prop = p["configurations"][name]
        se = hist["beta_se_si"][0]
        assert math.isclose(out["historical_mean_variance_reported_si2"], se ** 2, rel_tol=0, abs_tol=1e-45)
        for item in out["intervals"]:
            u_si = item["u_si"]
            sigma = prop["lambda_mle"] * math.sqrt(u_si ** 2 + se ** 2)
            assert math.isclose(item["sigma_pred_si"], sigma, rel_tol=0, abs_tol=1e-27)
            expected_half_width = out["coverage_factor_finite_sample"] * sigma
            assert math.isclose(item["half_width_si"], expected_half_width, rel_tol=0, abs_tol=1e-27)


def test_predictive_band_paper_output_present_and_combines_student_t_sensitivity():
    path = ROOT / "paper" / "tables" / "gaussian_predictive_band.tex"
    assert path.exists() and path.stat().st_size > 0
    text = path.read_text(encoding="utf-8")
    assert "Gaussian proportional common-scale (F)" in text
    assert "Gaussian plug-in (1.96)" in text
    assert "Archived Student-$t$ (all 16)" in text


def test_supporting_deterministic_outputs_present():
    for name in [
        "covariance_stress.json",
        "nist_mechanism_constraints.json",
        "nist_factorial_results.json",
        "nist_structure_profiles.json",
    ]:
        assert (R / name).exists()

def test_covariance_stress_uses_nominal_birge_and_exact_gls_mean_variance():
    d = load("covariance_stress.json")
    row = next(
        r for r in d["historical_grid"]
        if r["u_scale"] == 1.0 and r["corr_scale"] == 1.0
    )

    assert math.isclose(
        row["q_nominal"],
        195.67875422297618,
        rel_tol=1e-7,
        abs_tol=1e-10,
    )
    assert math.isclose(
        row["historical_birge_nominal"],
        3.6118209093934524,
        rel_tol=1e-7,
        abs_tol=1e-12,
    )
    assert math.isclose(
        row["nist_Q_gaussian"],
        3.8570594645839007,
        rel_tol=1e-7,
        abs_tol=1e-10,
    )

    assert "post_fit_Q" not in row
    assert "historical_birge" not in row
    nist = d["nist_u_scale_sensitivity_nominal_history"]

    assert len(nist) == 5
    assert [r["u_scale"] for r in nist] == [0.8, 0.9, 1.0, 1.1, 1.2]

    assert all(
        math.isfinite(r["nist_Q_gaussian"]) and
        math.isfinite(r["nist_tail_chi2"])
        for r in nist
    )

def test_prequential_reference_anchors():
    d = load("prequential_results.json")
    assert d["n_predictions"] == 10
    assert math.isfinite(d["cumulative_train_gls_minus_free"])


def test_precision_model_comparison_anchors():
    d = load("precision_model_compare.json")
    prop = next(x for x in d["models"] if x["model"] == "prop")
    power = next(x for x in d["models"] if x["model"] == "power")
    assert math.isclose(prop["bic_weight"], 0.579517768179824, rel_tol=0, abs_tol=1e-15)
    assert math.isclose(power["power_p"], 0.7833355701531679, rel_tol=1e-7, abs_tol=1e-15)
    assert d["bootstrap_replicates"] == 500
    assert d["bootstrap_seed"] == 20260908


def test_predictive_tail_validation_anchors():
    d = json.loads((ROOT / "results" / "machine_readable" / "predictive_tail_validation.json").read_text())
    h = d["hierarchical"]
    assert math.isclose(h["posterior_averaged_conditional_tail"], 0.6069617116805474, rel_tol=1e-7, abs_tol=1e-15)
    assert math.isclose(h["full_posterior_predictive_mixture_tail"], 0.656075, rel_tol=1e-7, abs_tol=1e-15)
    lam = d["lambda_only"]
    # This quadrature result differs by 8.54e-13 between the validated Linux and
    # Windows numerical environments.
    # Allow the observed cross-platform numerical difference rather than requiring
    # bitwise equality for this floating-point quadrature result.
    assert math.isclose(lam["lambda_only_reference_posterior_averaged_conditional_tail"], 0.4443871804332431, rel_tol=0, abs_tol=2e-12)
    assert math.isclose(lam["full_posterior_predictive_mixture_tail"], 0.44407, rel_tol=0, abs_tol=1e-12)
    assert lam["conditional_tail_quadrature_grid_points"] == 18000


def test_yaml_hash_map_matches_manifest():
    from g_metrology.data import CANONICAL_SOURCE_SHA256
    manifest = json.loads((ROOT / "data" / "MANIFEST.json").read_text())
    manifest_hashes = {r["path"]: r["sha256"] for r in manifest["files"]}
    for name, digest in CANONICAL_SOURCE_SHA256.items():
        assert manifest_hashes[name] == digest, f"{name}: data.py vs MANIFEST.json disagree"


def test_authoritative_data_manifest():
    manifest = json.loads((ROOT / "data" / "MANIFEST.json").read_text())
    for record in manifest["files"]:
        path = ROOT / "data" / record["path"]
        assert path.exists()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest == record["sha256"]
        assert path.stat().st_size == record["bytes"]

def test_paper_outputs_regenerate():
    manifest = json.loads(
        (ROOT / "paper" / "CLAIM_PROVENANCE.json").read_text()
    )

    for item in manifest["paper_outputs"]:
        assert (ROOT / item["output"]).exists(), item["output"]
        assert (ROOT / item["generator"]).exists(), item["generator"]
        for inp in item["inputs"]:
            assert (ROOT / inp).exists(), inp

    for generator in (
        ROOT / "paper" / "make_figures.py",
        ROOT / "paper" / "make_tables.py",
    ):
        result = subprocess.run(
            [sys.executable, str(generator)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, (
            f"{generator.name} failed:\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

    for item in manifest["paper_outputs"]:
        path = ROOT / item["output"]
        assert path.exists() and path.stat().st_size > 0, item["output"]

    for legacy in (
        ROOT / "paper" / "figures" / "fig2_prospective_predictive_contrast.pdf",
        ROOT / "paper" / "figures" / "fig2_prospective_predictive_contrast.png",
        ROOT / "paper" / "figures" / "fig4_variance_model_comparison.pdf",
        ROOT / "paper" / "figures" / "fig4_variance_model_comparison.png",
    ):
        assert not legacy.exists(), legacy

def test_paper_claim_provenance_graph():
    manifest = json.loads((ROOT / "paper" / "CLAIM_PROVENANCE.json").read_text())
    assert len(manifest["paper_outputs"]) == 8
    for item in manifest["paper_outputs"]:
        generator_text = (ROOT / item["generator"]).read_text(encoding="utf-8")
        assert "results" in generator_text
        assert item["output"].split("/")[-1].split(".")[0] in generator_text



def test_metrology_scale_conventions():
    from g_metrology.data import load_yaml, active_covariance_from_dataset
    ds = load_yaml(ROOT / "data" / "nist2026_configurations.yaml")
    assert ds.labels == ["copper_servo", "copper_free", "sapphire_servo", "sapphire_free"]
    assert ds.auxiliary is not None
    assert "table14_autocollimator_support_ppm" in ds.auxiliary
    assert "table15_torque_nNm" in ds.auxiliary
    assert "table17_uncertainty_budget_ppm" in ds.auxiliary
    assert "table19_bayesian_consensus" in ds.auxiliary
    C = active_covariance_from_dataset(ds)
    assert np.allclose(np.diag(C), ds.u**2, rtol=1e-12, atol=0)
    hist = load("historical_dispersion.json")
    f = hist["fits"]["all"]
    assert math.isclose(f["lambda_mle"], math.sqrt(f["q"] / 16), rel_tol=1e-12)
    assert math.isclose(f["birge_ratio"], math.sqrt(f["q"] / 15), rel_tol=1e-12)


def test_nist_claim_inputs_are_canonical():
    analysis_files = [
        ROOT / "analyses" / "nist_factorial" / "nist_factorial.py",
        ROOT / "analyses" / "mechanism_constraints" / "nist_mechanism_audit.py",
        ROOT / "analyses" / "nist_structure" / "nist_structure.py",
        ROOT / "analyses" / "prospective_nist" / "prospective_nist.py",
    ]
    forbidden_literals = ["6.673642e-11", "6.674021e-11", "6.672637e-11", "6.673636e-11", "6.67387e-11", "3.8e-15", "31.1979", "31.1962", "31.1842", "31.1828", "31.1856", "31.1836", "13.9799", "13.9778"]
    for path in analysis_files:
        text = path.read_text(encoding="utf-8")
        assert not any(value in text for value in forbidden_literals), f"hard-coded NIST datum in {path}"
    assert "g_metrology.nist_repro" in (ROOT / "analyses" / "nist_factorial" / "nist_factorial.py").read_text(encoding="utf-8")
    assert "g_metrology.nist_repro" in (ROOT / "analyses" / "mechanism_constraints" / "nist_mechanism_audit.py").read_text(encoding="utf-8")
    struct = (ROOT / "analyses" / "nist_structure" / "nist_structure.py").read_text(encoding="utf-8")
    assert "load_yaml" in struct and "active_covariance_from_dataset" in struct
    prospective = (ROOT / "analyses" / "prospective_nist" / "prospective_nist.py").read_text(encoding="utf-8")
    assert "load_yaml" in prospective and "active_covariance_from_dataset" in prospective
