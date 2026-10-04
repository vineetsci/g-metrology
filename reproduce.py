#!/usr/bin/env python3
"""Run the G-metrology reproducibility workflow in core, full, or archive mode."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ENV = {
    **os.environ,
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
    "G_METROLOGY_CUMULATIVE_SUITE": "1",
    "PYTHONPATH": os.pathsep.join(
        [str(ROOT / "src")]
        + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])
    ),
}


@dataclass(frozen=True)
class Stage:
    label: str
    command: tuple[str, ...]
    kind: str = "script"
    skip: str | None = None
    outputs: tuple[str, ...] = ()


CORE = [
    Stage("Historical dispersion", ("analyses/historical_dispersion/historical_dispersion.py",), outputs=("results/machine_readable/historical_dispersion.json",)),
    Stage("Prospective NIST prediction", ("analyses/prospective_nist/prospective_nist.py",), outputs=("results/machine_readable/prospective_nist.json",)),
    Stage("Gaussian proportional common-scale predictive band", ("analyses/gaussian_predictive_band/gaussian_predictive_band.py",), outputs=("results/machine_readable/gaussian_predictive_band.json",)),
    Stage("Torque offset and fractional-scale comparison", ("analyses/torque_offset/torque_offset.py",), outputs=("results/machine_readable/torque_offset.json",)),
    Stage("Precision model comparison", ("analyses/precision_model_comparison/precision_model_comparison.py",), outputs=("results/machine_readable/precision_model_compare.json",)),
    Stage("Manuscript figures", ("paper/make_figures.py",), outputs=("paper/figures/fig1_historical_archive.pdf", "paper/figures/fig2_nist_internal_dispersion.pdf", "paper/figures/fig3_torque_common_offset.pdf")),
    Stage("Manuscript and auxiliary tables", ("paper/make_tables.py",), outputs=("paper/tables/table1_historical_dispersion.tex", "paper/tables/table2_variance_models.tex", "paper/tables/table4_prospective_nist.tex", "paper/tables/table6_torque_models.tex", "paper/tables/gaussian_predictive_band.tex")),
    Stage("Core scientific regression tests", ("-m", "pytest", "-q", "tests/test_scientific_regression.py"), kind="pytest"),
]

FULL_ADDITIONS = [
    Stage("Covariance stress analysis", ("analyses/covariance_stress/covariance_stress.py",), outputs=("results/machine_readable/covariance_stress.json",)),
    Stage("Predictive-tail validation", ("analyses/predictive_tail_validation/predictive_tail_validation.py",), outputs=("results/machine_readable/predictive_tail_validation.json",)),
    Stage("NIST mechanism constraints", ("analyses/mechanism_constraints/nist_mechanism_audit.py",), outputs=("results/machine_readable/nist_mechanism_constraints.json",)),
    Stage("NIST factorial/additive structure", ("analyses/nist_factorial/nist_factorial.py",), outputs=("results/machine_readable/nist_factorial_results.json",)),
    Stage("Gaussian hierarchical random-effects sensitivity", ("analyses/hierarchical_sensitivity/hierarchical_meta.py",), outputs=("results/machine_readable/hierarchical_meta_results.json",)),
    Stage("Student-t nu-prior sensitivity", ("analyses/variance_sensitivity/nu_prior_sensitivity.py",), outputs=("results/machine_readable/nu_prior_sensitivity.json",)),
    Stage("NUTS reference integrity", ("analyses/sampler_validation/nuts_reference_integrity.py",)),
]

FULL_SBC_STAGE = Stage(
    "Simulation-based calibration (200 simulations)",
    ("analyses/sampler_validation/sbc.py",),
    kind="sbc",
    skip="sbc",
    outputs=("results/machine_readable/sbc_200.json",),
)

ARCHIVE_ADDITIONS = [
    Stage("NIST structural profiles (six candidate structures)", ("analyses/nist_structure/nist_structure.py",), outputs=("results/machine_readable/nist_structure_profiles.json",)),
    Stage("Prequential historical scoring", ("analyses/prequential/prequential.py",), outputs=("results/machine_readable/prequential_results.json",)),
    Stage("Prequential variance sensitivity", ("analyses/variance_sensitivity/prequential_variance.py",), outputs=("results/machine_readable/prequential_variance_results.json",)),
    Stage("Fresh NUTS sampling", ("analyses/sampler_validation/run_nuts.py",), skip="nuts", outputs=("results/nuts_reference/free.nc", "results/nuts_reference/codata.nc", "results/nuts_reference/run_manifest.json")),
    Stage("NUTS diagnostics", ("analyses/sampler_validation/nuts_diagnostics.py",), skip="nuts", outputs=("results/nuts_reference/nuts_diagnostics.json",)),
    Stage(
        "Simulation-based calibration (1,000 simulations; one run with 200-simulation checkpoint)",
        ("analyses/sampler_validation/sbc.py",),
        kind="sbc_archive",
        skip="sbc",
        outputs=("results/machine_readable/sbc_200.json", "results/machine_readable/sbc_1000.json"),
    ),
    Stage("Archived Student-t predictive-band integrity", ("-m", "pytest", "-q", "tests/test_archived_student_t_predictive_band.py"), kind="pytest"),
]

COMPLETE_PYTEST_STAGE = Stage(
    "Complete pytest regression and integrity suite",
    ("-m", "pytest", "-q", "tests"),
    kind="pytest",
)

PAPER_ONLY = [
    Stage("Manuscript figures", ("paper/make_figures.py",), outputs=("paper/figures/fig1_historical_archive.pdf", "paper/figures/fig2_nist_internal_dispersion.pdf", "paper/figures/fig3_torque_common_offset.pdf")),
    Stage("Gaussian proportional common-scale predictive band", ("analyses/gaussian_predictive_band/gaussian_predictive_band.py",), outputs=("results/machine_readable/gaussian_predictive_band.json",)),
    Stage("Manuscript and auxiliary tables", ("paper/make_tables.py",), outputs=("paper/tables/table1_historical_dispersion.tex", "paper/tables/table2_variance_models.tex", "paper/tables/table4_prospective_nist.tex", "paper/tables/table6_torque_models.tex", "paper/tables/gaussian_predictive_band.tex")),
]

SBC_WORKERS = 4
FULL_SBC_ARGS = (
    "--sims", "200", "--draws", "4000", "--burn", "2000",
    "--chains", "4", "--workers", str(SBC_WORKERS),
    "--output-name", "sbc_200.json",
)
ARCHIVE_SBC_ARGS = (
    "--sims", "1000", "--draws", "4000", "--burn", "2000",
    "--chains", "4", "--workers", str(SBC_WORKERS),
    "--output-name", "sbc_1000.json",
    "--checkpoint-sims", "200",
    "--checkpoint-output-name", "sbc_200.json",
)


def _format_elapsed(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:5.1f} s"
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{sec:02d}" if hours else f"{minutes:02d}:{sec:02d}"


def _run_process(command: list[str]) -> tuple[int, str, str, float]:
    start = time.perf_counter()
    proc = subprocess.Popen(
        command,
        cwd=ROOT,
        env=ENV,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stdout, stderr = proc.communicate()
    return proc.returncode, stdout, stderr, time.perf_counter() - start


def _run_stage(stage: Stage) -> bool:
    """Run a quiet analysis stage, exposing output only if it fails."""
    command = [sys.executable, str(ROOT / stage.command[0]), *stage.command[1:]]
    if stage.kind == "pytest":
        command = [sys.executable, *stage.command]

    code, stdout, stderr, elapsed = _run_process(command)
    ok = code == 0
    if not ok:
        if stdout.strip():
            print("--- standard output ---", flush=True)
            print(stdout.rstrip(), flush=True)
        if stderr.strip():
            print("--- standard error ---", flush=True)
            print(stderr.rstrip(), flush=True)

    if ok and stage.outputs:
        missing = []
        for relpath in stage.outputs:
            path = ROOT / relpath
            if not path.exists() or path.stat().st_size == 0:
                missing.append(relpath)
        if missing:
            ok = False
            print("--- missing or empty expected outputs ---", flush=True)
            for relpath in missing:
                print(relpath, flush=True)

    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {stage.label:<62} {_format_elapsed(elapsed):>8}", flush=True)
    return ok


def _run_sbc_stage(stage: Stage, args: list[str]) -> bool:
    """Run SBC with inherited stdout so its determinate progress line remains animated."""
    command = [sys.executable, str(ROOT / stage.command[0]), *args]
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=ROOT,
        env=ENV,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - start
    ok = proc.returncode == 0
    if not ok and proc.stderr.strip():
        print("--- standard error ---", flush=True)
        print(proc.stderr.rstrip(), flush=True)

    if ok and stage.outputs:
        missing = []
        for relpath in stage.outputs:
            path = ROOT / relpath
            if not path.exists() or path.stat().st_size == 0:
                missing.append(relpath)
        if missing:
            ok = False
            print("--- missing or empty expected outputs ---", flush=True)
            for relpath in missing:
                print(relpath, flush=True)

    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {stage.label:<62} {_format_elapsed(elapsed):>8}", flush=True)
    return ok


def _stages_for(mode: str) -> list[Stage]:
    if mode == "core":
        return list(CORE)

    # Full and Archive run the complete pytest suite at the end.
    # The core scientific regression file is therefore not run separately
    # in those modes, avoiding duplicate execution.
    core_analyses = list(CORE[:-1])

    if mode == "full":
        return core_analyses + list(FULL_ADDITIONS) + [
            FULL_SBC_STAGE,
            COMPLETE_PYTEST_STAGE,
        ]

    archive_stages = list(FULL_ADDITIONS)
    archive_stages.extend(ARCHIVE_ADDITIONS)
    return core_analyses + archive_stages + [COMPLETE_PYTEST_STAGE]

def _print_plan(mode: str, stages: list[Stage], skip_nuts: bool, skip_sbc: bool) -> None:
    print(f"G-metrology reproducibility suite — {mode.upper()}", flush=True)
    if mode == "core":
        print("Core: primary manuscript analyses and their verification.", flush=True)
    elif mode == "full":
        print("Full: core plus supplementary robustness and supporting analyses, including one 200-simulation SBC execution check.", flush=True)
    else:
        print("Archive: core plus supplementary, archived, and exploratory analyses, including fresh Bayesian sampling and exactly one 1,000-simulation SBC run; its first 200 simulations are saved as a checkpoint and the same run continues to 1,000.", flush=True)
    print("Task plan:", flush=True)
    for i, stage in enumerate(stages, 1):
        skipped = (stage.skip == "nuts" and skip_nuts) or (stage.skip == "sbc" and skip_sbc)
        marker = "-" if skipped else " "
        note = " — skipped by flag" if skipped else ""
        print(f"  {marker} {i:02d}. {stage.label}{note}", flush=True)
    print(flush=True)


def _run_mode(mode: str, skip_nuts: bool, skip_sbc: bool) -> int:
    stages = _stages_for(mode)
    _print_plan(mode, stages, skip_nuts, skip_sbc)
    executed = 0
    suite_start = time.perf_counter()
    for stage in stages:
        if stage.skip == "nuts" and skip_nuts:
            continue
        if stage.skip == "sbc" and skip_sbc:
            continue

        executed += 1
        if stage.kind == "sbc":
            ok = _run_sbc_stage(stage, list(FULL_SBC_ARGS))
        elif stage.kind == "sbc_archive":
            ok = _run_sbc_stage(stage, list(ARCHIVE_SBC_ARGS))
        else:
            ok = _run_stage(stage)

        if not ok:
            print(flush=True)
            print(f"SUMMARY: FAIL — stopped at {stage.label}.", flush=True)
            return 1

    elapsed = time.perf_counter() - suite_start
    print(flush=True)
    print(f"SUCCESS — all {executed} executed stages passed.", flush=True)
    print(f"TOTAL SUITE ELAPSED: {_format_elapsed(elapsed)}", flush=True)
    return 0


def _run_targeted_paper() -> int:
    for stage in PAPER_ONLY:
        if not _run_stage(stage):
            return 1
    print("SUCCESS — manuscript-output tasks passed.", flush=True)
    return 0


def _run_targeted_gaussian_predictive_band() -> int:
    stage = Stage("Gaussian proportional common-scale predictive band", ("analyses/gaussian_predictive_band/gaussian_predictive_band.py",), outputs=("results/machine_readable/gaussian_predictive_band.json",))
    if not _run_stage(stage):
        return 1
    print("SUCCESS — Gaussian predictive-band task passed.", flush=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the G-metrology reproducibility workflow.")
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--core", action="store_true", help="Core mode: primary manuscript analyses and their verification (default).")
    modes.add_argument("--full", action="store_true", help="Full mode: core plus supplementary robustness and supporting analyses, with one 200-simulation SBC execution check.")
    modes.add_argument("--archive", action="store_true", help="Archive mode: core plus supplementary, archived, and exploratory analyses, including fresh Bayesian sampling and exactly one 1,000-simulation SBC run.")
    modes.add_argument("--paper", action="store_true", help="Generate manuscript figures and tables only.")
    modes.add_argument("--gaussian-predictive-band", action="store_true", help="Generate the primary Gaussian proportional common-scale predictive band only.")
    ap.add_argument("--skip-nuts", action="store_true", help="With --archive, skip fresh NUTS sampling and NUTS diagnostics. Accepted with --full as a no-op.")
    ap.add_argument("--skip-sbc", action="store_true", help="With --full or --archive, skip the SBC stage for that mode.")
    args = ap.parse_args()

    if args.paper:
        if args.skip_nuts or args.skip_sbc:
            ap.error("--skip-nuts/--skip-sbc cannot be combined with --paper")
        return_code = _run_targeted_paper()
        raise SystemExit(return_code)
    if args.gaussian_predictive_band:
        if args.skip_nuts or args.skip_sbc:
            ap.error("--skip-nuts/--skip-sbc cannot be combined with --gaussian-predictive-band")
        return_code = _run_targeted_gaussian_predictive_band()
        raise SystemExit(return_code)
    if args.core and (args.skip_nuts or args.skip_sbc):
        ap.error("--skip-nuts/--skip-sbc require --full or --archive")

    mode = "archive" if args.archive else "full" if args.full else "core"
    raise SystemExit(_run_mode(mode, args.skip_nuts, args.skip_sbc))


if __name__ == "__main__":
    main()
