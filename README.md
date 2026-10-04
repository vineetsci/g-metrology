# G Metrology Reproducibility Package

This repository contains the data, software, analyses, tests, and manuscript-output generators required to reproduce the computational results of the accompanying the study _"Covariance scaling and a dual-mode torque diagnostic in measurements of the gravitational constant"_.

## Scientific scope

The study evaluates the adequacy of the quoted covariance model for a historical archive of G measurements, tests its predictive implications for a later lineage-related NIST configuration vector, using a BIPM-14-excluded historical training set as the primary prospective configuration, and analyzes free-versus-servo differences within the NIST apparatus. The full historical archive is retained as an explicit prospective robustness configuration.

The study does not claim a new value of G or identify a universal physical mechanism for the observed discrepancies. Statistical parameters such as lambda, tau, and nu are model parameters and are not interpreted as uniquely identified physical causes.

## Install

Create a clean Python environment and install the package with its test dependencies:

```bash
python -m pip install ".[test]"
```

For the pinned core/test environment:

```bash
python -m pip install -r requirements-core.lock
python -m pip install . --no-deps
```

For NUTS and full SBC stages:

```bash
python -m pip install -r requirements-bayesian.txt
```

## Reproduce the results

The reproducibility workflow has three cumulative execution modes. The default is the lightweight `--core` mode.

- `--core` runs the primary analyses, integrity checks, and regression tests needed for the core reproducible results.
- `--full` extends `--core` with the full simulation-based-calibration (SBC) stage.
- `--archive` extends `--full` with fresh NUTS sampling and archived-reference generation.

```bash
python reproduce.py --core
python reproduce.py --full
python reproduce.py --archive
```

`--skip-nuts` skips the fresh NUTS sampling stage in `--archive`. `--skip-sbc` skips the SBC stage in `--full` or `--archive`. Both flags affect execution only and do not delete existing saved results. On legacy hardware, combine them with `--archive` to shorten runtime.

Targeted commands remain available:

```bash
python reproduce.py --paper
python reproduce.py --gaussian-predictive-band
```

## Predictive artifacts

The primary predictive band is the Gaussian proportional common-scale model. The historical Student-t predictive band is preserved only as an archived artifact. Neither is a numbered manuscript table; see `CLAIMS.md` and `docs/paper_reproduction.md` for their status and provenance.

## Repository layout

- `data/` contains the authoritative scientific input datasets and their integrity manifest.
- `src/g_metrology/` contains the shared scientific implementation.
- `analyses/` contains executable analyses organized by scientific purpose.
- `results/machine_readable/` contains generated numerical results.
- `results/nuts_reference/` contains completed NUTS computational reference artifacts.
- `tests/` contains the standard pytest regression and integrity tests.
- `paper/` contains generators and outputs for manuscript figures and tables.
- `docs/` contains data provenance, model specification, reproducibility, limitations, and publication mapping.

## Traceability

`CLAIMS.md` links each manuscript result to its input data, analysis script, numerical result, and manuscript output. `docs/paper_reproduction.md` gives the paper-level map.

## Data

Each active input dataset occurs exactly once under `data/`, with SHA-256 values recorded in `data/MANIFEST.json`. Generated results and manuscript artifacts are downstream products, not alternative input sources. The software is released under the MIT License; the curated scientific input datasets originate from external published sources and retain their own licensing, attribution, and reuse terms. The MIT license does not relicense those external datasets.

## Terminal and progress display

Successful stages report only completion status and elapsed time; detailed output is shown only on failure. SBC is the only stage with a live progress display. Fresh NUTS runs have the native sampler progress bar disabled.

## Citation

The software citation metadata are provided in `CITATION.cff`.