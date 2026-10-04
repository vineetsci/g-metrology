# G Metrology Reproducibility Package

This repository contains the data, software, analyses, tests, and manuscript-output generators required to reproduce the computational results of the accompanying G-metrology study.

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

For NUTS and full SBC:

```bash
python -m pip install -r requirements-bayesian.txt
```

## Reproduce the results

The reproducibility workflow has three cumulative execution modes: `core`, `full`, `archive`. The default is the lightweight `core` mode.

```bash
python reproduce.py --core
python reproduce.py --full
python reproduce.py --archive
```

`--core` runs the primary manuscript analyses and generators, including the Gaussian proportional common-scale prospective predictive band, plus the core regression tests. `--full` adds supplementary robustness and supporting analyses and runs exactly one 200-simulation SBC job as an execution check. `--archive` adds supplementary, archived, and exploratory analyses and fresh NUTS, then runs exactly one 1,000-simulation SBC job; at 200 completed simulations it writes `sbc_200.json` as a checkpoint and continues the same run to 1,000, where it writes `sbc_1000.json`. The archived Student-t predictive-band artifact is checked as an archival object and is not regenerated as a current predictive model.

Targeted commands remain available:

```bash
python reproduce.py --paper
python reproduce.py --gaussian-predictive-band
```

`--skip-nuts` applies to `--archive` and skips fresh NUTS plus its dependent diagnostics; with `--full` it is accepted as a compatibility no-op because fresh NUTS is not in `full`. `--skip-sbc` applies to `--full` (skipping its 200-simulation job) and `--archive` (skipping its single 1,000-simulation job and its 200-simulation checkpoint). Skipping affects execution only and does not delete existing saved results. On legacy hardware, NUTS sampling and especially the 1,000-simulation SBC might take longer than desired; it is advised then to use the `--skip-nuts --skip-sbc` flags for the `--archive` suite.

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

`CLAIMS.md` links each manuscript-bearing result to its authoritative input data, executable analysis, numerical result, and manuscript output. `docs/paper_reproduction.md` gives the corresponding paper-level map.

## Data integrity

Each active input dataset occurs once under `data/`. SHA-256 values are recorded in `data/MANIFEST.json`. Generated results and manuscript artifacts are downstream products and are not alternative input sources.

## Data licensing

The software in this repository is released under the MIT License, as stated in `LICENSE`. The curated scientific input datasets in `data/` originate from external published sources and retain the licensing, attribution, and reuse terms applicable to those sources. The MIT license for the repository software does not relicense those external datasets.

## Terminal and progress display

Successful stages report only completion status and elapsed time; detailed output is shown only on failure. SBC is the only stage with a live progress display. Fresh NUTS runs with its native sampler progress bar disabled.

## Optional task skipping

Use `--skip-nuts` or `--skip-sbc` when iterating locally. These flags affect execution only and do not alter existing saved NUTS or SBC results. They may be combined. They are not part of the default full verification command.

## Citation

The software citation metadata are provided in `CITATION.cff`.

### Scientific provenance
The 2026 NIST/BIPM source file preserves the publication-bearing empirical material from Tables 14-19 in structured fields. Publication outputs are generated from machine-readable results and tracked by `paper/CLAIM_PROVENANCE.json`.
