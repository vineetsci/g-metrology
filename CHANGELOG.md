# Changelog

## 1.0.0

- Finalised the public reproducibility release for the accompanying G-metrology study and synchronised the active package identity to version 1.0.0.
- Updated `CLAIMS.md` and `paper/CLAIM_PROVENANCE.json` so that manuscript-bearing numerical claims and Tables 7-8 have explicit traceability to their generating analyses and machine-readable results.
- Added explicit machine-readable discriminator predictions for the dual-mode torque experiment at the three illustrative gravitational torque levels used in the manuscript.
- Completed the declared five-point NIST quoted-error covariance sensitivity and added regression protection requiring all five sensitivity points and finite numerical outputs.
- Preserved the Gaussian future-prediction result as the active predictive analysis while retaining the numerical Gaussian predictive-band table as auxiliary, non-manuscript material.
- Preserved the archived Student-t predictive artifact as historical provenance rather than as a primary manuscript analysis.
- Validated the final repository metadata and citation information, including author/ORCID and repository references, using cffconvert.
- Removed generated build artifacts and added repository ignore rules for caches, virtual environments, build outputs, and other local development files.
- Added `DATA_LICENSE.md` and clarified in the README that external scientific datasets retain their source licensing and attribution requirements.
- Updated documentation to use consistent final-release terminology and removed outdated references to previous versions.
- Updated the archived Student-t predictive-band documentation to describe its status independently of a specific release candidate.
- Strengthened the Student-t simulation-based calibration by evaluating SBC ranks for all four model parameters (\mu, \tau, \lambda, \nu), while retaining the existing 1,000-simulation archive protocol and the same sbc.py/reproduction pipeline.
- Aligned the 200-simulation Full SBC execution check with the Archive sampling configuration so that it uses the same 4-chain, 4,000-draw, 2,000-burn-in protocol.
- Added sampler-mixing diagnostics and ESS-aware rank sampling to the SBC output, without changing the underlying generative model or posterior target.
- Optimized the SBC likelihood evaluation without changing its mathematical target.
- Aligned the Student-t SBC generative and inference calculations with the canonical standardized-coordinate convention used by the model implementation.
- Added a dedicated SBC artifact regression test covering the four-parameter rank structure and sampler diagnostics.
- Removed duplicate execution of the core scientific regression test file in Full and Archive modes; the final complete pytest suite now provides the single consolidated verification stage.
- Added `tools/update_hash.py`, a maintainer utility that recomputes the SHA-256 hashes of the three authoritative input YAML files and updates `data/MANIFEST.json` and `src/g_metrology/data.py` after an explicit confirmation prompt.

## 0.9.5-rc6

- Renamed the primary Gaussian predictive-band analysis, machine-readable result, and Paper-suite auxiliary table to descriptive `gaussian_predictive_band` names and made the Gaussian proportional common-scale model the active predictive model in the reproduction suites.
- Renamed the historical Student-t predictive-band analysis, machine-readable result, documentation, and tests to descriptive `archived_student_t_predictive_band` names and removed the legacy registration-based predictive-band naming from the active package.
- Preserved the archived Student-t numerical predictive result while recording its historical RC5 generation provenance and archival status; the current Archive workflow verifies the artifact rather than regenerating it.
- Synchronized reproducible manuscript tables with Draft 7: Table 2 is the variance-model comparison, Table 4 is the prospective NIST test, and Table 6 is the torque model comparison with the manuscript's corresponding model/parameter structure.
- Retained the Gaussian predictive-band output as an auxiliary Paper-suite table, `paper/tables/gaussian_predictive_band.tex`; it is not a numbered manuscript table.
- Updated `CLAIMS.md`, `paper/CLAIM_PROVENANCE.json`, README/documentation, reproduction stages, targeted commands, and regression tests to use the new descriptive model names and artifact paths.
- Adjusted cumulative-suite regression execution to avoid regenerating manuscript figures/tables a second time immediately after the Paper stage; standalone `pytest -q tests` retains the paper-regeneration check.
- Bumped package metadata and archive root identity to 0.9.5-rc6.
- Improved the Gaussian predictive-band regression checks by deriving the reported center offset from the stored historical SI mean rather than hard-coding floating-point ppm values, while retaining fixed numerical anchors for established scientific results.
- Corrected the NIST 2026 Table 17 uncertainty-budget label `SM_MCR_Rt` to `SM_MCR_Rs` in `data/nist2026_configurations.yaml` to distinguish the source-mass mounting-circle radius from the test-mass mounting-circle radius; numerical values and calculations are unchanged.
- Updated the new hashes for `data/nist2026_configurations.yaml` in `data/MANIFEST.json`
- Removed the obsolete legacy-output cleanup machinery from `paper/make_figures.py` now that the retired figure outputs are no longer part of the package.
- Improved the paper-output regression test to report the underlying generator stdout/stderr when `make_figures.py` or `make_tables.py` fails, while retaining checks that retired figure outputs are absent.
- Finished and updated all references in `data/codata_2018_g.yaml` to have longer, more standard citations than just lastnames and years.
- Fully populated a separate section for manuscript-facing tables in `paper/README.md`
- Created a new file `updatehash.py` to update the hashes for the input files.

## 0.9.5-rc5

- Added the primary Gaussian prospective scalar predictive band for one future independent G determination, conditional on its quoted k=1 standard uncertainty.
- Uses the exact finite-sample F calibration under the declared Gaussian common-scale covariance model, with the profile MLE \(\hat{\lambda}=\sqrt{Q/N}\).
- Added illustrative predictive intervals at future quoted uncertainties of 20, 40, and 60 ppm for both the BIPM-14-excluded primary configuration and the complete historical archive.
- Added the 1.96 Gaussian known-scale plug-in interval as a diagnostic comparison.
- Added the machine-readable Gaussian predictive-band result and the auxiliary Paper-suite predictive-band table; the manuscript itself does not number this auxiliary table.
- Explicitly recorded that the predictive interval is conditional on the future determination's quoted k=1 uncertainty and that the finite-sample F calibration is exact only under the declared common-scale Gaussian model and independence assumptions.

## 0.9.5-rc4

- Added a Gaussian primary prospective scalar predictive band conditional on future k=1 quoted uncertainty, using the exact finite-sample F calibration under the declared common-scale model.
- Added side-by-side manuscript output with the frozen Student-t predictive band retained as a sensitivity analysis.

- Added explicit uniform-covariance and mean-only covariance conventions to the closed-form NIST prospective analysis.
- Added the mean-only NIST sensitivity result and machine-readable regression anchors.
- Added the exact finite-sample F reference for the uniform predictive convention:
  (nu/(N*m))*Q_uniform ~ F(m,nu) under the stated independent Gaussian common-scale assumptions.
- Retained the chi-square tail as a plug-in diagnostic rather than the primary finite-sample reference.
- Corrected the covariance-stress predictive mean-variance calculation to use the exact GLS term 1/(1^T C^{-1} 1).
- Replaced the post-fit covariance-stress `historical_birge` quantity with the nominal pre-inflation Birge ratio.
- Updated manuscript-table generation to expose the primary predictive convention and remove ambiguity about the historical scale shown.
- Added independent regression checks for the two NIST predictive covariance conventions and the finite-sample F transformation.
- Added regression protection against the obsolete post-fit Birge diagnostic.

## 0.9.5-rc3
- Added Fig. 2, `fig2_nist_internal_dispersion`, generated by `fig_nist_dispersion()`, and synchronized the paper-output provenance and reproduction metadata.
- Fixed the bug in line 47 of `analyses/nist_factorial/nist_factorial.py` which computes r1 = NIST_VALUES - X[:, :3] @ beta[:3], where beta was fit with the full 4-column (saturated) design. That was not the additive-model GLS fit.
- Added BIPM Mark II and NIST 4-configuration results to CLAIMS.md
- Added Plain GLS mean and its standard error under the reported covariance in `analyses/nist_factorial/nist_factorial.py`
- `analyses/historical_dispersion/historical_dispersion.py` now makes `historical_dispersion.json` save `beta_se_si` too now.
- GLS ±1 SE band was added to Figure 1 for consistency with Figure 2.
- updated `tests/test_scientific_regression.py::test_paper_claim_provenance_graph` from 6 to 7 in assert len(manifest["paper_outputs"])==7

## 0.9.5-rc2

- Corrected the CODATA-2018 dataset comments to state explicitly that the stored published standard uncertainties do not apply the CODATA-2018 k=3.9 expansion factor; the active analysis remains fixed at k=1.0.
- Synchronized the authoritative CODATA-2018 input SHA-256 and byte count in the data manifest after the comment-only dataset edit.
- Removed obsolete Fig. 2/Fig. 4 references from claim traceability and kept the paper-output regression contract aligned with the two surviving generated figures.
- Updated manuscript-output documentation wording for the reduced figure set.
- Bumped package metadata to 0.9.5-rc2.


## 0.9.5-rc1

- Reworked `fig1_historical_archive` to show pale-orange proportional-inflated uncertainty spans behind the unchanged blue reported-value/error-bar layer.
- Added the fitted historical inflation factor `lambda_MLE` to the Fig. 1 legend and updated the title to distinguish quoted from inflated uncertainties.
- Removed generation and packaged outputs for `fig2_prospective_predictive_contrast` and `fig4_variance_model_comparison`; the figure generator also removes stale copies on rerun.
- Updated manuscript-output provenance, paper documentation, reproduction-stage output contracts, and regression tests for the reduced figure set.
- Bumped package metadata to 0.9.5-rc1.

## 0.9.4-rc10

- Corrected the predictive-tail numerical regression tolerance from `1e-14` to `2e-12`, matching the observed cross-platform floating-point spread without changing the calculation.
- Prevented the predictive-tail validation from computing unused post-sampling log-posterior values (`return_logpost=False`), reducing unnecessary CPU work.
- Added the complete `pytest -q tests` suite as the final stage of Full and Archive modes, so regenerated outputs are tested after all analyses finish.
- Clarified that frozen SHA-256 enforcement applies to the three authoritative input datasets; hashes recorded inside the archived predictive-band artifact are provenance, not cross-platform byte-identity requirements.
- Bumped package metadata to 0.9.4-rc10.

## 0.9.4-rc9

- Corrected the precision-model parametric-bootstrap documentation/code contract to use 500 replicates and record the replicate count and seed in the machine-readable result.
- Replaced the hard-coded lambda-only predictive-tail reference with the same in-script log-lambda quadrature used to construct the historical posterior.
- Preserved the frozen NUTS pointwise log-likelihood generation and reference artifacts.
- Corrected claims/documentation wording where it implied a singular input dataset or qualitative rather than quantitative robustness.
- Clarified that the pinned Bayesian requirements are the target clean-reproduction environment; frozen artifacts retain their observed generation environment for provenance.
- Separated the completed MIT license metadata from other pending metadata in the release checklist.
- Bumped package metadata to 0.9.4-rc9.
