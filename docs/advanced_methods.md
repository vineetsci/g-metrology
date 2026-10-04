# Advanced methods and reference computations

The advanced analyses provide additional checks on model behavior, predictive definitions, sensitivity to the Student-t degrees-of-freedom prior, and computational implementation. The canonical NUTS pathway uses the multivariate Student-t hierarchical model. The separate hierarchical-sensitivity script is a Gaussian random-effects sensitivity analysis and is not the same model.

The saved NUTS calculations used four chains with 2,000 tuning draws and 4,000 retained draws per chain. Their diagnostics include rank-normalized R-hat, effective sample sizes, BFMI, divergences, tree-depth information, and related chain summaries.

Simulation-based calibration is a calibration check under the assumed hierarchical generative model. It does not validate the physical model or establish that the historical covariance is correct.

The prior-sensitivity calculations are stochastic Monte Carlo analyses. Numerical realizations may vary with the execution environment, so the results are interpreted using posterior summaries, diagnostics, and the stability of the scientific conclusion across plausible prior settings rather than by requiring identical individual Monte Carlo draws.