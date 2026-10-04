# Model specification

The historical analyses use the reported covariance structure with the stated active uncertainty convention. Where specified, extra dispersion is represented by a proportional covariance scale parameter lambda.

The primary prospective analysis fits the historical archive with BIPM-14 excluded because the later NIST configuration is lineage-related to the BIPM apparatus. The full historical archive is retained as a robustness configuration. The exclusion is an information-separation choice and does not establish orthogonal independence of the later NIST result.

The canonical Bayesian hierarchical model used by the NUTS reference and archived Student-t scalar predictive-band artifact uses a multivariate Student-t likelihood with degrees of freedom parameterized through `nu - 2`. The separate `analyses/hierarchical_sensitivity/hierarchical_meta.py` calculation is a Gaussian random-effects sensitivity analysis: it uses the same historical covariance structure and tau/lambda parameterization but a multivariate Gaussian likelihood and does not include `nu`. These are distinct analyses and should not be described interchangeably.

The predictive analysis uses each explicitly named historical training configuration to evaluate the same later NIST configuration vector. Where a published historical-to-NIST cross-covariance matrix is unavailable, zero cross-covariance is an explicit analysis assumption rather than an empirical claim.

The closed-form NIST predictive calculation evaluates two covariance conventions explicitly. Under the primary uniform convention, the fitted historical scale lambda^2 multiplies both the historical GLS-mean variance and the published NIST covariance. Under the mean-only sensitivity convention, lambda^2 multiplies only the historical GLS-mean variance and the NIST covariance remains unchanged.

For the uniform convention, if the historical training count is N, the historical residual degrees of freedom are nu=N-1, and the NIST vector dimension is m, then under independent Gaussian common-scale assumptions

    (nu / (N*m)) Q_uniform ~ F(m, nu).

This finite-sample F reference does not apply to the mean-only mixed-scale sensitivity.

Two predictive-tail quantities are kept distinct. The posterior-averaged conditional tail averages conditional tail probabilities evaluated separately for posterior draws. The posterior-predictive mixture tail uses posterior draws to generate replicated observations while holding a predeclared discrepancy reference fixed.

The free-minus-servo torque analysis uses the four configuration differences and their underlying Type-A uncertainty components. The fitted common offset is an apparatus-level descriptive quantity and does not identify a physical mechanism.
