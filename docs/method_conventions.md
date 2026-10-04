# Statistical and metrological scale conventions

The package uses two distinct scale quantities and does not treat them as interchangeable.

- `lambda_mle = sqrt(Q/N)` is the maximum-likelihood estimate of a single multiplicative covariance standard-deviation scale under the Gaussian proportional-covariance likelihood used by the historical dispersion calculation.
- `birge_ratio = sqrt(Q/nu)`, with `nu = N-M`, is the metrological consistency/expansion diagnostic used by CODATA and related comparison literature. For a single weighted mean, `nu = N-1`.

CODATA's G adjustments are the direct project precedent: the 2018 and 2022 adjustments use `nu = 16-1 = 15`, report the Birge ratio, and use a separate multiplicative expansion factor of 3.9 to reduce normalized residuals to the stated criterion. The 3.9 factor is therefore not renamed as `lambda_mle`.

For this project, metrology-facing discussion should use the Birge ratio when discussing consistency or uncertainty expansion; the statistical model should retain `lambda_mle` for its likelihood parameterization.

## Prospective covariance conventions

The primary closed-form prospective NIST calculation uses a uniform predictive covariance scaling convention: the historical proportional scale lambda^2 multiplies both the uncertainty of the historical GLS mean and the published NIST covariance.

A mean-only convention is retained as a sensitivity analysis. It inflates only the historical-mean variance and leaves the NIST covariance unchanged. This sensitivity is reported with the plug-in chi-square reference only because the mixed-scale covariance does not have the same simple finite-sample F reference distribution as the uniform model.

For the uniform convention, with N historical training measurements, nu=N-1 residual degrees of freedom, and m components in the NIST vector,

    (nu / (N*m)) Q_uniform ~ F(m, nu)

under the stated independent Gaussian common-scale assumptions.