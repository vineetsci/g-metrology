# Data provenance

The repository contains one authoritative copy of each active scientific input dataset.

`data/codata_2010_g.yaml` contains the historical 2010 reference dataset.

`data/codata_2018_g.yaml` contains the historical archive used for the principal covariance, historical-dispersion, and prospective analyses.

`data/nist2026_configurations.yaml` contains the four 2026 NIST configuration observations used in the prospective and apparatus-level analyses.

For the primary prospective test, the BIPM-14 historical determination is excluded from the training set because the later NIST configuration is lineage-related to the BIPM apparatus. This is an information-separation choice intended to reduce direct overlap between the historical reference distribution and the later configuration under assessment; it does not make the NIST result an orthogonally independent replication. The all-history fit is retained as an explicit robustness calculation.

The exclusion rule is defined by the measurement label in the canonical historical dataset rather than by numerical result ranking. The SHA-256 values of the three authoritative input datasets are recorded in `data/MANIFEST.json` and are the only repository-level byte-hash checks enforced against a frozen manifest. Generated results and manuscript artifacts are downstream products: their numerical values are regression-tested where appropriate, but they are not required to have the same byte hash across operating systems or numerical-library builds.


The NIST 2026 dataset also preserves the manuscript-bearing source material from Tables 14-19 in structured fields: Table 14 autocollimator support intervals; Table 15 raw free/servo torque values and Type-A uncertainties; Table 16 four final G determinations through the measurement rows; Table 17 uncertainty-budget components; Table 18 correlations through the dataset correlation matrix; Table 19 configuration-specific dark uncertainties, configuration effects, and final Bayesian consensus. Derived analyses must read these fields through the shared loader rather than retyping publication values in analysis scripts.
