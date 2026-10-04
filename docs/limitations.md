# Limitations

The historical and NIST measurements are not linked by a published numerical cross-covariance matrix. Analyses that use zero cross-covariance therefore state that choice explicitly as an assumption.

The 2026 NIST result is lineage-related to the BIPM apparatus and is not an orthogonal independent replication. The primary prospective analysis therefore excludes BIPM-14 from the historical training set to reduce direct overlap; this does not remove all lineage dependence.

The conclusion of the closed-form NIST predictive test depends materially on the chosen covariance convention. Uniformly scaling the historical and NIST covariance gives a non-extreme NIST vector, whereas scaling only the historical-mean variance leaves the vector discrepant. The data do not independently identify which convention is physically appropriate.

The hierarchical decomposition of extra dispersion into tau and lambda is weakly identified and is treated as a sensitivity analysis rather than as a uniquely measured physical decomposition.

The approximately common free-minus-servo torque offset is an apparatus-level descriptive result. The present data do not identify its physical cause.

A simple universal fixed-torque mapping between historical extra dispersion and the NIST offset is not supported. This does not exclude configuration-dependent or otherwise structured effects.

The proportional variance model is preferred among the prespecified low-dimensional comparisons considered, but the preference is not decisive evidence for universality.

The saved NUTS runs provide convergence diagnostics for those runs. A separate multi-start initialization experiment was not performed.
