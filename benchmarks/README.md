# Benchmarks

Benchmarks are executable evidence, not decorative numbers. Run them from the
repository root with the environment being reported.

## Meta-PGS rules

`meta_rules.py` reproduces the same-trait comparison in the algorithm notes
over 30 independent seeds:

```bash
python benchmarks/meta_rules.py
```

It writes every replicate, a mean/standard-deviation table, and machine-readable
provenance under `benchmarks/results/`. The simulation parameter
`shared_error_correlation` correlates score errors as a stylized consequence of
shared discovery information. It is deliberately not called sample overlap:
the simulator does not model overlapping people or discovery GWAS estimation.

The committed results are a regression reference, not a claim about real
cohorts. Re-run before changing a default or making a performance claim.

## Stack scaling

`stack_scaling.py` runs each Gaussian case in a fresh process, warms the
numerical kernels, and records wall time plus absolute peak resident memory:

```bash
python benchmarks/stack_scaling.py
```

The small defaults are suitable for routine validation. Supply representative
cohort dimensions explicitly before making a deployment claim, for example
`--cases 20000x250 50000x500`. Peak RSS includes the Python/NumPy/Numba runtime;
compare rows produced by the same command and environment.

## Summary-statistic calibration

`sumstat_calibration.py` checks three scientific contracts over independent
seeds:

1. `W_ld.T @ D @ W_ld` and `W_gwas.T @ z` against moments computed directly
   from discrete individual-level dosages.
2. Null-GWAS MSE used for path tuning against MSE in a third untouched GWAS.
3. The joint-Gaussian covariance used by PUMAS-style pseudotuning against
   empirical score-by-phenotype fourth moments, for Gaussian and binary traits.

```bash
python benchmarks/sumstat_calibration.py
```

It writes raw per-seed rows, a mean/standard-deviation table, and runtime
provenance under `benchmarks/results/`. The PUMAS covariance errors are
calibration diagnostics for a plug-in approximation, not prediction-accuracy
estimates. Re-run this benchmark before changing its assumptions or presenting
pseudotuning as empirically calibrated in a new setting.

**Table 1. Default calibration, 30 seeds and 10,000 simulated people.**

| Quantity | Mean |
|---|---:|
| Gram identity, maximum absolute error | 1.32e-14 |
| Cross-moment identity, maximum absolute error | 2.79e-15 |
| Null tuning MSE | 0.999952 |
| Null untouched-assessment MSE | 1.000048 |
| Assessment minus tuning MSE | 9.54e-5 |
| Gaussian plug-in covariance relative error | 0.0377 |
| Binary plug-in covariance relative error | 0.1099 |

Thus the algebraic identity is at floating-point error, while selecting on the
tuning GWAS lowers its apparent null MSE relative to untouched assessment. The
last two rows quantify the covariance approximation, not model accuracy.
