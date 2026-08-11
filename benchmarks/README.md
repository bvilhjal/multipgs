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

## Fit accuracy

`fit_accuracy.py` measures the headline claim — the learned combination beats
the best single input score on untouched individuals — over 30 seeds per
regime of `simulate_panel`, with the best score chosen on the training split:

```bash
python benchmarks/fit_accuracy.py
```

Replicates run in worker processes (`--jobs`, with BLAS threads pinned to one
per worker) to keep the grid within minutes; output rows are in grid order.
The oracle sits below `h2` because the evaluated predictions exclude
covariates while the simulated phenotype includes them.

**Table 2. Mean held-out R² over 30 seeds per regime (`n_causal=8`).**

| `n` | `K` | `h2` | best single | multi-PGS | oracle | uplift |
|---|---|---|---|---|---|---|
| 2,000 | 50 | 0.2 | 0.082 | 0.170 | 0.180 | +0.088 |
| 2,000 | 50 | 0.5 | 0.202 | 0.432 | 0.439 | +0.231 |
| 2,000 | 200 | 0.2 | 0.089 | 0.155 | 0.168 | +0.066 |
| 2,000 | 200 | 0.5 | 0.227 | 0.420 | 0.429 | +0.193 |
| 10,000 | 50 | 0.2 | 0.098 | 0.174 | 0.176 | +0.076 |
| 10,000 | 50 | 0.5 | 0.244 | 0.439 | 0.440 | +0.195 |
| 10,000 | 200 | 0.2 | 0.081 | 0.161 | 0.164 | +0.080 |
| 10,000 | 200 | 0.5 | 0.206 | 0.416 | 0.417 | +0.210 |

The combination closes essentially the whole gap between the best single
score and the genetic-value oracle in every regime, and `cv_r2` tracks the
held-out number. Support recall is 0.84–1.00 at precision 0.09–0.20: the
lasso spreads weight over correlated near-duplicates, which is the union
support CMSA is designed to produce, not a support-recovery guarantee.

## Null-gate calibration

`null_gate.py` records the nested gate's two operating characteristics over 30
seeds: how often the penalized stack passes when no score carries signal, and
how far `cv_r2` sits from untouched held-out R² when signal is present:

```bash
python benchmarks/null_gate.py
```

**Table 3. Gate behaviour at `K=50`, `h2=0.4`, 30 seeds.**

| `n` | null-model return rate (noise) | null `cv_r2` | signal `cv_r2` − held-out R² |
|---|---|---|---|
| 1,000 | 0.767 | −0.002 | −0.008 ± 0.052 |
| 5,000 | 0.867 | −0.000 | +0.002 ± 0.025 |

The one-standard-error gate is not a 5% test: it still passes a pure-noise
stack in about a fifth of seeds, so a passed gate alone is not evidence of
signal — read it together with `cv_r2`, which stays at zero on noise. Under
signal the internal estimate is unbiased to within a few thousandths,
consistent with the "possibly conservative" language in the theory notes.

## Real-LD score moments

Every other benchmark here builds its LD from a simulator, which leaves two
properties of practice unmeasured because a simulator does not produce them: a
real reference is block-heterogeneous, and a real score panel is rank-deficient
and near-collinear. `real_ld_gram.py` reports what `score_gram` and the fitter's
own moment validation see on a real reference:

```bash
python benchmarks/real_ld_gram.py --ld /path/to/ldpred3_ldref_hm3.npz \
    --scores pgs_catalog_scores/ --check-lowrank
```

The reference is supplied, not shipped. The one this was developed against is
Privé's bigsnpr HapMap3+ European (UK Biobank) LD, converted to ldpred3's block
format by `ldpred3/benchmarks/convert_bigsnpr_ldref.py`: 1,054,330 variants in
625 blocks, 406 of them dense (median 451 variants) and 219 low-rank (median
3,120 variants at median rank 890). Without `--scores` the panel is synthetic
over the real LD, which exercises representation and cost but says nothing
about real score collinearity; the two panel types are recorded separately in
the provenance and must not be compared.

Nothing here is an accuracy claim — it measures moments and what they cost.

## Summary-statistic and individual-level agreement

`sumstat_vs_individual.py` simulates three independent cohorts sharing
variants and component weights — one fits, one tunes, one assesses — and over
30 seeds records whether the moments-only route agrees with the
individual-level CMSA fit, and what each tuning regime costs:

```bash
python benchmarks/sumstat_vs_individual.py
```

**Table 4. Agreement and held-out R² on the untouched cohort, 30 seeds.**

| Quantity | Mean |
|---|---:|
| `beta_std` correlation, individual vs summary fit | 0.9996 |
| Held-out R², individual-level CMSA | 0.4981 |
| Held-out R², summary fit tuned independently | 0.4979 |
| Held-out R², summary fit with PUMAS pseudotuning | 0.4982 |
| Held-out R², summary fit tuned in-sample | 0.4973 |

With exact moments the two routes agree up to selection-routing differences
(nested CMSA versus a tuning GWAS). In this strong-signal, small-`K` regime
the tuning rule costs nothing measurable; treat that as evidence the
pseudotuning approximation is harmless here, not that it is free in general —
the plug-in error it rests on is what `sumstat_calibration.py` measures.
