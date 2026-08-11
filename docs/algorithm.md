# Algorithm notes

## The learned combination

### Model

With `S` the `n × K` matrix of polygenic scores, `C` the covariates and `y` the
target phenotype, multi-PGS fits

```
minimise  (1/2n) L(y, b0 + C·gamma + S·beta)
          + lambda · sum_k pf_k · [ alpha·|beta_k| + (1-alpha)/2 · beta_k^2 ]
```

over a decreasing sequence of `lambda`. `L` is squared error or the binomial
deviance. The covariates carry `pf = 0`: they are fitted inside the same
regression but never shrunk, so the scores are selected against what the
covariates cannot already explain. That is not equivalent to residualising `y`
on `C` first for the binomial case, and is only equivalent in the Gaussian case
by Frisch–Waugh–Lovell — which the test suite checks explicitly.

The combined score is `S·beta`. Covariates are deliberately excluded from it: an
accuracy figure that includes age and sex is not a polygenic-score accuracy.

### Cross-Model Selection and Averaging

Selection here follows CMSA (Privé, Aschard & Blum 2019), the procedure behind
`bigstatsr::big_spLinReg`. This is multipgs's choice, not the paper's: Albiñana
et al. fitted with `cv.glmnet` and assessed by fivefold cross-validation in
iPSYCH. What comes from the paper is the estimator — penalized regression of the
phenotype on a score panel, covariates at penalty factor 0 — not the routine
that picks its penalty. The steps are:

1. Split the training set into `n_folds` parts (stratified by case status for a
   binary phenotype, so a rare-disease fold cannot come out with no cases).
2. Compute one penalty grid on the full training set, at the unpenalized
   baseline. This grid belongs to the returned estimator; the independent
   assessment below constructs its own grid inside each outer-training set.
3. For each fold, fit the elastic-net path on the other folds and score it on
   the held-out part. Walk down the grid in blocks and stop after `n_abort`
   consecutive penalties fail to improve. Warm starts carry across blocks, so
   early stopping costs nothing over fitting the whole path.
4. Each fold keeps the coefficients at *its own* best `(alpha, lambda)`.
5. Average the fold coefficient vectors.

Two properties earn their keep. No separate tuning cohort is consumed. And the
averaging is a variance reduction: with `K` correlated scores and a lasso
penalty, *which* of a set of near-duplicate scores gets picked is close to
arbitrary, and averaging spreads weight over them instead of betting on one
draw.

### The nested gate

A CMSA fold chooses `(alpha, lambda)` on its held-out rows. Reusing that loss to
decide whether the whole stack carries signal would therefore tune and assess on
the same observations. Borrowing choices from another ordinary CMSA fold is not
enough either: that fold's model was trained on the putative assessment rows.

The gate consequently uses a separate nested assessment:

1. Split the cohort into `assessment_folds` outer parts.
2. For outer part `k`, use only the other rows to construct the penalty grid,
   run an inner CMSA, and fit the explicit unpenalized baseline.
3. Score the averaged inner-CMSA coefficients and the baseline on outer part
   `k`, which neither fit has seen.
4. Pool the two squared-error or deviance losses over all outer rows. Mean
   imputation, when requested, is also learned inside each outer-training set.

For Gaussian loss the reported gain is:

**Equation 1. Nested predictive R² gain.**

```
cv_r2 = (SSE_baseline - SSE_inner_CMSA) / SST_y
```

This is a predictive loss gain. It is deliberately not `incremental_r2`, which
refits a calibration coefficient against the assessment phenotype by OLS. The
baseline contains the covariates and every score with penalty factor zero.

The penalized stack passes only when its mean outer-fold loss gain exceeds one
standard error across outer folds. On a pass, the returned estimator is ordinary
CMSA: every final fold-selected vector enters the average. On a failure,
`multipgs` refits the full-data unpenalized baseline. Thus `null_model` means
"null penalized increment"; forced scores and covariates remain fitted rather
than being zeroed.

None of this sees sample overlap between the target cohort and the discovery
GWAS. That inflates every number here and is only excludable when the panel is
built.

### The solver

`multipgs/_coord.py`. Both families use the same soft-thresholded coordinate
update with a per-column penalty factor.

**Gaussian — covariance updates.** Let `D = K + P`, including covariates. The
solver touches only `G = XᵀX/n` and `r = Xᵀy/n`, so a path sweep costs
`O(D²)` regardless of `n`. Raw sufficient statistics are formed for a parent set
and its held-out parts; subtraction then obtains every training-fold Gram without
materializing another standardized `n × D` matrix. The initial BLAS work still
depends on `n`. `G` is symmetrised explicitly so the contiguous row `G[j]` can
stand in for the strided column.

**Binomial — IRLS with naive updates.** The weights change every outer
iteration, so no fixed Gram helps. Quadratic approximation outside, weighted
coordinate descent inside, against a column-major `X`, at `O(nK)` per sweep.
The weighted column sums of squares are chunked over rows so no `n × K`
temporary is ever formed.

Kernels are written twice — an explicit-loop version compiled by Numba and a
NumPy fallback performing the same arithmetic in the same order — with the
dispatch resolved once at import. A test asserts the two agree to 1e-10.

Correctness is pinned against `sklearn`'s `ElasticNet` to ~1e-7 across
`alpha ∈ {1.0, 0.5, 0.2}`, and the unpenalized-column handling against an
explicit Frisch–Waugh–Lovell reference.

### Cost

With `A = assessment_folds`, Gaussian sufficient-statistic construction is
`O(A n D²)` for the nested assessment as a whole, not one fresh `O(nD²)` Gram
for every fitted path. Each final or inner path then operates on `D × D`
statistics. Held-out Grams are constructed and subtracted one at a time, so
peak model memory is `O(nD + D²)`; no fold keeps a full standardized copy.
Binomial remains `O(nD)` per sweep with an IRLS loop and is the slower family.

The executable scaling artifact records wall time and absolute peak RSS for
representative cases. On the committed Python 3.13/NumPy 2.1.3 run, 5 final
folds, 3 assessment folds and 40 penalties took 0.049 s/149 MiB at
`n=5,000, K=100` and 1.004 s/658 MiB at `n=20,000, K=500`, after kernel warm-up.
See [`benchmarks/results/stack_scaling.csv`](../benchmarks/results/stack_scaling.csv)
and its provenance file; absolute RSS includes the Python, NumPy and Numba
runtime, so compare rows from one run rather than treating it as array bytes.

## Summary-statistic learned combination

For Gaussian stacking, the coordinate solver already needs only a Gram matrix
and a score–phenotype covariance vector. Raw allele-count scores have
dataset-specific standardized-genotype matrices `W_ld` and `W_gwas`, obtained
with the empirical dosage SD of the LD and GWAS sources respectively. With `D`
external LD and `z` standardized target-GWAS effects:

**Equation 2. Summary-level sufficient statistics.**

```
G_raw = W_ld.T @ D @ W_ld    c_raw = W_gwas.T @ z
```

The implementation standardizes each score by
`s_k = sqrt(G_raw[k,k])`, fits the existing covariance-update path, and converts
the selected vector back to raw-score coefficients. Consequently
`SumstatFit.beta`, `MultiPGSFit.beta`, and `combine_weights` share one contract;
`SumstatFit.beta_std` is the coefficient vector on unit-variance scores.

At the default `alpha=1`, the lasso acts on `K` whole component scores. The
quadratic form is inspired by SNP-level lassosum, but the variables are
different: the final variant weights combine the supplied raw score weights
and cannot leave their span. `ld_shrinkage` separately adds a diagonal repair to penalized score
coordinates; it is not folded into elastic-net `alpha`.

`score_gram` consumes LD blocks in exact variant order and accumulates only the
score columns active in each block. Peak working memory is
`O(block_size · K_active + K²)`, not `O(mK)`. PUMAS-style repeats factor the
`K × K` covariance once and reuse it for every noise draw.

When `D` and `z` come from the same individuals, Equation 2 is an exact sample
identity. With external LD it is a plug-in estimator: noisy `c` need not satisfy
the population Schur bound against the finite-reference `G`. Such discrepancies
are logged as diagnostics. Materially indefinite `G` still fails, and each
fitted quadratic must be bounded. Positive `ld_shrinkage` repairs penalized
singular directions; it cannot repair an unpenalized null direction with
nonzero linear signal, mismatched alleles, or a wrong effect scale.

Selection cannot learn from a direction to which its LD Gram assigns exactly
zero score variance. For `tune="none"` and PUMAS, the implementation projects
the unresolved component of `c` onto `range(G)` once, logs the discarded norm
and fraction, and uses the projected moment for fitting and pseudo-splitting.
Independent tuning projects both training and tuning cross-moments onto the
tuning Gram's range; a direction absent from the fitting Gram remains only if
the tuning Gram resolves it. `SumstatFit.c_raw` retains the observed training
vector, while `SumstatFit.r` records the moment actually fitted.

An independent `z_valid` chooses the path by minimum summary MSE; that minimum
remains a tuning statistic. Squared correlation is descriptive and is not the
selection rule, because it would reward a predictor with the wrong sign or
scale. Under PUMAS, both reported selection-path statistics average
pseudo-split refits rather than scoring the returned full-data coefficient
vector. A third untouched GWAS is required for regime-A assessment.
With one GWAS, `tune="pumas"` uses a joint-Gaussian/CLT plug-in covariance to
form pseudo-training and pseudo-tuning moments. It requires the explicit
`weights_independent_of_z=True` acknowledgement because holding a score trained
on the same `z` fixed would leak pseudo-tuning information. This is a two-way
pseudotuning device, not the recursive four-stage PUMAS-ensemble assessment.
`tune="none"` must be requested explicitly and is labelled in-sample reuse.

The executable calibration checks the moment identity on discrete dosages,
the null tuning-versus-assessment MSE gap, and Gaussian-versus-binary error of
the PUMAS covariance plug-in. Raw seeds, summaries, and provenance live in
[`benchmarks/results`](../benchmarks/results); regenerate them with
[`benchmarks/sumstat_calibration.py`](../benchmarks/sumstat_calibration.py).

## Score construction

`panel_from_catalog` reads the target genotypes **once**. Every scoring file is
harmonised against the variant table up front, the union of matched variants is
streamed from the `.bed` in blocks sized to keep a block near 64 MB, and all `K`
scores accumulate in that pass. Calling a single-score routine `K` times would
re-read the genotypes `K` times.

The panel also stores union variant metadata once. Each Catalog score retains
only a smallest-width index into that read-only table plus its weights, while
presenting the legacy mapping keys lazily. In the fully overlapping 8-score,
128-variant regression this uses 25% of the legacy NumPy-buffer bytes; the exact
ratio depends on overlap. Deployment consumes the shared representation directly
and does not materialize all score-specific metadata arrays.

Weight scales are tracked rather than assumed. PGS Catalog weights count alleles
(`Σ w_j g_ij`); LDpred3 weights apply to standardized genotypes `(g − 2f)/sd`.
The panel records which convention each score used, together with the target
cohort's per-variant allele frequency and dosage SD.

`combine_weights` puts everything on the standardized scale — allele-count
weights are multiplied by the recorded SD — because that is the scale
`ldpred3.score_from_weights` applies, so the file it writes can be handed
straight back to ldpred3. A test asserts the round trip reproduces the fitted
combination to a correlation of 1 within 1e-8.

## The training-free combination

For `K` scores of one trait from different discovery GWAS, with `z_k` the
standardized score and `rho_k` its expected correlation with the genetic value,
the optimal linear combination is `w ∝ C⁻¹ρ` where `C` is the correlation matrix
of the scores — estimable from the target genotypes with no phenotype at all.

Three rules, in increasing order of what they assume you know:

| `method` | `ρ` from | `C` |
|---|---|---|
| `sqrt_n_eff` | `sqrt(n_eff)` | assumed `I` |
| `expected_r2` | `sqrt(R²)` from the fitted architecture | assumed `I` |
| `decorrelated` | either | estimated from the panel |

`sqrt_n_eff` is the rule in `code/meta_prs.R` of the
[PGS-pipeline](https://github.com/olex2148/PGS-pipeline) accompanying Hansen
et al. Its justification is that in the power-limited regime accuracy grows as
`sqrt(N)`, so `sqrt(n_eff)` is proportional to expected accuracy.

Weighting by accuracy is *not* the exact inverse-variance combination, though.
Under the same independent-error model the GLS weights are
`w_k ∝ R_k / (1 - R_k²)`, which follows from `C⁻¹ρ` by Sherman–Morrison;
dropping the `1/(1-R_k²)` factor is exact only as every `R_k → 0`. That limit is
the power-limited regime the rule is for, so the approximation holds exactly
where its `sqrt(N)` justification does. [theory.md](theory.md) derives both.

### Choosing a meta-PGS rule

Write `x_k = N_k h²/M`, so score `k`'s accuracy against the genetic value is
`R_k = sqrt(x_k/(1+x_k))`. The exact optimum under independent errors is

```
w_k  ∝  R_k / (1 - R_k²)  =  R_k (1 + x_k)  ∝  sqrt(x_k) · sqrt(1 + x_k)
```

Against that, `sqrt_n_eff` supplies `sqrt(x_k)` and `expected_r2` supplies
`sqrt(x_k/(1+x_k))`. **Accuracy saturates and the optimal weight does not**, so
weighting by accuracy under-weights the best-powered GWAS more than weighting by
sample size does — across the panel below, by a factor 5 against 2.2.

Three same-trait scores (`n_eff` 150k/60k/20k, `h² = 0.4`, 5,000 causal
variants, so `x` = 12, 4.8, 1.6), evaluated against a simulated phenotype at
`n = 40,000`. The `shared` parameter correlates the scores' error terms; it is a
stylized consequence of shared discovery information, not a literal fraction
of overlapping samples.

**Table 1. Mean phenotypic r² over 30 seeds (standard deviations are 0.003–0.004).**

| shared error correlation | best single | `sqrt_n_eff` | `expected_r2` | decorr(`n_eff`) | decorr(`expected_r2`) |
|---|---|---|---|---|---|
| 0.0 | 0.369 | 0.378 | 0.370 | 0.195 | **0.379** |
| 0.3 | 0.369 | 0.365 | 0.356 | 0.184 | **0.372** |
| 0.6 | 0.369 | 0.354 | 0.342 | 0.177 | **0.372** |
| 0.8 | 0.369 | 0.347 | 0.334 | 0.175 | **0.379** |

Within this exact model, decorrelation with a well-specified `ρ` wins in every
regime and is the only combined rule that beats the best single score once the
shared error is positive. That is a simulation result, not an empirical claim
about a given overlap fraction. `sqrt_n_eff` beats `expected_r2` for the
saturation reason above. Decorrelation with `sqrt(n_eff)` is far worse than
doing nothing because `C⁻¹` amplifies a mis-specified `ρ` that a direct weighted
sum merely tolerates.

The complete per-seed rows, summary, command, versions, and platform are in
[`benchmarks/results`](../benchmarks/results); regenerate them with
[`benchmarks/meta_rules.py`](../benchmarks/meta_rules.py).

> An earlier version of this table reported the opposite ranking for the two
> `C = I` rules. It was measuring a defect in `simulate_same_trait_panel`, which
> set each score's accuracy to `sqrt(daetwyler_r2)` — the *phenotypic* r², i.e.
> `h·R_k` — which is precisely the quantity `expected_r2` consumes, so that rule
> was handed the true `ρ` by construction. Fixed, and the numbers above are from
> the corrected simulation.

`ridge` regularises `C` before inversion. Near-duplicate scores make it
ill-conditioned, and an unregularised inverse answers with two enormous weights
of opposite sign that cancel to noise.

## Screening and expected accuracy

`daetwyler_r2(h2, p, n_eff, n_variants)` implements

```
M   = n_variants · p
x   = n_eff · h2 / M
R²  = h2 · x / (1 + x)
```

the bound of Daetwyler et al. (2008) with `M` taken from the fitted
polygenicity rather than assumed a priori — Hansen et al.'s adaptation, which
lets the bound follow each trait's own inferred architecture. It is an upper
bound: causal variants are treated as known and independent, so LD-induced
dilution and between-cohort heterogeneity are both ignored.

`screen` applies the model-level Hansen et al. gates represented by
`Architecture`: `h²` in [0.01, 1], at least 20 of 50 chains converged, at least
60,000 variants after QC, and `n_eff > 10,000`. Upstream cohort and phenotype
eligibility criteria are outside this API. A failed convergence gate
can reflect discovery/LD-reference mismatch, weak information, or model failure;
the count identifies a problem but does not diagnose its cause.

`penalty_from_accuracy` turns expected accuracy into elastic-net penalty
factors, `pf_k = (gmean(a)/a_k)^power` for `a_k = sqrt(R²_k)`. Projection in
log space keeps every factor in `[1/clip, clip]` while retaining geometric mean
1, a neutral relative scale. The fit recomputes its penalty grid for these
factors; geometric-mean normalization does not make that grid invariant. This
adaptive weighting ignores genetic correlation with the target, so it can
over-penalise a low-powered GWAS of a trait genetically identical to yours;
that is why it is opt-in.

## References

- Albiñana C, et al. Multi-PGS enhances polygenic prediction by combining 937
  polygenic scores. *Nat Commun* 14, 4702 (2023).
  [doi:10.1038/s41467-023-40330-w](https://doi.org/10.1038/s41467-023-40330-w)
- Hansen OS, et al. Mapping Genetic Architecture of Thousands of Complex Traits
  Using GWAS Summary Statistics. *Research Square* (2026).
  [doi:10.21203/rs.3.rs-9415305/v1](https://doi.org/10.21203/rs.3.rs-9415305/v1)
- Privé F, Aschard H, Blum MGB. Efficient implementation of penalized regression
  for genetic risk prediction. *Genetics* 212, 65–74 (2019). — CMSA
- Friedman J, Hastie T, Tibshirani R. Regularization paths for generalized
  linear models via coordinate descent. *J Stat Softw* 33, 1–22 (2010).
- Daetwyler HD, Villanueva B, Woolliams JA. Accuracy of predicting the genetic
  risk of disease using a genome-wide approach. *PLoS ONE* 3, e3395 (2008).
- Lee SH, Goddard ME, Wray NR, Visscher PM. A better coefficient of
  determination for genetic profile analysis. *Genet Epidemiol* 36, 214–224
  (2012). — liability-scale R²
- Privé F, Arbel J, Vilhjálmsson BJ. LDpred2: better, faster, stronger.
  *Bioinformatics* 36, 5424–5431 (2020).
