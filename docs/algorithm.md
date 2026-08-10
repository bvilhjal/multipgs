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
2. Compute one penalty grid on the full training set, at the covariate-only fit.
   A per-fold grid would not be comparable, and averaging coefficients across
   incomparable grids is averaging across different problems. `cv.glmnet` does
   the same.
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

### The gate, and why the obvious one fails

A fold selects its penalty on a few dozen individuals. Under pure noise roughly
half of them beat their own covariate-only model by chance, so a per-fold gate
alone lets a dense model of nothing through — measured at 30 of 40 scores
selected on pure Gaussian noise with `n = 300`.

The pooled statistic fixes it, but only if it is computed honestly. Evaluating
each fold at the `(alpha, lambda)` *it* chose, on the *same* individuals it
chose them with, is selection on the assessment set: it reports **cv_r2 =
+0.014** for pure noise.

So the operating point comes from the other folds. For fold `k` the assessment
uses the median `lambda` index and modal `alpha` index selected by folds `j ≠ k`
— none of which saw fold `k`'s individuals at any stage. Both losses are means
over individuals, so pooling is the size-weighted mean of the per-fold losses
and no predictions need storing. Measured on pure noise across six seeds,
`cv_r2` is then **negative in every one**, and the fit returns the null model in
7 of 8 seeds.

`cv_r2` is reported *incremental over the covariate-only model*, which makes it
the same quantity `incremental_r2` reports in a held-out cohort. It is slightly
conservative about the returned model, which averages the folds and is usually a
little better than any single one.

None of this sees sample overlap between the target cohort and the discovery
GWAS. That inflates every number here and is only excludable when the panel is
built.

### The solver

`multipgs/_coord.py`. Both families use the same soft-thresholded coordinate
update with a per-column penalty factor.

**Gaussian — covariance updates.** The solver touches only `G = XᵀX/n` and
`r = Xᵀy/n`, so a sweep costs `O(K²)` regardless of `n`. That is the right trade
for multi-PGS: a few hundred to a few thousand scores in tens of thousands of
people. The Gram is formed once by BLAS per fold and the whole path is then
independent of the cohort size. `G` is symmetrised explicitly so the contiguous
row `G[j]` can stand in for the strided column.

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

Gaussian, per fold: one `O(nK²)` BLAS Gram, then a path independent of `n`.
For `K = 1000` a full sweep is ~10⁶ flops and the whole 100-penalty path runs in
seconds. Binomial is `O(nK)` per sweep with an IRLS loop around it, so it scales
with the cohort; expect it to be the slow one.

## Score construction

`panel_from_catalog` reads the target genotypes **once**. Every scoring file is
harmonised against the variant table up front, the union of matched variants is
streamed from the `.bed` in blocks sized to keep a block near 64 MB, and all `K`
scores accumulate in that pass. Calling a single-score routine `K` times would
re-read the genotypes `K` times.

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
variants, so `x` = 12, 4.8, 1.6), r² against a simulated phenotype as the
discovery cohorts are made to overlap (`simulate_same_trait_panel`, n = 40,000):

| overlap | best single | `sqrt_n_eff` | `expected_r2` | decorr(`n_eff`) | decorr(`expected_r2`) |
|---|---|---|---|---|---|
| none | 0.364 | 0.373 | 0.366 | 0.190 | **0.375** |
| moderate (0.3) | 0.364 | 0.361 | 0.352 | 0.179 | **0.368** |
| strong (0.6) | 0.364 | 0.350 | 0.339 | 0.173 | **0.367** |
| severe (0.8) | 0.364 | 0.343 | 0.330 | 0.170 | **0.374** |

Decorrelation with a well-specified `ρ` wins everywhere, and past mild overlap
it is the only rule still beating the best single score. `sqrt_n_eff` beats
`expected_r2` for the saturation reason above. Decorrelation with `sqrt(n_eff)`
is far worse than doing nothing, because `C⁻¹` amplifies a mis-specified `ρ`
that a direct weighted sum merely tolerates.

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

`screen` applies Hansen et al.'s inclusion gates, which they used across 1,523
GWAS Catalog traits: `h²` in [0.01, 1], at least 20 of 50 chains converged, at
least 60,000 variants after QC, and `n_eff > 10,000`. The convergence gate is
the informative one in practice — non-convergence usually means the discovery
GWAS and the LD reference disagree, most often a mixed-ancestry discovery
sample.

`penalty_from_accuracy` turns expected accuracy into elastic-net penalty
factors, `pf_k = (gmean(a)/a_k)^power` for `a_k = sqrt(R²_k)`, rescaled to
geometric mean 1 so the penalty grid is unchanged and only relative shrinkage
moves. It is an adaptive-lasso weighting whose prior comes from summary
statistics rather than a first-stage fit. It ignores genetic correlation with
the target, so it will over-penalise a low-powered GWAS of a trait genetically
identical to yours; that is why it is opt-in.

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
