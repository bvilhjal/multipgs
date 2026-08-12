# Theory

Why combining polygenic scores works, what the three fitting routes in this
package estimate, and what the resulting numbers mean. [algorithm.md](algorithm.md) is
the companion: implementation choices, costs, and the measurements behind the
defaults. [guide.md](guide.md) is how to run it.

Every reference here has been checked against the published source; the full
annotated list is in [references.md](references.md).

## Notation

Fixed once and used throughout.

| symbol | meaning |
|---|---|
| `n` | individuals in the target cohort |
| `K` | scores in the panel, indexed `k = 1..K` |
| `n_folds` | cross-validation folds (never `K`) |
| `f` | the focal, or target, trait |
| `g_k` | additive genetic value of trait `k`, scaled to variance 1 |
| `h_k²` | SNP heritability, so `cor(g_k, y_k) = h_k` |
| `z_k` | score `k`, standardized in the target cohort |
| `R_k` | `cor(z_k, g_k)` — accuracy of score `k` for **its own trait's genetic value** |
| `r_G(k,l)` | genetic correlation between traits `k` and `l` |
| `N_k` | effective sample size of GWAS `k`; `4/(1/n_case + 1/n_control)` for case/control |
| `M` | causal variants; the code uses `n_variants · p` from the fit |
| `x_k` | `N_k h_k² / M`, the power of GWAS `k` |
| `C` | `K × K` correlation matrix of the standardized scores in the target |
| `ρ` | `K`-vector, `ρ_k = cor(z_k, g_f)` |
| `eps_k` | unit-variance residual error in score `k` after its genetic component |

**Two quantities are both called "the accuracy of a PGS", and confusing them is
the usual error here:**

```
R_k        = cor(z_k, g_k)      genetic-value accuracy
h_k · R_k  = cor(z_k, y_k)      phenotypic accuracy
```

`multipgs.daetwyler_r2` returns the **phenotypic** r², `h²x/(1+x)`. So
`sqrt(daetwyler_r2) = h·R_k`, not `R_k`. Within one trait `h` is a constant and
cancels from any weight direction; across a panel of different traits it does
not.

## 1. Why multi-trait information helps

Three assumptions, stated so they can be checked rather than wished into being:

- **(A1) Residual-error model.** In the target population,
  `z_k = R_k·g_k + sqrt(1-R_k²)·eps_k`, where `E(eps_k)=0`,
  `var(eps_k)=1`, and `eps_k` is uncorrelated with every genetic value in the
  model, including `g_f`. This last condition is the substantive mediation and
  transport assumption; the algebraic decomposition with respect to `g_k`
  alone does not imply it.
- **(A2) Joint target-population architecture.** A genetic correlation matrix
  `r_G` over the `K+1` traits exists in the population where the scores will be
  used.
- **(A3) Cross-score residual independence.**
  `cor(eps_k, eps_l) = 0` for `k ≠ l`. Shared discovery individuals can violate
  this, but so can shared estimation, reference, imputation, or QC artefacts;
  disjoint discovery samples do not by themselves establish (A1) or transport.

Under (A1), the target relationship is one product. Without (A3), the score
correlation retains an additional residual term:

```
ρ_k = cor(z_k, g_f) = R_k · r_G(k,f)
C_kl = R_k·R_l·r_G(k,l)
       + sqrt(1-R_k²)·sqrt(1-R_l²)·cor(eps_k,eps_l)   (k ≠ l)
C_kk = 1
```

Assumption (A3) removes the second term and recovers the simple form.

A foreign score's usefulness is its accuracy for its own trait times the genetic
correlation. Both factors are at most 1, which is why `R_k` **bounds** a score's
value to you without **measuring** it — the fact `multipgs.architecture` rests
on, and the reason `penalty_from_accuracy` is a ranking heuristic.

All of `R_k`, `r_G`, `C`, and `ρ` are target-population quantities. The
discovery-trained weights define each score, but they do not make `ρ` a
property of the discovery GWAS alone. `meta_pgs` estimates `C` in the target
panel without a phenotype and constructs a proxy for `ρ` from discovery
metadata; using that proxy assumes the score direction and accuracy transport.

### The optimal combination

Choosing `w` to maximise the correlation of `w'z` with an unmeasured target
value is a linear selection index
([Smith 1936](https://doi.org/10.1111/j.1469-1809.1936.tb02143.x),
[Hazel 1943](https://doi.org/10.1093/genetics/28.6.476)). Any `w` gives

```
R2(w) = (w'ρ)² / (w'C w)
```

which is scale-free in `w`. Factor `C = LL'`, substitute `u = L'w`; Cauchy–Schwarz
gives `R2(w) ≤ ‖L⁻¹ρ‖² = ρ'C⁻¹ρ`, with equality when `u ∥ L⁻¹ρ`, i.e.

```
w = C⁻¹ρ ,        max R2 = ρ' C⁻¹ ρ
```

The genomic instance is wMT-SBLUP, whose weights solve the same system built
from `h²`, `r_G` and `N`
([Maier et al. 2018](https://doi.org/10.1038/s41467-017-02769-6)); the reduction
to correlation form above is this package's parameterisation, not that paper's.
The identical algebra combines correlated forecasts in econometrics
([Bates & Granger 1969](https://doi.org/10.1057/jors.1969.103)).

### The gain, in closed form

For the focal trait's own score (`R_f`) plus one auxiliary (`R_k`) at genetic
correlation `r`, substituting into `ρ'C⁻¹ρ` and subtracting `R_f²` gives the
gain exactly:

```
              r² · R_k² · (1 - R_f²)²
ΔR2  =  ---------------------------------
              1  -  r² · R_f² · R_k²
```

Checks: at `r = 1, R_k = 1` it returns `1 - R_f²` (the index becomes a perfect
predictor of `g_f`); at `r = 0` it returns 0. Three factors, each consequential:

1. **`r²`** — approximately quadratic in genetic correlation. A trait at
   `r_G = 0.3` contributes about 9% of what a genetically identical one would
   when the denominator changes little; the displayed exact expression should
   be used outside that low-accuracy regime.
2. **`R_k²`** — quadratic in the auxiliary score's own accuracy. The currency is
   the product `r²·R_k²`: a large GWAS of a weakly correlated trait and a small
   GWAS of a strongly correlated one can be worth the same.
3. **`(1 - R_f²)²`** — the gain is largest when your own score is weakest, and
   dies quadratically as it improves.

With `K` positively redundant auxiliary scores the gains are usually
sub-additive: correlated auxiliaries carry some information twice, and `C⁻¹`
removes the double count. This is not a theorem for every valid correlation
matrix. Suppressor configurations, including negatively correlated auxiliary
errors, can make the joint gain exceed the sum of the separate gains.

### Why the gain concentrates on underpowered traits

The Daetwyler bound
([Daetwyler et al. 2008](https://doi.org/10.1371/journal.pone.0003395)) gives
`R_k² = x_k/(1+x_k)`, so `1 - R_f² = 1/(1+x_f)` and the numerator of `ΔR2`
carries `(1+x_f)⁻²`. **The absolute gain falls off as the square of the focal
GWAS's own power.** At `r_G = 0.5`, `R_k² = 0.5`, `h_f² = 0.2`, `M = 20,000` so
`x_f = N_f/100,000` — arithmetic from the formula, not a simulation:

| `N_f` | `x_f` | `R_f²` alone | `ΔR2` | relative gain |
|---|---|---|---|---|
| 10,000 | 0.1 | 0.091 | 0.104 | **+115%** |
| 100,000 | 1 | 0.500 | 0.033 | +6.7% |
| 400,000 | 4 | 0.800 | 0.006 | +0.7% |
| 1,000,000 | 10 | 0.909 | 0.001 | +0.1% |

(Genetic-value scale; multiply by `h_f²` for phenotypic R². The relative column
is unaffected.) This is why multi-PGS helps most where single-trait scores are
weakest, and the shape of Albiñana et al.'s result: their ninefold R² increase
lands on ADHD, against roughly fourfold on average across the psychiatric
disorders analysed
([Albiñana et al. 2023](https://doi.org/10.1038/s41467-023-40330-w)). The same
asymmetry is long established in animal breeding, where multivariate prediction
raises accuracy for a low-heritability trait when a correlated high-heritability
one is available
([Jia & Jannink 2012](https://doi.org/10.1534/genetics.112.144246)).

**All else equal, a well-powered focal GWAS leaves less room to gain.** That is
a prediction of this model, not a limitation of the implementation.

### Three architectures

| | combine summary statistics | learned weights on scores | derived weights on scores |
|---|---|---|---|
| example | MTAG, wMT-SBLUP | **`multi_pgs_fit`**, **`multi_pgs_sumstats`** | **`meta_pgs`** |
| needs a phenotyped cohort | no | `multi_pgs_fit`: yes; `multi_pgs_sumstats`: no |
| needs `r_G` / `h²` up front | yes | no | no |
| assumption bought with | a correct multivariate model | enough individuals, or aligned target-GWAS and LD moment estimates | the scores target one trait |

MTAG combines *before* fitting, reweighting each trait's summary statistics by
the estimated genetic covariance
([Turley et al. 2018](https://doi.org/10.1038/s41588-017-0009-4)). `multipgs`
implements the two right-hand columns: they need no `r_G` estimate, at the cost
of a training phenotype, aligned summary-level moment estimates, or a
same-trait assumption, according to the fitting route.

## 2. Learned weights: penalized regression over a panel

### The objective

For one CMSA fold, let `T` be its training rows. Each score and covariate is
centered and scaled using **only** `T`:

```
S_tilde,ik = (S_ik - mean_T(S_k)) / sd_T(S_k)
X_tilde,ij = (C_x,ij - mean_T(C_x,j)) / sd_T(C_x,j)
```

Constant columns use scale 1 and become zero. On these fold-standardized
columns, `multi_pgs_fit` minimises

```
(1/2|T|) · L(y_T, a0 + X_tilde_T·η + S_tilde_T·θ)
    + λ · Σ_k pf_k · [ α·|θ_k| + (1-α)/2 · θ_k² ]
```

`L` is squared error or the binomial deviance
([Friedman, Hastie & Tibshirani 2010](https://doi.org/10.18637/jss.v033.i01)).
The penalty is therefore on comparable standardized-score coefficients
`θ`, not on coefficients in arbitrary raw PGS units. Covariates are represented
by the unpenalized `η` block, so the scores are selected against what the
covariates cannot already explain. Albiñana et al. likewise used unpenalized
covariates and penalized standardized scores.

Before fold coefficients are averaged, each fold is returned to raw units:
`β_k = θ_k/sd_T(S_k)`, `γ_j = η_j/sd_T(C_x,j)`, and the intercept subtracts
the corresponding training means. CMSA averages these raw coefficients and
intercepts. The reported `beta_std = beta·sd_full(S)` is a common full-training
scale for comparing the returned weights; it is not the coefficient vector
that every fold directly optimized.

The combined score is `S·β`. Covariates are excluded from it deliberately: an
accuracy figure that includes age and sex is not a polygenic-score accuracy.

### The same Gaussian objective from summary statistics

Let `w` denote the raw allele-count weights of the `K` component scores. In a
data source with per-variant dosage SD `s`, the corresponding weights on
standardized genotypes are `W_s = diag(s) @ w`. For LD source `D` and target
GWAS effects `u` (the API calls them `z`), the two Gaussian-objective moment
estimates are

**Score-space moments.**

```
G_raw = W_ld.T @ D @ W_ld    c_raw = W_gwas.T @ u
```

`multi_pgs_sumstats` divides each score by
`s_k = sqrt(G_raw[k,k])`, solves the same coordinate-descent problem with
`G_kl = G_raw,kl/(s_k s_l)` and `c_k = c_raw,k/s_k`, and returns raw-score
coefficients `beta`; `beta_std = beta * s` is the standardized-score vector.
Thus `w @ beta` is the raw allele-count deployment vector; `W_ld @ beta` is
the same score frozen to the LD source's standardized-genotype scale.

At `alpha=1`, the L1 penalty is on the `K` component-score coefficients. This
is the score-space analogue of lassosum's summary-statistic quadratic, but not
SNP-level lassosum: `||beta||_1` is not `||w @ beta||_1`, and the fitted SNP
effects cannot leave the span of the supplied component scores.

When `D` and `u` are computed from the same individuals, these are exact sample
moments. With an external LD reference they are plug-in estimates in one raw
score coordinate, not a single empirical covariance matrix: sampling noise can
put `c` outside `range(G)` or make `c.T @ pinv(G) @ c > var(y)`. Those are
diagnostics, not automatic proof of bad data. The solver still requires a
positive-semidefinite `G` and a bounded penalized objective; a positive
`ld_shrinkage` can make penalized singular directions well posed, but an
unpenalized null direction with nonzero linear signal cannot be repaired.

When one LD Gram supplies both the fit and its selection criterion
(`tune="none"` or PUMAS), the component of `c` in `null(G)` is not estimable
and is projected away; independent tuning instead projects onto the tuning
Gram's range. [algorithm.md](algorithm.md#summary-statistic-learned-combination)
has the mechanics, including what is logged and what `c_raw` retains.

An independent second GWAS selects the penalty by minimum summary MSE. Squared
correlation is not the selection rule: it would score an oppositely directed
predictor just as highly. That selected minimum is still tuning performance, so
assessment needs a third untouched GWAS. With only one GWAS, PUMAS-style
pseudotuning is a joint-Gaussian/CLT plug-in approximation — valid for choosing
the penalty, not for an accuracy claim. [guide.md
§4](guide.md#fitting-from-summary-statistics) states the operational contract.

### The coordinate update

With columns standardized, cycling over `j` and minimising in `β_j` alone gives
the soft-thresholded update

```
z_j  = (1/n)·Σ_i x_ij·(y_i - ŷ_i^(-j))
β_j ← soft(z_j, λ·α·pf_j) / (d_j + λ·(1-α)·pf_j)
soft(z, t) = sign(z)·max(|z| - t, 0)
```

`pf_j = 0` removes the threshold entirely and leaves an ordinary least-squares
update for that coordinate. `λ_max`, the smallest penalty at which every
penalized coefficient is still zero, is `max_j |z_j| / (α·pf_j)` evaluated at
the unpenalized-only fit — which is why `multipgs` solves the covariate and
forced-score baseline first and measures the gradient there. For ridge
(`α = 0`) no finite `λ`
zeroes the solution, so the grid starts at `α = 1e-3`, as glmnet does.

**Unpenalized covariates and Frisch–Waugh–Lovell.** In the Gaussian case,
setting `pf = 0` on the covariate columns is exactly equivalent to residualising
both `y` and every score on the covariates first and fitting without them
([Frisch & Waugh 1933](https://doi.org/10.2307/1907330),
[Lovell 1963](https://doi.org/10.1080/01621459.1963.10480682)); the test suite
pins this against an explicit QR reference. **The equivalence does not hold for
the binomial loss**, where the IRLS weights depend on the current fit, so
"regress the covariates out first" is a different and worse estimator there.

### CMSA, and why it is not the paper's procedure

Selection follows **Cross-Model Selection and Averaging**
([Privé, Aschard & Blum 2019](https://doi.org/10.1534/genetics.119.302019)):
each fold keeps the coefficients at *its own* best `(α, λ)`, and the returned
estimator averages those vectors;
[algorithm.md](algorithm.md#cross-model-selection-and-averaging) has the steps.

This is `multipgs`'s choice. **Albiñana et al. fitted with `cv.glmnet` and
assessed by fivefold cross-validation in iPSYCH.** What is taken from the paper
is the elastic-net stacking model; CMSA, fold-local standardization, and the
nested fallback rule are package choices rather than the paper's full
procedure.

Two properties earn CMSA its place. No separate tuning cohort is consumed. And
averaging is a variance reduction: with correlated scores and an L1 penalty,
*which* of a set of near-duplicates is picked is close to arbitrary — the
instability that motivates bagging
([Breiman 1996](https://doi.org/10.1023/A:1018054314350)) and stability selection
([Meinshausen & Bühlmann 2010](https://doi.org/10.1111/j.1467-9868.2010.00740.x))
— and averaging spreads weight across them instead of betting on one draw.

Two consequences worth stating plainly:

- The average's support is the **union** of the fold supports, so
  `fit.n_selected` is larger than any actually-fitted model's, and is not a
  count of "selected variables" in the lasso sense. It overstates density.
- `fit.selected()` reports nonzero predictive coefficients ordered by their
  standardized magnitude. It does not identify causal traits: correlated
  scores can exchange or share weight, and the association is conditional on
  this panel, phenotype coding, covariates, and training cohort.
- With `α = 0` nothing is ever exactly zero, so `n_selected == K`, `selected()`
  returns everything, and `dfmax` never binds. The selection language only means
  anything for `α > 0`.

### Optimism, and the nested cross-validated number

Choosing a tuning parameter and assessing performance are different tasks, and
data used for the first cannot be reused for the second
([Stone 1974](https://doi.org/10.1111/j.2517-6161.1974.tb00994.x)). Selecting on
the same data you assess on produces error estimates that look good on data with
no signal at all — on null data, tuned cross-validation error falls well below
chance while independent-test performance is exactly chance
([Varma & Simon 2006](https://doi.org/10.1186/1471-2105-7-91)); the general
point that every selection step must sit *inside* the resampling loop is
[Ambroise & McLachlan 2002](https://doi.org/10.1073/pnas.102102699).

An ordinary CMSA validation fold cannot also assess the stack: its phenotype
selected that fold's `(α, λ)`. Nor can the operating point simply be borrowed
from another ordinary fold, because that fold's model was trained on the first
fold's individuals. The whole selection procedure must be nested — the outer
loop, the heuristic fallback it feeds, and the `cv_r2` definition are in
[algorithm.md](algorithm.md#the-nested-gate).

What `cv_r2` *is*: a **predictive loss gain** of the nested stack over the
explicit unpenalized baseline on untouched outer rows, not the same estimator
as `incremental_r2`: the latter fits a calibration coefficient for the score
against assessment outcomes, whereas nested prediction must not. The baseline
contains covariates and every `unpenalized_score`, so forcing the target-trait
score changes the question to "what did the penalized auxiliary scores add?"

The one-standard-error fallback fed by this estimate is a conservative
deployment heuristic, not a hypothesis test. It supplies no p-value or
calibrated type-I error, and choosing the baseline does not establish that the
population increment is zero.

The outer estimators use fewer than `n` individuals and may therefore be more
shrunken than the returned estimator
([Hastie, Tibshirani & Friedman 2009](https://doi.org/10.1007/978-0-387-84858-7),
§7.10). The returned model also averages final folds rather than inner folds.
Under a stable learning curve those differences may make `cv_r2` conservative
about the model shipped, but the direction is not guaranteed and it remains an
internal estimate.

## 3. Derived weights: no phenotype required

When the `K` scores estimate the *same* genetic value, `C` can be estimated from
target genotypes and `ρ` can be **approximated** from external accuracy metadata,
so §1's `w = C⁻¹ρ` is computable with no phenotype. This is a transport
model, not an identity supplied by the metadata.

### Under independent errors

With `ρ_k = R_k` and `C = RR' + D`, `D = diag(1-R_k²)`, Sherman–Morrison gives

```
w_k  ∝  R_k / (1 - R_k²)  =  R_k(1 + x_k)  ∝  sqrt(x_k)·sqrt(1 + x_k)
max R2 = S/(1+S),      S = Σ_k R_k²/(1-R_k²) = Σ_k x_k
```

The second identity is worth pausing on: `R²/(1-R²) = x` exactly, so the
achievable `R²` depends on the discovery GWAS only through **`Σ x_k`** — the
combination behaves like one GWAS of the summed effective sample size. That is
the precise sense in which meta-analysing scores substitutes for meta-analysing
the studies.

`w_k ∝ R_k/(1-R_k²)` **is** the inverse-variance/GLS weighting. Weighting by
accuracy alone, `w ∝ ρ`, is the `C = I` approximation and drops the
`1/(1-R_k²)` factor; the two agree only as every `R_k → 0`.

### Which `C = I` rule, and a counter-intuitive answer

`sqrt_n_eff` supplies `sqrt(x_k)`; `expected_r2` supplies `sqrt(x_k/(1+x_k))`.
Against the optimum `sqrt(x_k)·sqrt(1+x_k)`, the distortions are `1/sqrt(1+x_k)`
and `1/(1+x_k)`.

**Accuracy saturates; the optimal weight does not.** So weighting by accuracy
under-weights a well-powered GWAS *more* than weighting by sample size does. For
`x = (12, 4.8, 1.6)` the distortion spans a factor 2.2 for `sqrt_n_eff` and 5.0
for `expected_r2`, and the simulation's measured ordering follows
([algorithm.md](algorithm.md#choosing-a-meta-pgs-rule) has the table).
The cruder statistic has the better-shaped one.

### What `decorrelated` costs

Estimating `C` from `n` individuals spends `K(K-1)/2` parameters, with
`Var(Ĉ_kl) ≈ (1-C_kl²)²/n`, and `C⁻¹` amplifies that error exactly as it
amplifies error in `ρ`. It is *not* free, even though it never touches the
phenotype — which is why `ridge` exists and why the `K²/n` ratio matters. In the
documented correlated-error simulation, it beats the best single score at every
positive shared-error setting. That simulation does not identify performance at
a literal sample-overlap fraction or guarantee the same ordering elsewhere.

In that benchmark, feeding it `sqrt(n_eff)` performs poorly: `C⁻¹` amplifies
the misspecification of `ρ`. Treat this as a model-specific warning, not a
universal ranking of the rules.

The theoretical `ρ` is signed. The implementation instead constructs a
nonnegative magnitude, `sqrt(expected_r2)`, so every component must first be
oriented to the same positive target direction. It cannot represent a genuinely
negative target correlation. `decorrelated` therefore requires independently
credible, transportable per-score target-correlation magnitudes; a Daetwyler or
sample-size proxy is not enough merely because `C` is well estimated.

### The same-trait assumption

`r_G(k,l) ≈ 1` is *assumed*, not known. Nominally identical phenotype GWAS can
have `r_G < 1` when definitions or ascertainment differ, so estimate or
sensitivity-test `r_G` rather than assuming exact identity. Applied to genuinely
different traits the assumption is wrong, and `multi_pgs_fit`, which learns
relevance instead, is the right tool.

## 4. What the numbers mean

### The Daetwyler bound

For a score built on `M` independent causal variants from a GWAS of `N`
individuals, distinguish accuracy for the genetic value from accuracy for the
phenotype:

```
R_G² = x/(1+x)                                  with  x = N h²/M
R_y² = h²·R_G² = h² / (1 + M/(N h²)) = h²·x/(1+x)
```

`multipgs.daetwyler_r2` returns `R_y²`, the phenotypic value; the selection-index
derivations above use `R_G`. `multipgs` takes `M = n_variants·p` from the fitted
polygenicity rather than assuming it, following Hansen et al.'s adaptation, so
the bound follows each trait's own architecture.

It is an **upper** bound, and the assumptions say why:

1. **Causal variants known and independent.** Real scores spread weight across
   variants in LD with the causal ones, which dilutes accuracy.
2. **No estimation error beyond sampling.** Phenotype measurement error,
   heterogeneity across contributing cohorts and imperfect imputation all
   reduce the realised value.
3. **One population, twice.** Discovery and target are assumed to share allele
   frequencies and LD. They frequently do not — see below.

### Ancestry

The bound above, `screen`, `penalty_from_accuracy` and
`meta_pgs(expected_r2=...)` are all **ancestry-blind**: `h²`, `p` and `N` are
properties of the discovery cohort. Accuracy in a target ancestry unmatched to
discovery falls substantially — European-derived scores lose most of their
accuracy in African-ancestry targets, with the reduction varying by trait
([Martin et al. 2019](https://doi.org/10.1038/s41588-019-0379-x)) — and mean
score levels shift in ways that are artefacts of allele-frequency differences
rather than real risk differences
([Martin et al. 2017](https://doi.org/10.1016/j.ajhg.2017.03.004)). The decay is
continuous, not categorical: accuracy declines with genetic distance even within
Europe ([Privé et al. 2022](https://doi.org/10.1016/j.ajhg.2021.11.008)).

`multi_pgs_fit`'s coefficients and `meta_pgs`'s `C`, being estimated in the
target cohort, *are* appropriate to it. The inputs they combine are not.
`multipgs` does not implement the ancestry-projection normalisation that
`pgsc_calc` applies ([Lambert et al. 2024](https://doi.org/10.1038/s41588-024-01937-x)).

### Liability scale

For a binary trait, R² on the observed 0/1 scale depends on the case fraction
sampled and is not comparable across studies. Under the liability threshold
model, let `π` be the population prevalence, `P` the sample case fraction,
`t = Φ⁻¹(1-π)`, `d = φ(t)` and `i = d/π`:

```
C_LT = [π(1-π)]² / (d² · P(1-P))
θ    = i·(P-π)/(1-π) · [ i·(P-π)/(1-π) - t ]
R²_liability = C_LT·R²_obs / (1 + C_LT·θ·R²_obs)
```

([Lee et al. 2012](https://doi.org/10.1002/gepi.21614)). The leading factor
`C_LT` alone is the heritability transformation
([Lee et al. 2011](https://doi.org/10.1016/j.ajhg.2011.02.002)), which is what
`ldpred3.h2_liability` applies; the `θ` term is the ascertainment correction and
matters once R² is appreciable. `multipgs.liability_r2` implements the full
expression, and a simulation in the test suite recovers a known liability-scale
R² to within 0.02.

### Score R² versus model R²

For a model containing covariates, the quantity attributable to the score is the
**incremental** R², `R²(covar + score) - R²(covar)`, which by
Frisch–Waugh–Lovell equals the squared partial correlation times
`(1 - R²_covar)`. Reporting the full-model R² credits the score with age and
sex. `multipgs.incremental_r2` computes the increment; `evaluate` reports it
automatically whenever covariates are supplied.

This is not Albiñana et al.'s normalized adjusted R²,
`(R²_full - R²_covar)/(1 - R²_covar)`. The two answer related but numerically
different questions, so label the metric when comparing results.

Ancestry principal components belong in that covariate set. Without them a score
can predict ancestry rather than disease
([Wray et al. 2013](https://doi.org/10.1038/nrg3457), whose remedy is stated for
the discovery analysis; including PCs in the *evaluation* model is standard
practice for the same reason).

### Direction, discrimination, and calibration

`r2` is a **squared** Pearson correlation. It gives the same value to a score
and its negative, so it cannot detect reversed alleles, phenotype coding, or
score orientation. Report the signed correlation (or at least verify its sign)
alongside R². For a binary outcome, a reversed score likewise produces AUC below
0.5 rather than useful discrimination.

Neither a weighted sum nor high R² implies calibrated absolute risk. Coefficient
scale and intercept can change across ancestry, prevalence, ascertainment, and
measurement settings. Estimate a calibration slope and intercept in appropriate
held-out target data before treating a combined score as a probability or risk
model; `.multi_pgs()` itself is a score, whereas `.predict()` is calibrated only
to the cohort in which that fit was trained.

### Sample overlap

If the cohort you evaluate in contributed to the GWAS behind an input score,
that score is partly fitted to the individuals it is being scored on. Inflation
generally increases with the overlap fraction, but its magnitude also depends
on discovery design, score construction, relatedness and effect-size
estimation. Overlap is frequently invisible in public summary statistics
([Wray et al. 2013](https://doi.org/10.1038/nrg3457)). Resampling *inside* the
target cohort, including nested cross-validation, cannot distinguish bias that
is common to every fold from genuine generalization. Named-cohort metadata and
external genome-wide diagnostics can flag some overlap patterns, but neither
proves absence; exclusion by design remains the defensible control.

### Reading an interval

`evaluate` bootstraps by default. One trap: `incremental_r2` and
`nagelkerke_r2` are truncated at zero, so for a null score the bootstrap
distribution piles up at exactly 0 and the lower bound is 0 **by construction**.
An interval like `[0.000, 0.004]` is not evidence of a small positive effect;
it is what no effect looks like.

## Further reading

- [algorithm.md](algorithm.md) — implementation, costs, and the measurements
  behind the defaults
- [guide.md](guide.md) — running it, and the failure modes in operational order
- [references.md](references.md) — the annotated bibliography
