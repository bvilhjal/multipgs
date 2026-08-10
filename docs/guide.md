# User guide

## 1. What you need

**To learn a combination** (`multi_pgs_fit`):

- **A panel of scores.** PGS Catalog scoring files, GWAS summary statistics to
  fit yourself, or an `n × K` matrix you already have. There is no minimum `K`;
  the method is worth reaching for from a handful of scores upward, and the
  original paper used 937.
- **A training cohort** with genotypes and the target phenotype. Hundreds of
  individuals will fit something; the cross-validated number will be wide.
- **Covariates**, if the cohort needs them — age, sex, genotyping batch,
  principal components.
- **Held-out individuals**, ideally. `fit.cv_r2` substitutes when you have none,
  with the caveats in §5.

**To combine without a phenotype** (`meta_pgs`): scores for the **same** trait
from different discovery GWAS, and each GWAS's effective sample size. Nothing
else.

### Effective sample size

For a case/control GWAS, `n_eff = 4 / (1/n_case + 1/n_control)`, which is
`ldpred3.n_eff_case_control`. For a continuous trait it is the sample size. Get
this wrong and every accuracy-derived weight and screening decision is wrong
with it.

### Remove related individuals first

CMSA folds are drawn at random (stratified only by case status). Siblings, twin
pairs and duplicate enrolments split across training and validation folds, and
`cv_r2` inflates accordingly. There is no group-aware split and no `groups=`
hook. Filter to an unrelated set — KING kinship < 0.0442, or your cohort's
standard — before fitting.

### Is it worth doing at all?

Often, but not always, and the theory says exactly when. The gain from a
correlated trait scales as `r_G² · R_k² · (1 - R_f²)²`, so it is **quadratic in
how weak your own score is**. At a genetic correlation of 0.5 with one auxiliary
trait, the relative gain runs from about +115% when your own GWAS is badly
underpowered to +0.7% when it is well powered
([theory.md §1](theory.md#why-the-gain-concentrates-on-underpowered-traits) has
the table and its derivation). If your own trait has a 500,000-sample GWAS
behind it, expect very little; if it has 20,000, expect a lot.

Separately, the combination costs roughly `(K/n)·(1 - R²)` in optimism from
estimating `K` weights in `n` people. Hundreds of individuals will fit
something, but with `K` in the hundreds it will fit mostly noise; the null gate
described in §4 exists for that case.

## 2. Building the panel

### From PGS Catalog scoring files

```python
from multipgs import panel_from_catalog

panel = panel_from_catalog("scores/", "cohort")     # a dir, files, or a mix
print(panel.summary())
```

The target genotypes are read once and all `K` scores accumulate in that pass.
Each file is harmonised to the target's counted allele through
`ldpred3.harmonize`: allele swaps flip the sign, strand flips are resolved where
the alleles allow, and palindromic A/T and C/G variants are dropped rather than
guessed at (`drop_ambiguous=False` to keep them, which you should not).

Read `panel.summary()` before anything else. It reports variants matched per
score and, more usefully, **weight mass matched** — the fraction of `Σw²` that
survived. Losing 5% of a score's variants means very different things depending
on whether they carried 0.1% or 40% of its weight. A score at 0.4 mass is
usually a genome-build mismatch, not a hard problem with the data.

Scores that match almost nothing are best dropped outright:

```python
panel = panel_from_catalog("scores/", "cohort", min_matched=1000,
                           on_error="skip")
```

### From GWAS summary statistics

```python
from multipgs import panel_from_sumstats

panel = panel_from_sumstats(
    {"height": "height.tsv.gz", "bmi": "bmi.tsv.gz"},
    "cohort", ld_cache="ld/cohort", method="auto")
```

Each trait is fitted with `ldpred3.run_ldpred3_prs`. **Pass `ld_cache`**: the LD
reference is built by the first trait and reused by the rest, which is the
difference between one LD build and `K` of them. Doing so sets
`subset_to_sumstats=False` so the blocks span the same variants for every trait.

This is also how you produce the target trait's own score, which belongs in the
panel like any other.

### Combining panels

Scores from both routes can be stacked side by side, and the panel remembers
which scale each came from:

```python
import numpy as np
from multipgs.panel import ScorePanel

both = ScorePanel(
    scores=np.hstack([cat.scores, own.scores]),
    sample_fid=cat.sample_fid, sample_iid=cat.sample_iid,
    score_ids=np.concatenate([cat.score_ids, own.score_ids]),
    standardized=np.concatenate([cat.standardized, own.standardized]),
    weights=cat.weights + own.weights, meta=cat.meta + own.meta, log={})
```

Both panels must be built on the same target so the rows correspond; if they are
not, use `cat.align(own)` first.

## 3. Screening the panel

Not every public GWAS produces a usable score. `multipgs.architecture` applies
the criteria of Hansen et al. (2026):

```python
from multipgs import architectures_from_panel, screen

archs = architectures_from_panel(panel, n_eff=n_eff_per_trait)
result = screen(archs)                  # h2 range, chain convergence, N, variants
print(result.summary())
panel = panel.select(result.keep)
```

Scores with no fitted architecture behind them — every PGS Catalog file — are
kept and flagged rather than failed, because failing them would empty a catalog
panel. Set `keep_unscreenable=False` if you want only fitted scores.

Screening is not the same as penalising. If you would rather keep a weak score
and shrink it harder:

```python
from multipgs import daetwyler_r2, penalty_from_accuracy

pf = penalty_from_accuracy(daetwyler_r2(h2, p, n_eff, n_variants))
fit = multi_pgs_fit(panel.scores, y, penalty_factor=pf, ...)
```

Read `penalty_from_accuracy`'s note first: it weights by each score's accuracy
for *its own* trait, which bounds but does not measure its relevance to yours.

## 4. Fitting

```python
from multipgs import multi_pgs_fit

fit = multi_pgs_fit(panel.scores, y, covar=covar, score_ids=panel.score_ids,
                    family="gaussian", alpha=1.0, n_folds=10, seed=1)
print(fit.summary())
```

The options worth knowing:

| Option | When to change it |
|---|---|
| `family` | `"binomial"` for case/control. Slower; the Gaussian path is Gram-based and independent of `n`, the binomial one is not. |
| `alpha` | `1.0` (lasso) follows the paper. A grid like `[1.0, 0.5, 0.1]` lets each fold choose; with many near-duplicate scores, a lower `alpha` spreads weight rather than picking one arbitrarily. |
| `n_folds` | 10 is the `bigstatsr` default. Fewer folds means noisier per-fold selection. |
| `unpenalized_scores` | The target trait's own score, if you want the combination to only ever add to it. It also changes what `cv_r2` means — see below. |
| `penalty_factor` | Per-score shrinkage; see §3. |
| `missing` | `"raise"` by default. `"mean"` fills each score's gaps with its column mean. |
| `seed` | Set it. Fold assignment is random, and without a seed the fit is not reproducible. |

Covariates are fitted **unpenalized inside the same regression**, which is not
the same as regressing them out first: the scores are selected against what the
covariates cannot already explain.

### Reading the result

```python
fit.multi_pgs(scores)        # the combined score -- what you evaluate
fit.predict(scores, covar)   # full linear predictor, incl. covariates
fit.selected(top=10)         # (score_id, beta_std, beta), largest first
fit.cv_r2                    # cross-validated, incremental over covariates
fit.n_folds_used             # folds that beat their own covariate-only model
```

Rank scores by `beta_std`, not `beta`: raw coefficients depend on whatever scale
each input score happened to arrive on.

With `unpenalized_scores`, the baseline that `cv_r2` is incremental *over*
includes those scores, because they are in the model at every penalty. So
`cv_r2` becomes the gain over your own trait's score rather than over the
covariates alone — usually the more interesting number, and a different one. On
one panel and seed the same fit reports 0.360 flat and 0.169 with the target
trait's own score unpenalized; both are correct, and they answer different
questions.

If `fit.beta` is all zero, cross-validation did not beat the covariate-only
model. That is a result — "these scores did not predict here" — not a bug, and
`fit.summary()` says so.

## 5. Evaluating, and the ways this goes wrong

```python
from multipgs import evaluate

# Pass the panel, not panel.scores: the score ids get checked against the fit.
print(evaluate(y_test, fit.multi_pgs(test_panel), covar=covar_test,
               family="binomial", prevalence=0.01))
```

**Check that the test panel is the same scores in the same order.** A bare
matrix is matched by *position*. A separately built panel can legitimately come
back with `K` columns in a different order — `on_error="skip"` may drop a
different file, a different `min_matched` may exclude a different score, and a
list of paths does not sort the way a directory does. Swapping two columns of a
ten-score panel took held-out r² from 0.448 to 0.075 in testing, with no error
raised. Passing the `ScorePanel` itself (or `score_ids=`) makes `multi_pgs`
check, and realigning is one call:

```python
test_panel = test_panel.select(list(fit.score_ids))
```

**Scores are only comparable across cohorts if you freeze the scale.** `beta` is
on the raw score scale, and a catalog score's raw scale depends on the cohort it
was computed in: missing calls are imputed to *that* cohort's column means, and
a different variant set may match. For a held-out cohort scored from its own
genotypes, go through `combine_weights` and `scaling="frozen"` (§6) rather than
rebuilding a panel and calling `multi_pgs` on it.

**Sample overlap is the failure that matters.** If individuals in your training
or test cohort contributed to the GWAS behind an input score, that score is
partly fitted to your own data, the combination will reward it for that, and
every number goes up. Many PGS Catalog scores are UK Biobank-derived, so a UK
Biobank target is the worst case — check each score's development samples in its
Catalog metadata. This must be excluded when the panel is built;
nothing downstream can detect it. `fit.cv_r2` is honest about the fitting
procedure and completely blind to this.

**Evaluate the score, not the model.** `r2` of a prediction that already
contains age and sex describes age and sex. With covariates, `incremental_r2`
is the quantity to report, and it is what `evaluate` adds automatically.

**Convert case/control R² to the liability scale.** On the 0/1 scale it depends
on how many cases were sampled and is not comparable to anyone else's number.
Pass `prevalence=`.

**Ancestry is not modelled anywhere in this package.** Most PGS Catalog scores
are European-derived, and accuracy falls substantially in a target ancestry
unmatched to discovery. Note the asymmetry: `multi_pgs_fit`'s coefficients and
`meta_pgs`'s `C` are estimated in *your* cohort and so are appropriate to it,
but `daetwyler_r2`, `screen`, `penalty_from_accuracy` and
`meta_pgs(expected_r2=…)` are ancestry-blind — `h²`, `p` and `n_eff` describe
the discovery cohort. Ancestry principal components in the covariate set are
mandatory, not optional, and `screen`'s convergence gate will fire on
mixed-ancestry discovery samples for a reason users often misread as a data
problem. See [theory.md §4](theory.md#ancestry).

**Report the interval — and know where it is bounded by construction.**
`evaluate` bootstraps by default. With a few thousand individuals the interval
on R² is usually wide enough to swallow the difference between the methods you
are comparing. One trap: `incremental_r2` and `nagelkerke_r2` are truncated at
zero, so for a useless score the bootstrap piles up at exactly 0 and the lower
bound is 0 *by construction*. `[0.000, 0.004]` is what no effect looks like, not
a small effect bounded away from zero.

## 6. Deploying

```python
from multipgs import combine_weights
from ldpred3 import score_from_weights

combine_weights(panel, fit, path="multi.weights")
result = score_from_weights("multi.weights", "new_cohort", scaling="frozen")
```

`combine_weights` collapses `K` weight sets and their coefficients into one
per-variant table, on the standardized scale ldpred3 applies. The new cohort is
scored with no reference to the `K` inputs and no need to rebuild the panel.

Use `scaling="frozen"`. It reuses the `AF_REF`/`SD_REF` written into the file,
so a cohort with different allele frequencies is still scored on the scale the
coefficients were fitted on. `scaling="target"` re-standardizes against the new
cohort, which silently changes what the weights mean.

The combined score equals the training-time score up to an additive constant.
That is irrelevant to R², AUC and ranking, and absorbed by the intercept of any
downstream regression.

## 7. When to use `meta_pgs` instead

Use it when the `K` scores all estimate the **same** genetic value and differ
only in which GWAS produced them — PGC, deCODE, a biobank, an internal
meta-analysis. Then no training cohort is needed:

```python
from multipgs import meta_pgs

combined = meta_pgs(panel, n_eff=[150_000, 60_000, 20_000])
prs = combined.multi_pgs(panel.scores)
```

Do **not** use it on a heterogeneous panel of different traits. Weighting scores
of different traits by their own sample sizes assumes a relevance that the
sample size cannot express; `multi_pgs_fit` learns it instead. The runnable
example shows this failure directly.

If the discovery GWAS overlap — a consortium meta-analysis usually contains a
cohort you are also using separately — use `method="decorrelated"`, and give it
`expected_r2` rather than `n_eff`. See
[algorithm.md](algorithm.md#choosing-a-meta-pgs-rule) for the measurements, and
[theory.md §3](theory.md#3-derived-weights-no-phenotype-required) for why the
cruder statistic has the better-shaped weights.
