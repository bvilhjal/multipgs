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
- **Held-out individuals**, ideally. `fit.cv_r2` is a nested internal estimate
  when you have none, with the caveats in §5.

**To combine without a phenotype** (`meta_pgs`): scores for the **same** trait
from different discovery GWAS, and each GWAS's effective sample size. Nothing
else.

**To learn a combination entirely from summary statistics**
(`multi_pgs_sumstats`): raw component-score definitions plus separate aligned
standardized-genotype weights for the target GWAS and ancestry-matched LD
reference. Their variant orders may differ, but score identities and counted
alleles may not. One target GWAS fits the path; an independent second GWAS can
tune it; a third untouched GWAS or cohort is needed to assess the selected
model. With only one GWAS, PUMAS-style pseudotuning is available under the
assumptions in §4.

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
the table and its derivation). Holding heritability, polygenicity, ancestry
transfer and auxiliary genetic correlation fixed, a larger focal GWAS usually
leaves less room to gain. Use `x = n_eff·h²/M`, not sample size alone, to judge
the regime.

For an unregularized fit, optimism is of order `(df/n)·(1 - R²)`. Elastic-net
`df` is effective model complexity, not necessarily `K`; nevertheless, large
`K` relative to `n` raises overfitting risk. Regularization and the nested gate
mitigate that risk but do not abolish it.

## 2. Building the panel

### From PGS Catalog scoring files

Acquire scoring files and the metadata needed to audit their provenance before
building the panel:

```bash
multipgs fetch --trait MONDO_0004989 --include-children \
    --out scores/ --build GRCh37 --cohort-overlap
```

This writes harmonized scoring files, `metadata.tsv`, and `n_eff.tsv`.
`--cohort-overlap` flags shared *named* discovery cohorts; incomplete Catalog
metadata means absence of a flag is not proof of disjoint samples. Pin the
genome build to the LD reference or target genotypes. The Python equivalents
are `search_scores`, `download_scores`, `write_score_metadata`, and
`cohort_overlap`.

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
    "cohort", ld_cache="ld/cohort", method="auto",
    infer=True, auto_chains=50)
```

Each trait is fitted with `ldpred3.run_ldpred3_prs`. **Pass `ld_cache`**: the LD
reference is built by the first trait and reused by the rest, which is the
difference between one LD build and `K` of them. Doing so sets
`subset_to_sumstats=False` so the blocks span the same variants for every trait.

`infer=True, auto_chains=50` requests the multi-chain architecture summary used
by §3 and records the exact attempted-chain count. It adds inference work; omit
both arguments if you only need scores and do not plan to screen architectures.
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

Both panels must be built on the same target so the rows correspond. Alignment
returns two panels, so write `cat, own = cat.align(own)` before concatenating.

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

For a fitted architecture, the defaults require `h²` in `[0.01, 1]`, at least
20 retained chains from at least 50 attempted, at least 60,000 post-QC variants,
and `n_eff > 10,000`. Missing data for an enabled gate fails that score. Current
LDpred3 results retain the kept-chain count but not always the attempted count;
Inference itself must be requested with `infer=True`. `panel_from_sumstats`
preserves an exact total when you also pass `auto_chains=` or
`infer_params={"n_chains": ...}`, and never guesses it from the kept count.
Hansen et al.'s fitted shrinkage-coefficient gate is available as
`min_shrinkage=0.4`, but is off by default because LDpred3's `shrink_corr` is a
different input, not that fitted quantity.

Screening is not the same as penalising. If you would rather keep a weak score
and shrink it harder:

```python
from multipgs import daetwyler_r2, penalty_from_accuracy

pf = penalty_from_accuracy(daetwyler_r2(h2, p, n_eff, n_variants))
fit = multi_pgs_fit(panel.scores, y, penalty_factor=pf,
                    score_ids=panel.score_ids, seed=1)
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

**Table 1. Fitting options that change the statistical contract.**

| Option | When to change it |
|---|---|
| `family` | `"binomial"` for case/control. Slower; after Gaussian sufficient statistics are formed, each path is independent of `n`, whereas binomial IRLS is not. |
| `alpha` | `1.0` is lasso and follows the authors' released [`e_net.R`](https://github.com/ClaraAlbi/paper_multiPGS/blob/main/code/e_net.R). The published Methods says `alpha=0`; that conflicts with the code. A grid like `[1.0, 0.5, 0.1]` lets each fold choose. |
| `n_folds` | 10 is the `bigstatsr` default. More folds train closer to the full sample but select on smaller, noisier validation folds; 5–10 is the ordinary range. |
| `assessment_folds` | Outer folds for nested performance assessment and the incremental-signal gate. The default 5 is separate from the final CMSA's `n_folds`; increasing it costs additional inner fits. |
| `unpenalized_scores` | Scores that belong in the baseline, usually the target trait's own score. Their coefficients remain fitted—not fixed at one—and they are retained if the penalized additions fail the gate. |
| `penalty_factor` | Per-score shrinkage; see §3. |
| `missing` | `"raise"` by default. `"mean"` learns imputation means inside each outer-training set for assessment, then from all rows for the final estimator. |
| `seed` | Set it. Fold assignment is random, and without a seed the fit is not reproducible. |

Covariates are fitted **unpenalized inside the same regression**, so scores are
selected against what the covariates cannot already explain. For Gaussian loss
this equals residualising both the phenotype and every score on the covariates;
for binomial loss it does not.

### Reading the result

```python
fit.multi_pgs(scores)        # the combined score -- what you evaluate
fit.predict(scores, covar)   # full linear predictor, incl. covariates
fit.selected(top=10)         # (score_id, beta_std, beta), largest first
fit.cv_r2                    # nested predictive gain over the baseline
fit.n_folds_used             # all CMSA folds on a pass; zero on fallback
```

Rank scores by `beta_std`, not `beta`: raw coefficients depend on whatever scale
each input score happened to arrive on.

`cv_r2 = (SSE_baseline - SSE_inner_CMSA) / SST` over untouched outer folds. It
is a predictive loss gain, not `incremental_r2`: the latter recalibrates the
supplied score by OLS in the assessment cohort. With `unpenalized_scores`, the baseline
includes those scores, so `cv_r2` becomes the gain over the fitted target-trait
score plus covariates rather than over covariates alone.

The penalized stack is returned only when the mean outer-fold gain exceeds one
standard error. Otherwise `fit.log["null_model"]` is present and the returned
fit is the full-data unpenalized baseline. "Null" describes the *increment*:
forced scores and covariate coefficients remain fitted, and `fit.beta` need not
be all zero. `fit.summary()` spells this out.

### Fitting from summary statistics

`multi_pgs_sumstats` fits the same Gaussian regression without a phenotyped
cohort, from the score-space moments `G = W_ld.T @ D @ W_ld` and
`c = W_gwas.T @ z`. With `alpha=1`, the lasso selects whole component scores —
the score-space analogue of lassosum, not SNP-level lassosum: the fitted SNP
effects stay in the span of the supplied score weights.

The component scores have one raw allele-count definition, but each data
source standardizes genotypes by its own empirical dosage SD. Align every
source separately, and check that the score identities match:

```python
from multipgs import (align_to_reference, multi_pgs_sumstats,
                      score_moments)

# Each matrix uses the empirical dosage SD of the source it accompanies.
W_ld, score_ids, _ = align_to_reference(
    scoring_files, ld_variants, sd=ld_dosage_sd)
W_train, train_ids, _ = align_to_reference(
    scoring_files, train_variants, sd=train_dosage_sd)
W_tune, tune_ids, _ = align_to_reference(
    scoring_files, tune_variants, sd=tune_dosage_sd)
W_test, test_ids, _ = align_to_reference(
    scoring_files, test_variants, sd=test_dosage_sd)
assert score_ids == train_ids == tune_ids == test_ids

fit = multi_pgs_sumstats(
    W_ld, z_train, ld_ref, weights_gwas=W_train, score_ids=score_ids,
    z_valid=z_tune, ld_valid=ld_ref, weights_gwas_valid=W_tune,
    weights_ld_valid=W_ld, tune="independent")

# Tuning chose lambda and any LD shrinkage. Assess on a third GWAS.
c_test, G_test, _ = score_moments(
    W_ld, z_test, ld_ref, weights_gwas=W_test)
result = fit.evaluate(c_test, G_test, regime="A")
print(result)
```

The contract:

- **Use mutually independent target-trait GWAS.** `z_train` fits the path,
  `z_tune` selects the penalty, and the selected minimum on `z_tune` is tuning
  performance, not evaluation. `z_test` — a third, untouched GWAS — is the
  assessment. Each `z` is on the standardized allele-correlation scale from
  `ldpred3.standardize_betas(beta, se, n_eff)`, not the raw per-allele beta.
- **Selection minimizes summary MSE.** Squared correlation is descriptive
  only; it would reward an oppositely directed predictor. Under PUMAS, the
  fit's MSE and R² fields average the pseudo-split refits used for selection,
  not the returned full-data `fit.beta`: score that fixed vector with
  `pseudo_r2(fit.beta, fit.gram, fit.r)`, or assess it with `fit.evaluate(...)`.
- **With one GWAS**, `tune="pumas", n_eff=..., weights_independent_of_z=True`
  draws pseudo-training and pseudo-tuning moments from a joint-Gaussian/CLT
  plug-in model. It supplies pseudotuning only — neither an independent
  assessment nor the recursive four-stage PUMAS-ensemble procedure — and the
  flag is an acknowledgement, not a check: every component score must have
  been built independently of that GWAS.
- **Alignment is per source.** `af=..., hwe_genotype_sd=True` requests the
  `sqrt(2 f (1-f))` approximation instead of an empirical `sd=`; it ignores
  imputation uncertainty and departures from equilibrium. If one cohort
  supplies both `z` and `D`, pass its aligned matrix in both roles explicitly.
  Do not silently reuse a matrix across populations because its shape matches.
- **The LD reference** must match the GWAS ancestry and raw score definitions.
  Its finite sample size bounds the resolvable rank, and compact LD encodings
  can introduce numerical indefiniteness. The solver rejects a materially
  indefinite `G` and any unbounded objective; population Schur conditions on
  noisy external `c` are logged as diagnostics, not enforced. Positive
  `ld_shrinkage` can make penalized singular directions well posed; it cannot
  repair mismatched alleles, ancestry, or effect units. `tune="none"` is
  available only when requested explicitly and is in-sample fitting, not
  validation.
- **Unresolvable signal is discarded and logged.** When the selection Gram is
  singular, the component of `c` it cannot identify is projected out. `fit.r`
  is the moment actually fitted, `fit.c_raw` the observed training moment, and
  the discarded norm and fraction are logged.
  [algorithm.md](algorithm.md#summary-statistic-learned-combination) has the
  mechanics; [theory.md §2](theory.md#the-same-gaussian-objective-from-summary-statistics)
  the derivation.

With moments computed from the same individuals, this route and the
individual-level fit agree to a coefficient correlation of 0.9996, and the
choice of tuning regime moved held-out R² by less than 0.001 in that simulated
regime — measured in
[`benchmarks/sumstat_vs_individual.py`](../benchmarks/sumstat_vs_individual.py).

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
partly fitted to your own data, the combination will reward it, and every
number goes up. Many PGS Catalog scores are UK Biobank-derived, so a UK
Biobank target is the worst case — check each score's development samples in
its Catalog metadata. Exclude overlap when the panel is built; nothing
downstream can detect it, including `fit.cv_r2`
([why](theory.md#sample-overlap)).

**Evaluate the score, not the model.** `r2` of a prediction that already
contains age and sex describes age and sex. With covariates, `incremental_r2`
is the quantity to report, and it is what `evaluate` adds automatically.

**Convert case/control R² to the liability scale.** On the 0/1 scale it depends
on how many cases were sampled and is not comparable to anyone else's number.
Pass `prevalence=`; [theory.md §4](theory.md#liability-scale) gives the model.

**Ancestry is not modelled anywhere in this package.** Many PGS Catalog scores
are European-derived, and accuracy falls substantially in a target ancestry
unmatched to discovery. `multi_pgs_fit`'s coefficients and `meta_pgs`'s `C` are
estimated in *your* cohort and so are appropriate to it, but `daetwyler_r2`,
`screen`, `penalty_from_accuracy` and `meta_pgs(expected_r2=…)` are
ancestry-blind — `h²`, `p` and `n_eff` describe the discovery cohort. Put
ancestry principal components in the covariates; [theory.md
§4](theory.md#ancestry) has the evidence.

**Report the interval — and know where it is bounded by construction.**
`evaluate` bootstraps by default; with a few thousand individuals the interval
on R² is usually wide enough to swallow the difference between the methods you
are comparing. And `incremental_r2` and `nagelkerke_r2` are truncated at zero:
for a useless score the bootstrap piles up at exactly 0, so `[0.000, 0.004]` is
what *no* effect looks like, not a small effect bounded away from zero
([theory.md §4](theory.md#reading-an-interval)).

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
Both `MultiPGSFit` and `SumstatFit` expose raw-score coefficients as `beta`, so
this deployment contract is the same for individual- and summary-level fits.

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
prs = combined.multi_pgs(panel)
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
