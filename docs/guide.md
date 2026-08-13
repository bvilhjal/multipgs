# User guide

## 1. Choose one of three routes

There is no validated universal threshold in `K` or `n` at which a route
becomes worthwhile. A solver returning coefficients establishes computational
feasibility, not generalization; accuracy claims require untouched assessment
data.

### Route A — individual-level learned combination

Use `multi_pgs_fit` when you have a target phenotype and a panel containing any
mixture of focal, related-trait, and irrelevant scores. You need target
genotypes or an existing `n × K` score matrix, appropriate covariates, and an
unrelated training sample. Untouched individuals are required for an external
accuracy claim; `fit.cv_r2` is a nested internal estimate when they are
unavailable. Build the panel in §2, optionally screen in §3, fit in §4, assess
in §5, and deploy in §6.

### Route B — summary-statistic learned combination

Use `multi_pgs_sumstats` when you have no phenotyped cohort but do have raw
component-score definitions, standardized target-GWAS effects, and an
ancestry-matched LD reference. Align each data source on its own genotype scale.
One target GWAS fits the path, an independent second GWAS can tune it, and a
third untouched GWAS or cohort is needed to assess the selected model. With one
GWAS, PUMAS-style pseudotuning is available under the assumptions in §4; it is
not external assessment.

### Route C — same-trait training-free combination

Use `meta_pgs` only when every consistently oriented score estimates the same
trait in the target population. Its requirements depend on the rule:

- `sqrt_n_eff` needs each discovery GWAS's effective sample size.
- `expected_r2` needs a credible, target-transportable phenotypic R² magnitude
  per score; Daetwyler values are model-based proxies, not measured target
  accuracy.
- `decorrelated` additionally estimates the score correlation matrix from the
  target panel and needs independently credible target-correlation magnitudes.
  Sample-size and Daetwyler proxies alone do not justify the inversion.

The API accepts squared, nonnegative magnitudes and therefore cannot represent
a genuinely negative target correlation. See §7 for the operational choice.

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

Theory identifies regimes in which gains are possible; the package validation
does not establish a universal sample-size cutoff or real-world superiority.
Under the stated model, the gain from a correlated trait scales as
`r_G² · R_k² · (1 - R_f²)²`, so it is **quadratic in how weak your own score
is**. At a genetic correlation of 0.5 with one auxiliary trait, the model's
relative gain runs from about +115% when your own GWAS is badly underpowered to
+0.7% when it is well powered
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
# metadata.tsv next to the files is attached automatically (n_eff, ancestry)
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

### Building component scores from GWAS summary statistics

```python
from multipgs import panel_from_sumstats, read_trait_table

# Prefer an external LD reference (or an existing cache), not the target .bed.
panel = panel_from_sumstats(
    {"height": "height.tsv.gz", "bmi": "bmi.tsv.gz"},
    "cohort", ld_prefix="ld/ref", ld_cache="ld/cohort", method="auto",
    infer=True, auto_chains=10, n_jobs=4, weights_dir="weights/")
```

Per-trait `n_eff`, case/control counts, method and alpha go in a table
(`TRAIT PATH N_EFF N_CASES N_CONTROLS METHOD ALPHA`) and are passed as
`traits="traits.tsv"`. `n_cases`/`n_controls` are converted with
`ldpred3.n_eff_case_control` before the fit. Architecture inference
(`infer=True`, `auto_chains=50`) is for screening, not the default score path.

Already-fitted ldpred3 weight files skip the Gibbs step:

```python
from multipgs import panel_from_weights

own = panel_from_weights("weights/", "cohort")
```

`panel_from_sumstats` calls `ldpred3.run_ldpred3_prs`. Give it `ld_cache` (and
preferably `ld_prefix`) so the first trait writes the blocks and the rest only
read them; that also sets `subset_to_sumstats=False`. After the first
successful write, `n_jobs` runs remaining traits in a thread pool.

`infer=True, auto_chains=50` is the Hansen screening path and is extra work.
Omit both if you only need scores. Include the target trait's own score in the
panel like any other.

### Combining panels

Scores from both routes can be stacked side by side, and the panel remembers
which scale each came from:

```python
both = cat.concat(own)
both.save("panel.npz")
```

`concat` matches individuals on `FID:IID` and refuses colliding score ids.
`save` / `load_panel` keep weights, scale flags, inference and `n_eff`, which
the TSV from `write_panel` does not.

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
from multipgs import daetwyler_r2, penalty_from_accuracy, penalty_from_relevance

pf = penalty_from_accuracy(daetwyler_r2(h2, p, n_eff, n_variants))
# with bipred r_G estimates: pf = penalty_from_relevance(r2, rg)
fit = multi_pgs_fit(panel.scores, y, penalty_factor=pf,
                    score_ids=panel.score_ids, seed=1)
```

`ldsc_rg_screen` (optional `multipgs[bipred]`) estimates `r_G` of each
auxiliary GWAS against a focal trait on a shared `ld_cache`. The LDSC χ² cap
is applied to those regression rows only.

Read `penalty_from_accuracy`'s note first: it weights by each score's accuracy
for *its own* trait. That is a ranking heuristic, not a bound on relevance to
the target, and can down-weight a low-powered score of a genetically identical
trait.

## 4. Fitting

```python
from multipgs import multi_pgs_fit

fit = multi_pgs_fit(panel.scores, y, covar=covar, score_ids=panel.score_ids,
                    family="gaussian", alpha=1.0, n_folds=10, seed=1)
print(fit.summary())
```

This uses Albiñana et al.'s elastic-net stacking model, not their complete
fitting and evaluation procedure. Their analysis standardized PGS and used
`cv.glmnet` with fivefold cross-validation. `multipgs` standardizes inside each
training fold, uses CMSA to select and average fold models, and adds the nested
heuristic fallback below. Their normalized adjusted R² is also not the package's
unnormalized `incremental_r2`; [theory.md §4](theory.md#score-r²-versus-model-r²)
gives both definitions.

The options worth knowing:

**Table 1. Fitting options that change the statistical contract.**

| Option | When to change it |
|---|---|
| `family` | `"binomial"` for case/control. Slower; after Gaussian sufficient statistics are formed, each path is independent of `n`, whereas binomial IRLS is not. |
| `alpha` | `1.0` is lasso and follows the authors' released [`e_net.R`](https://github.com/ClaraAlbi/paper_multiPGS/blob/main/code/e_net.R). The published Methods says `alpha=0`; that conflicts with the code. A grid like `[1.0, 0.5, 0.1]` lets each fold choose. |
| `n_folds` | 10 is the `bigstatsr` default. More folds train closer to the full sample but select on smaller, noisier validation folds; 5–10 is the ordinary range. |
| `assessment_folds` | Outer folds for nested performance assessment and the heuristic fallback gate. The default 5 is separate from the final CMSA's `n_folds`; increasing it costs additional inner fits. |
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
fit.log["solver_converged"]  # false if any numerical fit exhausted iterations
```

Rank scores by `beta_std`, not `beta`: raw coefficients depend on whatever scale
each input score happened to arrive on. `selected()` reports nonzero predictive
weights, not causal traits. CMSA support is the union of fold supports, and
correlated scores can exchange or share weight.

Read `fit.summary()` and require `fit.log["solver_converged"]` before trusting
the coefficients. If it is false, inspect `n_iteration_exhausted`,
`n_coordinate_descent_exhausted`, `n_irls_exhausted`, and
`n_baseline_not_converged`; increasing `max_iter` is the first numerical check,
not a guarantee that the model is otherwise well specified.

`cv_r2 = (SSE_baseline - SSE_inner_CMSA) / SST` over untouched outer folds. It
is a predictive loss gain, not `incremental_r2`: the latter recalibrates the
supplied score by OLS in the assessment cohort. With `unpenalized_scores`, the baseline
includes those scores, so `cv_r2` becomes the gain over the fitted target-trait
score plus covariates rather than over covariates alone.

The package uses a conservative heuristic: the penalized stack is returned only
when the mean outer-fold gain exceeds one standard error. It is not a hypothesis
test, supplies no p-value or calibrated type-I error, and fallback does not
establish zero population signal. On fallback, `fit.log["null_model"]` is
present and the returned fit is the full-data unpenalized baseline. "Null"
describes the *increment*: forced scores and covariate coefficients remain
fitted, and `fit.beta` need not be all zero. `fit.summary()` spells this out.

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
W_ld, score_ids, log_ld = align_to_reference(
    scoring_files, ld_variants, sd=ld_dosage_sd)
W_train, train_ids, log_train = align_to_reference(
    scoring_files, train_variants, sd=train_dosage_sd)
W_tune, tune_ids, log_tune = align_to_reference(
    scoring_files, tune_variants, sd=tune_dosage_sd)
W_test, test_ids, log_test = align_to_reference(
    scoring_files, test_variants, sd=test_dosage_sd)
assert score_ids == train_ids == tune_ids == test_ids
assert all(log["standardized"] for log in
           (log_ld, log_train, log_tune, log_test))

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
  imputation uncertainty and departures from equilibrium. Without `sd=` or
  that explicit approximation, `align_to_reference` leaves Catalog weights on
  the allele-count scale and records `log["standardized"] = False`; such output
  is suitable here only if the input weights were already standardized. If one
  cohort supplies both `z` and `D`, pass its aligned matrix in both roles
  explicitly. Do not silently reuse a matrix across populations because its
  shape matches.
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
from multipgs import combine_weights, evaluate
from ldpred3 import score_from_weights

combine_weights(panel, fit, path="multi.weights")
test_score = score_from_weights("multi.weights", "test_cohort",
                                scaling="frozen")
print(evaluate(y_test, test_score.scores, covar=covar_test,
               family="binomial", prevalence=0.01))
```

**Keep identity checks separate from scale transport.** For untouched rows of
the same already-built panel, `fit.multi_pgs(panel.scores[test_rows])` remains
on the fitted raw-score coordinate. A bare matrix is matched by *position*;
when a `ScorePanel` is passed, score ids are checked. A separately built panel
can return columns in a different order, and realigning identities is one call:

```python
test_panel = test_panel.select(list(fit.score_ids))
```

That check does not freeze score scale. For an independently scored cohort,
use the combined weight file and `scaling="frozen"` as above.

**Scores are only comparable across cohorts if you freeze the scale.** `beta` is
on the raw score scale, and a catalog score's raw scale depends on the cohort it
was computed in: missing calls are imputed to *that* cohort's column means, and
a different variant set may match. For a held-out cohort scored from its own
genotypes, go through `combine_weights` and `scaling="frozen"` (§6) rather than
rebuilding a panel and calling `multi_pgs` on it.

**Sample overlap can bias assessment.** If individuals in your training or test
cohort contributed to the GWAS behind an input score, that score is partly
fitted to your own data and accuracy can be inflated. The magnitude and even
visibility of the bias depend on discovery design, score construction,
relatedness, and overlap fraction. Many PGS Catalog scores are UK
Biobank-derived, so check each score's development samples in its Catalog
metadata. Exclude overlap when the panel is built: target-cohort resampling,
including `fit.cv_r2`, cannot diagnose contamination shared by every fold.
Named-cohort metadata or external genome-wide diagnostics may flag overlap
under favourable conditions, but exclusion by design is stronger
([why](theory.md#sample-overlap)).

**Evaluate the score, not the model.** `r2` of a prediction that already
contains age and sex describes age and sex. With covariates, `incremental_r2`
is the quantity to report, and it is what `evaluate` adds automatically.

**Check direction and calibration.** `r2` squares Pearson correlation, so a
score with reversed direction has exactly the same R². Verify the signed
correlation and, for binary outcomes, inspect whether AUC is below 0.5. A
weighted sum is not automatically an absolute-risk model: estimate calibration
slope and intercept in held-out target data before interpreting `.predict()` as
portable risk. `.multi_pgs()` is a score, not a probability.

**Convert case/control R² to the liability scale.** On the 0/1 scale it depends
on how many cases were sampled and is not comparable to anyone else's number.
Pass `prevalence=`; [theory.md §4](theory.md#liability-scale) gives the model.

**Ancestry is not modelled anywhere in this package.** Many PGS Catalog scores
are European-derived, and accuracy falls substantially in a target ancestry
unmatched to discovery. `multi_pgs_fit`'s coefficients and `meta_pgs`'s `C` are
cohort-specific estimates, but they do not repair non-portable discovery scores
or LD. `daetwyler_r2`, `screen`, `penalty_from_accuracy` and
`meta_pgs(expected_r2=…)` are
ancestry-blind — `h²`, `p` and `n_eff` describe the discovery cohort. Put
ancestry principal components in the covariates; [theory.md
§4](theory.md#ancestry) has the evidence.

**Report the interval — and know where it is bounded by construction.**
`evaluate` bootstraps target individuals while holding the component scores and
fitted weights fixed. Its intervals therefore condition on upstream discovery
GWAS, LD reference, score construction, and architecture estimates; they do not
propagate uncertainty from those stages. With a few thousand individuals an R²
interval may still be wide enough to cover the difference between methods.
Also, `incremental_r2` and `nagelkerke_r2` are truncated at zero: for a useless
score the bootstrap piles up at exactly 0, so `[0.000, 0.004]` is what *no*
effect looks like, not a small effect bounded away from zero
([theory.md §4](theory.md#reading-an-interval)).

## 6. Deploying

```python
from multipgs import check_weights, combine_weights
from ldpred3 import score_from_weights

combine_weights(panel, fit, path="multi.weights")
check_weights(panel, fit, "multi.weights", "train_cohort")
result = score_from_weights("multi.weights", "new_cohort", scaling="frozen")
```

Or from the CLI, after `panel --out panel.npz` and `fit --panel panel.npz`:

```bash
multipgs combine --panel panel.npz --fit fit.tsv --out multi.weights \
    --check --plink train
multipgs score --weights multi.weights --plink test --out test.prs
```

`multipgs score` always uses frozen scaling.

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

Discovery-GWAS overlap — for example, when a consortium meta-analysis contains
a cohort also used separately — does not by itself justify
`method="decorrelated"`. Its inverse correlation matrix amplifies errors in the
accuracy vector. Use `method="expected_r2"` for `daetwyler_r2` proxies. Reserve
`method="decorrelated"` for independently credible per-score target
correlations, with every component score oriented consistently; pass the
squared correlations as `expected_r2`. The API takes nonnegative magnitudes and
cannot encode a genuinely negative target correlation. See
[algorithm.md](algorithm.md#choosing-a-meta-pgs-rule) for the measurements, and
[theory.md §3](theory.md#3-derived-weights-no-phenotype-required) for why the
cruder statistic has the better-shaped weights.
