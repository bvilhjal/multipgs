# Python API

Every name below is importable directly from `multipgs`. Use `help(name)` for
signatures and detailed notes.

```python
from multipgs import (panel_from_catalog, multi_pgs_fit, multi_pgs_sumstats,
                      evaluate, combine_weights)
```

## Combining scores

**Table 1. Combiners.**

| Name | Purpose |
|---|---|
| `multi_pgs_fit` | learn a combination from a training phenotype (CMSA elastic net) |
| `multi_pgs_sumstats` | learn a Gaussian combination from target GWAS statistics and external LD |
| `meta_pgs` | combine consistently oriented same-trait scores with no phenotype, from `n_eff` or an expected target-accuracy proxy |
| `MultiPGSFit`, `SumstatFit`, `MetaPGS` | fitted combinations on the common raw-score coefficient contract |
| `FoldFit` | one CMSA fold's selected alpha/lambda, held-out and baseline losses, sparsity, use flag, and convergence counters |

`MultiPGSFit.multi_pgs(scores)` is the combined score — what you evaluate.
`.predict(scores, covar)` is the full linear predictor including covariates, and
is a different thing; see [guide.md §5](guide.md#5-evaluating-and-the-ways-this-goes-wrong).
`.cv_r2` is the nested outer-fold predictive gain over the explicit
unpenalized baseline; it is not the OLS-recalibrated `incremental_r2` and does
not provide a hypothesis test. The one-standard-error fallback is a conservative
heuristic with no p-value or calibrated type-I error.

Before interpreting coefficients, require `fit.log["solver_converged"]`. A false
value means at least one baseline or path point exhausted its iteration limit;
the log provides path-point, coordinate-descent, IRLS, and baseline counters,
and each `FoldFit` retains its own convergence state.

`MultiPGSFit.selected()` orders nonzero coefficients by `|beta_std|`. These are
predictive weights, not causal trait effects; CMSA support is the union of fold
supports and correlated scores can exchange weight.

Pass a `ScorePanel` to `.multi_pgs()` rather than a bare matrix and the score
ids are checked against the fit. A bare matrix is matched by **position**, and a
separately built panel can have the same columns in a different order.
That identity check does not freeze the score scale. For an independently
scored cohort, use `combine_weights` and score the resulting weight file with
`scaling="frozen"`; direct `.multi_pgs()` is for rows already on the fitted raw
score coordinate.

## Summary-statistic fitting

**Table 2. Score-space sufficient statistics.**

| Name | Purpose |
|---|---|
| `align_to_reference` | harmonize component weights to one data source's variant order and optionally convert them to standardized-genotype scale |
| `score_gram` | compute `G = W_ld.T @ D @ W_ld`, streaming LD blocks and sparse scores |
| `score_moments` | compute `(c, G)` from separate GWAS- and LD-scaled weights |
| `multi_pgs_sumstats`, `SumstatFit` | fit and retain the score-space lasso/elastic-net path |
| `pseudo_r2` | fixed-vector summary-statistic R²; it does not establish independence |
| `evaluate_sumstat`, `SumstatEval` | evaluate a fixed combination and retain its declared provenance |
| `subsample_score_moments` | joint-Gaussian/CLT plug-in pseudo-split used for PUMAS-style tuning |
| `REGIMES` | descriptions of external assessment, pseudotuning, and in-sample reuse |

The complete workflow — the alignment code, the independent train/tune/test
GWAS contract, PUMAS pseudotuning, and the projection of `c` onto the LD
Gram's range (`fit.r` is fitted, `fit.c_raw` observed) — is in
[guide.md §4](guide.md#fitting-from-summary-statistics), with the derivation
in [theory.md §2](theory.md#2-learned-weights-penalized-regression-over-a-panel)
and the implementation in [algorithm.md](algorithm.md#summary-statistic-learned-combination).

`align_to_reference` converts raw Catalog weights only when empirical `sd=` is
supplied or `hwe_genotype_sd=True` is requested. Otherwise its output remains
on the allele-count scale and `log["standardized"]` is false; it is suitable for
`score_gram` only when the input weights were already standardized.

## Building the panel

**Table 3. Panel construction.**

| Name | Purpose |
|---|---|
| `panel_from_catalog` | score PGS Catalog files against a target, in one genotype pass |
| `panel_from_sumstats` | fit each GWAS with LDpred3 and score it on the target; `n_jobs` parallelizes traits after the LD cache exists |
| `ScorePanel` | the `n × K` matrix, its per-variant weights and provenance |
| `combine_weights` | collapse a panel plus a fit into one deployable weight file |
| `read_panel`, `write_panel` | plain-text score matrices (`FID IID <scores...>`) |

`ScorePanel` methods: `.select(columns)` by index, id or mask; `.align(other)` to
match two panels on `FID:IID`; `.summary()` for matched-variant and weight-mass
counts; `.index_of(score_id)`.

## Scoring files

**Table 4. PGS Catalog I/O.**

| Name | Purpose |
|---|---|
| `read_scoring_file` | parse a scoring file, its `#key=value` header and its quirks |
| `ScoringFile` | parsed variants, weights, metadata and parse log |
| `harmonize_scoring_file` | align one to a genotype variant table via `ldpred3.harmonize` |

Odds-ratio weights are log-transformed on read; non-additive rows
(`is_dominant`, `is_haplotype`, ...) are dropped and counted; harmonized
`hm_*` columns are preferred when present.

## PGS Catalog acquisition

**Table 5. Catalog network API.**

| Name | Purpose |
|---|---|
| `search_scores`, `ScoreRecord` | find scores by trait, PGS ids, PMID, or Catalog publication |
| `download_scores` | download harmonized scoring files for GRCh37 or GRCh38 |
| `write_score_metadata` | write score-keyed discovery and publication metadata plus effective sample size |
| `cohort_overlap` | flag score pairs sharing named discovery cohorts; a lower bound, not proof of sample overlap |

## Screening and expected accuracy

**Table 6. Architecture.**

| Name | Purpose |
|---|---|
| `daetwyler_r2` | expected phenotypic r² of a score for its own trait, approximating effective independent effects as `n_variants · p` |
| `Architecture` | per-score `h²`, polygenicity, inferred r², total/kept chains, `n_eff`, and optional fitted shrinkage |
| `architectures_from_panel` | read those back out of an LDpred3-built panel |
| `screen`, `ScreenResult` | represented model-level Hansen et al. gates, with per-score reasons |
| `penalty_from_accuracy` | expected accuracy → elastic-net penalty factors |

`meta_pgs(method="sqrt_n_eff")` requires same-trait, consistently oriented
scores and discovery effective sample sizes. `method="expected_r2"` instead
requires credible, target-transportable phenotypic R² magnitudes; Daetwyler
output is a model proxy, not measured target accuracy.

`meta_pgs(method="decorrelated")` constructs a nonnegative vector from
`sqrt(expected_r2)`, or from `sqrt(n_eff)` when `expected_r2` is omitted, before
applying the inverse score-correlation matrix. Every score must therefore be
oriented to the same positive target direction, and the supplied magnitudes
must be independently credible in the target population. The API cannot encode
a negative target correlation; sample-size or Daetwyler proxies are not
automatically suitable for decorrelation.

## Evaluation

**Table 7. Individual-level metrics.**

| Name | Purpose |
|---|---|
| `evaluate`, `EvalResult` | every applicable metric with bootstrap intervals |
| `r2` | squared correlation of score and phenotype |
| `incremental_r2` | R² added over covariates — the quantity to report when they exist |
| `auc` | Mann–Whitney AUC, ties counted as ½ |
| `nagelkerke_r2` | pseudo-R² over the covariate-only logistic model |
| `liability_r2` | observed → liability scale (Lee et al. 2012, with the θ term) |

`r2` is sign-blind because it squares Pearson correlation. Verify signed
correlation (and AUC direction for binary outcomes) separately. None of these
metrics makes a combined score an automatically calibrated absolute-risk
prediction; calibration slope and intercept require suitable held-out target
data.

Bootstrap intervals from `evaluate` resample target individuals while holding
the supplied scores and fitted weights fixed. They do not propagate uncertainty
from discovery GWAS, LD references, score construction, or architecture
estimation.

`liability_r2` includes the ascertainment correction; `ldpred3.h2_liability`
applies only the leading factor, which is right for a heritability and
understates the shrinkage for a large R².

## Simulation

**Table 8. Synthetic data.**

| Name | Purpose |
|---|---|
| `simulate_panel` | correlated scores, a few of which drive the phenotype |
| `simulate_same_trait_panel` | several scores for *one* trait, with optional shared error correlation |
| `simulate_target` | a small PLINK fileset plus matching scoring files |
| `SimPanel` | the simulated problem, with the answer in `beta_true` |

## Command line

**Table 9. Commands.**

| Command | Purpose |
|---|---|
| `fetch` | acquire PGS Catalog scoring files and metadata |
| `panel` | construct an individual-level score matrix |
| `fit` | fit the individual-level learned combination |
| `meta` | derive a same-trait combination without a phenotype |
| `evaluate` | evaluate one score against individual-level phenotypes |

See `multipgs <command> --help`, or [guide.md](guide.md). `python -m multipgs`
is the same entry point. Summary-statistic fitting and evaluation are currently
Python API only; there is no `sumstats` command.
