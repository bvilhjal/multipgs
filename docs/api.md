# Python API

Every name below is importable directly from `multipgs`. Use `help(name)` for
the complete signature and validation rules.

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
| `meta_pgs` | combine same-trait scores with no phenotype, from `n_eff` or fitted accuracy |
| `MultiPGSFit`, `SumstatFit`, `MetaPGS` | fitted combinations on the common raw-score coefficient contract |
| `FoldFit` | what one CMSA fold selected, and how it did |

`MultiPGSFit.multi_pgs(scores)` is the combined score — what you evaluate.
`.predict(scores, covar)` is the full linear predictor including covariates, and
is a different thing; see [guide.md §5](guide.md#5-evaluating-and-the-ways-this-goes-wrong).
`.cv_r2` is the nested outer-fold predictive gain over the explicit
unpenalized baseline; it is not the OLS-recalibrated `incremental_r2`.

Pass a `ScorePanel` to `.multi_pgs()` rather than a bare matrix and the score
ids are checked against the fit. A bare matrix is matched by **position**, and a
separately built panel can have the same columns in a different order.

## Summary-statistic fitting

**Table 2. Score-space sufficient statistics.**

| Name | Purpose |
|---|---|
| `align_to_reference` | harmonize component weights to one data source's variant order and standardized-genotype scale |
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

## Building the panel

**Table 3. Panel construction.**

| Name | Purpose |
|---|---|
| `panel_from_catalog` | score PGS Catalog files against a target, in one genotype pass |
| `panel_from_sumstats` | fit each GWAS with LDpred3 and score it on the target |
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
| `daetwyler_r2` | expected r² of a score for its own trait, from `h²`, `p`, `n_eff` |
| `Architecture` | per-score `h²`, polygenicity, inferred r², total/kept chains, `n_eff`, and optional fitted shrinkage |
| `architectures_from_panel` | read those back out of an LDpred3-built panel |
| `screen`, `ScreenResult` | represented model-level Hansen et al. gates, with per-score reasons |
| `penalty_from_accuracy` | expected accuracy → elastic-net penalty factors |

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
Python API only; there is no `sumstats` command in 0.3.0.
