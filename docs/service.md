# Running without a cohort

This is the contract for driving multipgs from a service that holds summary
statistics and a prepared LD reference and **never sees target genotypes** —
the position [SMARTpred](https://github.com/bvilhjal/SMARTpred) is in for every
mode it serves. It is not a second implementation: it is the subset of the
ordinary API that survives that boundary, plus the three places where the
boundary changes what a number means.

Read [guide.md §4](guide.md#fitting-from-summary-statistics) first for the
estimator itself. This page only adds what the missing cohort costs.

## 1. What crosses the boundary

**Table 1. The three fitting routes against a genotype-free caller.**

| Route | Needs individuals? | Available |
|---|---|---|
| `fit_prepared_panel` | no | yes — the packaged route: cache, prepared trait, component weight files in, weight file out |
| `multi_pgs_sumstats` | no | yes — target GWAS plus the LD reference is exactly its input |
| `meta_pgs` | no, with `scores=None` | yes — same-trait panels only; `method="decorrelated"` is not available |
| `multi_pgs_fit` | **yes** | no — CMSA holds out people, and there are none |

**Table 2. Getting the `K` component scores.**

| Function | Available | Why |
|---|---|---|
| `align_weights_to_reference` | yes | harmonizes LDpred3 weight files — the service's own earlier fits — to the reference, and refuses one built on a different panel |
| `align_to_reference` | yes | harmonizes PGS Catalog files to the reference variant table |
| LDpred3 posterior means | yes | fit each auxiliary GWAS against the same LD cache; the weights are already standardized |
| `panel_from_sumstats` | no | it fits *and scores* against a PLINK/BGEN target |
| `panel_from_catalog`, `panel_from_weights` | no | both are one pass over target dosages |

`architecture.screen`, `daetwyler_r2`, `penalty_from_accuracy`,
`penalty_from_relevance` and `rg.ldsc_rg_screen` are all summary-level and
cross unchanged. So do `score_gram`, `score_moments`, `pseudo_r2` and
`evaluate_sumstat`.

## 2. The entry point

`fit_prepared_panel` is the whole route in one call, the counterpart of
`gwfm.fit_prepared_trait`: the caller runs LDpred3's QC, harmonisation and
LD-consistency screen itself, stores the resulting `PreparedTrait`, and hands
it over with the LD cache and the component weight files.

```python
from multipgs import fit_prepared_panel

result = fit_prepared_panel(
    ld_cache,                       # path, or a caller-owned PreparedLDCache
    focal_trait,                    # PreparedTrait for the target GWAS
    ["job_a.weights.tsv", "job_b.weights.tsv"],
    score_ids=["T2D", "BMI"],
    tune="pumas", weights_independent_of_z=True,
    progress=lambda done, total, stage: report(stage, done, total))

result.write_weights("multi.weights")
result.coefficient_table()           # per-component rows, ready to render
```

It opens the cache (borrowing a prepared one and leaving it open), builds the
reference variant table from the cache's own allele, coordinate and frequency
metadata, aligns each component, scatters the trait's `z` into cache order,
fits, and folds the result into one deployable weight file. `PreparedPanelFit`
carries `regime`, `n_eff_policy`, `align_log`, `trait_log` and
`cache_provenance` alongside the fit, so a results page reports what was done
rather than only what came out.

**The components must be weight files fitted against this same reference.**
`align_weights_to_reference` — which `fit_prepared_panel` uses, and which is
public for callers that want the columns themselves — checks each file's own
`AF_REF`/`SD_REF` against the reference at the variants it matched, and
refuses a file that disagrees. Identifiers can match while the frequencies do
not, and the resulting Gram would describe a reference none of the scores were
built on. `check_scale=False` disables the guard for a file predating those
columns; nothing else downstream can detect the mistake.

### Doing it by hand

Component weights and the target `z` must be on **one** index space — the
reference's own variant order, which is the row order of `D`. A prepared trait
carries `indices` into the full cache, so scattering into cache-length vectors
is the whole alignment step:

```python
import numpy as np
from multipgs import multi_pgs_sumstats, combine_weights, ScorePanel

components = [(trait.indices, weights_k) for trait, weights_k in fitted]
z = np.zeros(n_cache)
z[focal.indices] = focal.z

fit = multi_pgs_sumstats(
    components, z, ld_blocks, weights_gwas=components,
    score_ids=score_ids, n_variants_ld=n_cache,
    tune="pumas", n_eff=focal_n_eff, weights_independent_of_z=True)
combine_weights(ScorePanel.weights_only(tables, score_ids), fit,
                path="multi.weights")
```

`ld_blocks` is the reference as `(corr_block, idx)` pairs tiling `0..m-1`,
which is what `ldpred3.interop.load_ld_blocks` returns and what
`subset_ld_blocks` produces from a `PreparedLDCache`. Pass the block list, not
a densified matrix: `score_gram` streams it one block at a time, so peak
memory is `O(block_size · K)`.

`ScorePanel.weights_only` exists for exactly this: `combine_weights` reads
weight tables, `standardized` flags and score ids, never score columns, and
this is the supported way to say so rather than fabricating an empty cohort.

## 3. What the missing cohort changes

**Genotype SD.** `G` and `c` are defined with each source's *empirical* dosage
SD (`W_ld` and `W_gwas` in [guide.md](guide.md#fitting-from-summary-statistics)).
A service has neither the target's nor the GWAS's dosages. Two consequences:

- Components that are LDpred3 posterior means are **already** on the
  standardized-genotype scale, so no SD multiplication applies and the same
  matrix is passed in both roles. This is why `weights_gwas` is mandatory even
  when identical — passing it explicitly is the acknowledgement.
- Raw PGS Catalog weights count alleles and must be converted. With no
  dosages, the only conversion available is the HWE approximation:
  `align_to_reference(..., af=reference_af, hwe_genotype_sd=True)`. It ignores
  imputation uncertainty and departures from equilibrium, `log["scale_source"]`
  records `hwe_from_af`, and the deployed weight file inherits that
  approximation. Record it wherever the result is shown.

**The deployed file's reference scale.** `combine_weights` writes ldpred3's
`ID CHR POS A1 A2 WEIGHT AF_REF SD_REF` layout, and with HWE input the `SD_REF`
column is `sqrt(2 f (1-f))` from the reference panel, not a target dosage SD.
That is the same limitation the univariate weight files carry, and the same
advice applies: score with `scaling="target"` unless the HWE assumption is
defensible for the target cohort.

**What can never be reported.** AUC, any participant-level bootstrap interval,
exact logistic fitting and arbitrary covariate adjustment need individuals. No
summary-statistic algebra recovers them, so a genotype-free mode must not
display them.

## 4. Regimes, with one uploaded GWAS

The regime is the difference between a publishable number and a meaningless
one, and it is invisible in the number itself
([guide.md §4](guide.md#fitting-from-summary-statistics)). A service is
normally given **one** target GWAS, which caps what it can honestly show:

- `tune="pumas"` with `n_eff` gives pseudotuning — **regime B**. It also
  requires `weights_independent_of_z=True`, which is an acknowledgement, not a
  check: every component score must have been built without the uploaded GWAS.
  A panel of PGS Catalog scores derived from the same biobank as the upload
  violates this, and nothing in the code can detect it.
- `tune="none"` reuses the fitting moments for selection — **regime C**, an
  upper bound.
- **Regime A needs a second, untouched GWAS or cohort.** Without one, no
  accuracy claim is available, only a fitted model.

`fit.log["regime"]`, `SumstatEval.regime` and `SumstatEval.is_assessment`
carry this; surface them rather than the R² alone. Sample overlap between the
upload and the panel's discovery GWAS is a separate and larger hazard that
resampling cannot reveal — see
[theory.md](theory.md#sample-overlap) and `fetch.cohort_overlap`.

## 5. LD reference encodings

`score_gram` contracts each block through `ldpred3.interop.ld_crossproducts`,
so every representation LDpred3 supports is consumed in its own form — a
low-rank `LowRankLD` factor keeps its `W^T D W = (U^T W)^T (U^T W) + (rW)^T W`
form and is never widened to a dense block.

Compact and quantized encodings can be numerically indefinite. The solver
checks the score Gram globally and **fails closed** on a materially negative
eigenvalue rather than returning coefficients from an unbounded objective; a
near-zero negative direction is clamped and logged. Positive `ld_shrinkage`
stabilises resolved small-eigenvalue directions, and cannot repair a mismatched
build, ancestry or effect scale. Register the reference the mode will use and
check that a real panel fits on it before releasing the mode; a cache
generation that is fine for one estimator is not automatically fine for a
`K`-by-`K` Gram built from it.

## 6. Progress

`multi_pgs_sumstats(progress=...)` is called as `progress(done, total, stage)`
from the calling thread:

| stage | reports |
|---|---|
| `ld` | blocks contracted for the fitting Gram |
| `ld_tuning` | blocks contracted for a separate tuning reference |
| `shrinkage` | `ld_shrinkage` grid points completed |

Streaming LD dominates a genome-wide panel; the `K`-by-`K` coordinate descent
is seconds beside it, so the `ld` stages are the ones worth showing. `total` is
the exact block count for a concrete block list and `None` for a lazy stream,
which is never consumed to obtain one. `score_gram` and `score_moments` take
the same callback without the `stage` argument.

`align_to_reference(progress=...)` reports `progress(done, total, label)` per
scoring file, and `fetch.download_scores(progress=...)` per download.
