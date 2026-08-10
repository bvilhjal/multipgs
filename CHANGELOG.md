# Changelog

## 0.1.0

First release of `multipgs`. The repository was previously named `pypcma` and
held a Python 2 implementation of principal-component meta-analysis, last
touched in 2017; none of it ran under Python 3 and all of it has been removed.
Git history retains every line at `5bba37b`, and GitHub redirects the old
`bvilhjal/pypcma` URL.

### Combining scores

- `multi_pgs_fit` — the multi-PGS estimator of Albiñana et al. (Nat Commun 14,
  4702, 2023): a penalized regression of the phenotype on `K` polygenic scores,
  selected by Cross-Model Selection and Averaging. Gaussian and binomial
  families, unpenalized covariates fitted inside the same regression, an
  elastic-net `alpha` grid searched per fold, per-score penalty factors, and
  scores that can be forced to stay in the model.
- `meta_pgs` — the training-free combination for scores of the *same* trait
  from different discovery GWAS. `sqrt_n_eff` (the rule in `meta_prs.R` of the
  PGS-pipeline accompanying Hansen et al.), `expected_r2`, and `decorrelated`,
  which additionally discounts scores for information they share.

### Building and deploying panels

- `panel_from_catalog` — score PGS Catalog files against a target cohort in a
  single pass over the genotypes, harmonising alleles through
  `ldpred3.harmonize`.
- `panel_from_sumstats` — fit each GWAS with `ldpred3.run_ldpred3_prs` against
  one shared, cached LD reference.
- `read_scoring_file` — PGS Catalog format, including log-transforming
  odds-ratio weights, dropping non-additive rows, and preferring harmonized
  `hm_*` coordinates.
- `combine_weights` — collapse a panel and a fit into one per-variant weight
  file that `ldpred3.score_from_weights` reads directly, reproducing the fitted
  combination to a correlation of 1 within 1e-8.

### Screening

- `daetwyler_r2`, `screen`, `penalty_from_accuracy` — the summary-statistic
  inclusion criteria of Hansen et al. (Research Square, 2026), applied there
  across 1,523 GWAS Catalog traits, plus the conversion of expected accuracy
  into per-score shrinkage.

### Evaluation

- `evaluate` with bootstrap intervals, `incremental_r2`, `auc`,
  `nagelkerke_r2`, and `liability_r2` including the Lee et al. (2012)
  ascertainment correction.

### Documentation

- [`docs/theory.md`](docs/theory.md) — why combining scores works, derived: the
  closed form for the gain from a correlated trait, the selection-index optimum
  and its Sherman–Morrison solution, the Daetwyler bound and the liability-scale
  conversion, and the failure modes (sample overlap, ancestry, score-vs-model
  R², bootstrap bounds that are zero by construction).
- [`docs/references.md`](docs/references.md) — 91 annotated references. Each was
  checked against the published record — DOI resolved, authors/year/venue and
  the specific claim confirmed against the publisher page, PubMed or Crossref —
  before being cited. `tests/test_docs.py` fails if any DOI cited in the docs or
  the source is not among them.

### Corrections to earlier drafts of this release

Found while writing the above, all in material written earlier in this same
release:

- **The Albiñana et al. title** does not end "…from the PGS Catalog"; it ends at
  "…937 polygenic scores". Corrected in three places.
- **That paper did not use CMSA.** It fitted with `cv.glmnet` and assessed by
  fivefold cross-validation in iPSYCH. CMSA is this package's choice; the docs
  claimed it was the paper's.
- **`w ∝ ρ` is not "the inverse-variance combination".** Under the same
  independent-error model the GLS weights are `R_k/(1-R_k²)`; the two agree only
  as every `R_k → 0`. The claim now states the approximation and its range.
- **`simulate_same_trait_panel` set each score's accuracy to
  `sqrt(daetwyler_r2)`** — the *phenotypic* r², i.e. `h·R_k` — where the
  genetic-value accuracy `R_k` belongs. That is precisely the quantity
  `meta_pgs(method="expected_r2")` consumes, so the simulation handed that one
  rule the true `ρ` by construction. With it fixed, **`sqrt_n_eff` beats
  `expected_r2`**, the reverse of what the first table in this release reported.
  The reason is structural: accuracy saturates and the optimal weight does not,
  so `sqrt(x)` is a better-shaped weight than `sqrt(x/(1+x))`.
- **`daetwyler_r2` returns the phenotypic r²**, now said explicitly, since the
  distinction is what the previous item turned on.
- **`multi_pgs()` matched score columns by position only.** Swapping two columns
  of a ten-score panel moved held-out r² from 0.448 to 0.075 with no error.
  It now checks ids when given a `ScorePanel` or `score_ids=`.
- **`penalty_from_accuracy`'s `clip`** bounds each factor to `[1/clip, clip]`,
  so the largest-to-smallest ratio is `clip**2`, not `clip`.
- **"Most PGS Catalog scores are UK Biobank-derived"** could not be substantiated
  as stated; reworded to "many", with the actionable instruction to check each
  score's development samples.
- **`screen` does not implement one of Hansen et al.'s gates** — the
  shrinkage-coefficient ≥ 0.4 requirement — because ldpred3 does not expose that
  quantity. Now said in the docstring rather than left implied.

### Notes on two decisions that measurement forced

- **The fit's gate is a pooled, leave-one-fold-out assessment, not a per-fold
  one.** Gating each fold on its own held-out loss selected 30 of 40 scores on
  pure noise; pooling the folds but evaluating each at the penalty *it* chose
  still reported `cv_r2 = +0.014` for noise. Taking each fold's operating point
  from the other folds gives a negative `cv_r2` on noise in every seed tested
  and returns the null model in 7 of 8. `cv_r2` is reported incremental over
  the covariate-only model, so it is comparable to `incremental_r2` in a
  held-out cohort.
- **`meta_pgs(method="decorrelated")` wants `expected_r2`, not `n_eff`.**
  `sqrt(n_eff)` tracks accuracy only while `Nh²/M ≪ 1`; past saturation it
  overstates the largest GWAS, and `C⁻¹` amplifies the error. Measured across
  overlap levels, decorrelation with `n_eff` loses to the plain weighted sum,
  while decorrelation with `expected_r2` wins everywhere. Both are in
  [docs/algorithm.md](docs/algorithm.md#choosing-a-meta-pgs-rule).
