# Changelog

## Unreleased

Correctness and documentation:

- Independent summary-statistic tuning now constructs an orthonormal basis in
  training-standardised coordinates before projecting moments. Previously,
  unequal training and tuning score scales could turn the nominal projection
  into a non-idempotent transformation, corrupting moments even when the
  tuning Gram was full rank.
- The training-free examples now use `method="expected_r2"` for Daetwyler
  accuracy proxies. `method="decorrelated"` is reserved for independently
  credible, consistently oriented per-score target correlations.

The performance changes below have no intended change in results. Fitted
coefficients, intercepts, cross-validated losses, selected penalties and
per-fold selections are bit-identical to 0.3.0 across the Gaussian and
binomial families. What
moves, all from reassociated sums: `FoldFit.loss` and the summary-statistic
path diagnostics by about one unit in the last place, and accumulated score
columns by about 2e-12 relative.

End to end, `multi_pgs_sumstats` is 1.9x faster at `K=100` and 2.7x at
`K=900` (2.7s to 1.0s) with same-data tuning, 1.2x to 1.9x with PUMAS.
`multi_pgs_fit` gains only 3-12%: its cost is dominated by forming the Gaussian
sufficient statistics, not by the path.

- `panel_from_catalog` accumulates each genotype block with one dense product
  over the scores that touch it, instead of gathering columns per score. The
  gather copies, so it was memory-bound and repeated `K` times. Measured 48x on
  one block at 50,000 samples and 500 scores. The crossover is near a block
  density of 0.005 and the gather is retained below it, so sparse panels do not
  regress.
- Summary-statistic path selection symmetrizes the selection Gram once per fit
  rather than once per candidate, and scores the whole path in one batched
  product. This was `O(K^2)` per path point, per shrinkage value, per PUMAS
  repeat — 642 ms per path at `K=900`, now 4 ms.
- `score_gram` reads an ldpred3 low-rank (LR8) block through its factor,
  `W'DW = (U'W)'(U'W) + (residual * W)'W`, instead of asking `ld_matmul` to
  project back up to the block's variant dimension. Reproducibly 1.5x to 1.7x
  per low-rank block; genome-wide on the bigsnpr HapMap3+ reference the gain is
  diluted to roughly 1.1x-1.3x, because 406 of its 625 blocks are dense and
  unchanged, and run-to-run variance on the development machine is too large to
  put a firmer number on it. Dense and int8 blocks, and any representation
  added later, still route through `ld_matmul`. Int8 factors are dequantized
  first; new tests pin that, the equivalence with the dense form, and mixed
  references, none of which the suite previously covered.
- The CMSA inner loop scores a whole block of the penalty path with one
  product rather than one per penalty. Selection remains sequential, so the
  abort counter and tie-breaking are unchanged.
- `multi_pgs_sumstats` parses its sparse LD-basis weights once instead of
  twice. This saves the second parse, not peak memory: measured peak allocation
  along the Gram path is unchanged.
- Four real-data benchmarks. `real_ld_simulation.py` draws summary statistics
  from the real reference (`z ~ N(D beta, D/n)`), which buys three independent
  GWAS of one trait — so a regime A label is checkable, which nested consortium
  releases never allow — and a closed-form truth, `R2 = (w'D beta)^2 / (w'D w)`.
  Its reference-mismatch arm fits from a simulated finite panel drawn from the
  true `D`, reproducing rank deficiency rather than mere added noise.
  `overlap_inflation.py` sets sample overlap to a known
  `rho_s = rho_p N_shared / sqrt(N1 N2)`, which is also the estimand of the
  cross-trait LDSC intercept, and reports both the inflation it causes and
  whether bipred's `ldsc_rg` detects it. `real_meta_rules.py` replaces
  `meta_rules.py`'s stylized sharing knob with the observed off-diagonal of a
  real score covariance. `sumstat_cost.py` gives the stage-by-stage cost of the
  summary-statistic pipeline at real dimensions.
- Both simulation benchmarks scale `h2` to the selected chromosomes' share of
  the reference by default (`--h2-genome-wide`). Holding `h2` fixed while
  subsetting does not simulate a smaller slice of the same trait: it packs a
  genome's heritability into a few per cent of the variants and inflates every
  per-variant effect, which moves the LDSC residual scatter and hence the
  intercept's precision for the wrong reason. `--h2-in-subset` restores the
  literal reading.
- `benchmarks/real_ld_gram.py` reports score-space moments and their cost on a
  real genome-wide LD reference: block census, Gram spectrum and rank, dead
  scores, wall time and peak memory, and an optional check that the low-rank
  fast path agrees with routing every block through `ld_matmul` (it does, to
  5e-16). It faults the memory-mapped payload in before timing, because
  otherwise the first route measured pays to page in a gigabyte and the second
  looks twice as fast.

Two things measured and deliberately **not** changed. Numba on the sparse
scatters: `np.add.at` costs about 20 ms per 20 million entries in current
NumPy, so its 2.2x is worth 11 ms in a 25-second genome-wide fit, and
`np.bincount` is slower than either. `fastmath` on the coordinate-descent
sweeps: 0.85x-0.99x, that is no faster, with bit-identical coefficients — the
inner loop is an AXPY that Numba already vectorizes, with no reduction to
reassociate. `_coord.py` keeps `_jit_nogil`, which costs nothing and preserves
determinism.

## 0.3.0

Summary-statistic multi-PGS fitting and reproducible PGS Catalog acquisition.

### Summary-statistic Multi-PGS

- `multi_pgs_sumstats` fits a lasso or elastic-net combination from
  `W_ld.T @ D @ W_ld` and `W_gwas.T @ z`, without individual-level genotypes
  or phenotypes. The penalty acts on whole component scores; this is a
  lassosum-inspired score-space estimator, not SNP-level lassosum.
- An independent `z_valid` tunes the path but does not assess the selected
  model. `score_moments`, `evaluate_sumstat`, and `SumstatFit.evaluate` measure
  a fixed combination against a third untouched GWAS; every result records its
  declared provenance.
- Path selection minimizes summary MSE. Squared pseudo-R² remains descriptive
  and cannot select an oppositely directed predictor.
- `tune="pumas"` supplies joint-Gaussian/CLT plug-in pseudotuning from one GWAS.
  It is documented as an approximation, requires component weights constructed
  independently of that GWAS, and is not reported as external validation.
- `SumstatFit.beta` follows the package-wide raw-score coefficient contract;
  `beta_std` supplies standardized-score coefficients, and
  `frozen_variant_weights` and `combine_weights` expose the LD-frozen and raw
  panel deployment routes without mixing their scales.
- Score moments stream disjoint LD blocks and sparse score columns. Duplicate
  sparse entries are coalesced, materially indefinite LD fails, and an
  unbounded external-LD objective cannot return runaway coefficients.
- GWAS and LD inputs carry separate, explicit standardized-genotype weight
  matrices. This preserves one raw-score coordinate when their empirical
  genotype standard deviations or variant sets differ.
- Same-LD in-sample and PUMAS tuning project unresolved GWAS signal onto the LD
  Gram's estimable range. Independent tuning projects both training and tuning
  moments onto the tuning Gram's range, so it may retain a direction absent
  from the fitting reference only when the tuning reference resolves it. All
  discarded fractions are logged.
- `align_to_reference` prefers the empirical dosage SD used by the
  accompanying GWAS or LD source. HWE scaling from allele frequency remains
  available only through the explicit `hwe_genotype_sd=True` approximation.
- `benchmarks/sumstat_calibration.py` records the individual-level moment
  identity, null tuning-versus-assessment MSE gap, and Gaussian/binary error of
  the PUMAS covariance plug-in with per-seed provenance.

### PGS Catalog acquisition

- `search_scores` finds scores by trait, score identifiers, PMID, or Catalog
  publication; `download_scores` retrieves harmonized scoring files for a
  named genome build.
- `write_score_metadata` writes discovery sample size, ancestry, method, and
  publication metadata, while `cohort_overlap` flags named discovery cohorts
  shared by score pairs.
- `multipgs fetch` exposes the acquisition workflow, including response
  caching, metadata-only runs, child-trait inclusion, and overlap reporting.
- Download verification now requires matching score metadata, a valid scoring
  table, and at least one variant; a bad cached file is removed so the next run
  can retrieve it again.

### Documentation and release

- The README, guide, algorithm notes, API map, and bibliography now distinguish
  fitting, tuning, and untouched assessment for both individual-level and
  summary-statistic workflows.
- The package version is `0.3.0`; the public summary-statistic API remains
  Python-only in this release.

## 0.2.0

Correctness, integrity, and packaging hardening after the first release.

### Stacking

- `cv_r2` is now a genuinely nested outer-fold predictive gain over the fitted
  unpenalized baseline. Grid construction, imputation, inner CMSA, and fitting
  do not see the corresponding assessment rows.
- Final CMSA averages every fold-selected vector. If the nested signal gate
  fails, the full-data baseline is returned, preserving covariates and forced
  scores; `null_model` now means a null *penalized increment*.
- Ridge has an explicit unpenalized candidate, missing-value means are learned
  within each outer-training set, and duplicate normalized identifiers fail.
- Gaussian fold fitting reuses stable, chunked sufficient statistics while
  retaining only one held-out Gram at a time.

### Panels, metadata, and command line

- Sum-statistic panels validate both FID and IID, preserve explicit inference
  controls, and never infer attempted chains from the retained count.
- Catalog panels store one read-only union metadata table with compact per-score
  indices. The fully overlapping regression uses 25% of the former NumPy-buffer
  bytes, while panels remain pickleable and deepcopyable.
- Score metadata and penalty vectors align by score ID throughout the Python and
  command-line APIs; duplicate rows, columns, samples, and normalized IDs fail.
- `meta --method expected_r2` is reachable, and `MetaPGS` validates score order
  just like a learned fit.

### Architecture, documentation, and distribution

- `screen` implements the represented model-level gates: heritability, retained
  and attempted chains, post-QC variants, strict effective sample size, and an
  optional fitted shrinkage coefficient distinct from `shrink_corr`.
- Penalty-factor projection now satisfies both its individual bounds and unit
  geometric mean, and rejects unclipped numerical underflow to zero.
- Scientific claims were reconciled with the implementation and reproducible
  30-seed meta-rule and stack-scaling benchmarks were added with provenance.
- Source distributions now include the documentation, examples, changelog,
  benchmarks, and tests. CI covers Python 3.9, 3.13, and 3.14.

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
