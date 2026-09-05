# Changelog

## Unreleased

Version `0.3.4.dev1`.

- Advance to `ldpred3>=0.7.12,<0.8`, restoring a resolvable combination with
  current bipred/GWFM and sharing bounded compact-LD score-Gram contractions.
- Consume sequential trait outcomes immediately and bound pending plus ready
  parallel fits. Failures no longer wait for every later fit; skipped failures
  do not retain tracebacks and their working arrays. One-pass scoring remains.

- **Re-baselined onto the LDpred3 0.6 line** (`ldpred3>=0.6,<0.7` in
  `pyproject.toml` and the README install contract). The 0.5 ceiling had
  been overtaken by events: the packaged methods-report evidence was
  generated under ldpred3 0.6.0, out of the range the package declared.
  The one observable API change on multipgs' seams is that an external
  PLINK LD reference is now read through a single lazy metadata scan plus
  block-wise dosage streaming instead of `ldpred3.pipeline.load_genotypes`;
  the read-the-reference-once guarantee for disjoint traits is re-verified
  against the new hooks in
  `tests/test_panel.py::test_real_sumstat_panel_validates_once_and_batch_scores`,
  and the full suite plus regenerated report evidence now run against
  ldpred3 0.6.5. The 0.5 line is no longer supported.
- `enet_path_gaussian` raises when exactly one of `beta_init`/`grad_init`
  is passed, instead of silently discarding the half warm start and
  falling back to the unpenalized fit.
- The nested CV assessment draws its outer-fold partition from a spawned
  child seed rather than the same `default_rng(seed)` stream as the CMSA
  folds; the two partitions no longer coincide for a given seed. Nested-CV
  outputs (`cv_r2` and downstream) differ from previous runs at a fixed
  seed -- intended -- while the CMSA partition is unchanged.
- `panel_from_catalog` records the effective `min_matched` in its build
  log (per-score matched counts were already in `meta`/`summary()`).
- `combine_weights` sorts chromosomes naturally (1, 2, ..., 10, 11, X)
  instead of by string.
- `evaluate` counts bootstrap replicates it could not use (single-class
  binomial resamples, metrics that raised) as `n_boot_skipped` on
  `EvalResult`, so silent interval shrinkage is visible.
- `read_scoring_file` fixes the delimiter from the header line instead of
  re-sniffing per row; a space-joined data row under a tab header now
  counts as unparsable rather than silently misaligning columns.
- `multi_pgs_fit` documents that 1-D `scores` means one individual with K
  scores (ambiguous when n == K), and that `missing="mean"` learns
  imputation means before the CMSA fold split, so fold-level (alpha,
  lambda) selection is mildly optimistic while `cv_r2` stays honest.
- `panel_from_sumstats` writes a new `ld_cache` reference-wide
  (`subset_to_sumstats=False`) and then loads it with LDpred3's default
  per-trait subset. Passing `subset_to_sumstats=False` against an
  existing cache is rejected; LDpred3 only allows that flag while
  building fresh LD.
- The optional genetic-correlation screen's Bipred floor is
  `bipred>=0.3.9.dev0,<0.4`, advanced from `>=0.3.8` alongside the 0.3.4
  line's coordinated Bipred 0.3.9 (recorded here explicitly).

- `liability_r2` computes the liability threshold as `-inv_cdf(K)` rather than
  `inv_cdf(1 - K)`. The identity `Phi^-1(1-K) = -Phi^-1(K)` is exact, so no
  realistic prevalence changes by even one bit; the old spelling lost digits
  below `K ~ 1e-9` and raised out of `NormalDist` for `K <= 1.1e-16`. Matches
  `ltpred.thresholds` and `ldpred3.scale`.
- `multipgs fit` and `multipgs evaluate` treat `-9` (and any other numeric
  code passed via the new `--missing-code` option) as a missing value in
  `--pheno`, `--covar` and `--scores`: such rows are dropped and counted like
  any other non-finite input instead of scored as data. Pass
  `--missing-code none` to restore the old behaviour.
- Header detection in CLI tables now requires every trailing field of the
  candidate header row to be non-numeric, so an individual literally keyed
  `ID`/`FID` in a headerless file is no longer silently swallowed as a header.
- `save_panel`, `load_panel` and `read_panel` agree on the `.npz` suffix:
  saving a path without one writes (and reports) `path.npz`, and reading a
  suffix-less path finds it. TSV round trips document that scale flags are
  not preserved.
- `panel --out-panel` reports the path actually written, and `panel` warns
  when an option has no effect for the chosen source branch (for example
  `--standardize` with `--weights`). The `--ld-prefix`/`--ld-cache`
  requirement message names `--traits` as well as `--sumstats`.
- `_verify_scoring_file` also converts corrupt (not merely truncated) gzip
  streams into its readable-gzip error instead of leaking `zlib.error`.
- `_request_json` rejects `retries=0` explicitly instead of raising
  `NameError` from an empty retry loop.
- `read_scoring_file` judges harmonized-coordinate provenance across rsID,
  chromosome and position together, recording `mixed_coordinate_warning` when
  they disagree, rather than keying on the position column alone.
- `combine_weights` skips zero-weight entries before per-variant
  reconciliation; streamed target scoring lowers the block-size floor so the
  ~64 MB dosage budget holds at biobank sample sizes.
- The guide states that target-cohort AF/SD/imputation means are computed on
  all individuals including future CV folds (unsupervised transductive use).
- Frozen-scoring round-trip tests compare against the precision the weights
  file format guarantees (`%.8g`) instead of an in-memory tolerance.
- Panel constructors and `ScorePanel` itself reject duplicate stringified
  score ids. `index_of` and `select` share that stringified lookup and fail
  when an id is not unique, instead of first-hit versus last-wins.
- Headerless CLI tables with three or more columns are `FID IID <values...>`,
  including numeric PLINK identifiers. IID is no longer promoted into the
  design matrix.
- Binomial coordinate descent uses the same shared `max_iter` sweep budget as
  the Gaussian Gram path. `on_error="skip"` now covers harmonisation failures,
  and catalog / weight / sumstat directory scans use route-specific suffixes.
- `align_to_reference` wraps a mapping variant table into an ldpred3
  `VariantTable` before calling `harmonize`.
- **Summary-statistic QC defaults now applied by LDpred3 on multipgs' behalf.**
  `panel_from_catalog` and the other fitting entry points call
  `ldpred3.run_ldpred3_prs`, and LDpred3 0.5.3 changed three of its defaults.
  multipgs passes none of them explicitly, so every panel fitted since the
  `>=0.5.3.dev0` floor inherits all three, with no change to multipgs' own
  code:

  1. `impute_n=True` — per-variant effective sample sizes are imputed from the
     standard errors and reference allele frequencies rather than the reported
     total being reused for every variant.
  2. The low-N filter is anchored on the **median** per-variant N rather than
     the file maximum, keeping variants at `N >= 0.7 * median(N)`. On a
     summary-statistic file that meta-analyses two arms of unequal size, the
     old max anchor discarded most of the file; a measured two-arm case
     retained 300/300 variants where the max anchor retained 60/300.
  3. `gc_correct=True` — where LD-score regression establishes an intercept
     significantly below one, the standard errors are rescaled by
     `sqrt(intercept)` to undo genomic control. This changes `beta/se`, and so
     every p-value-keyed filter downstream of it.

  Weights and scores therefore move relative to multipgs 0.3.3 for the same
  inputs. Pass the corresponding keyword through the `run_ldpred3_prs` kwargs
  to restore the previous behaviour. See the LDpred3 changelog for the
  derivations; this entry exists because the defaults are documented there and
  the behaviour change lands here.

- `multipgs/_numba.py` now binds its Numba decorators (`_jit_nogil`,
  `_jit_parallel`, `prange`, `HAVE_NUMBA`) from LDpred3's new public
  `ldpred3.shim` code-level surface instead of owning a duplicate of the same
  try/except shim — one implementation across the sibling packages.
  `warn_no_numba` stays local (its message names multipgs' own kernels), and
  the LDpred3 floor advances to `>=0.5.3.dev0,<0.6` for the new module.
- `multi_pgs_fit` and `multi_pgs_sumstats` warn once per process when the
  pure-Python no-Numba fallback is active (`multipgs._numba.warn_no_numba`):
  the NumPy solver twins compute the same answers but much slower, and the
  fallback was previously silent.
- Panel construction against one target inherits LDpred3's per-variant-table
  harmonisation index cache (LDpred3 changelog), so `panel_from_catalog` no
  longer rebuilds the O(variants) matching index per scoring file.
- Began the 0.3.4 development line, coordinated with LDpred3 0.5 and Bipred
  0.3.9. Real cache-to-GWAS alignment now passes LDpred3's actual
  `VariantTable`, and dense score panels remain dense during Gram and
  cross-moment calculations instead of allocating three COO arrays per
  non-zero entry. Multi-trait summary-statistic panels fully validate their
  shared cache once, prepare target metadata once, fit against those shared
  read-only contexts, and score all fitted columns in one target-dosage pass.
  Catalog files, saved weights, and fitted panels all stream the selected BGEN
  union through LDpred3's byte-capped dosage iterator instead of loading an
  `n_samples x n_variants` matrix. Existing reference-wide caches therefore
  incur one LD validation and one selected-payload decode; a new in-sample
  cache adds its necessary LD-construction read.
  Multipgs now owns its small Numba decorator shim instead of importing LDpred3
  internals. Ecosystem CI resolves LDpred3, Bipred, GWFM, and Multipgs together
  and exercises their public interoperability seams on changes, manual runs,
  and a weekly upstream-drift schedule.
- The packaged methods PDF is a research note: estimand,
  selection-index theory, three information routes, and the
  simulation evidence that survives its design, rather than a
  project-validation brochure.
- Docs and the methods note now distinguish this package from
  MIXPRS: multi-trait (or metadata same-trait) score stacking
  versus a same-trait ensemble of multi-population methods.

## 0.3.3 - 2026-08-13

Correctness and release integrity:

- LDpred3 weight panels now distinguish target-standardized files from frozen
  files, impute missing frozen dosages to the frozen reference mean, and fold
  components from different AF/SD references onto one declared deployment
  basis. Deployment checks require unit slope and negligible centred error;
  correlation alone no longer passes a rescaled predictor.
- Architecture gates compose: enabling expected-R² screening no longer skips
  the requested genetic-correlation gate, and an rG-only record is screenable.
  Noisy finite LDSC estimates are clipped to the correlation parameter space
  before relevance penalties are ranked.
- Summary-statistic paths propagate coordinate-descent exhaustion instead of
  silently selecting an unfinished solution. Fixed-vector evaluation uses the
  observed fitted direction, external moment incoherence has an explicit
  diagnostic policy, and sparse variant indices must be exact integers.
- CLI combination requires the exact panel score-ID set, BGEN summary panels
  receive `--sample`, and NPZ panels reject duplicate samples and score IDs.
- Panel concatenation cannot detach metadata or weight tables from their score
  columns. Bootstrap controls, liability-scale R² inputs, and trusted-only NPZ
  loading are validated or stated explicitly.
- Genetic-correlation screening reuses one validated LD-cache load across all
  traits. Independent summary tuning reuses identical Gram work where safe,
  and `max_iter` is once again a linear bound on coordinate sweeps.
- The optional rG extra now requires `bipred>=0.3.8`, the first release that
  provides the chi-square row mask used by `ldsc_rg_screen`.
- CI uses an explicit credential for the private compatible LDpred3 revision.
  The methods PDF embeds the current source digest and has a checked SHA-256
  sidecar, so fresh TeX evidence can no longer mask a stale packaged PDF.

## 0.3.2 - 2026-08-13

- Full panels persist to `.npz` (`save_panel` / `load_panel` / `ScorePanel.save`)
  with weights, scale flags, inference and `n_eff`. `concat` joins two panels
  on `FID:IID`. `panel_from_weights` scores a directory of ldpred3 weight files
  in one pass. `check_weights` requires frozen scoring to reproduce the fit.
- `panel_from_sumstats` accepts a per-trait table (`n_eff` / cases / method /
  alpha), `ld_prefix`, `weights_dir` and an optional preflight. The CLI requires
  `--ld-prefix` or an existing `--ld-cache`, writes `.npz` when `--out` ends
  that way, and adds `combine` / `score` (frozen). Catalog `metadata.tsv` is
  attached automatically.
- Optional `multipgs[bipred]`: `ldsc_rg_screen` estimates `r_G` of auxiliary
  GWAS against a focal trait on a shared ldpred3 cache (χ² cap on LDSC rows
  only). `penalty_from_relevance` ranks by `r2 · r_G²`; `screen(min_abs_rg=)`
  is the matching gate. ldpred3 dependency is `>=0.4.7,<0.5`.
- Nested Gaussian assessment reuses the full-data parent Gram by subtracting
  held-out rows. The numerical origin is a function of `X` only (`origin_y` is
  identically zero), and training `X'y` is formed on the training rows, so an
  outer-assessment phenotype cannot leak into the path. Fold-local mean
  imputation still rebuilds.
- `panel_from_sumstats(..., n_jobs=)` fits independent traits concurrently
  after the first successful trait has written `ld_cache`. The default `1` is
  sequential.
- PUMAS pseudo-training refits that share a Gram run as one compiled
  coordinate-descent batch (`enet_path_gaussian_batch`) instead of a Python
  loop over repeats. Boundedness is still applied per repeat.
- Rank, range projection and the cleaned covariance of a score Gram now share
  one correlation-scale eigendecomposition.
- `stack.py` and `sumstat.py` are split along named seams (`_cmsa`, `_stats`,
  `_gram`, `_moments`, `_pumas`, `_align`, `_evaluate`, `_validate`). Public
  imports are unchanged.
- The guide now describes `penalty_from_accuracy` as a ranking heuristic, not
  a bound on target-trait relevance, matching the implementation.
- Missing Catalog ancestry is written `NA` in `EUR_PERCENT`, not `0`. A
  recorded 0% share is still `0`.
- Source distributions include `tests/`, as claimed since 0.2.0. CI asserts
  the test tree is in the sdist.
- The API map no longer pins "no `sumstats` command" to a version number.

## 0.3.1 - 2026-08-12

Correctness and documentation:

- Independent summary-statistic tuning now constructs an orthonormal basis in
  training-standardised coordinates before projecting moments. Previously,
  unequal training and tuning score scales could turn the nominal projection
  into a non-idempotent transformation, corrupting moments even when the
  tuning Gram was full rank.
- The training-free examples now use `method="expected_r2"` for Daetwyler
  accuracy proxies. `method="decorrelated"` is reserved for independently
  credible, consistently oriented per-score target correlations.
- `multi_pgs_fit` validates its numerical controls, builds a separate lambda
  grid for each alpha, and reports coordinate-descent and IRLS iteration
  exhaustion in fold results and the fit log.
- `expected_r2` is constrained to `[0, 1]`, and
  `align_to_reference` now states explicitly when returned weights remain on
  the raw allele-count scale.
- The theory and guide now match fold-local standardisation, distinguish the
  package's CMSA and nested heuristic from the published workflow, and keep
  predictive selection separate from causal interpretation.
- Benchmark summaries are checked against raw rows; future runs record source
  identity; distribution checks cover the full benchmark evidence and exclude
  ignored real-data inputs.
- The packaged LaTeX/PDF artifact is now a methods and validation report,
  focused on the project, its equations, and committed test results.

Apart from the correctness changes above, the performance changes below have
no intended change in results. For scalar-alpha fits, fitted coefficients,
intercepts, cross-validated losses, selected penalties, and per-fold
selections are bit-identical to 0.3.0 across the Gaussian and binomial
families. Mixed-alpha fits intentionally differ because each alpha now uses a
correctly anchored lambda grid. Reassociated sums can also move
`FoldFit.loss` and summary-statistic path diagnostics by about one unit in the
last place, and accumulated score columns by about 2e-12 relative.

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
