# Benchmarks

Benchmarks are executable evidence, not decorative numbers. Run them from the
repository root with the environment being reported.

## Meta-PGS rules

`meta_rules.py` reproduces the same-trait comparison in the algorithm notes
over 30 independent seeds:

```bash
python benchmarks/meta_rules.py
```

It writes every replicate, a mean/standard-deviation table, and machine-readable
provenance under `benchmarks/results/`. The simulation parameter
`shared_error_correlation` correlates score errors as a stylized consequence of
shared discovery information. It is deliberately not called sample overlap:
the simulator does not model overlapping people or discovery GWAS estimation.

The committed results are a regression reference, not a claim about real
cohorts. Re-run before changing a default or making a performance claim.

## Stack scaling

`stack_scaling.py` runs each Gaussian case in a fresh process, warms the
numerical kernels, and records wall time plus absolute peak resident memory:

```bash
python benchmarks/stack_scaling.py
```

The small defaults are suitable for routine validation. Supply representative
cohort dimensions explicitly before making a deployment claim, for example
`--cases 20000x250 50000x500`. Peak RSS includes the Python/NumPy/Numba runtime;
compare rows produced by the same command and environment.

## Summary-statistic calibration

`sumstat_calibration.py` checks three scientific contracts over independent
seeds:

1. `W_ld.T @ D @ W_ld` and `W_gwas.T @ z` against moments computed directly
   from discrete individual-level dosages.
2. Null-GWAS MSE used for path tuning against MSE in a third untouched GWAS.
3. The joint-Gaussian covariance used by PUMAS-style pseudotuning against
   empirical score-by-phenotype fourth moments, for Gaussian and binary traits.

```bash
python benchmarks/sumstat_calibration.py
```

It writes raw per-seed rows, a mean/standard-deviation table, and runtime
provenance under `benchmarks/results/`. The PUMAS covariance errors are
calibration diagnostics for a plug-in approximation, not prediction-accuracy
estimates. Re-run this benchmark before changing its assumptions or presenting
pseudotuning as empirically calibrated in a new setting.

**Table 1. Default calibration, 30 seeds and 10,000 simulated people.**

| Quantity | Mean |
|---|---:|
| Gram identity, maximum absolute error | 1.32e-14 |
| Cross-moment identity, maximum absolute error | 2.79e-15 |
| Null tuning MSE | 0.999952 |
| Null untouched-assessment MSE | 1.000048 |
| Assessment minus tuning MSE | 9.54e-5 |
| Gaussian plug-in covariance relative error | 0.0377 |
| Binary plug-in covariance relative error | 0.1099 |

Thus the algebraic identity is at floating-point error, while selecting on the
tuning GWAS lowers its apparent null MSE relative to untouched assessment. The
last two rows quantify the covariance approximation, not model accuracy.

## Fit accuracy

`fit_accuracy.py` measures the headline claim — the learned combination beats
the best single input score on untouched individuals — over 30 seeds per
regime of `simulate_panel`, with the best score chosen on the training split:

```bash
python benchmarks/fit_accuracy.py
```

Replicates run in worker processes (`--jobs`, with BLAS threads pinned to one
per worker) to keep the grid within minutes; output rows are in grid order.
The oracle sits below `h2` because the evaluated predictions exclude
covariates while the simulated phenotype includes them.

**Table 2. Mean held-out R² over 30 seeds per regime (`n_causal=8`).**

| `n` | `K` | `h2` | best single | multi-PGS | oracle | uplift |
|---|---|---|---|---|---|---|
| 2,000 | 50 | 0.2 | 0.082 | 0.170 | 0.180 | +0.088 |
| 2,000 | 50 | 0.5 | 0.202 | 0.432 | 0.439 | +0.231 |
| 2,000 | 200 | 0.2 | 0.089 | 0.155 | 0.168 | +0.066 |
| 2,000 | 200 | 0.5 | 0.227 | 0.420 | 0.429 | +0.193 |
| 10,000 | 50 | 0.2 | 0.098 | 0.174 | 0.176 | +0.076 |
| 10,000 | 50 | 0.5 | 0.244 | 0.439 | 0.440 | +0.195 |
| 10,000 | 200 | 0.2 | 0.081 | 0.161 | 0.164 | +0.080 |
| 10,000 | 200 | 0.5 | 0.206 | 0.416 | 0.417 | +0.210 |

The combination closes essentially the whole gap between the best single
score and the genetic-value oracle in every regime, and `cv_r2` tracks the
held-out number. Support recall is 0.84–1.00 at precision 0.09–0.20: the
lasso spreads weight over correlated near-duplicates, which is the union
support CMSA is designed to produce, not a support-recovery guarantee.

## Null-gate calibration

`null_gate.py` records the nested gate's two operating characteristics over 30
seeds: how often the penalized stack passes when no score carries signal, and
how far `cv_r2` sits from untouched held-out R² when signal is present:

```bash
python benchmarks/null_gate.py
```

**Table 3. Gate behaviour at `K=50`, `h2=0.4`, 30 seeds.**

| `n` | null-model return rate (noise) | null `cv_r2` | signal `cv_r2` − held-out R² |
|---|---|---|---|
| 1,000 | 0.767 | −0.002 | −0.008 ± 0.052 |
| 5,000 | 0.867 | −0.000 | +0.002 ± 0.025 |

The one-standard-error gate is not a 5% test: it still passes a pure-noise
stack in about a fifth of seeds, so a passed gate alone is not evidence of
signal — read it together with `cv_r2`, which stays at zero on noise. Under
signal the internal estimate is unbiased to within a few thousandths,
consistent with the "possibly conservative" language in the theory notes.

## Real-LD score moments

Every other benchmark here builds its LD from a simulator, which leaves two
properties of practice unmeasured because a simulator does not produce them: a
real reference is block-heterogeneous, and a real score panel is rank-deficient
and near-collinear. `real_ld_gram.py` reports what `score_gram` and the fitter's
own moment validation see on a real reference:

```bash
python benchmarks/real_ld_gram.py --ld /path/to/ldpred3_ldref_hm3.npz \
    --scores pgs_catalog_scores/ --check-lowrank
```

The reference is supplied, not shipped. The one this was developed against is
Privé's bigsnpr HapMap3+ European (UK Biobank) LD, converted to ldpred3's block
format by `ldpred3/benchmarks/convert_bigsnpr_ldref.py`: 1,054,330 variants in
625 blocks, 406 of them dense (median 451 variants) and 219 low-rank (median
3,120 variants at median rank 890). Without `--scores` the panel is synthetic
over the real LD, which exercises representation and cost but says nothing
about real score collinearity; the two panel types are recorded separately in
the provenance and must not be compared.

Nothing here is an accuracy claim — it measures moments and what they cost.

## Real LD with known truth

`real_ld_simulation.py` draws the summary statistics rather than the people.
Given the real reference's `D` and a chosen true effect vector `beta`, a
discovery GWAS of effective size `n` has `z ~ N(D beta, D / n)`, sampled block
by block from a factor of `D` — free for the low-rank blocks ldpred3 already
stores as `U U' + diag(residual)`. Two things follow that no other benchmark
here can offer:

- **Three independent GWAS of one trait**, so fitting, tuning and assessment
  are genuinely separate and a regime A label is checkable. Real data almost
  never supplies this: successive consortium releases are nested
  meta-analyses.
- **Closed-form truth with no Monte Carlo error of its own.** A combination
  with per-variant weights `w` has `R2 = (w' D beta)^2 / (w' D w)`, because
  `cov(Xw, y) = w'D beta` and `var(Xw) = w'D w`. The estimator sees `z`; the
  truth uses `D beta`.

The reference-mismatch arm reuses the same machinery. Rather than perturbing
`D` arbitrarily, the fitting reference is a *simulated finite panel* — `n_ref`
individuals drawn from the true `D` — which reproduces the failure a real small
reference has, rank deficiency, not merely added noise.

```bash
python benchmarks/real_ld_simulation.py \
    --ld /path/to/ldpred3_ldref_hm3.npz --chrom 21 22 --seeds 20
```

Two results deserve emphasis. The regime A number tracks the truth to a few
thousandths at every reference size, so the label does what it claims. But the
accuracy a user would *report* from their own imperfect reference does not: at
`n_ref = 50` the fit itself degrades only modestly while the reported R² inflates
by more than 0.12, which nothing in the output reveals. Reference quality is
therefore far more dangerous to the reported number than to the fit.

The Gram is `K x K`, so reference size only starts to matter as `K` grows
toward `n_ref`: at `K = 8` a 500-individual reference is indistinguishable from
the true `D`.

## Sample-overlap inflation, and whether LDSC can see it

`overlap_inflation.py` answers the first item in the README's trust checklist,
which is asserted everywhere and measured nowhere. Overlap is simulated as
correlated sampling error at a known `rho_s = rho_p N_shared / sqrt(N1 N2)` —
which is also exactly the estimand of cross-trait LD Score regression's
intercept, so the knob, the truth and the detector all address one number.

```bash
python benchmarks/overlap_inflation.py \
    --ld /path/to/ldpred3_ldref_hm3.npz --chrom 18 19 20 21 22 --seeds 20
```

True accuracy is unaffected by overlap by construction; the reported accuracy
is not, and their difference is the inflation. The detector arm uses
[bipred](https://github.com/bvilhjal/bipred)'s `ldsc_rg`; without it the
inflation arm still runs. bipred reports no standard error for the intercept,
so the null is calibrated from the `overlap = 0` replicates and the detection
rate at each true overlap is a power curve.

**Read the detector arm only at a realistic variant count.** The intercept's
sampling spread falls roughly as the square root of the number of variants in
the regression, so a single chromosome understates genome-wide LDSC by close to
an order of magnitude. Every row records the variant count it used; compare
rows only at equal counts, and sweep `--chrom` rather than quoting one number.

## Summary-statistic and individual-level agreement

`sumstat_vs_individual.py` simulates three independent cohorts sharing
variants and component weights — one fits, one tunes, one assesses — and over
30 seeds records whether the moments-only route agrees with the
individual-level CMSA fit, and what each tuning regime costs:

```bash
python benchmarks/sumstat_vs_individual.py
```

**Table 4. Agreement and held-out R² on the untouched cohort, 30 seeds.**

| Quantity | Mean |
|---|---:|
| `beta_std` correlation, individual vs summary fit | 0.9996 |
| Held-out R², individual-level CMSA | 0.4981 |
| Held-out R², summary fit tuned independently | 0.4979 |
| Held-out R², summary fit with PUMAS pseudotuning | 0.4982 |
| Held-out R², summary fit tuned in-sample | 0.4973 |

With exact moments the two routes agree up to selection-routing differences
(nested CMSA versus a tuning GWAS). In this strong-signal, small-`K` regime
the tuning rule costs nothing measurable; treat that as evidence the
pseudotuning approximation is harmless here, not that it is free in general —
the plug-in error it rests on is what `sumstat_calibration.py` measures.

## Meta-PGS rules on a real same-trait panel

`meta_rules.py` above compares the weighting rules in simulation, where
everything the discovery studies share is the single knob
`shared_error_correlation` — which, as that section says, is not sample
overlap. `real_meta_rules.py` replaces the knob with the measurement. For a
directory of PGS Catalog scoring files for **one** trait, the shared structure
is the off-diagonal of `W' D W` on a real LD reference, and that is exactly the
matrix `method="decorrelated"` inverts:

```bash
python benchmarks/real_meta_rules.py \
    --ld /path/to/ldpred3_ldref_hm3.npz \
    --scores pgs_catalog_height/ \
    --metadata pgs_catalog_height_metadata.tsv \
    --gwas /path/to/GIANT_HEIGHT_2014.txt.gz --gwas-format giant_height \
    --regime A --h2 0.5 --m-causal 12000 \
    --gwas-cohorts ARIC,FHS,EGCUT,ERF,HealthABC,InCHIANTI,B58C,ALSPAC
```

It writes one row per weighting rule, one per score, one per score *pair*, a
panel-level summary and provenance under `benchmarks/results/`. The pair file
is the point: it puts each pair's observed score correlation beside the Jaccard
index of the two scores' declared discovery cohorts from
`multipgs.cohort_overlap`, so whether the Catalog's metadata anticipates the
correlation the LD reference shows can be read directly instead of assumed.
A pair whose correlation the reference cannot support — either score matching
no polymorphic reference variant — is written `nan` with
`correlation_measured = False` and left out of every distribution statistic,
rather than recorded as a correlation of zero; `n_pairs_correlation_measured`
says how many of the `K(K-1)/2` survived. The same applies per score: a run
reports `n_variants_aligned` and `gwas_weight_coverage` for each, warns when
either is thin, and warns when the largest `n_eff` is a tie, because
`best_single_max_n_eff` is then resolved by filename order and is the yardstick
the rest are read against.

Nothing is downloaded during a run. The scoring files, their metadata table
(`multipgs.write_score_metadata`), the LD reference and the target GWAS are all
supplied, and the sha256 of the reference and the GWAS goes into the
provenance. No results are committed, because every number depends on inputs
this repository does not ship. `--chrom` and `--max-scores` exist for smoke
tests; a one-chromosome run estimates the same correlations from a twentieth of
the variants and its accuracies are not comparable with a genome-wide one.

Four things this deliberately does not claim. It is **one draw** — one panel,
one trait, one reference, one target GWAS — so it cannot rank the rules the way
30 seeds can; read differences against the gap between the two single-score
baselines. The **regime is declared, never inferred**: meta-PGS weights are
untuned, so an untouched target GWAS gives a real regime A number, but that
precondition is about *people*, and `--gwas-cohorts` makes the benchmark report
how many panel scores name a cohort behind the target GWAS. A non-zero count
under `--regime A` is recorded as `regime_a_contested` rather than quietly
accepted, and because declared cohorts are only a lower bound, a zero count is
weak evidence — check the target's cohort list against each score's publication
before reporting an A. Omitting `--gwas-cohorts` under `--regime A` skips that
check rather than passing it, and `regime_a_cohort_check` distinguishes
`not_checked` from `no_declared_overlap` so the two can never be read as the
same thing. Third, the regime labels the **target GWAS and not the LD
reference**: `decorrelated` inverts the correlation implied by the same Gram
matrix that forms every R² denominator, and every rule standardizes by its
diagonal, so no rule here is independent of the one reference — testing that
needs a second reference, not a second GWAS. Finally, `--h2` and `--m-causal`
are **declared assumptions**: Catalog scores carry no fitted architecture, so
without them the two `expected_r2` rules do not run at all, and with them those
two rows are conditional on the two numbers being right for the trait.

## Real LD, simulated phenotype, individual-level ground truth

`fit_accuracy.py` above measures the headline claim on `simulate_panel`, whose
phenotype is an exact linear function of the observed score columns
(`g = Z @ beta_true`). The panel spans the genetic value by construction, so
the stack cannot help but win. `real_ld_gram.py` uses a real reference but
estimates no accuracy. `real_ld_holdout.py` closes the gap between them: real
genotypes, real LD, real P+T component scores from a real marginal GWAS, a
genetic value that lies *outside* the panel's span, and an untouched cohort of
individuals to measure against.

Only the phenotype is simulated, because knowing the genetic value exactly is
the whole point. Individuals are partitioned into **four disjoint cohorts** —
one builds the component scores, one supplies the target trait's own GWAS `z`,
one is the LD reference, one is untouched — and the run reports the split
sizes. A target trait and `K-1` genetically correlated auxiliary traits are
simulated over the real genotypes with a controllable `--rg` and a controllable
per-trait discovery sample size, which is the setting a multi-PGS exists for:
auxiliary traits whose scores carry structure the target's own underpowered
GWAS cannot resolve.

```bash
python benchmarks/real_ld_holdout.py \
    --bfile /path/to/1000G_phase3_common_norel \
    --chrom 20 21 22 --seeds 30 --traits 8 \
    --rg 0.8 0.6 0.6 0.4 0.4 0.2 0.2
```

Nothing is downloaded during a run: the PLINK genotypes are supplied and the
sha256 of every input file goes into the provenance. `--chrom`,
`--max-variants` and `--seeds` exist for smoke tests, and the committed CSV is
one — two seeds over 1,657 chromosome 22 variants, which proves the pipeline
runs and measures nothing. Re-run at full scale before quoting any number. The
run prints and records its full coverage accounting — how many variants the
requested chromosomes hold, how many the window kept, and how many each QC
criterion dropped — so a truncated smoke window can never be mistaken for whole
chromosomes; `scores_variant_capped` does the same for component scores
truncated by `--max-score-variants`. Both tuning arms are deterministic given
`--seeds`, the PUMAS pseudo-split included, so the recorded command reproduces
the CSV exactly.

Every column is labelled with its evidence class. `plugin_pseudo_r2` is what
the fit itself reported: **regime B** under `--tune pumas`, **regime C** under
`--tune none`, and `plugin_optimism` is that number minus the held-out one, so
the bias is visible rather than asserted. `sumstat_r2_regime_a` is
`evaluate_sumstat` with `regime="A"` passed explicitly and justified by
disjointness — its moments come from the test cohort, which supplied no
genotypes to the reference, no phenotypes to any GWAS and no individuals to
score construction or tuning. The held-out R² columns — `multi_r2_gv`,
`multi_r2_pheno`, `best_single_r2_gv`, `best_single_r2_pheno`, `uplift_gv` and
`multi_incremental_r2_gv` — are individual-level assessment on untouched
people, which is stronger than regime A and outside the A/B/C vocabulary
altogether; the single score they are compared against is chosen by summary
pseudo-R² on the fit's own moments, never by its held-out performance. Two
columns are **not** held out and are not assessment of anything:
`panel_ceiling_r2_gv` is a least-squares fit of the genetic value on the panel
*in the test cohort itself*, an in-sample upper bound that exists to give
`uplift_gv` a scale, and `oracle_r2_pheno` is `r2(y, g)`, the `h2` ceiling.
Neither is a competitor to `multi_r2_gv` and neither may be quoted as an
accuracy. PUMAS's independence contract holds by construction: every component
score is built in the score cohort, and `z` comes from a disjoint one, which is
why the two cohorts exist.

The smoke run already shows one mechanism worth naming, because it is a
property of the design rather than of the sample size. `gram_variance_ratio` is
`(a' G a) / var(score)` measured in the same cohort — the only place a regime A
summary number can disagree with the individual-level number, since the
numerator `a'c` is an exact covariance and only the denominator passes through
the LD reference.

**Table 5. Smoke scale (2–3 seeds, 1,657 variants). Not a result — a
diagnostic.**

| configuration | `gram_variance_ratio` | multi-PGS vs best single, R²(g) |
|---|---:|---|
| all super-populations, `--ld-block-size 500` | 0.33–0.45 | 0.34 vs 0.40 |
| all super-populations, one dense block | 0.99 | 0.40 vs 0.40 |
| `--super-pop EUR`, `--ld-block-size 500` | 0.85–1.29 | 0.42 vs 0.32 |

Block-diagonal LD asserts that variants in different blocks are uncorrelated.
That is very nearly true within one population and false in a pooled one, where
allele-frequency differences correlate every pair of variants genome-wide. The
pooled reference therefore misses real score variance, every plug-in R² built
on it is inflated, and the same misspecified `G` is what the fit itself
optimizes against. Removing the block truncation alone recovers the ratio; so
does using one population.

Which is the honest reading of the whole benchmark: 1000 Genomes has 2,490
unrelated individuals, so split four ways each cohort is a few hundred people
and every absolute R² is what the estimator achieves with a discovery GWAS and
an LD reference three orders of magnitude smaller than the ones this method is
for. Pooled, the sample is also ancestrally heterogeneous — real LD, but not
the LD of any one population — which is why `multi_incremental_r2_gv` reports
R² incremental to genotype principal components and `gram_variance_ratio`
reports the reference's fidelity. `--super-pop EUR` fixes the LD structure and
leaves cohorts of about a hundred. The two configurations differ in cohort size
and reference quality at once and must not be compared with each other. This
measures the estimator under real LD; it is not a biobank-scale accuracy claim
and nothing here should be read as one.

## Summary-statistic cost at real dimensions

`stack_scaling.py` above times the individual-level fit and warns that its
defaults are too small for a deployment claim. Nobody had supplied
representative dimensions, and the summary-statistic route — the one that is
supposed to scale to a catalog-sized panel over a genome-wide reference — had
no cost benchmark at all. `sumstat_cost.py` is that benchmark. It holds a real
LD reference fixed, sweeps `K` and per-score variant support, and runs each
case in a fresh subprocess with the kernels warmed and BLAS pinned to one
thread, reusing `stack_scaling.py`'s process accounting:

```bash
python benchmarks/sumstat_cost.py \
    --ld /path/to/ldpred3_ldref_hm3.npz \
    --cases 100x5000 300x5000 900x5000 300x200000 3000x30 3000x5
```

That is the full-scale run: the whole 1,054,330-variant reference, about twenty
minutes and up to roughly 5 GB resident. The reference is supplied, not
shipped, and its sha256 goes in the provenance. With `--scores DIR` each case
aligns its first `K` real PGS Catalog scoring files instead, which is the only
configuration in which the `panel` stage measures anything about alignment.

Wall time and absolute peak RSS are reported per stage — panel, sparse parse,
`score_gram`, cross-moment, `_validate_moments`, and the path fit — because the
stages scale differently and the crossover is the useful result. The Gram grows
with the scores active per LD block and stops growing once every score touches
every large block, so beyond a few thousand variants of support it is nearly
indifferent to support; validation is an `O(K^3)` eigendecomposition and
indifferent to support entirely. The `dominant_stage` column says which one won,
so a user can predict their own run rather than extrapolate from a toy.

The memory model is explicit and its constant is measured, not assumed. The
sparse `(index, weight)` parse materializes an `int64` index, an `int64` score
column and a `float64` value per non-zero entry — 24 bytes, and
`bytes_per_nonzero` comes back at exactly 24.0. A dense `float32` `(m, K)`
matrix costs 4 bytes per cell, so sparse is cheaper only below `4/24 = 1/6`
density: about 175,700 variants of support per score on this reference. A
900-score panel of *sparse* catalog scores at 5,000 variants each is 108 MB in
that form and nobody notices. A 900-score panel of genome-wide dense HapMap3
scores sits at density 1, where the same form is 21.2 GiB against 3.5 GiB
dense — and the pipeline parses `weights_ld` and `weights_gwas` separately, so
it holds two copies at its peak: 42 GiB, which is the real constraint on
running a catalog-scale dense panel. The benchmark materializes both
representations in a separate LD-free worker and checks the arithmetic. Note
what it also shows: `_weight_columns` given a dense matrix immediately calls
`np.nonzero` and builds the very same COO arrays, so today a dense panel handed
to multipgs costs strictly more than a sparse one. The crossover says which
representation a future dense Gram path ought to consume; it is not yet an
available saving.

**What this cannot establish.** It measures cost and nothing else. There is no
accuracy number anywhere in its output — for accuracy see `fit_accuracy.py`,
`sumstat_vs_individual.py` and `sumstat_calibration.py`. The timed fit uses
`tune="pumas"`, regime B, or `tune="none"`, regime C, but that label selects
which code path is being timed and is not the provenance of an accuracy
estimate, because there is no accuracy estimate. `tune="independent"` is not
timed: from the source it adds one further parse and Gram pass over the tuning
weights, so its extra cost is the measured `parse` and `gram` columns again.
Without `--scores` the panel is synthetic over the real LD, which reproduces
the reference's block structure and the panel's shape but not real score
collinearity; coordinate descent converges faster on near-orthogonal scores
than on the near-duplicate catalog scores multi-PGS actually combines, so a
synthetic `path_residual_seconds` is a lower bound, and synthetic and catalog
rows must never be compared. `z` is synthetic in every configuration, rescaled
so the strongest single score has a fixed plug-in marginal correlation, purely
so the path solve does representative work; `n_selected` is reported only to
show the path was not trivially empty. `path_residual_seconds` is the one stage
not measured directly — total fitter time minus the stages it repeats — so it
also carries two further `O(K^3)` factorizations and can go slightly negative
on a reference small enough to fit in cache, which means that case is too small
for the residual to be informative. Peak RSS is the worker's absolute
high-water mark including the Python, NumPy and Numba runtime and the loaded
reference, and the per-stage columns are cumulative high-water marks; compare
only rows from the same command and environment. Each case runs once, so
adjacent rows differing by a few percent differ by noise.

The committed CSVs are a `--max-blocks 40` smoke run on a truncated reference,
recorded as truncated in the provenance. They prove the script runs. They are
not deployment numbers — produce those with the full-scale command above, in
the environment being reported.

## LD reference quality and `ld_shrinkage`

`multi_pgs_sumstats` carries a second tuning knob nothing else here measures.
`ld_shrinkage` (`delta`) is lassosum's separate LD-stabilisation term,
`G_delta = G + delta * P`; it defaults to `[0.0]` and the fitter only warns when
the selected value lands on the edge of a supplied grid. That default decides
whether the summary-statistic route is usable without a biobank-scale LD
reference, which is the situation most people are in.

`ld_reference_shrinkage.py` holds the score panel, the target GWAS, the delta
grid and every fitting option fixed and varies exactly one thing — the reference
behind `G`. Reference 1 is the supplied biobank-scale artifact. Reference 2 is
computed here from the 1000 Genomes EUR individuals with
`ldpred3.compute_ld_blocks`, restricted to the variants both references share
and built on the *same block boundaries*, so the two Grams are the same
quadratic form under two estimates of `D`. `--subsample-sizes` adds smaller
random subsamples of those same individuals, which isolates sample size from
population composition:

```bash
python benchmarks/ld_reference_shrinkage.py \
    --ld /path/to/ldpred3_ldref_hm3.npz --ld-n-ref 362320 \
    --bfile /path/to/1000G_phase3_common_norel \
    --gwas /path/to/GIANT_BMI_2015.txt.gz \
    --panel-gwas /path/to/jointGwasMc_LDL.txt.gz \
                 /path/to/jointGwasMc_HDL.txt.gz \
                 /path/to/jointGwasMc_TG.txt.gz \
                 /path/to/cad.add.160614.website.txt \
    --subsample-sizes 250 100
```

Every large input is supplied by flag and never downloaded, and each one's
sha256 goes in the provenance. Restricting the references to the panel's union
support is exact rather than an approximation — a variant carrying no weight in
any component score has a zero row in `W` — and is done only to keep the dense
1000 Genomes blocks in memory; `--chrom` splits the genome further when even
that does not fit. `--scores DIR` swaps the p-value-thresholded panel for real
PGS Catalog scoring files.

Per reference it records the score Gram's condition number and minimum
eigenvalue on correlation coordinates, its numerical rank, the fraction of `c`
the fit discards as unidentifiable, the selected `delta`, the number of scores
selected and the selection MSE, plus the fitter's full per-delta audit. The
audit is what answers the transfer question, and it answers it the only honest
way: a `delta` chosen on one reference is scored against the *borrowing*
reference's own criterion, never against the lender's.

Everything here is **regime B** with the default `--tune pumas` (PUMAS
pseudo-validation chose the model) or **regime C** with `--tune none` (fitted
and selected on one unsplit GWAS). No **regime A** number is produced and none
can be: `evaluate_sumstat` is deliberately never called, because that needs a
third untouched GWAS of the target trait and this benchmark has none. Nothing
here is an accuracy claim. Selection MSE is a plug-in moment against each
reference's own Gram in that reference's own coordinates, so it is comparable
across the delta grid *within* a reference and never *between* references. The
one column whose name invites the mistake is `plugin_pseudo_r2`, the fitter's
descriptive R² at the selected point: it is computed against the same moments
the coefficients were fitted on, so it is optimistic by construction, carries
the row's own regime, and is not a held-out R². The module docstring lists the
rest of what this cannot establish, including the PUMAS independence
assumption, which is declared and unverifiable.

Where coverage is lost it is counted rather than absorbed: the provenance's
`coverage` block records how many reference variants failed to intersect the
genotypes and why, what fraction of the support the target GWAS actually
reported (the rest carry `z = 0`), every row each GWAS reader dropped and for
what reason, and any `--subsample-sizes` value too large or too small to draw,
which is otherwise the difference between an `n_ref` trend and a two-point
anecdote. The partition claim is checked rather than asserted — the blocks
`compute_ld_blocks` returns must reproduce reference 1's restricted boundaries
exactly, and the run aborts if they do not, since a differing partition would
confound the one thing being varied.

`delta` only has something to repair once `K` approaches the reference's
effective rank in score space, so a small panel can legitimately select
`delta = 0` on every reference; read the selected value together with
`gram_rank` and `correlation_condition_number` rather than on its own.

The committed CSVs are a `--chrom 22` smoke run, recorded as such in the
provenance. They prove the script runs end to end and writes its three tables.
They are not a genome-wide result, and the direction of any effect in them is
not evidence of anything — produce that with the full-scale command above.
