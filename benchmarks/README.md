# Benchmarks

Benchmarks are executable evidence, not decorative numbers. Run them from the
repository root with the environment being reported.

Peak-RSS benchmarks use `resource.getrusage` on POSIX and
`GetProcessMemoryInfo` on Windows. Absolute RSS and timings include different
runtime and operating-system overheads, so compare them only within a
documented platform and run; provenance records that platform.

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

### Reading the committed artifacts

Each replicated benchmark commits raw rows, a derived summary, and JSON
provenance. New runs record the repository commit, whether the relevant source
tree was dirty, the SHA-256 of the producer script, and one stable digest over
all package and benchmark Python sources plus `pyproject.toml`. The latter
identifies the exact code even for an intentional pre-commit validation run.
Historical artifacts without that identity remain labelled as such rather
than being cosmetically rewritten.

The 0.3.3 release reruns `fit_accuracy`, `meta_rules`, `null_gate`,
`stack_scaling`, `sumstat_calibration`, `sumstat_vs_individual`, the bounded
`sumstat_cost` smoke run, and `real_ld_gram` under ldpred3 0.4.7. Their source
digests bind them to the code that produced them. The three external-data
artifacts described below remain historical because their private inputs were
not available for a tractable current-version rerun.

Run `python benchmarks/check_results.py` to recompute committed summary means
from raw rows and verify the headline README values and provenance structure.

Three historical real-data runs (`overlap_inflation`, `real_ld_simulation`, and
`real_meta_rules`) report `ldpred3=0.3.1`, below multipgs's current supported
range `ldpred3>=0.5.0.dev1,<0.6`. That may be the code actually imported or stale editable
installation metadata; the old provenance does not contain enough information
to distinguish them. Those rows remain useful as historical scientific
diagnostics, but are not evidence that the current release was validated
against its declared dependency range. Re-run them before making a release or
compatibility claim.

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
| Gram identity, maximum absolute error | 7.77e-15 |
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
accuracy a user would *report* from their own imperfect reference does not. In
the committed `K=120` run, reducing `n_ref` from infinity to 50 lowers true
multi-PGS R² from 0.18636 to 0.16842, while evaluation against that same small
reference reports 0.21288: inflation of 0.04446 R², or 26.4% relative to truth.
Reference quality is therefore more dangerous to the reported number than to
the fitted predictor in this design. This five-point sweep does not establish a
general reference-size threshold; that threshold depends on `K`, score
collinearity, and the LD spectrum.

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
| `beta_std` correlation, individual vs summary fit | 0.9999 |
| Held-out R², individual-level CMSA | 0.4981 |
| Held-out R², summary fit tuned independently | 0.4981 |
| Held-out R², summary fit with PUMAS pseudotuning | 0.4983 |
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
provenance. One result set is committed as a historical regression artifact,
but its external inputs are not shipped. Its LD reference and GWAS are named
and hashed; its score directory and metadata path are named, but the scoring
files were not individually hashed, so the panel cannot be reconstructed
byte-for-byte from the committed artifact alone. Future runs also hash the
metadata and every candidate scoring file. `--chrom` and `--max-scores` exist
for smoke tests; a one-chromosome run estimates the same correlations from a
twentieth of the variants and its accuracies are not comparable with a
genome-wide one.

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

**Table 5. Checked rule rows for the committed 24-score CAD run.** All rows use
the same target GWAS and LD reference. The run is declared regime C; the oracle
single additionally selects on those evaluation moments.

| Rule | R² |
|---|---:|
| best single by maximum `n_eff` | 0.08616 |
| equal weight | 0.17536 |
| `sqrt_n_eff` | 0.16416 |
| `expected_r2` | 0.17729 |
| decorrelated `n_eff` | 0.00017 |
| decorrelated `expected_r2` | 0.00226 |
| best single selected on evaluation moments | 0.45670 |

The checked artifact therefore shows `decorrelated_expected_r2` performing
78.4 times worse than the same expected accuracies without decorrelation. This
is one overlap-contaminated regime-C draw, not a population estimate or a
general rule ranking. It does show why a positive, sample-size-derived accuracy
proxy must not be treated as a known per-score correlation when inverting a
near-singular score covariance.

### Estimating the architecture instead of declaring it

`--h2` and `--m-causal` set every score's Daetwyler `expected_r2`, so a declared
architecture is a declared ranking. `--h2-ldsc` estimates the heritability from
the target GWAS by LD Score regression; `--h2-auto` additionally runs
LDpred3-auto's multi-chain sampler, which is the more accurate of the two where
there is signal and, crucially, is the only one that identifies **polygenicity**
— LDSC's slope is `N h2 ell / M` whatever the causal fraction, so no LDSC run
can ever replace `--m-causal`. Auto is preferred unless its `h2` falls below
`--h2-near-zero`, where a sampler with nothing to condition on degrades and the
regression does not.

The historical committed run printed its LDSC and LDpred3-auto estimates and
intervals to the console but did not persist them in CSV or JSON. Consequently,
the precise architecture values from that run cannot be independently checked
from this repository and should not be quoted as artifact-backed results.
Future runs write the declared, estimated, and actually used architecture to
both `real_meta_rules_summary.csv` and the provenance JSON, including the auto
intervals, chain counts, predictive R², and its Daetwyler upper reference.

The interpretation remains structural: in a same-trait panel every score
shares one architecture, so `h2` and `p` only warp the common
`n_eff`-to-accuracy mapping and normalized weights may move little. They can
matter much more in a cross-trait panel, where each score has its own
architecture. LDSC and auto estimates are model-dependent diagnostics, not
ground-truth trait heritability.

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

That command is the intended full-scale design over the whole
1,054,330-variant reference; no result from it is committed. The reference is
supplied, not shipped, and its sha256 goes in the provenance. With
`--scores DIR` each case aligns its first `K` real PGS Catalog scoring files
instead, which is the only configuration in which the `panel` stage measures
anything about alignment.

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
900-score panel of sparse catalog scores at 5,000 variants each is 108 MB in
that form. A 900-score genome-wide dense HapMap3 panel is 3.5 GiB instead of
21.2 GiB as COO. The current dense path preserves the caller's floating-point
matrix, validates it in bounded row chunks and streams dense block slices
through the Gram calculation; parsing and the Gram do not enumerate non-zero
coordinates or build COO copies. The reproducibility digest does enumerate
non-zeros in bounded chunks, never as genome-wide coordinate arrays. The
LD-free worker checks both the storage arithmetic and that the parser shares
the dense input. The committed 0.3.3 smoke CSV predates this dense path, so
rerun the command above for current memory evidence.

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
