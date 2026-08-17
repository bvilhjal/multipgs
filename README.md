# multipgs

**multipgs** combines many polygenic scores into one. Many traits have smaller
discovery GWAS than genetically correlated phenotypes. Multi-PGS can borrow
from informative auxiliary scores, with the largest potential gains when the
focal score is underpowered and the fitted combination generalizes.

The package provides three fitting routes, the machinery to acquire and build
their input scores, and the step that folds a fitted combination back into a
single per-variant weight file you can deploy. It is built on
[ldpred3](https://github.com/bvilhjal/ldpred3) for genotype I/O, allele
harmonisation, and the LDpred model behind score construction.

(Formerly `pypcma`, a Python 2 PC meta-analysis; that code was removed on
2026-08-10 — see the [changelog](CHANGELOG.md), git history at `5bba37b`.)

## Three fitting routes

**Table 1. Choose by the information you have.**

| Method | Learns from | Appropriate scores |
|---|---|---|
| `multi_pgs_fit` | phenotypes in an individual-level training cohort | any traits; mostly irrelevant is fine |
| `multi_pgs_sumstats` | target-trait GWAS statistics plus an external LD reference | any traits; mostly irrelevant is fine |
| `meta_pgs` | effective sample size or expected target-accuracy proxy, with no target phenotype | the **same** trait from different GWAS |

`multi_pgs_fit` uses the elastic-net score-stacking model of Albiñana et al.,
[*Multi-PGS enhances polygenic prediction by combining 937 polygenic
scores*](https://doi.org/10.1038/s41467-023-40330-w) (Nat Commun 14, 4702,
2023), but it is not a reproduction of their complete procedure. This package
standardizes within training folds, selects and averages coefficients by CMSA,
and applies a separate nested heuristic fallback gate. The paper used
`cv.glmnet` and fivefold cross-validation. Its reported adjusted R²,
`(R²_full - R²_covar)/(1 - R²_covar)`, is also not this package's
`incremental_r2 = R²_full - R²_covar`. The `sqrt_n_eff`
training-free rule is in [`code/meta_prs.R` of the accompanying
PGS-pipeline](https://github.com/olex2148/PGS-pipeline), while Hansen et al.,
[*Mapping Genetic Architecture of Thousands of Complex Traits Using GWAS
Summary Statistics*](https://doi.org/10.21203/rs.3.rs-9415305/v1) (Research
Square, 2026), supplies the model-level architecture-screening criteria. The paper itself
does not document the meta-PGS rule.

This is not MIXPRS
([Xu et al. 2026](https://doi.org/10.1038/s41588-026-02637-4)). MIXPRS
combines two *same-trait* multi-population methods (JointPRS-auto and SDPRX)
with non-negative least squares after a data-fission split of one target GWAS.
Use MIXPRS when the panel is method-by-ancestry versions of one trait. Use
this package when the panel is many traits, or same-trait GWAS to be combined
by sample-size or accuracy rules (`meta_pgs`). The shared summary-statistic
quadratic and PUMAS-style split are Zhao et al.; they do not make the
estimands the same. See [theory.md](docs/theory.md#three-objects-that-get-combined).

## Install

Python 3.9–3.14. Numba is strongly recommended. These sibling packages are not
on PyPI, so Git installs need authenticated GitHub read access. Install the
coordinated LDpred3 0.5 line first; install Bipred only for the optional
genetic-correlation screen:

```bash
python -m pip install "ldpred3[fast] @ git+https://github.com/bvilhjal/ldpred3.git@master"
python -m pip install "bipred[fast] @ git+https://github.com/bvilhjal/bipred.git@main"  # optional
python -m pip install "multipgs[fast] @ git+https://github.com/bvilhjal/multipgs.git"
```

For development against sibling checkouts:

```bash
python -m pip install -e "../ldpred3[fast]"
python -m pip install -e "../bipred[fast]"   # optional r_G screen
python -m pip install -e ".[fast,test]"
python -m pip check
```

The declared contract is `ldpred3>=0.5.3.dev0,<0.6` (and, for the optional
screen, `bipred>=0.3.9.dev0,<0.4`). A resolver failure means the sibling
checkouts are genuinely from different API generations; update them together.

## Runnable example

From a checkout, with no data of your own:

```bash
python -m examples.minimal
```

It simulates a small cohort, builds a panel from PGS Catalog-format scoring
files, fits the combination, evaluates it against the best single score, and
writes the deployable weight file. Its complete source is
[`examples/minimal.py`](examples/minimal.py).

## Learned combination

```python
from multipgs import panel_from_catalog, multi_pgs_fit, evaluate

panel = panel_from_catalog("pgs_catalog_scores/", "train_cohort")
fit = multi_pgs_fit(panel.scores, y_train, covar=covariates,
                    score_ids=panel.score_ids, seed=1)
print(fit.summary())
print(fit.selected(top=10))          # largest nonzero standardized weights
```

For an independent cohort, first collapse the training-panel combination to one
weight file and score the cohort on that frozen coordinate system:

```python
from multipgs import combine_weights
from ldpred3 import score_from_weights

combine_weights(panel, fit, path="multi.weights")
test_score = score_from_weights("multi.weights", "test_cohort",
                                scaling="frozen")
print(evaluate(y_test, test_score.scores, covar=covar_test))
```

The same `multi.weights` file is the deployment artifact. Do not rebuild a
panel in the new cohort and pass it to `fit.multi_pgs`: cohort-specific score
scales can silently change the fitted coordinate system.

## Summary-statistic learned combination

`multi_pgs_sumstats` fits the same Gaussian stack without a phenotyped
cohort, from score-space moment estimates `G = W_ld.T @ D @ W_ld` and
`c = W_gwas.T @ z`. With `alpha=1`, its lasso selects whole component scores —
the score-space analogue of lassosum, not SNP-level lassosum. The contract:

- The training, tuning, and assessment target GWAS must be mutually
  independent: one fits the path, an optional second selects the penalty, and
  a third untouched GWAS or cohort is needed for an accuracy claim. With a
  single GWAS, `tune="pumas"` provides pseudotuning, not external assessment.
- Each data source's component weights are aligned with that source's
  empirical dosage SD. `align_to_reference` performs that conversion only when
  `sd=` is supplied (or the explicit HWE approximation is requested); check
  `log["standardized"]` before using its output as standardized-score weights.

The full workflow, including the alignment code, is in
[the guide](docs/guide.md#fitting-from-summary-statistics).

## Training-free combination

When several GWAS exist for *one* trait and there is no cohort to train on:

```python
from multipgs import meta_pgs, daetwyler_r2

expected_r2 = daetwyler_r2(h2, p, n_eff, n_variants)  # phenotypic R² proxy
combined = meta_pgs(panel, expected_r2=expected_r2, method="expected_r2")
prs = combined.multi_pgs(panel)
```

`method="sqrt_n_eff"` needs only the discovery sample sizes.
`method="expected_r2"` uses expected phenotypic R² proxies directly and is the
appropriate choice for `daetwyler_r2` output when its transport assumptions are
defensible. `method="decorrelated"` additionally uses the panel's score
correlations to discount shared information, but its matrix inverse amplifies
errors in the supplied target-correlation magnitudes. Reserve it for
independently credible, transportable magnitudes, with every component score
oriented consistently; pass their squares as `expected_r2`. The API cannot
encode a negative target correlation. Sample-size and Daetwyler proxies do not
meet that bar. The derivation is in
[theory.md §3](docs/theory.md#3-derived-weights-no-phenotype-required) and the
measurements in
[docs/algorithm.md](docs/algorithm.md#choosing-a-meta-pgs-rule).

## Command line

```bash
multipgs fetch --trait MONDO_0004989 --out scores/ --cohort-overlap
multipgs panel --catalog scores/ --plink train --out panel.npz
multipgs fit --panel panel.npz --pheno pheno.tsv --covar covar.tsv --out fit.tsv
multipgs combine --panel panel.npz --fit fit.tsv --out multi.weights --check --plink train
multipgs score --weights multi.weights --plink test --out test.prs
multipgs evaluate --scores test.prs --pheno test_pheno.tsv --family binomial
```

Every command matches individuals on `FID:IID` rather than assuming row order.
Summary-statistic fitting and evaluation currently use the Python API; there is
no summary-statistic fitting command.

## Before trusting a number

1. **Exclude sample overlap by construction.** If your cohort contributed to
   the GWAS behind an input score, accuracy can be inflated. Target-cohort
   resampling cannot reveal contamination shared by every fold. Named-cohort
   metadata and external diagnostics may flag overlap, but absence of a flag is
   not proof of independence. Many PGS Catalog scores are UK Biobank-derived —
   check each score's development samples.
   ([why](docs/theory.md#sample-overlap))
2. Put scoring files, LD reference, and target genotypes on the same genome
   build, and check the matched-variant and weight-mass counts in
   `panel.summary()`.
3. Remove related individuals before fitting — random folds split relatives
   across training and validation, inflating `cv_r2`.
4. Report accuracy in individuals who trained neither the input scores nor the
   combination. `fit.cv_r2` nests all selection inside outer folds, but it is
   an internal estimate, not a hypothesis test, and is blind to point 1.
5. With covariates, report `incremental_r2` — an R² that includes age and sex
   is not a polygenic-score accuracy.
6. Check direction and calibration as well as R². Squared correlation is
   unchanged by a sign reversal, and a combined score is not automatically a
   calibrated risk prediction in a new cohort.
7. For case/control traits, convert to the liability scale with a stated
   prevalence, or the number is not comparable to anyone else's.
   ([how](docs/theory.md#liability-scale))
8. Match ancestry between discovery and target as closely as you can, and put
   principal components in the covariates. Nothing here models ancestry.
   ([theory](docs/theory.md#ancestry))
9. Treat bootstrap intervals as conditional on the supplied scores and fitted
   weights. They do not propagate discovery-GWAS, LD-reference, score-building,
   or architecture-estimation uncertainty.

The [guide](docs/guide.md) expands each point.

## Documentation

- [Methods note](report/multipgs_methods.pdf) — estimand, selection-index
  theory, three information routes, and the simulation evidence that survives
  its design; the editable [LaTeX source](report/multipgs_methods.tex) is
  included too
- [User guide](docs/guide.md) — inputs, workflow, options, and the ways this
  goes wrong
- [Theory](docs/theory.md) — why combining scores works, what each fitting route
  estimates, and what the resulting numbers mean, derived
- [Algorithm notes](docs/algorithm.md) — the solver, the costs, and the
  measurements behind the defaults
- [Benchmark harness](benchmarks/README.md) — commands, per-seed results, and
  machine-readable provenance
- [References](docs/references.md) — annotated references, each checked
  against the published record
- [Python API map](docs/api.md)
- [Changelog](CHANGELOG.md)
- [Contributing](CONTRIBUTING.md) — development setup and contribution guidance

## Citation

Cite the method your analysis actually used: Albiñana et al. 2023 for the
learned combination, Hansen et al. 2026 for the screening criteria,
`PGS-pipeline/code/meta_prs.R` for the `sqrt_n_eff` rule, Privé, Aschard & Blum
2019 for CMSA, Mak et al. 2017 for the SNP-level quadratic that inspired the
score-space summary fit, Zhao et al. 2021 when its PUMAS-style pseudotuning is
used, and
[ldpred3](https://github.com/bvilhjal/ldpred3) for score construction. Cite
Xu et al. 2026 only if the analysis actually ran MIXPRS. There is
no multipgs paper; do not invent one.

## License

[MIT](LICENSE) © Bjarni Vilhjálmsson
