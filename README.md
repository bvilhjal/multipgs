# multipgs

**multipgs** combines many polygenic scores into one. Many traits have smaller
discovery GWAS than genetically correlated phenotypes. Multi-PGS can borrow
from informative auxiliary scores, with the largest potential gains when the
focal score is underpowered and the fitted combination generalizes.

The package provides the two combiners, the machinery to build the score matrix
they consume, and the step that folds a fitted combination back into a single
per-variant weight file you can deploy. It is built on
[ldpred3](https://github.com/bvilhjal/ldpred3) for genotype I/O, allele
harmonisation, and the LDpred model behind score construction.

> This repository was called `pypcma` until 2026-08-10, when it held a Python 2
> implementation of principal-component meta-analysis. That code is gone (see
> the [changelog](CHANGELOG.md); git history retains it at `5bba37b`), and
> GitHub redirects the old URL.

## Two combiners

**Table 1. The two combination problems.**

| | `multi_pgs_fit` | `meta_pgs` |
|---|---|---|
| **Needs** | a training cohort with phenotypes | nothing but the scores and their GWAS sizes |
| **Input scores** | any traits, mostly irrelevant is fine | the **same** trait, different discovery GWAS |
| **Weights** | learned by penalized regression | derived from effective sample size or fitted accuracy |
| **Method** | Cross-Model Selection and Averaging | accuracy-derived weights, optionally decorrelated |

`multi_pgs_fit` is the estimator of Albiñana et al.,
[*Multi-PGS enhances polygenic prediction by combining 937 polygenic
scores*](https://doi.org/10.1038/s41467-023-40330-w) (Nat Commun 14, 4702,
2023). Its penalty is selected by CMSA, which is this package's choice — the
paper used `cv.glmnet` with fivefold cross-validation. The `sqrt_n_eff`
training-free rule is in [`code/meta_prs.R` of the accompanying
PGS-pipeline](https://github.com/olex2148/PGS-pipeline), while Hansen et al.,
[*Mapping Genetic Architecture of Thousands of Complex Traits Using GWAS
Summary Statistics*](https://doi.org/10.21203/rs.3.rs-9415305/v1) (Research
Square, 2026), supplies the model-level architecture-screening criteria. The paper itself
does not document the meta-PGS rule.

## Install

Python 3.9–3.14. Numba is strongly recommended. ldpred3 is not on PyPI, so its
Git install needs authenticated GitHub read access:

```bash
python -m pip install "ldpred3[fast] @ git+https://github.com/bvilhjal/ldpred3.git@dcde5737f720642105c0e1c79878219304fa3012"
python -m pip install "multipgs[fast] @ git+https://github.com/bvilhjal/multipgs.git"
```

For development against sibling checkouts:

```bash
python -m pip install -e "../ldpred3[fast]"
python -m pip install -e ".[fast,test]"
```

If the second command reports that it cannot satisfy `ldpred3>=0.4.5`, the
editable ldpred3 install has stale recorded metadata — a checkout that moved
past the version pip last saw. Re-running the first command refreshes it. pip
resolves against that metadata, not against the code it will actually import.

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
print(fit.selected(top=10))          # which traits carried the signal
```

Then measure it in individuals who trained neither the scores nor the
combination:

```python
test = panel_from_catalog("pgs_catalog_scores/", "test_cohort")
print(evaluate(y_test, fit.multi_pgs(test), covar=covar_test))
```

And collapse it to one weight file, which is what you deploy:

```python
from multipgs import combine_weights
from ldpred3 import score_from_weights

combine_weights(panel, fit, path="multi.weights")
scores = score_from_weights("multi.weights", "another_cohort", scaling="frozen")
```

## Training-free combination

When several GWAS exist for *one* trait and there is no cohort to train on:

```python
from multipgs import meta_pgs, daetwyler_r2

accuracy = daetwyler_r2(h2, p, n_eff, n_variants)     # from LDpred3 fits
combined = meta_pgs(panel, expected_r2=accuracy, method="decorrelated")
prs = combined.multi_pgs(panel)
```

`method="sqrt_n_eff"` needs only the discovery sample sizes.
`method="decorrelated"` additionally discounts scores for information they share
— for example when discovery studies reuse cohorts — but wants
`expected_r2` rather than `n_eff`; the reason, with measurements, is in
[`docs/algorithm.md`](docs/algorithm.md#choosing-a-meta-pgs-rule).

## Command line

```bash
multipgs panel --catalog scores/ --plink train --out scores.tsv
multipgs fit --scores scores.tsv --pheno pheno.tsv --covar covar.tsv --out fit.tsv
multipgs evaluate --scores test_scores.tsv --pheno test_pheno.tsv --family binomial
```

Every command matches individuals on `FID:IID` rather than assuming row order.

## Before trusting a number

1. **Exclude sample overlap by construction.** If your cohort contributed to the
   GWAS behind an input score, that score is partly fitted to your data, the
   combination will reward it, and every accuracy here is inflated. Many PGS
   Catalog scores are UK Biobank-derived — check each score's development
   samples in its Catalog metadata. No cross-validation inside the target
   cohort can detect this, because every fold shares the contamination.
2. Put scoring files, LD reference, and target genotypes on the same genome
   build, and check the matched-variant and weight-mass counts in
   `panel.summary()`.
3. Remove related individuals before fitting — folds are random, and relatives
   split across them inflate `cv_r2`.
4. Report accuracy in individuals who trained neither the input scores nor the
   combination. `fit.cv_r2` nests grid construction, tuning, imputation and
   fitting inside outer folds, but it is still an internal estimate and is
   blind to point 1.
5. With covariates, report `incremental_r2` — an R² that includes age and sex is
   not a polygenic-score accuracy.
6. For case/control traits, convert to the liability scale with a stated
   prevalence, or the number is not comparable to anyone else's.
7. Match ancestry between discovery and target as closely as you can, and put
   principal components in the covariates. Nothing here models ancestry.

## Documentation

- [User guide](docs/guide.md) — inputs, workflow, options, and the ways this
  goes wrong
- [Theory](docs/theory.md) — why combining scores works, what each combiner
  estimates, and what the resulting numbers mean, derived
- [Algorithm notes](docs/algorithm.md) — the solver, the costs, and the
  measurements behind the defaults
- [Benchmark harness](benchmarks/README.md) — commands, per-seed results, and
  machine-readable provenance
- [References](docs/references.md) — 91 annotated references, each checked
  against the published record
- [Python API map](docs/api.md)
- [Changelog](CHANGELOG.md)

## Citation

Cite the method your analysis actually used: Albiñana et al. 2023 for the
learned combination, Hansen et al. 2026 for the screening criteria,
`PGS-pipeline/code/meta_prs.R` for the `sqrt_n_eff` rule, Privé, Aschard & Blum
2019 for CMSA, and
[ldpred3](https://github.com/bvilhjal/ldpred3) for score construction. There is
no multipgs paper; do not invent one.

## License

[MIT](LICENSE) © Bjarni Vilhjálmsson
