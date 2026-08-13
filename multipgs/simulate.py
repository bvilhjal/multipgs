"""Synthetic data for tests, examples and quick sanity checks.

Nothing here models real genetic architecture. :func:`simulate_panel` builds a
score matrix with the *structure* that makes multi-PGS interesting — many
correlated scores, a handful of which carry the signal — which is what the
estimator has to cope with. :func:`simulate_target` writes a small PLINK
fileset and matching PGS Catalog scoring files so the I/O path can be exercised
end to end without a biobank.
"""

from __future__ import annotations

import gzip
import os
from dataclasses import dataclass

import numpy as np


__all__ = ["simulate_panel", "SimPanel", "simulate_same_trait_panel",
           "simulate_target"]


@dataclass
class SimPanel:
    """A simulated training problem with a known answer."""

    scores: np.ndarray
    y: np.ndarray
    covar: np.ndarray
    beta_true: np.ndarray
    genetic_value: np.ndarray
    score_ids: np.ndarray
    n_eff: np.ndarray


def simulate_panel(*, n=2000, n_scores=50, n_causal=5, h2=0.4, n_factors=4,
                   n_covar=2, family="gaussian", prevalence=0.1, seed=None):
    """A score matrix whose phenotype depends on a few of its columns.

    Scores share ``n_factors`` latent factors, so they are correlated the way
    real trait scores are, and each is put on its own arbitrary scale and
    offset — a combiner that quietly assumes standardized inputs will fail
    here, which is the point.

    ``h2`` is the fraction of the *conditional/residual* phenotype variance
    (of the residual liability for ``family="binomial"``) explained by the
    ``n_causal`` scores that matter. Random covariate effects are added after
    that unit-variance residual is formed, so their variance is not included
    in the denominator.
    """
    rng = np.random.default_rng(seed)
    n, n_scores, n_causal = int(n), int(n_scores), int(n_causal)
    if not 0 < n_causal <= n_scores:
        raise ValueError("need 0 < n_causal <= n_scores")
    if not 0.0 < h2 < 1.0:
        raise ValueError("h2 must be in (0, 1)")

    loadings = rng.normal(size=(n, int(n_factors)))
    mixing = rng.normal(size=(int(n_factors), n_scores)) * 0.6
    S = loadings @ mixing + rng.normal(size=(n, n_scores))
    S = S * rng.uniform(0.5, 3.0, size=n_scores) + rng.normal(size=n_scores) * 5

    beta_true = np.zeros(n_scores)
    beta_true[:n_causal] = rng.normal(size=n_causal)
    Z = (S - S.mean(axis=0)) / S.std(axis=0)
    g = Z @ beta_true
    g_scale = float(g.std())
    if not np.isfinite(g_scale) or g_scale <= 1e-12:
        raise ValueError("simulation produced a constant genetic value; "
                         "increase n")
    # Keep the reported coefficients and genetic value on the same unit-
    # variance scale: Z @ beta_true is exactly genetic_value.
    beta_true /= g_scale
    g /= g_scale

    covar = rng.normal(size=(n, int(n_covar))) if n_covar else np.zeros((n, 0))
    covar_effect = covar @ (rng.normal(size=covar.shape[1]) * 0.3) \
        if covar.shape[1] else np.zeros(n)

    liab = np.sqrt(h2) * g + np.sqrt(1.0 - h2) * rng.normal(size=n)
    if family == "gaussian":
        y = liab + covar_effect
    elif family == "binomial":
        if not 0.0 < prevalence < 1.0:
            raise ValueError("prevalence must be in (0, 1)")
        full = liab + covar_effect
        y = (full > np.quantile(full, 1.0 - prevalence)).astype(float)
    else:
        raise ValueError("family must be 'gaussian' or 'binomial'")

    # Effective sample sizes that track the true importance, so meta_pgs has
    # something meaningful to weight by in tests.
    n_eff = np.full(n_scores, 5_000.0)
    n_eff[:n_causal] = np.linspace(80_000, 20_000, n_causal)

    return SimPanel(
        scores=S, y=y, covar=covar, beta_true=beta_true, genetic_value=g,
        score_ids=np.array([f"PGS{j:06d}" for j in range(n_scores)],
                           dtype=object),
        n_eff=n_eff)


def simulate_same_trait_panel(*, n=2000, n_eff=(150_000, 60_000, 20_000),
                              h2=0.4, m_causal=5_000, n_variants=1_000_000,
                              shared=0.0, family="gaussian", prevalence=0.1,
                              seed=None):
    """Several scores for **one** trait, from discovery GWAS of different sizes.

    This is the situation :func:`multipgs.meta_pgs` is for, and the one
    :func:`simulate_panel` deliberately is not: every score estimates the same
    genetic value ``g``, and score ``k``'s correlation with it is
    ``sqrt(x/(1+x))`` for ``x = n_eff[k]·h2/m_causal`` — the Daetwyler accuracy
    on the **genetic-value** scale, which is not what
    :func:`multipgs.daetwyler_r2` returns.

    ``shared`` (in ``[0, 1)``) correlates the scores' *error* terms, standing in
    for discovery cohorts that overlap — a consortium meta-analysis containing
    a cohort you are also using on its own. Raise it to give
    ``method="decorrelated"`` something that ``sqrt_n_eff`` weighting cannot
    handle.

    Returns a :class:`SimPanel` whose ``beta_true`` holds each score's true
    correlation with ``g``.
    """
    rng = np.random.default_rng(seed)
    n = int(n)
    n_eff = np.asarray(n_eff, dtype=float).ravel()
    K = n_eff.size
    if K == 0:
        raise ValueError("need at least one discovery GWAS")
    if not 0.0 <= shared < 1.0:
        raise ValueError("shared must be in [0, 1)")

    # Accuracy against the *genetic value*, cor(z_k, g) = sqrt(x/(1+x)), not
    # against the phenotype. daetwyler_r2 returns the phenotypic r², which is
    # h² x/(1+x), so taking its square root here would set every score's
    # accuracy to h·R_k — a factor h too low, and, worse, exactly the quantity
    # meta_pgs(method="expected_r2") consumes, which would hand that one rule
    # the true rho by construction and rig any comparison between rules.
    x = np.asarray(n_eff, dtype=float) * h2 / float(m_causal)
    rho = np.sqrt(np.clip(x / (1.0 + x), 0.0, 0.99))
    g = rng.normal(size=n)
    common = rng.normal(size=n)
    S = np.empty((n, K))
    for k in range(K):
        err = (np.sqrt(shared) * common
               + np.sqrt(1.0 - shared) * rng.normal(size=n))
        S[:, k] = rho[k] * g + np.sqrt(1.0 - rho[k] ** 2) * err
    S = S * rng.uniform(0.5, 3.0, size=K) + rng.normal(size=K) * 5

    liab = np.sqrt(h2) * g + np.sqrt(1.0 - h2) * rng.normal(size=n)
    if family == "gaussian":
        y = liab
    elif family == "binomial":
        y = (liab > np.quantile(liab, 1.0 - prevalence)).astype(float)
    else:
        raise ValueError("family must be 'gaussian' or 'binomial'")

    return SimPanel(
        scores=S, y=y, covar=np.zeros((n, 0)), beta_true=rho,
        genetic_value=g,
        score_ids=np.array([f"GWAS{k + 1}" for k in range(K)], dtype=object),
        n_eff=n_eff)


def simulate_target(prefix, *, n=300, n_variants=400, n_scores=5,
                    n_per_score=None, maf_range=(0.05, 0.5), missing=0.0,
                    scoring_dir=None, seed=None, gzip_scores=False):
    """Write a small PLINK fileset and matching PGS Catalog scoring files.

    Returns a dict with ``prefix``, ``scoring_files``, ``variants``,
    ``true_scores`` (the ``n x n_scores`` matrix implied by the written weights
    on raw dosages) and ``sample_iid``.
    """
    from ldpred3.genotype_io import SampleTable, VariantTable, write_plink

    rng = np.random.default_rng(seed)
    n, n_variants, n_scores = int(n), int(n_variants), int(n_scores)
    if n_per_score is None:
        n_per_score = max(5, n_variants // 4)
    n_per_score = min(int(n_per_score), n_variants)

    maf = rng.uniform(*maf_range, size=n_variants)
    dosage = rng.binomial(2, maf, size=(n, n_variants)).astype(np.int8)
    if missing > 0:
        mask = rng.random((n, n_variants)) < float(missing)
        dosage[mask] = -1

    alleles = np.array([("A", "G"), ("C", "T"), ("G", "A"), ("T", "C")],
                       dtype=object)
    pick = rng.integers(0, len(alleles), size=n_variants)
    a1 = np.array([alleles[i][0] for i in pick], dtype=object)
    a2 = np.array([alleles[i][1] for i in pick], dtype=object)
    ids = np.array([f"rs{100000 + j}" for j in range(n_variants)], dtype=object)
    chrom = np.array([str(1 + (j % 2)) for j in range(n_variants)],
                     dtype=object)
    pos = np.array([1000 + 50 * j for j in range(n_variants)], dtype=np.int64)

    variants = VariantTable(chrom=chrom, id=ids, cm=np.zeros(n_variants),
                            pos=pos, a1=a1, a2=a2)
    iid = np.array([f"IND{i:04d}" for i in range(n)], dtype=object)
    samples = SampleTable(fid=iid.copy(), iid=iid,
                          sex=np.zeros(n, dtype=np.int8),
                          pheno=np.full(n, -9.0))
    prefix = str(prefix)
    write_plink(prefix, dosage, variants, samples)

    scoring_dir = os.path.dirname(prefix) if scoring_dir is None \
        else str(scoring_dir)
    os.makedirs(scoring_dir, exist_ok=True)

    # Raw-dosage scores implied by the weights, with missing calls mean-imputed
    # exactly as panel_from_catalog does, so a test can compare against them.
    d = dosage.astype(float)
    miss = d < 0
    if miss.any():
        d[miss] = np.nan
        col_mean = np.nanmean(d, axis=0)
        col_mean = np.where(np.isfinite(col_mean), col_mean, 0.0)
        d = np.where(np.isnan(d), col_mean, d)

    paths, true_scores = [], np.zeros((n, n_scores))
    for k in range(n_scores):
        idx = np.sort(rng.choice(n_variants, size=n_per_score, replace=False))
        w = rng.normal(size=idx.size) * 0.1
        pgs_id = f"PGS{k + 1:06d}"
        path = os.path.join(scoring_dir, f"{pgs_id}_hmPOS_GRCh37.txt"
                            + (".gz" if gzip_scores else ""))
        _write_scoring_file(path, pgs_id, ids[idx], chrom[idx], pos[idx],
                            a1[idx], a2[idx], w, gzip_scores)
        paths.append(path)
        true_scores[:, k] = d[:, idx] @ w

    return {"prefix": prefix, "scoring_files": paths, "variants": variants,
            "true_scores": true_scores, "sample_iid": iid, "dosage": dosage}


def _write_scoring_file(path, pgs_id, ids, chrom, pos, a1, a2, weight, use_gz):
    header = [
        "###PGS CATALOG SCORING FILE - simulated by multipgs.simulate",
        "#format_version=2.0",
        f"#pgs_id={pgs_id}",
        f"#pgs_name={pgs_id}_sim",
        "#trait_reported=Simulated trait",
        "#genome_build=GRCh37",
        f"#variants_number={len(ids)}",
        "#weight_type=beta",
        "hm_rsID\thm_chr\thm_pos\teffect_allele\tother_allele\teffect_weight",
    ]
    lines = list(header)
    for j in range(len(ids)):
        lines.append(f"{ids[j]}\t{chrom[j]}\t{int(pos[j])}\t{a1[j]}\t{a2[j]}"
                     f"\t{weight[j]:.8g}")
    text = "\n".join(lines) + "\n"
    if use_gz:
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(text)
    else:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
    return path
