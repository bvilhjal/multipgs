#!/usr/bin/env python3
"""Real LD, simulated summary statistics, exact ground truth.

Two things this repository cannot currently establish, both for the same
reason, and both fixed by simulating the *summary statistics* rather than
people.

**Does the learned combination actually beat its best input score?**
``benchmarks/fit_accuracy.py`` says yes, but its phenotype is an exact linear
function of the observed score columns, so the panel spans the genetic value by
construction and the stack cannot lose. There is no linkage disequilibrium in
it at all, no discovery sample size, and no genetic correlation — the three
things that decide the answer in practice.

**Does a regime A number track the truth?**
:func:`multipgs.evaluate_sumstat` labels an accuracy regime A only when the
combination is scored against a GWAS untouched by fitting and tuning. Real data
rarely supplies three mutually independent GWAS of one trait — successive
consortium releases are nested meta-analyses — so the label has never been
checked against a known answer.

Drawing summary statistics directly from a real LD reference solves both. With
``D`` the reference's correlation matrix and ``beta`` a chosen true
standardized effect vector, a discovery GWAS of effective size ``n`` has

.. math::

    z \\sim N(D\\beta,\\; D / n),

which is sampled block by block from a factor of ``D`` — free for the low-rank
blocks ldpred3 already stores as ``U U' + diag(residual)``. Three independent
draws give a fitting, a tuning and a genuinely untouched assessment GWAS. No
genotypes, no phenotypes, and the real reference's actual LD.

The ground truth is then **closed form, with no Monte Carlo error of its own**.
A combination with per-variant weights ``w`` on the standardized-genotype scale
predicts a unit-variance trait with

.. math::

    R^2 = \\frac{(w' D \\beta)^2}{w' D w},

because ``cov(Xw, y) = w'D\\beta`` and ``var(Xw) = w'Dw``. The estimator sees
``z``; the truth uses ``D\\beta``. Their difference is exactly what is being
measured.

**The reference-mismatch arm.** The same machinery answers what
``ld_shrinkage`` is for. Rather than perturbing ``D`` arbitrarily, the fitting
reference is a *simulated finite panel*: ``n_ref`` individuals drawn from the
true ``D``, whose sample correlation is used to fit and tune while truth and
the assessment GWAS keep the true ``D``. That reproduces the failure a real
small reference actually has — rank deficiency once ``n_ref`` falls below the
block size, not merely added noise — and makes "how many reference individuals
do I need, and what shrinkage compensates" a question with a measured answer.
``--reference-n inf`` uses the true ``D`` and is the matched-reference control.

**What this cannot establish.** Nothing here is a real-data accuracy claim. The
effects are drawn from a stated prior, the component scores are built by one
fixed rule, and the trait is Gaussian with unit variance; a real trait is none
of those. The LD is real, the sampling model for ``z`` is the standard
large-sample normal approximation, and the truth is exact *given* those. Read
this as the estimator being checked against a known answer under real LD, not
as an accuracy anyone will reproduce in a cohort.

    python benchmarks/real_ld_simulation.py \\
        --ld /path/to/ldpred3_ldref_hm3.npz --chrom 21 22 --seeds 20

The reference this was developed against is Privé's bigsnpr HapMap3+ European
(UK Biobank) LD, converted by ``ldpred3/benchmarks/convert_bigsnpr_ldref.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import multipgs
from benchmarks._provenance import benchmark_identity
from multipgs import evaluate_sumstat, multi_pgs_sumstats
from multipgs.sumstat import score_gram


FIELDS = (
    "reference_n", "seed", "true_r2_multi", "true_r2_best_single",
    "true_r2_oracle", "uplift", "regime_a_r2", "regime_a_minus_true",
    "regime_a_r2_own_reference", "own_reference_minus_true",
    "plugin_r2", "plugin_minus_true", "selected_delta", "n_selected",
    "gram_rank", "gram_min_correlation_eigenvalue", "discarded_c_fraction",
    "fit_seconds",
)


def _version(name):
    try:
        value = getattr(importlib.import_module(name), "__version__", None)
        if value is not None:
            return str(value)
    except ImportError:
        pass
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _sha256(path, chunk=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path, chroms):
    """Real LD blocks for the requested chromosomes, re-tiled from zero.

    ``score_gram`` requires blocks that tile ``0..m-1`` exactly once, so a
    chromosome subset has to be renumbered rather than carry its original
    offsets. The full reference's variant count is returned alongside, because
    a chromosome subset that keeps the same ``h2`` is not a smaller version of
    the same trait — see :func:`_true_effects`.
    """
    from ldpred3 import load_ld_blocks

    blocks, _ = load_ld_blocks(str(path))
    payload = np.load(str(path), allow_pickle=True)
    chrom = np.asarray(payload["chrom"], dtype=object)
    sizes = np.asarray(payload["sizes"], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)])

    wanted = None if chroms is None else {str(c) for c in chroms}
    kept, offset = [], 0
    for i, (corr, idx) in enumerate(blocks):
        if wanted is not None and str(chrom[starts[i]]) not in wanted:
            continue
        size = int(idx.size)
        kept.append((corr, np.arange(offset, offset + size)))
        offset += size
    if not kept:
        raise SystemExit(f"no LD blocks on chromosome(s) {sorted(wanted)}")
    return kept, offset, int(starts[-1])


def _block_factor(corr):
    """``L`` with ``L L' = D`` for one block, in that block's representation.

    A low-rank block already carries its own factor, so sampling from it costs
    nothing: ``U U' + diag(residual)`` factors as ``[U, diag(sqrt(residual))]``.
    A dense block needs a decomposition, and an eigenvalue floor rather than a
    Cholesky jitter loop — a stored correlation block rounded to int8 or
    float32 need not be numerically positive definite, and clipping its
    spectrum says exactly zero in the directions it has lost rather than
    inventing variance there.
    """
    from ldpred3 import LowRankLD

    from multipgs._ldpred3_compat import dequantize_ld

    block = dequantize_ld(corr)
    if isinstance(block, LowRankLD):
        factor = np.asarray(block.U, dtype=np.float64)
        residual = np.sqrt(np.maximum(
            np.asarray(block.residual_diag, dtype=np.float64), 0.0))
        return factor, residual
    dense = np.asarray(block, dtype=np.float64)
    dense = 0.5 * (dense + dense.T)
    values, vectors = np.linalg.eigh(dense)
    keep = values > 1e-10 * max(float(values[-1]), np.finfo(float).tiny)
    return vectors[:, keep] * np.sqrt(values[keep]), None


def _blockwise(blocks, vector, op):
    """Apply a per-block operation to a genome-length vector."""
    out = np.zeros_like(vector)
    for corr, idx in blocks:
        out[idx] = op(corr, vector[idx])
    return out


def _ld_times(blocks, vector):
    """``D v`` using each block's own representation."""
    from ldpred3 import ld_matmul

    return _blockwise(blocks, vector,
                      lambda corr, part: np.asarray(
                          ld_matmul(corr, part), dtype=float))


def _draw_z(blocks, factors, beta, n_eff, rng):
    """One GWAS: ``z ~ N(D beta, D / n)``, sampled block by block."""
    mean = _ld_times(blocks, beta)
    noise = np.zeros_like(beta)
    scale = 1.0 / np.sqrt(float(n_eff))
    for (_, idx), (factor, residual) in zip(blocks, factors):
        draw = factor @ rng.standard_normal(factor.shape[1])
        if residual is not None:
            draw = draw + residual * rng.standard_normal(residual.size)
        noise[idx] = draw
    return mean + scale * noise


def _simulated_reference(blocks, factors, n_ref, rng):
    """Sample correlation of ``n_ref`` individuals drawn from the true ``D``.

    This is what a real reference panel of that size looks like, including
    being singular once ``n_ref`` drops below a block's variant count — the
    failure mode an additive perturbation of ``D`` would miss entirely.
    """
    out = []
    for (corr, idx), (factor, residual) in zip(blocks, factors):
        k = int(idx.size)
        draws = rng.standard_normal((n_ref, factor.shape[1])) @ factor.T
        if residual is not None:
            draws = draws + rng.standard_normal((n_ref, k)) * residual
        draws -= draws.mean(axis=0)
        sd = draws.std(axis=0)
        sd = np.where(sd > 0, sd, 1.0)
        draws /= sd
        sample = draws.T @ draws / n_ref
        sample = 0.5 * (sample + sample.T)
        np.fill_diagonal(sample, 1.0)
        out.append((sample, idx))
    return out


def _true_effects(m, h2, polygenicity, rng, *, m_genome=None):
    """Sparse standardized effects for a trait of heritability ``h2``.

    Rescaling uses the causal variants' own variance rather than ``beta' D
    beta``: the latter is the realized genetic variance under this reference
    and would make h2 depend on which variants happened to be drawn.

    ``m_genome`` is what makes chromosome subsets comparable to each other, and
    getting it wrong quietly ruins any sweep over the variant count. Holding
    ``h2`` fixed while restricting to one chromosome does not simulate a
    smaller slice of the same trait — it packs a whole genome's heritability
    into a few per cent of the variants, raising the per-variant effect size by
    the inverse of that fraction. Everything sensitive to per-variant
    heritability then moves for the wrong reason: the LDSC regression's slope,
    its residual scatter, and hence the precision of its intercept. Passing the
    full reference's variant count scales ``h2`` down to the subset's share, so
    the per-variant heritability matches the genome-wide trait and a sweep over
    chromosomes varies only the number of variants.
    """
    if m_genome:
        h2 = h2 * (float(m) / float(m_genome))
    n_causal = max(1, int(round(polygenicity * m)))
    beta = np.zeros(m)
    where = rng.choice(m, size=n_causal, replace=False)
    beta[where] = rng.standard_normal(n_causal)
    beta[where] *= np.sqrt(h2 / np.sum(beta[where] ** 2))
    return beta


def _panel(blocks, factors, beta_target, n_scores, rg, n_eff_aux, rng):
    """One component score per auxiliary trait, plus the target's own.

    Each auxiliary trait's effects share a correlation ``rg`` with the target's,
    and its score is that trait's own marginal GWAS estimate. Marginal effects
    are the simplest score-construction rule that responds to discovery sample
    size, which is what has to vary here; a better rule would raise every arm
    without changing what is being compared.
    """
    m = beta_target.size
    weights, ids = [], []
    for k in range(n_scores):
        if k == 0:
            beta_k = beta_target
            n_k = n_eff_aux[0]
        else:
            independent = np.zeros(m)
            support = np.flatnonzero(beta_target)
            independent[support] = rng.standard_normal(support.size)
            independent *= (np.linalg.norm(beta_target)
                            / max(np.linalg.norm(independent),
                                  np.finfo(float).tiny))
            r = float(rg[k])
            beta_k = r * beta_target + np.sqrt(max(1.0 - r * r, 0.0)) * independent
            n_k = n_eff_aux[k]
        z_k = _draw_z(blocks, factors, beta_k, n_k, rng)
        support = np.flatnonzero(z_k)
        weights.append((support, z_k[support]))
        ids.append(f"trait_{k}" if k else "target_own")
    return weights, ids


def _true_r2(blocks, variant_weights, beta_target):
    """``(w' D beta)^2 / (w' D w)`` — exact, given the true ``D``."""
    d_beta = _ld_times(blocks, beta_target)
    d_w = _ld_times(blocks, variant_weights)
    numerator = float(variant_weights @ d_beta)
    denominator = float(variant_weights @ d_w)
    if denominator <= 0.0:
        return float("nan")
    return numerator * numerator / denominator


def _replicate(blocks, factors, m, args, reference_n, seed, m_genome=None):
    rng = np.random.default_rng(seed)
    beta_target = _true_effects(m, args.h2, args.polygenicity, rng,
                                m_genome=m_genome)

    rg = np.concatenate([[1.0], rng.uniform(*args.rg, size=args.n_scores - 1)])
    n_aux = np.concatenate([[args.n_eff_target],
                            rng.uniform(*args.n_eff_aux,
                                        size=args.n_scores - 1)])
    weights, ids = _panel(blocks, factors, beta_target, args.n_scores, rg,
                          n_aux, rng)

    z_fit = _draw_z(blocks, factors, beta_target, args.n_eff_target, rng)
    z_tune = _draw_z(blocks, factors, beta_target, args.n_eff_target, rng)
    z_assess = _draw_z(blocks, factors, beta_target, args.n_eff_target, rng)

    if np.isinf(reference_n):
        fit_blocks = blocks
    else:
        fit_blocks = _simulated_reference(blocks, factors, int(reference_n), rng)

    started = time.perf_counter()
    fit = multi_pgs_sumstats(
        weights, z_fit, fit_blocks, weights_gwas=weights, score_ids=ids,
        z_valid=z_tune, ld_valid=fit_blocks, weights_gwas_valid=weights,
        weights_ld_valid=weights, tune="independent",
        n_variants_ld=m, n_variants_ld_valid=m,
        ld_shrinkage=args.ld_shrinkage, n_lambda=args.n_lambda)
    fit_seconds = time.perf_counter() - started

    # Ground truth for the fitted combination, and for each single score, all
    # under the TRUE D. The best single score is chosen on the tuning GWAS, not
    # on the truth, or the comparison would be rigged against the stack.
    combined = np.zeros(m)
    for (idx, w), coefficient in zip(weights, fit.beta):
        if coefficient != 0.0:
            combined[idx] += w * coefficient
    true_multi = _true_r2(blocks, combined, beta_target)

    tuning_scores = []
    for idx, w in weights:
        single = np.zeros(m)
        single[idx] = w
        d_single = _ld_times(blocks, single)
        denominator = float(single @ d_single)
        numerator = float(single @ z_tune)
        tuning_scores.append(numerator * numerator / denominator
                             if denominator > 0 else -np.inf)
    chosen = int(np.argmax(tuning_scores))
    best_single = np.zeros(m)
    best_single[weights[chosen][0]] = weights[chosen][1]
    true_best = _true_r2(blocks, best_single, beta_target)

    # The oracle is the true effect vector itself, whose R2 is h2 by
    # construction only in the absence of LD; under real LD it is what the best
    # possible linear predictor achieves here.
    true_oracle = _true_r2(blocks, beta_target, beta_target)

    c_assess = np.array([float(w @ z_assess[idx]) for idx, w in weights])
    gram_assess, _ = score_gram(weights, blocks, n_variants=m)
    assessment = evaluate_sumstat(fit.beta, c_assess, gram_assess,
                                  var_y=1.0, regime="A")
    # The line above assesses against the TRUE covariance, which isolates the
    # question "does a regime A number track the truth". A user does not have
    # the true D: they would form the assessment Gram from the same imperfect
    # reference they fitted with. That number is what they would publish, so it
    # is reported alongside — the gap between the two columns is the error a
    # bad reference puts into the *reported* accuracy, on top of the damage it
    # already did to the fit.
    if np.isinf(reference_n):
        own_reference_r2 = assessment.r2
    else:
        gram_own, _ = score_gram(weights, fit_blocks, n_variants=m)
        own_reference_r2 = evaluate_sumstat(
            fit.beta, c_assess, gram_own, var_y=1.0, regime="A").r2

    return {
        "reference_n": ("inf" if np.isinf(reference_n) else int(reference_n)),
        "seed": seed,
        "true_r2_multi": true_multi,
        "true_r2_best_single": true_best,
        "true_r2_oracle": true_oracle,
        "uplift": true_multi - true_best,
        "regime_a_r2": assessment.r2,
        "regime_a_minus_true": assessment.r2 - true_multi,
        "regime_a_r2_own_reference": own_reference_r2,
        "own_reference_minus_true": own_reference_r2 - true_multi,
        "plugin_r2": fit.pseudo_r2,
        "plugin_minus_true": fit.pseudo_r2 - true_multi,
        "selected_delta": fit.log["ld_shrinkage"],
        "n_selected": fit.n_selected,
        "gram_rank": fit.log["gram_rank"],
        "gram_min_correlation_eigenvalue":
            fit.log["gram_min_correlation_eigenvalue"],
        "discarded_c_fraction": fit.log["discarded_ld_null_c_fraction"],
        "fit_seconds": fit_seconds,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ld", required=True, type=Path)
    parser.add_argument("--chrom", nargs="*", default=["21", "22"],
                        help="chromosomes to use; omit the flag's values for all")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-scores", type=int, default=20)
    parser.add_argument("--h2", type=float, default=0.3)
    parser.add_argument("--h2-genome-wide", action="store_true", default=True,
                        help="interpret --h2 as the whole trait's heritability "
                             "and give the selected chromosomes their share, "
                             "so runs at different --chrom are comparable")
    parser.add_argument("--h2-in-subset", dest="h2_genome_wide",
                        action="store_false",
                        help="interpret --h2 as the heritability carried by "
                             "the selected chromosomes alone")
    parser.add_argument("--polygenicity", type=float, default=0.01)
    parser.add_argument("--n-eff-target", type=float, default=50_000.0)
    parser.add_argument("--n-eff-aux", type=float, nargs=2,
                        default=(50_000.0, 400_000.0),
                        help="range of auxiliary discovery sizes; auxiliary "
                             "GWAS are typically larger than the target's, "
                             "which is why borrowing from them can pay")
    parser.add_argument("--rg", type=float, nargs=2, default=(0.1, 0.7))
    parser.add_argument("--reference-n", nargs="*", default=["inf", "2000", "500"],
                        help="'inf' is the matched-reference control; finite "
                             "values simulate a reference panel of that size")
    parser.add_argument("--ld-shrinkage", type=float, nargs="*",
                        default=[0.0, 0.001, 0.01, 0.1])
    parser.add_argument("--n-lambda", type=int, default=50)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args(argv)
    started = time.perf_counter()

    chroms = args.chrom if args.chrom else None
    blocks, m, m_genome = _load(args.ld, chroms)
    if not args.h2_genome_wide:
        m_genome = None
    print(f"LD: {len(blocks)} blocks, {m:,} variants "
          f"(chromosomes {' '.join(chroms) if chroms else 'all'})")
    factors = [_block_factor(corr) for corr, _ in blocks]
    print(f"sampling factors built in {time.perf_counter() - started:.1f}s")

    references = [float("inf") if str(v).lower() in ("inf", "true", "matched")
                  else float(v) for v in args.reference_n]
    rows = []
    for reference_n in references:
        for seed in range(args.seeds):
            row = _replicate(blocks, factors, m, args, reference_n, seed,
                             m_genome)
            rows.append(row)
        label = "inf" if np.isinf(reference_n) else int(reference_n)
        recent = [r for r in rows if r["reference_n"] == label]
        print(f"  reference_n={label}: true multi R2 "
              f"{np.mean([r['true_r2_multi'] for r in recent]):.4f}, "
              f"best single {np.mean([r['true_r2_best_single'] for r in recent]):.4f}, "
              f"regime A minus true "
              f"{np.mean([r['regime_a_minus_true'] for r in recent]):+.4f}, "
              f"own-reference minus true "
              f"{np.mean([r['own_reference_minus_true'] for r in recent]):+.4f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "real_ld_simulation.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for reference_n in references:
        label = "inf" if np.isinf(reference_n) else int(reference_n)
        subset = [r for r in rows if r["reference_n"] == label]
        entry = {"reference_n": label, "n_seeds": len(subset)}
        for field in FIELDS:
            if field in ("reference_n", "seed"):
                continue
            values = np.asarray([r[field] for r in subset], dtype=float)
            finite = values[np.isfinite(values)]
            entry[f"{field}_mean"] = (float(finite.mean()) if finite.size
                                      else float("nan"))
            entry[f"{field}_sd"] = (float(finite.std(ddof=1))
                                    if finite.size > 1 else float("nan"))
        summary_rows.append(entry)
    summary_path = args.output_dir / "real_ld_simulation_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary_rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(summary_rows)

    provenance = {
        "source": benchmark_identity(__file__),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.perf_counter() - started,
        "command": [sys.executable, str(Path(__file__).resolve()),
                    *(sys.argv[1:] if argv is None else argv)],
        "platform": platform.platform(),
        "python": platform.python_version(),
        "multipgs": multipgs.__version__,
        "numpy": np.__version__,
        "ldpred3": _version("ldpred3"),
        "numba": _version("numba"),
        "ld_reference": {"path": str(args.ld), "sha256": _sha256(args.ld),
                         "n_blocks": len(blocks), "n_variants": m},
        "parameters": {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in vars(args).items()},
        "note": ("Real LD, simulated summary statistics, closed-form truth. "
                 "The assessment GWAS is drawn independently of the fitting "
                 "and tuning GWAS, which is what makes the regime A label "
                 "checkable here and is precisely what real successive GWAS "
                 "releases do not provide. No real-data accuracy is claimed."),
    }
    with (args.output_dir / "real_ld_simulation_provenance.json").open(
            "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")

    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
