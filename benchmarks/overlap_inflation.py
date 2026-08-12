#!/usr/bin/env python3
"""How much does sample overlap inflate a summary-statistic accuracy, and can
bivariate LDSC see it?

The first item in this package's "before trusting a number" checklist is that
sample overlap inflates every accuracy it reports, and that no cross-validation
inside the target cohort can detect it, because every fold shares the
contamination. That is asserted throughout the documentation and measured
nowhere. It is also the one contract a user cannot verify from their own
outputs: a contaminated regime A number looks exactly like a clean one.

Overlap is measurable here for the same reason the rest of this simulation
works — the summary statistics are drawn, so their sampling error is
controllable. Two GWAS sharing ``N_s`` individuals with phenotypic correlation
``rho_p`` have correlated sampling error, and in the standardized
parameterization used throughout multipgs that correlation is exactly

.. math::

    \\rho_s = \\rho_p N_s / \\sqrt{N_1 N_2},

which is also precisely the estimand of cross-trait LD Score regression's
intercept. So one knob sets the overlap, the truth knows it, and the detector
is aimed at the same number.

Three things are recorded against that knob:

* **The damage.** True accuracy of the fitted combination is closed form under
  the real LD and is unaffected by overlap; the *reported* accuracy is not.
  Their difference is the inflation, in R² units, that the checklist warns
  about without quantifying.
* **The detector.** :func:`bipred.ldsc.ldsc_rg`'s ``gcov_intercept`` estimates
  ``rho_s`` directly. bipred reports no standard error for it, so the null is
  calibrated from the ``overlap = 0`` replicates instead, and the detection
  rate at each true overlap is a power curve. The smallest overlap resolved is
  the practical answer to "can I rule this out from summary statistics alone".
* **The gap between them.** The useful comparison is not either column but
  their ratio: the overlap that already costs a user a materially inflated
  number, against the overlap the detector can actually see. If the first is
  well below the second — and cross-trait LDSC's intercept is known to be a
  blunt instrument — then a non-significant intercept is not evidence of
  no overlap, and the documentation should not let anyone read it that way.

**Detection power depends on both the number of variants and the
heritability, so the detector arm means nothing quoted at one setting.** The
intercept is separated from the slope by the spread of LD scores across ``M``
variants, so its precision improves with ``M``. Its residual scatter, though,
grows with the per-variant heritability: ``var(z_1 z_2)`` carries terms in
``N h^2 ell / M``, so a more heritable or better-powered pair of GWAS makes
the intercept *harder* to pin down, not easier. The two move in opposite
directions and both must be stated.

That interaction is also a trap for this benchmark's own chromosome sweep, and
:func:`real_ld_simulation._true_effects` handles it: holding ``h2`` fixed while
restricting to one chromosome does not simulate a smaller slice of the same
trait, it concentrates a whole genome's heritability into a few per cent of the
variants and inflates every per-variant effect. A sweep run that way varies
``M`` and per-variant ``h2`` at once and cannot attribute the result to either.
``--h2-genome-wide`` (the default) scales ``h2`` to the selected chromosomes'
share so the sweep isolates ``M``; ``--h2-in-subset`` restores the literal
reading. Every row records ``n_variants`` and the realized ``h2_realized``;
compare rows only where both agree.

**What this cannot establish.** Not the overlap in any real analysis. It
measures a detector's sensitivity and an estimator's sensitivity to a
simulated, exactly-known contamination, under real LD. Real overlap arrives
with population structure, assortative mating, differing phenotype definitions
and unstated cohort composition, none of which are here; the inflation it
causes may be larger or differently shaped. Nor is bivariate LDSC evaluated as
an r_g estimator — only its intercept is read, which is the part sensitive to
overlap.

This benchmark needs `bipred <https://github.com/bvilhjal/bipred>`_ for its
LD Score regression; without it the detector columns are skipped and the
inflation arm still runs.

    python benchmarks/overlap_inflation.py \\
        --ld /path/to/ldpred3_ldref_hm3.npz --chrom 20 21 22 --seeds 20
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[0]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import multipgs
from benchmarks._provenance import benchmark_identity
from multipgs import evaluate_sumstat, multi_pgs_sumstats
from multipgs.sumstat import score_gram
from real_ld_simulation import (_block_factor, _ld_times, _load, _sha256,
                                _true_effects, _true_r2, _version)


FIELDS = (
    "overlap", "n_variants", "h2_realized", "seed", "true_r2_multi", "reported_r2_multi",
    "inflation_multi", "true_r2_single", "reported_r2_single",
    "inflation_single", "ldsc_gcov_intercept", "ldsc_intercept_error",
    "ldsc_detected", "n_selected", "fit_seconds",
)


def _draw_correlated(blocks, factors, beta_a, beta_b, n_a, n_b, overlap, rng):
    """Two GWAS whose sampling errors correlate at ``overlap``.

    The shared component is drawn once per block and mixed into both draws, so
    the induced noise correlation is exactly ``overlap`` and does not depend on
    the block's representation. Genetic signal is untouched: overlap is a
    property of the people, not of the effects.
    """
    mean_a = _ld_times(blocks, beta_a)
    mean_b = _ld_times(blocks, beta_b)
    noise_a = np.zeros_like(beta_a)
    noise_b = np.zeros_like(beta_b)
    mix = np.sqrt(max(1.0 - overlap, 0.0))
    root = np.sqrt(max(overlap, 0.0))
    for (_, idx), (factor, residual) in zip(blocks, factors):
        rank = factor.shape[1]
        shared = rng.standard_normal(rank)
        own_a = rng.standard_normal(rank)
        own_b = rng.standard_normal(rank)
        draw_a = factor @ (root * shared + mix * own_a)
        draw_b = factor @ (root * shared + mix * own_b)
        if residual is not None:
            k = residual.size
            shared_r = rng.standard_normal(k)
            draw_a = draw_a + residual * (root * shared_r
                                          + mix * rng.standard_normal(k))
            draw_b = draw_b + residual * (root * shared_r
                                          + mix * rng.standard_normal(k))
        noise_a[idx] = draw_a
        noise_b[idx] = draw_b
    return (mean_a + noise_a / np.sqrt(n_a),
            mean_b + noise_b / np.sqrt(n_b))


def _predicted_intercept_se(ell, m, n_a, n_b, h2_a, h2_b, intercept=0.0):
    """Analytic sampling SE of the cross-trait LDSC intercept.

    Cross-trait LD Score regression fits
    ``E[z_1j z_2j] = (sqrt(N1 N2) rho_g / M) ell_j + a`` with
    ``a = rho_p N_shared / sqrt(N1 N2)``, so the intercept is the overlap this
    benchmark sets. Its precision follows from the residual scatter about that
    line. Univariate LDSC gives each trait's per-variant variance,
    ``Var(z_t) = 1 + N_t h2_t ell / M``; for bivariate normal ``z``,
    ``Var(z_1 z_2) = Var(z_1) Var(z_2) + Cov(z_1, z_2)^2``, and at the null
    slope the covariance is ``a``. Ordinary least squares then gives

        Var(a_hat) = mean(sigma2_eps) * (1 / m_eff + xbar^2 / Sxx).

    The two drivers pull against each other, which is why neither ``M`` nor
    ``h2`` characterises sensitivity alone: more variants sharpen the intercept
    through ``1 / m_eff``, while a more heritable or better-powered pair of GWAS
    blunts it through ``N h2 ell / M`` inside the residual.

    ``m_eff`` is **not** the variant count. LD correlates neighbouring ``z``, so
    the regression carries far fewer independent observations than rows — which
    is why LDSC estimates its own errors by block jackknife rather than by this
    formula.

    Read this as a scaling law, not a calibrated standard error. Taking
    ``m_eff = M / mean(ell)`` lands within a factor of about two of the
    simulated null spread over the range tested — closer at one chromosome,
    roughly twofold high at five — where treating all ``M`` rows as independent
    understates it threefold. Two things it does not model: ``m_eff`` is a
    crude stand-in for the true effective count, and the implementation this is
    compared against fits weighted, not ordinary, least squares. What it does
    capture is how sensitivity *moves* — improving with ``M``, degrading with
    ``N h2`` — which is the part needed to decide whether a non-significant
    intercept means anything in a given setting. The measured null spread in
    the run's own output is the number to trust.
    """
    ell = np.asarray(ell, dtype=float)
    x = np.sqrt(n_a * n_b) * ell / m
    sigma2 = ((1.0 + n_a * h2_a * ell / m) * (1.0 + n_b * h2_b * ell / m)
              + intercept ** 2)
    m_eff = max(float(m) / max(float(ell.mean()), 1e-12), 2.0)
    scale = m_eff / ell.size
    sxx = float(np.sum((x - x.mean()) ** 2)) * scale
    leverage = 1.0 / m_eff + (float(x.mean()) ** 2 / sxx if sxx > 0 else 0.0)
    return float(np.sqrt(float(np.mean(sigma2)) * leverage)), m_eff


def _ldsc_intercept(z_a, z_b, ld_scores, n_a, n_b, n_blocks):
    """Cross-trait LDSC intercept, or ``None`` when bipred is unavailable.

    bipred reports no standard error for this intercept, so its sampling
    spread is not read off one fit. It is calibrated instead from the
    ``overlap = 0`` replicates, which is the honest null for exactly this
    estimator on exactly this reference and panel — a better reference
    distribution than a jackknife would give, and the reason that arm is
    always run.
    """
    try:
        from bipred.ldsc import ldsc_rg
    except ImportError:
        return None
    # These marginal effects are already on the standardized scale ldsc_rg
    # documents; its own signed conversion to z-scores handles the rest.
    return float(ldsc_rg(z_a, z_b, ld_scores, n_a, n_b,
                         n_blocks=n_blocks).gcov_intercept)


def _replicate(blocks, factors, ld_scores, m, args, overlap, seed,
               m_genome=None):
    rng = np.random.default_rng(seed)
    beta_target = _true_effects(m, args.h2, args.polygenicity, rng,
                                m_genome=m_genome)

    # The contamination that matters: each component score's discovery GWAS
    # shares people with the GWAS the combination is later scored against. The
    # score is then partly fitted to the very sampling noise it is assessed on.
    weights, ids = [], []
    z_assess = None
    z_discovery = None
    for k in range(args.n_scores):
        r = 1.0 if k == 0 else float(rng.uniform(*args.rg))
        if k == 0:
            beta_k = beta_target
        else:
            independent = np.zeros(m)
            support = np.flatnonzero(beta_target)
            independent[support] = rng.standard_normal(support.size)
            independent *= (np.linalg.norm(beta_target)
                            / max(np.linalg.norm(independent),
                                  np.finfo(float).tiny))
            beta_k = r * beta_target + np.sqrt(max(1.0 - r * r, 0.0)) * independent
        z_k, z_shared = _draw_correlated(
            blocks, factors, beta_k, beta_target, args.n_eff_discovery,
            args.n_eff_assess, overlap, rng)
        if z_assess is None:
            # One assessment GWAS for the whole panel, drawn jointly with the
            # first score's discovery so their overlap is the stated one. Later
            # scores reuse it, which is the realistic case: a panel of scores
            # all trained in a biobank, scored against that same biobank.
            z_assess = z_shared
            z_discovery = z_k
        support = np.flatnonzero(z_k)
        weights.append((support, z_k[support]))
        ids.append(f"trait_{k}" if k else "target_own")

    z_fit = _draw_correlated(blocks, factors, beta_target, beta_target,
                             args.n_eff_assess, args.n_eff_assess, 0.0, rng)[0]

    started = time.perf_counter()
    fit = multi_pgs_sumstats(
        weights, z_fit, blocks, weights_gwas=weights, score_ids=ids,
        n_variants_ld=m, tune="none", n_lambda=args.n_lambda)
    fit_seconds = time.perf_counter() - started

    combined = np.zeros(m)
    for (idx, w), coefficient in zip(weights, fit.beta):
        if coefficient != 0.0:
            combined[idx] += w * coefficient
    true_multi = _true_r2(blocks, combined, beta_target)

    gram, _ = score_gram(weights, blocks, n_variants=m)
    c_assess = np.array([float(w @ z_assess[idx]) for idx, w in weights])
    # Labelled A because every model choice is fixed before this GWAS is
    # touched. That is the whole point: the label is defensible on its own
    # terms and the number is still wrong, because the label is about model
    # selection and overlap is about people.
    reported_multi = evaluate_sumstat(fit.beta, c_assess, gram, var_y=1.0,
                                      regime="A").r2

    single = np.zeros(m)
    single[weights[0][0]] = weights[0][1]
    true_single = _true_r2(blocks, single, beta_target)
    unit = np.zeros(len(weights))
    unit[0] = 1.0
    reported_single = evaluate_sumstat(unit, c_assess, gram, var_y=1.0,
                                       regime="A").r2

    intercept = (_ldsc_intercept(z_discovery, z_assess, ld_scores,
                                 args.n_eff_discovery, args.n_eff_assess,
                                 args.ldsc_blocks)
                 if args.ldsc else None)

    return {
        "overlap": overlap,
        "n_variants": m,
        "h2_realized": float(np.sum(beta_target ** 2)),
        "seed": seed,
        "true_r2_multi": true_multi,
        "reported_r2_multi": reported_multi,
        "inflation_multi": reported_multi - true_multi,
        "true_r2_single": true_single,
        "reported_r2_single": reported_single,
        "inflation_single": reported_single - true_single,
        "ldsc_gcov_intercept": intercept,
        "ldsc_intercept_error": (None if intercept is None
                                 else intercept - overlap),
        "ldsc_detected": None,
        "n_selected": fit.n_selected,
        "fit_seconds": fit_seconds,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ld", required=True, type=Path)
    parser.add_argument("--chrom", nargs="*", default=["21", "22"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--n-scores", type=int, default=20)
    parser.add_argument("--h2", type=float, default=0.3)
    parser.add_argument("--h2-genome-wide", action="store_true", default=True,
                        help="interpret --h2 as the whole trait's heritability "
                             "and give the selected chromosomes their share")
    parser.add_argument("--h2-in-subset", dest="h2_genome_wide",
                        action="store_false")
    parser.add_argument("--polygenicity", type=float, default=0.01)
    parser.add_argument("--n-eff-discovery", type=float, default=100_000.0)
    parser.add_argument("--n-eff-assess", type=float, default=100_000.0)
    parser.add_argument("--rg", type=float, nargs=2, default=(0.1, 0.7))
    parser.add_argument("--overlap", type=float, nargs="*",
                        default=[0.0, 0.05, 0.1, 0.25, 0.5, 1.0],
                        help="rho_p * N_shared / sqrt(N1 N2); 1.0 is the same "
                             "cohort twice")
    parser.add_argument("--n-lambda", type=int, default=50)
    parser.add_argument("--ldsc", action="store_true", default=True)
    parser.add_argument("--no-ldsc", dest="ldsc", action="store_false")
    parser.add_argument("--ldsc-blocks", type=int, default=200)
    parser.add_argument("--output-dir", type=Path,
                        default=_HERE / "results")
    args = parser.parse_args(argv)
    started = time.perf_counter()

    chroms = args.chrom if args.chrom else None
    blocks, m, m_genome = _load(args.ld, chroms)
    if not args.h2_genome_wide:
        m_genome = None
    factors = [_block_factor(corr) for corr, _ in blocks]
    from ldpred3 import ld_scores as ld_score_fn
    ld_scores = np.asarray(ld_score_fn(blocks), dtype=float)
    print(f"LD: {len(blocks)} blocks, {m:,} variants; factors and LD scores in "
          f"{time.perf_counter() - started:.1f}s")

    rows = []
    for overlap in args.overlap:
        for seed in range(args.seeds):
            rows.append(_replicate(blocks, factors, ld_scores, m, args,
                                   float(overlap), seed, m_genome))
        recent = [r for r in rows if r["overlap"] == float(overlap)]
        message = (f"  overlap={overlap:<5}: inflation "
                   f"{np.mean([r['inflation_multi'] for r in recent]):+.4f} R2")
        if recent[0]["ldsc_gcov_intercept"] is not None:
            message += (" | LDSC intercept "
                        f"{np.mean([r['ldsc_gcov_intercept'] for r in recent]):+.5f}")
        print(message)

    # Detection is calibrated on the overlap = 0 replicates: an intercept
    # counts as detected when it exceeds what this estimator produces on this
    # panel and reference when there is genuinely no overlap at all.
    null = np.asarray([r["ldsc_gcov_intercept"] for r in rows
                       if r["overlap"] == 0.0
                       and r["ldsc_gcov_intercept"] is not None], dtype=float)
    threshold = (1.96 * float(null.std(ddof=1))
                 if null.size > 1 else float("nan"))
    if args.ldsc:
        realized = float(np.mean([r["h2_realized"] for r in rows]))
        predicted, m_eff = _predicted_intercept_se(
            ld_scores, m, args.n_eff_discovery, args.n_eff_assess,
            realized, realized)
        print(f"predicted intercept SD {predicted:.5f} "
              f"(m_eff {m_eff:,.0f} of {m:,} variants, h2 {realized:.4g})")
    if np.isfinite(threshold):
        for row in rows:
            if row["ldsc_gcov_intercept"] is not None:
                row["ldsc_detected"] = int(abs(row["ldsc_gcov_intercept"])
                                           > threshold)
        print(f"measured null intercept SD {null.std(ddof=1):.5f}, "
              f"detection threshold {threshold:.5f}")
        for overlap in args.overlap:
            subset = [r["ldsc_detected"] for r in rows
                      if r["overlap"] == float(overlap)
                      and r["ldsc_detected"] is not None]
            if subset:
                print(f"  overlap={overlap:<5}: detected "
                      f"{int(np.sum(subset))}/{len(subset)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.output_dir / "overlap_inflation.csv"
    with raw_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for overlap in args.overlap:
        subset = [r for r in rows if r["overlap"] == float(overlap)]
        entry = {"overlap": float(overlap), "n_variants": m,
                 "n_seeds": len(subset)}
        for field in FIELDS:
            if field in ("overlap", "n_variants", "h2_realized", "seed"):
                continue
            values = np.asarray([np.nan if r[field] is None else r[field]
                                 for r in subset], dtype=float)
            finite = values[np.isfinite(values)]
            entry[f"{field}_mean"] = (float(finite.mean()) if finite.size
                                      else float("nan"))
            entry[f"{field}_sd"] = (float(finite.std(ddof=1))
                                    if finite.size > 1 else float("nan"))
        summary_rows.append(entry)
    summary_path = args.output_dir / "overlap_inflation_summary.csv"
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
        "bipred": _version("bipred"),
        "ld_reference": {"path": str(args.ld), "sha256": _sha256(args.ld),
                         "n_blocks": len(blocks), "n_variants": m},
        "parameters": {k: (str(v) if isinstance(v, Path) else v)
                       for k, v in vars(args).items()},
        "note": ("Overlap is simulated as correlated sampling error at a known "
                 "rho_s = rho_p N_shared / sqrt(N1 N2), which is also the "
                 "estimand of the cross-trait LDSC intercept. True accuracy is "
                 "closed form and unaffected by overlap; the reported accuracy "
                 "is not, and their difference is the inflation. No real "
                 "analysis's overlap is estimated here."),
    }
    with (args.output_dir / "overlap_inflation_provenance.json").open(
            "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")

    print(summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
