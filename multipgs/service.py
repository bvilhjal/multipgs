"""Fit a multi-PGS combination for a caller that holds no genotypes.

:func:`fit_prepared_panel` is the pipeline entry point for a service that runs
LDpred3's QC, harmonisation and LD-consistency screen itself and stores the
resulting :class:`~ldpred3.prepare.PreparedTrait`, exactly as
:func:`gwfm.fit_prepared_trait` is for fine-mapping. The component scores are
**LDpred3 weight files already fitted against this same LD cache** — a
service's own earlier runs — so no score is rebuilt and nothing is scored
against a cohort. The whole job is one streamed Gram, one ``K``-by-``K`` lasso,
and one folded weight file.

The estimand, the tuning regimes and the alignment contract are unchanged from
:func:`multipgs.multi_pgs_sumstats`; ``docs/service.md`` states what the absent
cohort costs. Two things this function does *not* do, deliberately:

* It does not assume the component scores are independent of the target GWAS.
  ``tune="pumas"`` holds them fixed, so it cannot remove that leakage, and
  ``weights_independent_of_z`` stays an explicit acknowledgement rather than
  something inferred from the inputs. A caller that lets a user combine a
  score fitted on the uploaded GWAS must refuse it itself.
* It does not claim an assessment. With one target GWAS the honest labels are
  regime B (pseudotuning) or regime C (same-data reuse); regime A needs a
  second untouched GWAS. :attr:`PreparedPanelFit.regime` records which.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

from ._align import Component, align_components
from ._gram import accuracy_blocks

__all__ = ["fit_prepared_panel", "PreparedPanelFit", "Component"]

#: Tuning mode -> the regime the resulting numbers belong to.
_TUNE_REGIME = {"none": "C", "pumas": "B", "independent": "B"}


@contextmanager
def _opened(ld_cache, validate):
    """Borrow a caller-owned prepared cache, or own one path-based load."""
    from ldpred3.interop import PreparedLDCache, prepare_ld_cache

    if isinstance(ld_cache, PreparedLDCache):
        if ld_cache.closed:
            raise ValueError("prepared ld_cache is closed")
        yield ld_cache
        return
    context = prepare_ld_cache(ld_cache, validate=validate)
    try:
        yield context
    finally:
        context.close()


def _reference(cache):
    """The cache's variant table, reference AF and HWE dosage SD.

    A service never sees target dosages, so the deployment scale is the
    reference's own frequency and the HWE ``sqrt(2 f (1-f))`` approximation --
    the same one its univariate weight files already carry.
    """
    from ldpred3.interop import VariantTable

    meta = cache.metadata
    if not hasattr(meta, "get"):
        raise ValueError("ld_cache metadata is not a mapping")
    ids = np.asarray(cache.variant_ids)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("ld_cache variant IDs must be a non-empty vector")
    missing = [key for key in ("counted_allele", "other_allele", "chrom",
                               "pos", "reference_af") if meta.get(key) is None]
    if missing:
        raise ValueError(
            f"ld_cache lacks {', '.join(missing)}; a multi-PGS panel needs the "
            "reference's alleles, coordinates and frequencies. Rebuild the "
            "cache with a current ldpred3 ld_out=")
    fields = {}
    for key in ("counted_allele", "other_allele", "chrom", "pos",
                "reference_af"):
        values = np.asarray(meta[key])
        if values.ndim != 1 or values.size != ids.size:
            raise ValueError(
                f"ld_cache metadata {key!r} has length {values.size}, "
                f"expected {ids.size}")
        fields[key] = values
    af = np.asarray(fields["reference_af"], dtype=float)
    variants = VariantTable(
        id=ids, chrom=fields["chrom"], pos=fields["pos"],
        cm=np.zeros(ids.size, dtype=float),
        a1=fields["counted_allele"], a2=fields["other_allele"])
    return variants, af, np.sqrt(2.0 * af * (1.0 - af))


def _target_moment(trait, n_cache):
    """The target trait's standardized effects scattered into cache order.

    A prepared trait stores only its usable variants, sparsely indexed into the
    full cache. ``c = W^T z`` needs ``z`` on the same rows as the component
    weights, and a variant the trait dropped contributes nothing -- which is
    zero here, not a missing value.
    """
    indices = np.asarray(getattr(trait, "indices"), dtype=np.int64).ravel()
    z = np.asarray(getattr(trait, "z"), dtype=float).ravel()
    if indices.size != z.size:
        raise ValueError(f"the trait has {indices.size} indices for {z.size} "
                         "standardized effects")
    if indices.size == 0:
        raise ValueError("the trait has no usable variants")
    declared = getattr(trait, "n_cache", None)
    if declared is not None and int(declared) != int(n_cache):
        raise ValueError(
            f"the trait was prepared against a {int(declared)}-variant "
            f"reference but this cache holds {int(n_cache)}; they are not the "
            "same LD reference")
    if indices.min() < 0 or indices.max() >= n_cache:
        raise ValueError("trait indices fall outside the LD cache")
    if not np.all(np.isfinite(z)):
        raise ValueError("the trait's standardized effects are not all finite")
    full = np.zeros(int(n_cache), dtype=float)
    full[indices] = z
    return full, indices


def _scalar_n_eff(trait, n_eff):
    """One effective sample size for PUMAS, and how it was obtained.

    A prepared trait carries a per-variant ``n_eff`` from an imputed or
    meta-analysed GWAS. The pseudo-split is a single scalar draw, so this
    reports the median rather than letting a caller assume a constant N.
    """
    if n_eff is not None:
        value = float(n_eff)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("n_eff must be finite and strictly positive")
        return value, {"policy": "supplied", "n_eff": value}
    values = np.asarray(getattr(trait, "n_eff", None), dtype=float).ravel()
    if values.size == 0 or not np.any(np.isfinite(values)):
        raise ValueError(
            "tune='pumas' needs n_eff and the prepared trait carries none; "
            "pass n_eff= explicitly")
    finite = values[np.isfinite(values)]
    value = float(np.median(finite))
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("the trait's n_eff median is not positive")
    return value, {"policy": "trait_median", "n_eff": value,
                   "min": float(finite.min()), "max": float(finite.max()),
                   "n_variants": int(finite.size)}


@dataclass
class PreparedPanelFit:
    """A multi-PGS combination fitted from summary statistics and one LD cache.

    ``fit`` is the :class:`~multipgs.sumstat.SumstatFit`; ``panel`` is the
    cohort-free :class:`~multipgs.panel.ScorePanel` of component weight tables
    and ``weights`` the folded per-variant deployment table, on LDpred3's
    standardized scale with the reference's own ``AF_REF``/``SD_REF``.

    ``regime`` is the first thing to read. It is ``"B"`` for pseudotuning or an
    independent tuning GWAS and ``"C"`` when selection reused the fitting
    moments; neither is a clean assessment, and no arrangement of one target
    GWAS produces one.

    ``accuracy_u``/``accuracy_v`` are the per-LD-block ``w_b' z_b`` and
    ``w_b' D_b w_b`` of the *combined* score, with ``accuracy_groups`` labelling
    each block's chromosome. They are kept raw and dependency-free because the
    plug-in ratio they sum to is a point estimate with no correction, no
    interval and no null; a caller that has :mod:`ppb` can hand them straight
    to ``ppb.r2_block_jackknife``, ``ppb.sign_flip_null`` and (with the two
    sums) ``ppb.corrected_r2`` to get all three. See
    :meth:`accuracy_totals`.
    """

    fit: object
    score_ids: list
    panel: object
    weights: dict
    regime: str
    n_eff: float
    n_eff_policy: dict
    align_log: dict
    trait_log: dict
    cache_provenance: dict
    accuracy_u: object = None
    accuracy_v: object = None
    accuracy_groups: object = None
    log: dict = field(default_factory=dict)

    @property
    def beta(self):
        """Raw-score combination coefficients, one per component."""
        return self.fit.beta

    @property
    def n_selected(self):
        return int(self.fit.n_selected)

    def coefficient_table(self):
        """Component rows for rendering: id, coefficient, and what it carries.

        ``beta`` is on the raw score scale and ``beta_std`` on the
        standardized one, which is the scale to compare components on.
        ``n_variants`` and ``weight_mass`` come from the alignment, so a
        component that matched the reference poorly is visible next to its
        coefficient rather than only in a log.
        """
        log = self.align_log or {}
        mass = log.get("weight_mass_matched") or {}
        per_component = log.get("components") or {}
        beta = np.asarray(self.fit.beta, dtype=float)
        beta_std = np.asarray(self.fit.beta_std, dtype=float)
        rows = []
        for j, sid in enumerate(self.score_ids):
            table = self.panel.weights[j]
            detail = per_component.get(str(sid), {})
            rows.append({
                "score_id": sid,
                "beta": float(beta[j]),
                "beta_std": float(beta_std[j]),
                "selected": bool(beta[j] != 0.0),
                "n_variants": int(np.asarray(table["weight"]).size),
                # Only computable where the component's own weights are
                # known; a Catalog scoring file reports none, and that is a
                # None rather than a zero.
                "weight_mass": (None if mass.get(str(sid)) is None
                                else float(mass[str(sid)])),
                # How this component's weights reached the standardized scale,
                # so an HWE-approximated score is never shown as equivalent to
                # one whose reference scale was verified.
                "kind": detail.get("kind"),
                "scale_source": detail.get("scale_source"),
                "scale_verified": bool(detail.get("scale_verified")),
            })
        rows.sort(key=lambda row: -abs(row["beta_std"]))
        return rows

    def accuracy_totals(self):
        """``(numerator, denominator, n_blocks)`` of the plug-in accuracy.

        ``self.n_eff`` completes the triple ``ppb.corrected_r2`` needs; it is
        resolved from the trait whether or not tuning used it.

        ``numerator**2 / (denominator * var_y)`` is the combined score's
        plug-in R², and the pair is what ``ppb.corrected_r2`` takes for its
        finite-sample correction. ``None`` when the block decomposition was
        not computed.
        """
        if self.accuracy_u is None or self.accuracy_v is None:
            return None
        u = np.asarray(self.accuracy_u, dtype=float)
        v = np.asarray(self.accuracy_v, dtype=float)
        return float(u.sum()), float(v.sum()), int(u.size)

    def write_weights(self, path):
        """Write the folded combination as an LDpred3 weight file.

        ``ID CHR POS A1 A2 WEIGHT AF_REF SD_REF``, deployable with
        ``ldpred3.score_from_weights``. ``SD_REF`` is the reference's HWE
        approximation, not a target dosage SD, so score with
        ``scaling="target"`` unless that assumption holds for the target.
        """
        from ldpred3.interop import write_weights

        w = self.weights
        write_weights(
            str(path), id=w["id"], chrom=w["chrom"], pos=w["pos"],
            effect_allele=w["a1"], other_allele=w["a2"], weight=w["weight"],
            af=w["af"], sd=w["sd"], sd_source="hwe")
        return str(path)

    def summary(self):
        meaning = ("pseudotuning or independent tuning, not an assessment"
                   if self.regime == "B"
                   else "selection reused the fitting moments; an upper bound")
        lines = [
            f"multi-PGS from {len(self.score_ids)} component score(s): "
            f"{self.n_selected} selected",
            f"  regime {self.regime} — {meaning}",
            f"  n_eff {self.n_eff:,.0f} ({self.n_eff_policy['policy']})",
        ]
        top = [row for row in self.coefficient_table() if row["selected"]][:6]
        if top:
            lines.append("  weights: " + ", ".join(
                f"{row['score_id']} {row['beta_std']:+.3g}" for row in top))
        if self.align_log.get("all_zero_scores"):
            lines.append("  all-zero component(s): "
                         + ", ".join(self.align_log["all_zero_scores"]))
        if self.align_log.get("n_hwe_approximated"):
            lines.append(
                f"  {self.align_log['n_hwe_approximated']} component(s) on the "
                "HWE-approximated scale, not a measured dosage SD")
        return "\n".join(lines)


def fit_prepared_panel(ld_cache, trait, components, *, score_ids=None,
                       tune="pumas", n_eff=None,
                       weights_independent_of_z=False, sd_source=None,
                       check_scale=True, validate="full", progress=None,
                       **fit_kwargs):
    """Combine LDpred3 weight files for a trait prepared against ``ld_cache``.

    Parameters
    ----------
    ld_cache : str or PreparedLDCache
        The LD reference. A caller-owned prepared cache is borrowed and left
        open; a path is opened under ``validate`` and closed again.
    trait : PreparedTrait
        The **target** trait, prepared against this same cache. Only its
        ``indices``, ``z`` and (for PUMAS) ``n_eff`` are read; no phenotype
        and no genotypes are involved.
    components : sequence of str, WeightsTable, or Component
        The panel. A bare path or table is an LDpred3 weight file fitted
        against this same reference, and ``check_scale`` verifies that claim
        against the file's own ``AF_REF``/``SD_REF``. A
        :class:`~multipgs.Component` with ``kind="scoring_file"`` is a PGS
        Catalog-format file counting raw alleles, converted with the
        reference's HWE ``sqrt(2 f (1-f))`` approximation because no target
        dosages exist to measure the real dosage SD. Mixing the two is
        allowed and recorded per component: the folded weight file inherits
        the weaker of the two scale assumptions, so which components carry
        which has to remain visible.
    score_ids : sequence of str, optional
        Component names; default is each file's stem.
    tune : {"pumas", "none"}
        ``"pumas"`` draws a pseudo-split from the single target GWAS — regime
        B — and requires ``weights_independent_of_z=True``. ``"none"`` selects
        on the fitting moments, which is regime C.
    weights_independent_of_z : bool
        Acknowledgement that no component score was built using the GWAS
        behind ``trait``. It is not checkable here: a caller that offers a
        user their own earlier fits must exclude the target's own.
    **fit_kwargs
        Forwarded to :func:`multipgs.multi_pgs_sumstats` (``alpha``,
        ``penalty_factor``, ``ld_shrinkage``, ``n_lambda``, ``rng`` ...).

    Returns
    -------
    PreparedPanelFit
    """
    from .panel import ScorePanel, combine_weights
    from .sumstat import multi_pgs_sumstats

    if tune not in ("pumas", "none"):
        raise ValueError(
            "tune must be 'pumas' or 'none' here: an independent tuning GWAS "
            "needs its own aligned panel and LD, so use multi_pgs_sumstats "
            f"directly for that. Got {tune!r}")
    components = list(components)
    if not components:
        raise ValueError("a multi-PGS panel needs at least one component score")

    def _stage(name):
        if progress is None:
            return None
        return lambda done, total, *rest: progress(done, total, name)

    with _opened(ld_cache, validate) as cache:
        variants, af, sd = _reference(cache)
        n_cache = int(af.size)
        described = list(components)
        if score_ids is not None:
            named = list(score_ids)
            if len(named) != len(described):
                raise ValueError(f"score_ids has {len(named)} entries for "
                                 f"{len(described)} components")
            described = [
                item if isinstance(item, Component)
                else Component(item, score_id=name)
                for item, name in zip(described, named)]
        pairs, ids, tables, align_log = align_components(
            described, variants, af=af, sd=sd, sd_source=sd_source,
            check_scale=check_scale, progress=_stage("align"))
        if not pairs:
            raise ValueError(
                "no component score aligned to this LD reference; "
                f"{align_log.get('errors') or 'see the alignment log'}")
        z, trait_index = _target_moment(trait, n_cache)
        # Always resolved, not only for PUMAS: the pseudo-split needs it, and
        # so does any finite-sample correction of the plug-in accuracy, which
        # a caller may apply whatever the tuning mode was. `used_for_tuning`
        # records which of the two it served here.
        scalar_n_eff, n_eff_policy = _scalar_n_eff(trait, n_eff)
        n_eff_policy = dict(n_eff_policy, used_for_tuning=(tune == "pumas"))
        fit = multi_pgs_sumstats(
            pairs, z, cache.blocks, weights_gwas=pairs, score_ids=ids,
            n_variants_ld=n_cache, tune=tune,
            n_eff=scalar_n_eff if tune == "pumas" else None,
            weights_independent_of_z=weights_independent_of_z,
            progress=progress, **fit_kwargs)
        # The combined weights on the reference basis, decomposed per LD
        # block while the cache is still open. Done here because the blocks
        # are gone once it closes, and because a point estimate with no
        # interval and no null is the weakest part of this whole route.
        accuracy_u = accuracy_v = accuracy_groups = None
        try:
            collapsed = fit.frozen_variant_weights(pairs,
                                                   n_variants_ld=n_cache)
            accuracy_u, accuracy_v, accuracy_groups = accuracy_blocks(
                collapsed, z, cache.blocks, chrom=np.asarray(variants.chrom),
                progress=_stage("accuracy"))
        except Exception as exc:                              # noqa: BLE001
            # A missing interval must not cost the fit itself; record why.
            accuracy_error = f"{type(exc).__name__}: {exc}"
        else:
            accuracy_error = None
        cache_provenance = {
            "n_variants": n_cache,
            "n_ref": _scalar(cache.metadata.get("n_ref")),
            "schema_version": _scalar(cache.metadata.get("schema_version")),
            "ld_shrunk": bool(_scalar(cache.metadata.get("ld_shrunk")) or 0),
        }

    panel = ScorePanel.weights_only(tables, ids)
    weights = combine_weights(panel, fit)
    return PreparedPanelFit(
        fit=fit, score_ids=list(ids), panel=panel, weights=weights,
        regime=_TUNE_REGIME[tune], n_eff=scalar_n_eff,
        n_eff_policy=n_eff_policy, align_log=align_log,
        trait_log={"n_variants": int(trait_index.size),
                   "n_cache": n_cache,
                   "coverage": float(trait_index.size) / n_cache},
        cache_provenance=cache_provenance,
        accuracy_u=accuracy_u, accuracy_v=accuracy_v,
        accuracy_groups=accuracy_groups,
        log={"tune": tune, "n_components": len(ids),
             "accuracy_blocks_error": accuracy_error,
             "weights_independent_of_z": bool(weights_independent_of_z),
             "n_combined_variants": int(np.asarray(weights["id"]).size)})


def _scalar(value):
    """One number out of a cache metadata entry stored as a length-1 array."""
    if value is None:
        return None
    array = np.asarray(value).ravel()
    return None if array.size == 0 else array.item(0)
