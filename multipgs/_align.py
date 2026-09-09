"""Align Catalog scoring files to an LD-reference variant table."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np


def _as_variant_table(variants):
    """Accept an ldpred3 ``VariantTable`` or wrap a mapping with the same keys.

    ``ldpred3.harmonize`` needs attribute access (``variants.a1``,
    ``len(variants)``). A catalog union mapping is converted once here.
    """
    if all(hasattr(variants, name)
           for name in ("id", "chrom", "pos", "a1", "a2")):
        return variants
    try:
        ids = variants["id"]
        chrom = variants["chrom"]
        pos = variants["pos"]
        a1 = variants["a1"]
        a2 = variants["a2"]
    except (TypeError, KeyError, IndexError) as exc:
        raise TypeError(
            "variants must be an ldpred3 VariantTable or a mapping with "
            "id, chrom, pos, a1 and a2") from exc
    from ldpred3.genotype_model import VariantTable
    n = len(np.asarray(ids).ravel())
    cm = (variants["cm"] if isinstance(variants, Mapping) and "cm" in variants
          else np.zeros(n))
    return VariantTable(
        chrom=np.asarray(chrom), id=np.asarray(ids), cm=np.asarray(cm),
        pos=np.asarray(pos), a1=np.asarray(a1), a2=np.asarray(a2))


def align_to_reference(scoring_files, variants, *, sd=None, af=None,
                       hwe_genotype_sd=False, drop_ambiguous=True,
                       on_error="raise", progress=None):
    """Align PGS Catalog scoring files to an LD reference's variant table.

    Returns allele-aligned weights. They are on the **standardized** genotype
    scale only when ``sd`` is supplied or ``hwe_genotype_sd=True`` is requested;
    otherwise catalog weights remain on their raw allele-count scale and the log
    records ``standardized=False``. :func:`score_gram` needs standardized-scale
    weights, so unscaled output is suitable only when the input weights were
    already standardized. HWE ``sqrt(2 f (1-f))`` is available as an explicit
    approximation because it ignores imputation uncertainty and departures from
    HWE.

    Parameters
    ----------
    scoring_files : sequence of str or ScoringFile
    variants : VariantTable or mapping
        The LD reference's variant table, with ``id chrom pos a1 a2``, in the
        reference's own order — the row order of ``D``. A mapping is wrapped
        into an ldpred3 ``VariantTable`` before harmonisation.
    sd : array_like, optional
        Empirical dosage standard deviation per reference variant. Preferred.
    af : array_like, optional
        Reference allele frequency of ``a1`` per variant. Used only when
        ``hwe_genotype_sd=True``.
    hwe_genotype_sd : bool
        Explicitly approximate dosage SD by ``sqrt(2 f (1-f))`` from ``af``.
    on_error : {"raise", "skip"}

    Returns
    -------
    (pairs, score_ids, log) : (list of (index, weight), list of str, dict)
        ``pairs`` can go straight to :func:`score_gram` and
        :func:`multi_pgs_sumstats` only when ``log["standardized"]`` is true or
        the supplied scoring weights were already standardized.
    """
    from .catalog import harmonize_scoring_file, read_scoring_file

    if on_error not in ("raise", "skip"):
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")

    variants = _as_variant_table(variants)
    n_variants = len(variants)
    scale_sd = None
    scale_source = None
    if sd is not None:
        scale_sd = np.asarray(sd, dtype=float).ravel()
        if scale_sd.size != n_variants:
            raise ValueError(f"sd has {scale_sd.size} entries for {n_variants} "
                             "reference variants")
        if not np.all(np.isfinite(scale_sd)) or np.any(scale_sd < 0.0):
            raise ValueError("sd must be finite and non-negative")
        scale_source = "empirical_sd"
    if af is not None:
        f = np.asarray(af, dtype=float).ravel()
        if f.size != n_variants:
            raise ValueError(f"af has {f.size} entries for {n_variants} "
                             "reference variants")
        if not np.all(np.isfinite(f)) or np.any((f < 0.0) | (f > 1.0)):
            raise ValueError("af must be finite and lie in [0, 1]")
        if hwe_genotype_sd:
            if scale_sd is not None:
                raise ValueError("give sd or request HWE scaling from af, not both")
            scale_sd = np.sqrt(2.0 * f * (1.0 - f))
            scale_source = "hwe_from_af"
    elif hwe_genotype_sd:
        raise ValueError("hwe_genotype_sd=True requires af")
    if af is not None and scale_sd is None:
        raise ValueError("af alone does not define empirical dosage SD; pass sd, "
                         "or set hwe_genotype_sd=True to request the HWE "
                         "approximation explicitly")

    files = list(scoring_files)
    pairs, ids, errors = [], [], {}
    matched = []
    for i, item in enumerate(files):
        label = getattr(item, "pgs_id", None) or str(item)
        try:
            scoring = item if hasattr(item, "weight") else read_scoring_file(item)
            idx, w, log = harmonize_scoring_file(scoring, variants,
                                                 drop_ambiguous=drop_ambiguous)
            if scale_sd is not None:
                # A catalog weight counts alleles; on standardized genotypes the
                # same score is w * sd. A monomorphic reference variant has
                # sd = 0 and contributes nothing, which is the truth here.
                w = w * scale_sd[idx]
            pairs.append((idx, w))
            ids.append(scoring.pgs_id)
            matched.append(int(log.get("n_matched", idx.size)))
        except Exception as exc:                      # noqa: BLE001
            if on_error == "raise":
                raise
            errors[label] = str(exc)
        if progress is not None:
            progress(i, len(files), label)

    log = {"n_requested": len(files), "n_aligned": len(pairs),
           "n_failed": len(errors), "n_reference_variants": n_variants,
           "standardized": scale_sd is not None,
           "scale_source": scale_source}
    if matched:
        log["n_matched_median"] = int(np.median(matched))
        log["n_matched_min"] = int(min(matched))
    if errors:
        log["errors"] = errors
    if scale_source == "hwe_from_af":
        log["warning"] = (
            "weights used HWE sqrt(2 f (1-f)) rather than empirical dosage SD; "
            "this is an approximation and may be wrong for imputed variants")
    elif scale_sd is None:
        log["warning"] = (
            "no empirical dosage SD was supplied and no HWE conversion was "
            "requested, so catalog weights were not converted to the "
            "standardized-genotype scale; this is correct only for weights "
            "that were already on it")
    return pairs, ids, log


def reference_weight_table(index, weight, variants, af, sd) -> dict:
    """One deployable weight table on the reference's own scale.

    :func:`multipgs.combine_weights` needs per-variant ``af``/``sd`` to write a
    frozen file, and every component of one panel must share them or the folded
    file describes no single scale. Taking both from the *reference* rather
    than from each component's own file is what makes a mixed panel -- some
    components fitted here, some downloaded -- fold onto one coordinate.
    """
    index = np.asarray(index, dtype=np.int64)
    return {
        "id": np.asarray(variants.id)[index],
        "chrom": np.asarray(variants.chrom)[index],
        "pos": np.asarray(variants.pos)[index],
        "a1": np.asarray(variants.a1)[index],
        "a2": np.asarray(variants.a2)[index],
        "weight": np.asarray(weight, dtype=float),
        "af": np.asarray(af, dtype=float)[index],
        "sd": np.asarray(sd, dtype=float)[index],
    }


def _weights_as_sumstats(table):
    """Wrap a :class:`ldpred3.WeightsTable` so ``harmonize`` can align it.

    The allele logic — rsID then ``chrom:pos`` matching, sign flips for swapped
    alleles, strand resolution, palindromic drops — is LDpred3's, and a weight
    file needs exactly the same handling as a GWAS. ``se``/``n_eff`` are
    placeholders: harmonisation never reads them, and a weight carries neither.
    """
    from ldpred3.interop import Sumstats

    m = int(np.asarray(table.weight).size)
    return Sumstats(
        id=np.asarray(table.id), chrom=np.asarray(table.chrom),
        pos=np.asarray(table.pos), ea=np.asarray(table.a1),
        oa=np.asarray(table.a2), beta=np.asarray(table.weight, dtype=float),
        se=np.ones(m), n_eff=np.ones(m), eaf=np.full(m, np.nan),
        info=np.full(m, np.nan))


def align_weights_to_reference(weight_files, variants, *, af, sd,
                               score_ids=None, sd_source=None,
                               drop_ambiguous=True, on_error="raise",
                               check_scale=True, scale_atol=1e-6,
                               progress=None):
    """Align LDpred3 weight files to an LD reference's variant table.

    The component-panel counterpart of :func:`align_to_reference`, for scores
    that were **fitted against this same reference** rather than downloaded
    from the PGS Catalog. LDpred3 posterior-mean weights are already on the
    standardized-genotype scale, so no dosage-SD conversion applies and the
    result goes straight to :func:`multipgs.score_gram` and
    :func:`multipgs.multi_pgs_sumstats`.

    Parameters
    ----------
    weight_files : sequence of str or WeightsTable
        One LDpred3 weight file per component score.
    variants : VariantTable or mapping
        The reference's variant table in its own order — the row order of
        ``D``. A mapping is wrapped, as in :func:`align_to_reference`.
    af, sd : array_like
        Reference allele frequency of ``a1`` and dosage SD, per reference
        variant. These define the deployment scale written into the combined
        weight file, and are what the per-file columns are checked against.
    score_ids : sequence of str, optional
        Names for the columns; defaults to each file's stem.
    sd_source : str, optional
        Forwarded to :func:`ldpred3.read_weights` for files written before the
        ``SD_SOURCE`` column existed. Those carry HWE ``sqrt(2f(1-f))`` SDs,
        which is not what the absent column implies.
    check_scale : bool
        Require each file's own ``AF_REF``/``SD_REF`` to agree with ``af``/``sd``
        at the variants it matched. This is the guard against combining a score
        fitted on a *different* LD panel: the identifiers can still match while
        the frequencies do not, and the resulting Gram would describe a
        reference none of the scores were built on. Allele-swapped matches are
        compared after flipping ``af`` to ``1 - af``.

    Returns
    -------
    (pairs, ids, tables, log)
        ``pairs`` are ``(index, weight)`` columns on the reference's index
        space; ``tables`` are the per-score weight tables for
        :meth:`multipgs.ScorePanel.weights_only`, carrying the reference's own
        ``af``/``sd`` so a combined file deploys on one consistent scale.
    """
    from ldpred3.interop import harmonize, read_weights

    if on_error not in ("raise", "skip"):
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")
    variants = _as_variant_table(variants)
    n_variants = len(variants)
    af = np.asarray(af, dtype=float).ravel()
    sd = np.asarray(sd, dtype=float).ravel()
    for name, values in (("af", af), ("sd", sd)):
        if values.size != n_variants:
            raise ValueError(f"{name} has {values.size} entries for "
                             f"{n_variants} reference variants")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must be finite")
    if np.any((af < 0.0) | (af > 1.0)):
        raise ValueError("af must lie in [0, 1]")
    if np.any(sd < 0.0):
        raise ValueError("sd must be non-negative")

    files = list(weight_files)
    if score_ids is not None:
        requested = [str(s) for s in score_ids]
        if len(requested) != len(files):
            raise ValueError(f"score_ids has {len(requested)} entries for "
                             f"{len(files)} weight files")
    else:
        requested = None

    pairs, ids, tables, errors, matched = [], [], [], {}, []
    mismatched, mass = {}, []
    for i, item in enumerate(files):
        label = (requested[i] if requested is not None
                 else _stem_of(item))
        try:
            table = (item if hasattr(item, "weight")
                     else read_weights(item, sd_source=sd_source))
            h = harmonize(_weights_as_sumstats(table), variants,
                          drop_ambiguous=drop_ambiguous)
            idx = np.asarray(h.var_index, dtype=np.int64)
            w = np.asarray(h.beta, dtype=float)
            if idx.size == 0:
                raise ValueError(
                    "no variant matched the LD reference; check the genome "
                    "build and that this score was fitted on this reference")
            if check_scale:
                detail = _scale_disagreement(table, h, af, sd, scale_atol)
                if detail is not None:
                    mismatched[label] = detail
                    raise ValueError(
                        f"this weight file does not describe the supplied "
                        f"reference: {detail}. Combining it would build a Gram "
                        "for a reference none of the scores were fitted on")
            # Squared-weight mass, as in ``harmonize_scoring_file``: losing 5%
            # of a score's variants means very different things depending on
            # whether they carried 0.1% or 40% of its weight.
            source = np.asarray(table.weight, dtype=float)
            total = float(source @ source)
            mass.append(float(w @ w) / total if total > 0.0 else 0.0)
            pairs.append((idx, w))
            ids.append(label)
            tables.append(
                reference_weight_table(idx, w, variants, af, sd))
            matched.append(int(idx.size))
        except Exception as exc:                      # noqa: BLE001
            if on_error == "raise":
                raise
            errors[label] = str(exc)
        if progress is not None:
            progress(i + 1, len(files), label)

    log = {"n_requested": len(files), "n_aligned": len(pairs),
           "n_failed": len(errors), "n_reference_variants": n_variants,
           "standardized": True, "scale_source": "ldpred3_weight_file",
           "scale_checked": bool(check_scale)}
    if matched:
        log["n_matched_median"] = int(np.median(matched))
        log["n_matched_min"] = int(min(matched))
        log["weight_mass_matched_min"] = float(min(mass))
        log["weight_mass_matched"] = {sid: value
                                      for sid, value in zip(ids, mass)}
    if errors:
        log["errors"] = errors
    if mismatched:
        log["reference_mismatch"] = mismatched
    empty = [sid for sid, table in zip(ids, tables)
             if not np.any(table["weight"])]
    if empty:
        # A score that contributes nothing still occupies a Gram column and a
        # penalty factor; naming it is better than letting it look selected-out.
        log["all_zero_scores"] = empty
    return pairs, ids, tables, log


def _stem_of(item):
    """A score id for a weight file: its filename stem, or the object's repr."""
    import os

    path = getattr(item, "path", None) or (item if isinstance(item, (str, os.PathLike)) else None)
    if path is None:
        return "score"
    return os.path.splitext(os.path.basename(os.fspath(path)))[0]


def _scale_disagreement(table, harmonized, af, sd, atol):
    """Describe how a weight file's own reference scale differs, or ``None``.

    A file written before ``AF_REF``/``SD_REF`` existed carries neither and
    cannot be checked; that is reported by the caller's ``scale_checked``
    rather than treated as a disagreement.
    """
    af_src = getattr(table, "af_ref", None)
    sd_src = getattr(table, "sd_ref", None)
    if af_src is None or sd_src is None:
        return None
    af_src = np.asarray(af_src, dtype=float)
    sd_src = np.asarray(sd_src, dtype=float)
    if af_src.size == 0 or not np.any(np.isfinite(af_src)):
        return None
    src = np.asarray(harmonized.src_index, dtype=np.int64)
    dst = np.asarray(harmonized.var_index, dtype=np.int64)
    flipped = np.asarray(harmonized.flipped, dtype=bool)
    # A swapped match counts the other allele, so its frequency is 1 - f.
    want_af = np.where(flipped, 1.0 - af_src[src], af_src[src])
    finite = np.isfinite(want_af) & np.isfinite(af[dst])
    if finite.any():
        gap = np.abs(want_af[finite] - af[dst][finite])
        worst = int(np.argmax(gap))
        if gap[worst] > atol:
            return (f"reference allele frequency differs by up to "
                    f"{float(gap[worst]):.4g} (file "
                    f"{float(want_af[finite][worst]):.4g} against reference "
                    f"{float(af[dst][finite][worst]):.4g})")
    finite_sd = np.isfinite(sd_src[src]) & np.isfinite(sd[dst])
    if finite_sd.any():
        gap = np.abs(sd_src[src][finite_sd] - sd[dst][finite_sd])
        worst = int(np.argmax(gap))
        if gap[worst] > atol:
            return (f"reference dosage SD differs by up to "
                    f"{float(gap[worst]):.4g}")
    return None


@dataclass
class Component:
    """One component score of a panel, and how its weights are scaled.

    ``kind="weights"`` is an LDpred3 weight file -- typically a fit made
    against this same reference -- whose own ``AF_REF``/``SD_REF`` can be
    checked against it. ``kind="scoring_file"`` is a PGS Catalog-format file
    counting raw alleles, which has no reference scale of its own: it must be
    converted with the HWE ``sqrt(2 f (1-f))`` approximation from the
    reference's frequencies, and that approximation is recorded per component
    rather than averaged into one panel-wide claim.

    The distinction is not cosmetic. A verified component's scale was measured
    on the reference; an approximated one ignores imputation uncertainty and
    departures from equilibrium, and a panel mixing the two folds into a file
    whose accuracy inherits the weaker assumption.
    """

    path: object
    kind: str = "weights"
    score_id: object = None

    def __post_init__(self):
        if self.kind not in ("weights", "scoring_file"):
            raise ValueError(
                f"component kind must be 'weights' or 'scoring_file', got "
                f"{self.kind!r}")


#: How each component kind's weights reached the standardized-genotype scale.
SCALE_SOURCES = {
    "weights": "ldpred3_weight_file",
    "scoring_file": "hwe_from_af",
}


def align_components(components, variants, *, af, sd, sd_source=None,
                     check_scale=True, drop_ambiguous=True,
                     on_error="raise", progress=None):
    """Align a mixed panel of component scores to one LD reference.

    Accepts :class:`Component` descriptors, or bare paths for the
    LDpred3-weight-file case. Every component is returned on the reference's
    index space and standardized-genotype scale, with ``log["scale_source"]``
    naming per score how it got there -- so a caller can show which components
    carry a verified reference scale and which are HWE-approximated instead of
    presenting them as equivalent.

    Returns ``(pairs, ids, tables, log)``, the same shape as
    :func:`align_weights_to_reference`.
    """
    if on_error not in ("raise", "skip"):
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")
    described = [item if isinstance(item, Component) else Component(item)
                 for item in components]
    if not described:
        raise ValueError("a panel needs at least one component score")
    variants = _as_variant_table(variants)
    af = np.asarray(af, dtype=float).ravel()
    sd = np.asarray(sd, dtype=float).ravel()

    pairs, ids, tables = [], [], []
    scale_source, per_component, errors = {}, {}, {}
    matched, mass = [], {}
    for position, item in enumerate(described):
        try:
            if item.kind == "weights":
                sub_pairs, sub_ids, sub_tables, sub_log = \
                    align_weights_to_reference(
                        [item.path], variants, af=af, sd=sd,
                        score_ids=None if item.score_id is None
                        else [item.score_id],
                        sd_source=sd_source, check_scale=check_scale,
                        drop_ambiguous=drop_ambiguous)
            else:
                # Raw allele counts: the only conversion available without
                # target dosages is the reference's HWE approximation, and it
                # is requested explicitly rather than defaulted.
                sub_pairs, sub_ids, sub_log = align_to_reference(
                    [item.path], variants, af=af, hwe_genotype_sd=True,
                    drop_ambiguous=drop_ambiguous)
                if not sub_log.get("standardized"):
                    raise ValueError(
                        "the scoring file was not converted to the "
                        "standardized-genotype scale; its weights cannot "
                        "enter a Gram built on that scale")
                if item.score_id is not None:
                    sub_ids = [str(item.score_id)]
                sub_tables = [
                    reference_weight_table(index, weight, variants, af, sd)
                    for index, weight in sub_pairs]
            if not sub_pairs:
                raise ValueError(
                    sub_log.get("errors")
                    or "no variant aligned to this LD reference")
            pairs.extend(sub_pairs)
            ids.extend(str(name) for name in sub_ids)
            tables.extend(sub_tables)
            sub_mass = sub_log.get("weight_mass_matched") or {}
            for name in sub_ids:
                key = str(name)
                scale_source[key] = SCALE_SOURCES[item.kind]
                matched.append(int(sub_log.get("n_matched_median") or 0))
                # Squared-weight mass is only computable where the source's
                # own weights are known; the Catalog path does not report it.
                mass[key] = (float(sub_mass[key]) if key in sub_mass
                             else None)
                per_component[key] = {
                    "kind": item.kind,
                    "scale_source": SCALE_SOURCES[item.kind],
                    "scale_verified": item.kind == "weights" and check_scale,
                    "n_matched": int(sub_log.get("n_matched_median") or 0),
                    "weight_mass": mass[key],
                }
        except Exception as exc:                              # noqa: BLE001
            label = str(item.score_id or item.path)
            if on_error == "raise":
                raise ValueError(f"{label}: {exc}") from exc
            errors[label] = str(exc)
        if progress is not None:
            progress(position + 1, len(described),
                     str(item.score_id or item.path))

    if len(set(ids)) != len(ids):
        raise ValueError(
            "component score ids must be unique across the whole panel; "
            f"got {ids}")
    approximated = sorted(name for name, source in scale_source.items()
                          if source == "hwe_from_af")
    log = {"n_requested": len(described), "n_aligned": len(pairs),
           "n_failed": len(errors),
           "n_reference_variants": int(af.size),
           "standardized": True,
           # Whether the reference-scale check was requested. It can only
           # ever apply to the weight-file components; a scoring file has no
           # reference scale of its own to check.
           "scale_checked": bool(check_scale),
           "scale_source": scale_source,
           "components": per_component,
           "n_hwe_approximated": len(approximated)}
    if matched:
        log["n_matched_median"] = int(np.median(matched))
        log["n_matched_min"] = int(min(matched))
    if mass:
        log["weight_mass_matched"] = dict(mass)
        known = [value for value in mass.values() if value is not None]
        log["weight_mass_matched_min"] = float(min(known)) if known else None
    empty = [name for name, table in zip(ids, tables)
             if not np.any(table["weight"])]
    if empty:
        log["all_zero_scores"] = empty
    if errors:
        log["errors"] = errors
    if approximated:
        log["warning"] = (
            f"{len(approximated)} component(s) were converted to the "
            "standardized-genotype scale with the reference's HWE "
            "sqrt(2 f (1-f)) approximation, which ignores imputation "
            "uncertainty and departures from equilibrium: "
            + ", ".join(approximated))
    return pairs, ids, tables, log
