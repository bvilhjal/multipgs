"""Align Catalog scoring files to an LD-reference variant table."""

from __future__ import annotations

from collections.abc import Mapping

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
