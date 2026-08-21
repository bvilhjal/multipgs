"""Read PGS Catalog scoring files and align them to a target genotype.

A `PGS Catalog <https://www.pgscatalog.org/>`_ scoring file is a TSV with a
``#key=value`` metadata header::

    ###PGS CATALOG SCORING FILE
    #format_version=2.0
    #pgs_id=PGS000001
    #trait_reported=Breast cancer
    #genome_build=GRCh37
    #weight_type=beta
    rsID       chr_name  chr_position  effect_allele  other_allele  effect_weight
    rs2016394  2         19320803      G              A             -0.045

Harmonized (``*_hmPOS_*``) files add ``hm_rsID``/``hm_chr``/``hm_pos`` on a
stated build; those columns are preferred when present, because they are the
only ones guaranteed to be on the build the filename advertises.

Three details that quietly ruin scores if ignored, and are handled here:

* **Odds ratios.** ``weight_type=OR`` (or ``HR``) means the column is a *ratio*,
  not a log-effect. Summing ratios is meaningless; they are log-transformed on
  read, and the transformation is recorded in the log.
* **Non-additive rows.** ``is_dominant``/``is_recessive``/``is_haplotype``/
  ``is_diplotype``/``is_interaction`` mark variants that an additive dosage
  score cannot represent. They are dropped and counted rather than silently
  treated as additive.
* **Allele orientation.** Alignment to the target's counted allele goes through
  :func:`ldpred3.harmonize`, so allele swaps flip the sign, strand flips are
  resolved where they can be, and palindromic A/T and C/G variants are dropped
  by default rather than guessed at.

Catalog weights are per **allele count**, so a catalog score is
``sum_j w_j * dosage_ij`` on raw dosages — not on standardized genotypes, which
is the convention LDpred3's own weights use. :mod:`multipgs.panel` keeps the two
apart; if you score a catalog file yourself, pass ``standardize=False``.
"""

from __future__ import annotations

import gzip
import io
import os
from dataclasses import dataclass, field

import numpy as np


__all__ = ["read_scoring_file", "ScoringFile", "harmonize_scoring_file",
           "scoring_file_id"]

# Column name -> canonical field. Harmonized names win over the originals.
_COLUMNS = {
    "id": ("hm_rsid", "rsid", "rsids", "snpid", "variant_id"),
    "chrom": ("hm_chr", "chr_name", "chromosome", "chr"),
    "pos": ("hm_pos", "chr_position", "position", "bp"),
    "ea": ("effect_allele", "a1"),
    "oa": ("other_allele", "hm_inferotherallele", "a2"),
    "weight": ("effect_weight", "weight", "beta"),
}
_NON_ADDITIVE = ("is_dominant", "is_recessive", "is_haplotype", "is_diplotype",
                 "is_interaction")
_RATIO_WEIGHTS = ("or", "hr", "odds_ratio", "hazard_ratio")
_TRUE = ("true", "t", "yes", "y", "1")
_MISSING_ALLELE = ("", ".", "NA", "N/A", "NULL")


@dataclass
class ScoringFile:
    """One parsed scoring file.

    ``weight`` counts ``ea``. ``meta`` holds the ``#key=value`` header verbatim;
    ``log`` records what parsing dropped or transformed.
    """

    id: np.ndarray
    chrom: np.ndarray
    pos: np.ndarray
    ea: np.ndarray
    oa: np.ndarray
    weight: np.ndarray
    meta: dict = field(default_factory=dict)
    log: dict = field(default_factory=dict)
    path: str = ""

    def __len__(self):
        return int(self.weight.size)

    @property
    def pgs_id(self):
        """``pgs_id`` from the header, falling back to the file name."""
        return self.meta.get("pgs_id") or scoring_file_id(self.path)

    @property
    def trait(self):
        return self.meta.get("trait_reported", "")

    @property
    def genome_build(self):
        """Build of the coordinates actually used (harmonized build if any)."""
        return self.log.get("build_used") or self.meta.get("genome_build", "")

    def __str__(self):
        return (f"{self.pgs_id}: {len(self)} variants"
                + (f", {self.trait}" if self.trait else "")
                + (f", build {self.genome_build}" if self.genome_build else ""))


def scoring_file_id(path):
    """``PGS000123`` from ``.../PGS000123_hmPOS_GRCh37.txt.gz``."""
    base = os.path.basename(str(path))
    for suffix in (".txt.gz", ".tsv.gz", ".txt", ".tsv", ".gz"):
        if base.endswith(suffix):
            base = base[: -len(suffix)]
            break
    return base.split("_")[0] if base.startswith("PGS") else base


def _open_text(path):
    path = str(path)
    if path.endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8",
                                errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def _pick(header, names):
    lower = {name.strip().lower(): i for i, name in enumerate(header)}
    for want in names:
        if want in lower:
            return lower[want], want
    return None, None


def read_scoring_file(path, *, prefer_harmonized=True, drop_non_additive=True):
    """Parse a PGS Catalog scoring file (plain or gzipped).

    Parameters
    ----------
    path : str
    prefer_harmonized : bool
        Use ``hm_rsID``/``hm_chr``/``hm_pos`` when the file has them. Turn this
        off only if you mean to match on the author-submitted coordinates.
    drop_non_additive : bool
        Drop dominant/recessive/haplotype/interaction rows. With ``False`` they
        are kept and treated as additive, which is wrong for every one of them;
        the option exists to reproduce a pipeline that did that, not because it
        is defensible.

    Returns
    -------
    ScoringFile
    """
    meta, header, rows = {}, None, []
    with _open_text(path) as fh:
        for line in fh:
            if line.startswith("#"):
                stripped = line.lstrip("#").strip()
                if "=" in stripped:
                    key, _, value = stripped.partition("=")
                    meta[key.strip().lower()] = value.strip()
                continue
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            fields = line.split("\t") if "\t" in line else line.split()
            if header is None:
                header = fields
                continue
            rows.append(fields)

    if header is None:
        raise ValueError(f"{path}: no column header found (only comment lines)")

    idx = {}
    used = {}
    for field_name, names in _COLUMNS.items():
        candidates = names if prefer_harmonized else \
            tuple(n for n in names if not n.startswith("hm_")) + names
        i, chosen = _pick(header, candidates)
        idx[field_name] = i
        used[field_name] = chosen
    # Harmonized files can carry both columns, with ``other_allele`` populated
    # only on some rows. Choosing one column for the whole file silently loses
    # the row-wise inference supplied by the Catalog.
    inferred_oa_idx, inferred_oa_name = _pick(
        header, ("hm_inferotherallele",))
    if idx["weight"] is None:
        raise ValueError(f"{path}: no effect_weight column in {header}")
    if idx["ea"] is None:
        raise ValueError(f"{path}: no effect_allele column in {header}")
    if idx["id"] is None and (idx["chrom"] is None or idx["pos"] is None):
        raise ValueError(f"{path}: needs either an rsID column or both "
                         f"chromosome and position, found {header}")

    flag_cols = [i for i, name in enumerate(header)
                 if name.strip().lower() in _NON_ADDITIVE]
    n_read = len(rows)
    n_flagged = n_bad_weight = n_inferred_oa = 0
    ids, chroms, poss, eas, oas, ws = [], [], [], [], [], []
    width = len(header)
    for fields in rows:
        if len(fields) < width:
            fields = fields + [""] * (width - len(fields))
        if flag_cols and any(fields[i].strip().lower() in _TRUE
                             for i in flag_cols):
            n_flagged += 1
            if drop_non_additive:
                continue
        try:
            w = float(fields[idx["weight"]])
        except ValueError:
            n_bad_weight += 1
            continue
        if not np.isfinite(w):
            n_bad_weight += 1
            continue
        ids.append(fields[idx["id"]].strip() if idx["id"] is not None else "")
        chroms.append(fields[idx["chrom"]].strip()
                      if idx["chrom"] is not None else "")
        poss.append(fields[idx["pos"]].strip()
                    if idx["pos"] is not None else "0")
        eas.append(fields[idx["ea"]].strip().upper())
        oa = (fields[idx["oa"]].strip().upper()
              if idx["oa"] is not None else "")
        if (oa in _MISSING_ALLELE and inferred_oa_idx is not None
                and inferred_oa_idx != idx["oa"]):
            inferred = fields[inferred_oa_idx].strip().upper()
            if inferred not in _MISSING_ALLELE:
                oa = inferred
                n_inferred_oa += 1
        oas.append(oa)
        ws.append(w)

    pos = np.array([int(p) if p.lstrip("-").isdigit() else 0 for p in poss],
                   dtype=np.int64)
    weight = np.array(ws, dtype=float)

    # Coordinate provenance must be judged across every matched field, not
    # position alone: a file pairing hm_pos with an original-build chr_name
    # would otherwise inherit hmpos_build metadata it does not satisfy. The
    # Catalog ships hm_* columns all-or-nothing, so disagreement signals a
    # malformed or hand-edited header.
    coordinate_used = [str(used[name]).startswith("hm_")
                       for name in ("id", "chrom", "pos") if used[name]]
    harmonized_columns = bool(coordinate_used) and all(coordinate_used)
    log = {"n_rows": n_read, "n_kept": int(weight.size),
           "n_non_additive": n_flagged, "n_unparsable_weight": n_bad_weight,
           "n_inferred_other_allele": n_inferred_oa,
           "columns_used": {k: v for k, v in used.items() if v},
           "harmonized_columns": harmonized_columns}
    if coordinate_used and not harmonized_columns:
        log["mixed_coordinate_warning"] = (
            "some but not all of the rsID/chromosome/position columns are "
            "harmonized; coordinates may mix genome builds")
    if log["harmonized_columns"]:
        log["build_used"] = (meta.get("hmpos_build")
                             or meta.get("harmonized_build")
                             or meta.get("genome_build", ""))
    if n_inferred_oa:
        log["columns_used"]["oa_fallback"] = inferred_oa_name

    weight_type = meta.get("weight_type", "").strip().lower()
    if weight_type in _RATIO_WEIGHTS:
        if np.any(weight <= 0):
            raise ValueError(
                f"{path}: weight_type={weight_type!r} says these are ratios, "
                f"but {int(np.sum(weight <= 0))} are non-positive, so they "
                f"cannot be. Fix the header or the file.")
        weight = np.log(weight)
        log["log_transformed"] = True

    return ScoringFile(
        id=np.array(ids, dtype=object), chrom=np.array(chroms, dtype=object),
        pos=pos, ea=np.array(eas, dtype=object),
        oa=np.array(oas, dtype=object), weight=weight, meta=meta, log=log,
        path=str(path))


def harmonize_scoring_file(scoring, variants, *, drop_ambiguous=True):
    """Align a :class:`ScoringFile` to a genotype variant table.

    Delegates the allele logic to :func:`ldpred3.interop.harmonize` — matching
    by rsID then ``chrom:pos``, sign-flipping swapped alleles, resolving strand
    flips where the alleles allow it, and dropping palindromic variants.

    Returns ``(var_index, weight, log)``: positions into ``variants`` and the
    weights aligned to that table's A1 allele.
    """
    from ldpred3.interop import Sumstats, harmonize

    m = len(scoring)
    if m == 0:
        return (np.zeros(0, dtype=np.int64), np.zeros(0),
                {"n_weights": 0, "n_matched": 0, "weight_mass_matched": 0.0})
    ss = Sumstats(id=scoring.id, chrom=scoring.chrom, pos=scoring.pos,
                  ea=scoring.ea, oa=scoring.oa, beta=scoring.weight,
                  se=np.ones(m), n_eff=np.ones(m), eaf=np.full(m, np.nan),
                  info=np.full(m, np.nan))
    h = harmonize(ss, variants, drop_ambiguous=drop_ambiguous)

    total = float(scoring.weight @ scoring.weight)
    kept = np.asarray(scoring.weight)[h.src_index] if len(h) else np.zeros(0)
    log = dict(h.log)
    log.update({
        "n_weights": m, "n_matched": int(len(h)),
        # Squared-weight mass is the better completeness number than a bare
        # count: losing 5% of variants matters very differently depending on
        # whether they carried 0.1% or 40% of the score's weight. It assumes
        # linkage equilibrium, so it is a proxy, not an accuracy loss.
        "weight_mass_matched": (float(kept @ kept) / total if total > 0
                                else 0.0),
    })
    return np.asarray(h.var_index, dtype=np.int64), np.asarray(h.beta), log
