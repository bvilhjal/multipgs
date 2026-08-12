#!/usr/bin/env python3
"""Meta-PGS weighting rules on a real same-trait panel, with observed sharing.

``benchmarks/meta_rules.py`` compares :func:`multipgs.meta_pgs`'s rules in
simulation, and everything the discovery studies share there is one stylized
knob, ``shared_error_correlation``. ``benchmarks/README.md`` is careful about
what that knob is not: the simulator models neither overlapping people nor
discovery-GWAS estimation, so it cannot say what the rules do when the shared
structure is the real thing.

On a real panel the shared structure is not a knob. For ``K`` PGS Catalog
scores of **one** trait it is the off-diagonal of the score covariance
``W' D W`` on a real LD reference, and it is directly observable. That matrix is
precisely what ``method="decorrelated"`` inverts in order to discount a score
for information it already shares with the panel, so this benchmark replaces
the knob with the measurement:

* align the scoring files to a real LD reference with
  :func:`multipgs.align_to_reference`, form ``G = W_ld' D W_ld`` with
  :func:`multipgs.score_moments`, and report the distribution of the implied
  ``K x K`` score correlations — the real shared-information structure — over
  the pairs the reference can support, and how many it cannot;
* score ``sqrt_n_eff``, ``expected_r2`` and both ``decorrelated`` variants
  against a real target GWAS with :func:`multipgs.evaluate_sumstat`, next to an
  equal-weight baseline and two single-score baselines;
* cross-tabulate the observed correlations against
  :func:`multipgs.cohort_overlap` on the scores' declared discovery cohorts.
  Whether a declaration in the Catalog predicts the correlation the LD
  reference actually shows is a finding in its own right, and it is recorded
  per score pair.

**What this can establish.** The magnitude and spread of real same-trait score
correlation on a real European LD reference, over the score pairs that
reference can actually support; how much of it declared cohort metadata
anticipates; and the ordering of the weighting rules on one target GWAS, at the
panel and reference supplied. Nothing about the rules is assumed here —
``meta_pgs`` itself computes the weights.

**What it cannot establish.**

* *A rule ranking.* One panel, one trait, one LD reference and one target GWAS
  give one draw. The simulation benchmark varies its knob over seeds; this one
  cannot, because the real quantity has no seed. Differences smaller than the
  gap between the two single-score baselines should not be read as a ranking —
  and check first that ``best_single_max_n_eff`` is not a coin toss:
  ``n_scores_tied_max_n_eff`` counts how many scores share the largest
  ``n_eff``, and above one that baseline is decided by filename order, not by
  the panel.
* *That the rules are independent of the LD reference.* The regime label below
  is about the target **GWAS**. The reference is a second input and it is used
  twice: ``decorrelated`` inverts the correlation implied by ``G``, every rule
  standardizes by ``sqrt(diag(G))``, and the R² denominator is ``beta' G beta``
  on that same ``G``. So the weights are fitted on the moments the accuracy is
  measured against — not the phenotype moments, which is why no regime is
  violated, but a direction in which this ``G`` understates score variance
  raises R² for the two ``decorrelated`` rows and nothing here would show it.
  A second reference, not a second GWAS, is what would test that.
* *That the numbers are regime A.* ``--regime`` is required and never guessed.
  Meta-PGS weights touch no phenotype and are not tuned, so a target GWAS
  untouched by every score's discovery gives a genuine **regime A** number per
  rule. That precondition is about *people*, not about model selection: if a
  score's discovery cohorts contributed to the target GWAS, the score is partly
  fitted to the individuals it is being scored on, the label is not defensible
  and inflation is invisible in the number. Declare the target's cohorts with
  ``--gwas-cohorts`` and this reports how many panel scores name one of them;
  a non-zero count with ``--regime A`` is recorded as ``regime_a_contested``
  and warned about, not silently accepted. Declared cohorts are a lower bound
  (see :func:`multipgs.cohort_overlap`), so a zero count is weak evidence:
  check the target GWAS's own cohort list against each score's publication
  before reporting an A. Omitting ``--gwas-cohorts`` under ``--regime A`` does
  not pass that check, it skips it, and ``regime_a_cohort_check`` says which of
  the two happened — ``not_checked`` is not ``no_declared_overlap``.
  ``best_single_oracle`` is chosen by its accuracy on the evaluation moments
  themselves and is therefore labelled **regime C** whatever ``--regime`` says.
* *That h2 and polygenicity are estimated.* Catalog scores arrive as weights
  with no model behind them, so :func:`multipgs.architectures_from_panel`
  returns ``nan`` for all of them and ``expected_r2`` cannot be derived from
  the panel. ``--h2`` and ``--m-causal`` are therefore *declared assumptions*
  fed to :func:`multipgs.daetwyler_r2`; they are recorded in the provenance and
  the two ``expected_r2`` rules are simply not run without them. The rules that
  use them are conditional on those two numbers being right for this trait.
* *Anything about a target cohort.* Every accuracy here is a summary-statistic
  plug-in against an external LD reference, in the reference's ancestry. It is
  not held-out R² in individuals, and none of it survives a change of ancestry.
  It is also on whatever scale the target GWAS's ``z`` is on: for a case/control
  study that is the observed scale at the study's own case fraction, not
  liability.
* *That a low number means a weak score.* Three things attenuate a score here
  and none of them is the score: variants of the score the reference does not
  carry (``n_variants_aligned``), variants the reference carries but the target
  GWAS does not (``gwas_weight_coverage``, since an absent ``z`` enters ``c``
  as an exact zero), and a panel restricted with ``--chrom``. All three are
  reported per score, the run warns when any of them bites, and a score the
  reference leaves with no variance at all is excluded from every reported
  correlation rather than recorded as uncorrelated. Compare rules within one
  run; never compare a score's number across runs with different coverage.

**The two weight matrices.** ``G`` uses the LD reference's genotype SD and
``c`` uses the target GWAS's, which is why :func:`multipgs.score_moments` takes
them separately. Both are the HWE approximation ``sqrt(2 f (1-f))``: from the
reference's own allele frequencies for ``W_ld``, and from the GWAS's frequency
column for ``W_gwas``. Public GWAS report a reference-panel frequency rather
than their own, so ``W_gwas`` is an approximation twice over, and with a format
carrying no frequency column at all (``--gwas-columns af=``) the two matrices
coincide — which asserts that the GWAS and the reference have the same dosage
SD. ``gwas_af_column_used`` records which of the two happened.

**Scale, and why MSE is nearly redundant.** ``MetaPGS`` weights are defined only
up to a positive constant, so ``R2`` is comparable across rules and MSE is not
until a scale is fixed. Each combination is rescaled to unit variance *under the
LD reference* — what a deployed PGS is standardized to, and computed without
touching the target GWAS. After that rescaling ``beta' G beta = 1``, so
``MSE = var_y + 1 - 2 beta'c`` while ``R2 = (beta'c)^2 / var_y`` (and ``var_y``
is 1 throughout, the standardized scale ``z`` is on): MSE is strictly
decreasing in ``beta'c`` and carries exactly one thing R² does not, the *sign*
of ``beta'c``, which is why ``beta_c`` is a column of its own. A combination anticorrelated with the trait scores a respectable
R² and an MSE above ``var_y``. Read ``r2``, and read ``mse`` only to check it is
below ``var_y``.

Full-scale run (UK Biobank-derived height panel against the pre-UK-Biobank
GIANT 2014 height GWAS, which is the pairing that makes the regime A label
arguable in the first place)::

    python benchmarks/real_meta_rules.py \\
        --ld /path/to/ldpred3_ldref_hm3.npz \\
        --scores pgs_catalog_height/ \\
        --metadata pgs_catalog_height_metadata.tsv \\
        --gwas /path/to/GIANT_HEIGHT_2014.txt.gz --gwas-format giant_height \\
        --regime A --h2 0.5 --m-causal 12000 \\
        --gwas-cohorts ARIC,FHS,EGCUT,ERF,HealthABC,InCHIANTI,B58C,ALSPAC

The scoring files, their metadata table (from
:func:`multipgs.write_score_metadata`), the LD reference and the GWAS are all
supplied: nothing is downloaded here, and the sha256 of every large input goes
into the provenance. ``--chrom`` restricts the whole computation to one
chromosome, which is how this is smoke-tested; a one-chromosome panel estimates
the same correlations from a twentieth of the variants and its accuracies are
not comparable with a genome-wide run.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Measure the checkout beside this artifact even when no editable install is
# active. This also keeps version provenance tied to the code being exercised.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import multipgs
from benchmarks._provenance import benchmark_identity
from multipgs import (ScoreRecord, align_to_reference, cohort_overlap,
                      daetwyler_r2, evaluate_sumstat, meta_pgs, score_moments)


# Column names of the public GWAS this was developed against. A study is a
# three-line entry rather than eight command-line flags because the mapping is
# a property of the file, not of the run; --gwas-columns overrides any field for
# a study not listed here.
_GWAS_FORMATS = {
    "giant_height": {"id": "MarkerName", "ea": "Allele1", "oa": "Allele2",
                     "af": "Freq.Allele1.HapMapCEU", "beta": "b", "se": "SE",
                     "n": "N"},
    "giant_bmi": {"id": "SNP", "ea": "A1", "oa": "A2", "af": "Freq1.Hapmap",
                  "beta": "b", "se": "se", "n": "N"},
    "gwasmc": {"id": "rsid", "ea": "A1", "oa": "A2",
               "af": "Freq.A1.1000G.EUR", "beta": "beta", "se": "se",
               "n": "N"},
    "cardiogram": {"id": "markername", "ea": "effect_allele",
                   "oa": "noneffect_allele", "af": "effect_allele_freq",
                   "beta": "beta", "se": "se_dgc", "n": None},
}

RULE_FIELDS = ("rule", "regime", "r2", "beta_c", "mse", "n_nonzero_weights",
               "n_negative_weights", "condition_number",
               "combined_score_sd_reference", "discarded_c_null_fraction",
               "top_score", "top_weight")


def _version(name):
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


def _open_text(path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r", encoding="utf-8", errors="replace")


def _reference(path, chrom):
    """Load an ldpred3 LD cache, optionally restricted to one chromosome.

    ldpred3 blocks tile ``0..m-1`` contiguously and never straddle a
    chromosome, and the reference is in chromosome order, so one chromosome is
    a contiguous run of both variants and blocks. Re-basing the surviving block
    indices to zero is therefore the whole of the restriction, and it keeps the
    contract :func:`multipgs.score_gram` validates.
    """
    from ldpred3 import load_ld_blocks
    from ldpred3.genotype_io import VariantTable

    started = time.perf_counter()
    blocks, variant_ids = load_ld_blocks(str(path))
    payload = np.load(str(path), allow_pickle=True)
    ids = np.asarray(payload["ids"], dtype=object)
    chroms = np.asarray(payload["chrom"], dtype=object)
    positions = np.asarray(payload["pos"], dtype=np.int64)
    a1 = np.asarray(payload["counted_allele"], dtype=object)
    a2 = np.asarray(payload["other_allele"], dtype=object)
    af = np.asarray(payload["reference_af"], dtype=float)
    total = int(len(variant_ids))

    lo, hi = 0, total
    if chrom is not None:
        wanted = np.flatnonzero(np.asarray(chroms, dtype=str) == str(chrom))
        if wanted.size == 0:
            raise SystemExit(f"chromosome {chrom!r} is not in {path}")
        lo, hi = int(wanted[0]), int(wanted[-1]) + 1
        if wanted.size != hi - lo:
            raise SystemExit(f"chromosome {chrom!r} is not a contiguous run of "
                             "reference variants; this restriction assumes the "
                             "reference is in chromosome order")
        kept = []
        for corr, idx in blocks:
            start, stop = int(idx[0]), int(idx[-1]) + 1
            if start >= lo and stop <= hi:
                kept.append((corr, np.arange(start - lo, stop - lo)))
        blocks = kept
        ids, chroms, positions = ids[lo:hi], chroms[lo:hi], positions[lo:hi]
        a1, a2, af = a1[lo:hi], a2[lo:hi], af[lo:hi]

    variants = VariantTable(chrom=np.asarray(chroms, dtype=object),
                            id=np.asarray(ids, dtype=object),
                            cm=np.zeros(hi - lo),
                            pos=positions,
                            a1=np.asarray(a1, dtype=object),
                            a2=np.asarray(a2, dtype=object))
    meta = {"n_variants_total": total, "n_variants": hi - lo,
            "n_blocks": len(blocks), "af": af,
            "n_ref": (int(np.asarray(payload["n_ref"]).ravel()[0])
                      if "n_ref" in payload else None),
            "load_seconds": time.perf_counter() - started}
    return blocks, variants, meta


def _read_metadata(path):
    """Read a :func:`multipgs.write_score_metadata` table into ScoreRecords.

    Only ``N_EFF``, ``COHORTS`` and ``TRAIT`` are used, but they are put back
    into real :class:`multipgs.ScoreRecord` objects so that
    :func:`multipgs.cohort_overlap` is exercised as shipped rather than
    reimplemented against a dict.
    """
    records = {}
    with _open_text(path) as handle:
        header = handle.readline().rstrip("\n").split("\t")
        if not header or header[0] != "SCORE":
            raise SystemExit(f"{path} is not a multipgs score metadata table "
                             "(its first column must be SCORE)")
        for line in handle:
            if not line.strip():
                continue
            fields = line.rstrip("\n").split("\t")
            row = dict(zip(header, fields))
            cohorts = [c for c in (row.get("COHORTS") or "").split(",") if c]
            efo = [e for e in (row.get("EFO") or "").split(",") if e]
            n_eff = row.get("N_EFF", "NA")
            records[row["SCORE"]] = ScoreRecord(
                pgs_id=row["SCORE"],
                trait_reported=row.get("TRAIT", ""),
                efo_ids=tuple(sorted(efo)),
                n_eff=float("nan") if n_eff in ("", "NA") else float(n_eff),
                cohorts=frozenset(cohorts))
    if not records:
        raise SystemExit(f"{path} contains no scores")
    return records


def _read_gwas(path, spec, keep_ids, n_eff_default):
    """Stream one GWAS, keeping only rows whose rsID the reference carries.

    The rsID filter runs before any parsing of alleles or effects, because the
    files this is aimed at are millions of rows and, restricted to a HapMap3+
    reference or a single chromosome, almost all of them are discarded. The
    surviving rows are handed to ``ldpred3.harmonize``, which owns the allele
    logic; duplicating it here is exactly how effect signs get silently
    flipped.
    """
    ids, ea, oa, beta, se, n_eff, af = [], [], [], [], [], [], []
    n_rows = n_unparsable = n_no_id = 0
    with _open_text(path) as handle:
        head = handle.readline().rstrip("\n")
        sep = "\t" if "\t" in head else None
        header = head.split(sep) if sep else head.split()
        columns = {name: i for i, name in enumerate(header)}
        missing = [key for key, name in spec.items()
                   if name is not None and name not in columns]
        if missing:
            raise SystemExit(
                f"{path} has no column(s) "
                + ", ".join(f"{spec[k]!r} (for {k})" for k in missing)
                + f"; its header is {header}")
        c_id = columns[spec["id"]]
        c_ea, c_oa = columns[spec["ea"]], columns[spec["oa"]]
        c_beta, c_se = columns[spec["beta"]], columns[spec["se"]]
        c_n = columns[spec["n"]] if spec.get("n") else None
        c_af = columns[spec["af"]] if spec.get("af") else None
        width = max(c_id, c_ea, c_oa, c_beta, c_se,
                    c_n if c_n is not None else 0,
                    c_af if c_af is not None else 0) + 1
        for line in handle:
            n_rows += 1
            fields = line.split(sep) if sep else line.split()
            if len(fields) <= c_id:
                # Too short to even read an rsID. Counted apart from
                # n_unparsable, which must mean "matched the reference and was
                # then lost" — that is the count that costs coverage.
                n_no_id += 1
                continue
            rsid = fields[c_id]
            if rsid not in keep_ids:
                continue
            if len(fields) < width:
                n_unparsable += 1
                continue
            try:
                b = float(fields[c_beta])
                s = float(fields[c_se])
                n = float(fields[c_n]) if c_n is not None else n_eff_default
                f = float(fields[c_af]) if c_af is not None else float("nan")
            except ValueError:
                n_unparsable += 1
                continue
            ids.append(rsid)
            ea.append(fields[c_ea].strip().upper())
            oa.append(fields[c_oa].strip().upper())
            beta.append(b)
            se.append(s)
            n_eff.append(n)
            af.append(f)
    log = {"n_rows": n_rows, "n_kept_by_id": len(ids),
           "n_unparsable": n_unparsable, "n_no_id_column": n_no_id,
           "has_af_column": c_af is not None}
    return (np.asarray(ids, dtype=object), np.asarray(ea, dtype=object),
            np.asarray(oa, dtype=object), np.asarray(beta, dtype=float),
            np.asarray(se, dtype=float), np.asarray(n_eff, dtype=float),
            np.asarray(af, dtype=float), log)


def _harmonize_gwas(raw, variants, reference_af):
    """Standardized marginal effects, GWAS allele frequency, and coverage.

    ``covered`` marks the reference variants the GWAS actually supplied a usable
    effect for. Everywhere else ``z`` stays 0, which enters ``c = W_gwas' z`` as
    an assertion that the variant is uncorrelated with the trait rather than as
    missing data — so a score supported mostly on uncovered variants comes out
    looking weak. Nothing downstream can distinguish the two; only this mask
    can, which is why it is returned and reported per score.
    """
    from ldpred3 import standardize_betas
    from ldpred3.harmonize import harmonize
    from ldpred3.sumstats import Sumstats

    ids, ea, oa, beta, se, n_eff, af, log = raw
    m = len(np.asarray(variants.id))
    if ids.size == 0:
        raise SystemExit("no GWAS variant matched the reference by rsID; check "
                         "the genome build and the --gwas-format columns")
    stats = Sumstats(id=ids, chrom=np.full(ids.size, "", dtype=object),
                     pos=np.zeros(ids.size, dtype=np.int64), ea=ea, oa=oa,
                     beta=beta, se=se, n_eff=n_eff,
                     eaf=af, info=np.full(ids.size, np.nan))
    h = harmonize(stats, variants, drop_ambiguous=True)
    if h.var_index.size == 0:
        raise SystemExit("every GWAS variant was dropped by harmonisation")

    # multipgs requires the standardized (allele-correlation) scale; a raw
    # per-allele beta here would silently produce wrong weights, and nothing
    # downstream can detect it.
    z_std, _ = standardize_betas(h.beta, h.se, h.n_eff)
    z = np.zeros(m)
    z[h.var_index] = z_std
    covered = np.zeros(m, dtype=bool)
    covered[h.var_index] = True

    # An allele swap makes the reference's counted allele the other one, so its
    # frequency is 1 - eaf; a strand flip leaves the frequency alone, which is
    # why `flipped` and not `n_strand_flipped` is the right condition.
    gwas_af = np.array(reference_af, dtype=float)
    matched_af = af[h.src_index]
    matched_af = np.where(h.flipped, 1.0 - matched_af, matched_af)
    usable = np.isfinite(matched_af) & (matched_af > 0.0) & (matched_af < 1.0)
    gwas_af[h.var_index[usable]] = matched_af[usable]

    out = dict(log)
    out.update({f"harmonize_{k}": v for k, v in h.log.items()})
    out["n_gwas_af_used"] = int(np.count_nonzero(usable))
    out["n_standardized"] = int(h.var_index.size)
    out["reference_coverage"] = float(np.mean(covered)) if m else float("nan")
    return z, gwas_af, covered, out


def _surrogate_cohort(correlation):
    """A deterministic ``(K+1) x K`` matrix whose correlation is exactly ``C``.

    :func:`multipgs.meta_pgs` reads ``C`` from individual-level scores, and
    there are no individuals here — only the correlation the LD reference
    implies. Rather than reimplement the ``C^{-1} rho`` solve and benchmark a
    copy of the rule instead of the rule, this hands ``meta_pgs`` a surrogate
    cohort carrying exactly that correlation: with ``Q'Q = n I`` and columns
    summing to zero, ``Z = Q L'`` has sample correlation ``L L' = C``. ``Q`` is
    a DCT-II basis, so the construction is exact and involves no sampling
    noise; the residual is reported as ``surrogate_max_correlation_error``.

    A real reference's correlation can be slightly indefinite, and a factor
    exists only for a PSD matrix, so negative eigenvalues are clipped and the
    diagonal renormalized. The clipped mass is returned and must be small for
    the surrogate to represent the panel that was measured.
    """
    k = correlation.shape[0]
    values, vectors = np.linalg.eigh(0.5 * (correlation + correlation.T))
    clipped = float(np.abs(np.sum(values[values < 0.0])))
    factor = vectors * np.sqrt(np.maximum(values, 0.0))
    diagonal = np.sqrt(np.maximum(np.einsum("ij,ij->i", factor, factor),
                                  np.finfo(float).tiny))
    factor = factor / diagonal[:, None]
    n = k + 1
    rows = np.arange(n)[:, None]
    cols = np.arange(1, k + 1)[None, :]
    basis = np.sqrt(2.0) * np.cos(np.pi * cols * (rows + 0.5) / n)
    return basis @ factor.T, clipped, factor @ factor.T


def _single_score(index, score_sd):
    """Raw-score weights selecting one component score and nothing else."""
    beta = np.zeros(score_sd.size)
    beta[index] = 1.0 / max(score_sd[index], np.finfo(float).tiny)
    return beta


def _ldsc_h2_from_gwas(z, covered, ld_scores, n_eff, m_reference):
    """SNP heritability of the target GWAS, by LD Score regression.

    ``--h2`` otherwise arrives as an assertion, and it propagates: it sets
    every score's Daetwyler ``expected_r2``, which is what the ``expected_r2``
    and ``decorrelated_expected_r2`` rules weight by. A declared heritability
    is therefore a declared ranking, and estimating it from the same GWAS the
    rules are scored against at least makes it a measurement.

    ``ldpred3.ldsc_h2`` wants chi-square. These effects are on the standardized
    scale, where the exact relation is ``z^2 = N b^2 / (1 - b^2)``, not the
    weak-effect approximation ``N b^2`` -- the difference matters precisely at
    the large-effect variants that carry the most leverage in this regression.

    Only variants the GWAS actually covered enter. Everywhere else ``z`` is
    zero by construction, and feeding those in as genuine null chi-squares
    would drag the intercept down and the slope with it.
    """
    from ldpred3 import ldsc_h2

    covered = np.asarray(covered, dtype=bool)
    b = np.asarray(z, dtype=float)[covered]
    ell = np.asarray(ld_scores, dtype=float)[covered]
    safe = np.clip(b * b, 0.0, 1.0 - 1e-12)
    chisq = n_eff * safe / (1.0 - safe)
    # m_snps is the reference map the LD scores were summed over, not the
    # covered subset: the regressor is N * ell / M, so passing the subset size
    # would inflate the slope by M_reference / covered.
    return ldsc_h2(chisq, ell, n_eff, m_snps=m_reference)


def _auto_h2_and_polygenicity(blocks, z, n_eff, chains, iters, cores, seed=0):
    """``(h2, p)`` and their credible intervals from LDpred3-auto.

    Two quantities, both otherwise supplied by assertion. ``daetwyler_r2``
    needs a heritability *and* a polygenicity, and LD Score regression
    identifies only the first — its slope is ``N h2 ell / M`` whatever the
    causal fraction. LDpred3-auto's sampler infers both jointly, with credible
    intervals, so ``--m-causal`` stops being a number somebody made up.

    This is the more accurate estimator of the two where there is signal to
    find, which is why the caller prefers it. It is also the one that degrades
    first when there is not: a sampler asked to apportion a heritability
    indistinguishable from zero has nothing to condition on, chains wander, and
    the multi-chain filter can discard most of them. Near zero the caller falls
    back to LD Score regression, which has no such failure mode because it is
    a regression, not a sampler. ``n_chains_kept`` is returned so that decision
    is auditable rather than silent.
    """
    from ldpred3 import ldpred3_auto_infer

    result = ldpred3_auto_infer(blocks, np.asarray(z, dtype=float), n_eff,
                                n_chains=int(chains), num_iter=int(iters),
                                burn_in=int(iters), ncores=int(cores),
                                seed=seed)
    return result


def _prune_redundant(correlation, n_eff, alive, threshold):
    """Greedily keep one score from each set correlated above ``threshold``.

    A real same-trait panel accumulates near-duplicates: the same method
    re-deposited across Catalog releases, or two studies differing only by a
    later data freeze. The CAD panel this was written against contains a pair
    at correlation 1.0000 and a third at 0.9957. Rules that invert the score
    covariance cannot survive that, and rules that do not invert it still spend
    weight several times on one piece of information.

    Redundancy is measured here by the **score** correlation, the off-diagonal
    of ``W' D W`` normalised — which is what actually enters the fit. It is not
    the genetic correlation of the underlying traits. Estimating that would
    take each score's discovery GWAS marginal effects and bivariate LD Score
    regression; the PGS Catalog distributes weights, not summary statistics, so
    it is unavailable here. The two answer different questions in any case:
    genetic correlation is a property of the traits, while this is a property
    of the score vectors as the panel actually holds them, including whatever
    shrinkage each method applied.

    Ties are broken by ``n_eff``, so the better-powered member of a redundant
    set survives, and by index where ``n_eff`` is unknown, which keeps the
    result deterministic rather than dependent on dictionary order.
    """
    k = correlation.shape[0]
    order = sorted(range(k), key=lambda j: (-(n_eff[j] if np.isfinite(n_eff[j])
                                              else -np.inf), j))
    keep, dropped = [], {}
    for j in order:
        if not alive[j]:
            continue
        clash = next((i for i in keep
                      if abs(correlation[i, j]) > threshold), None)
        if clash is None:
            keep.append(j)
        else:
            dropped[j] = (clash, float(correlation[clash, j]))
    return sorted(keep), dropped


def _evaluate(beta, c, gram, regime, var_y):
    """Rescale to unit combined-score variance in the reference, then score.

    The rescaling uses the LD reference only, never the target GWAS, so it
    cannot move the evaluation regime; it exists because R2 is scale-invariant
    while MSE is not, and an arbitrary ``||w||=1`` normalisation would make the
    MSE column meaningless.
    """
    quadratic = float(beta @ gram @ beta)
    sd = float(np.sqrt(quadratic)) if quadratic > 0.0 else 0.0
    scaled = beta / sd if sd > 0.0 else beta
    return evaluate_sumstat(scaled, c, gram, var_y=var_y, regime=regime), sd


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ld", required=True, type=Path,
                        help="ldpred3 LD cache (.npz) written by save_ld_blocks")
    parser.add_argument("--scores", required=True, type=Path,
                        help="directory of PGS Catalog scoring files, all for "
                             "the same trait")
    parser.add_argument("--metadata", required=True, type=Path,
                        help="score metadata table from write_score_metadata; "
                             "supplies N_EFF and the declared COHORTS")
    parser.add_argument("--gwas", required=True, type=Path,
                        help="target-trait GWAS summary statistics")
    parser.add_argument("--gwas-format", required=True,
                        choices=sorted(_GWAS_FORMATS),
                        help="column layout of --gwas")
    parser.add_argument("--gwas-columns", default=None,
                        help="override columns as key=name pairs, e.g. "
                             "'id=SNP,beta=BETA'; keys are id ea oa af beta "
                             "se n")
    parser.add_argument("--gwas-n-eff", type=float, default=None,
                        help="effective sample size when --gwas has no N "
                             "column (required for cardiogram)")
    parser.add_argument("--gwas-cohorts", default="",
                        help="comma-separated short names of the cohorts "
                             "behind --gwas, checked against every score's "
                             "declared discovery cohorts")
    parser.add_argument("--regime", required=True, choices=("A", "B", "C"),
                        help="provenance of the target GWAS; never inferred")
    parser.add_argument("--h2", type=float, default=None,
                        help="declared trait heritability, enabling the two "
                             "expected_r2 rules via daetwyler_r2")
    parser.add_argument("--m-causal", type=float, default=None,
                        help="declared number of causal variants, with --h2")
    parser.add_argument("--ridge", type=float, default=1e-3,
                        help="meta_pgs ridge on C before inversion")
    parser.add_argument("--h2-auto", action="store_true",
                        help="also estimate h2 and polygenicity from --gwas "
                             "with LDpred3-auto's multi-chain sampler, and "
                             "prefer it over LDSC unless h2 is near zero")
    parser.add_argument("--h2-auto-chains", type=int, default=10)
    parser.add_argument("--h2-auto-iter", type=int, default=200)
    parser.add_argument("--h2-auto-cores", type=int, default=4)
    parser.add_argument("--h2-near-zero", type=float, default=0.01,
                        help="below this the LDpred3-auto estimate is not "
                             "trusted and the LDSC one is used instead")
    parser.add_argument("--h2-ldsc", action="store_true",
                        help="estimate the target trait's SNP heritability "
                             "from --gwas by LD Score regression instead of "
                             "taking --h2 on trust")
    parser.add_argument("--prune-correlation", type=float, default=None,
                        help="drop a score whose |correlation| with an "
                             "already-kept score exceeds this, keeping the "
                             "larger-n_eff member of each redundant set")
    parser.add_argument("--high-correlation", type=float, default=0.5,
                        help="|correlation| above which a score pair counts as "
                             "highly correlated in the cohort cross-tab")
    parser.add_argument("--min-score-variants", type=int, default=10,
                        help="warn about any score the reference supports with "
                             "fewer than this many variants; its correlations "
                             "and accuracy are then noise, not a measurement")
    parser.add_argument("--min-gwas-coverage", type=float, default=0.8,
                        help="warn about any score this fraction or less of "
                             "whose aligned weight the target GWAS covers")
    parser.add_argument("--chrom", default=None,
                        help="restrict the reference, panel and GWAS to one "
                             "chromosome (smoke tests)")
    parser.add_argument("--max-scores", type=int, default=None,
                        help="use only the first N scoring files (smoke tests)")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args(argv)
    started = time.perf_counter()

    spec = dict(_GWAS_FORMATS[args.gwas_format])
    if args.gwas_columns:
        for item in args.gwas_columns.split(","):
            if "=" not in item:
                parser.error(f"--gwas-columns entry {item!r} is not key=name")
            key, name = item.split("=", 1)
            if key not in spec:
                parser.error(f"--gwas-columns key {key!r} is not one of "
                             + ", ".join(sorted(spec)))
            spec[key] = name or None
    if spec.get("n") is None and args.gwas_n_eff is None:
        parser.error(f"--gwas-format {args.gwas_format} has no sample-size "
                     "column, so --gwas-n-eff is required")
    if args.h2_ldsc and args.m_causal is None:
        parser.error("--h2-ldsc estimates h2 but daetwyler_r2 also needs "
                     "--m-causal; polygenicity is not identified by LD Score "
                     "regression")
    if (args.h2 is None) != (args.m_causal is None) and not args.h2_ldsc:
        parser.error("--h2 and --m-causal must be given together")
    if args.h2 is not None and not 0.0 < args.h2 <= 1.0:
        parser.error("--h2 must lie in (0, 1]; daetwyler_r2 returns nan "
                     "outside it and every expected_r2 rule would then fail "
                     "with an unrelated message")
    if args.m_causal is not None and args.m_causal <= 0.0:
        parser.error("--m-causal must be positive")
    if args.max_scores is not None and args.max_scores < 2:
        parser.error("--max-scores must be at least 2; a meta-PGS comparison "
                     "needs a panel")

    blocks, variants, meta = _reference(args.ld, args.chrom)
    m = meta["n_variants"]
    print(f"reference: {m:,} variants in {meta['n_blocks']} blocks "
          f"(of {meta['n_variants_total']:,} genome-wide), "
          f"loaded in {meta['load_seconds']:.1f}s")

    # The metadata table is a .tsv and conventionally sits beside the scoring
    # files it describes, so it matches this glob; excluding it by path keeps it
    # out of the panel rather than relying on read_scoring_file to reject it.
    metadata_path = args.metadata.resolve()
    files = sorted(p for p in Path(args.scores).iterdir()
                   if p.name.endswith((".txt", ".txt.gz", ".tsv", ".tsv.gz"))
                   and p.resolve() != metadata_path)
    if not files:
        raise SystemExit(f"no scoring files found in {args.scores}")
    n_files_found = len(files)
    if args.max_scores is not None and args.max_scores < n_files_found:
        files = files[:args.max_scores]
        print(f"panel: --max-scores {args.max_scores} keeps the first "
              f"{len(files)} of {n_files_found} scoring files in "
              f"{args.scores} (filename order); the rest are not in this run")
    reference_af = meta["af"]
    pairs_ld, score_ids, align_log = align_to_reference(
        [str(p) for p in files], variants, af=reference_af,
        hwe_genotype_sd=True, on_error="skip")
    if len(pairs_ld) < 2:
        raise SystemExit("fewer than two scoring files aligned to the "
                         "reference; a meta-PGS comparison needs a panel")
    print(f"panel: {len(pairs_ld)} of {len(files)} scores aligned, "
          f"median {align_log.get('n_matched_median')} variants matched, "
          f"minimum {align_log.get('n_matched_min')}")
    for label, reason in sorted(align_log.get("errors", {}).items()):
        print(f"  WARNING: {label} did not align and is not in the panel: "
              f"{reason}")

    records = _read_metadata(args.metadata)
    missing = [s for s in score_ids if s not in records]
    if missing:
        raise SystemExit("no metadata row for " + ", ".join(missing[:5])
                         + f" ({len(missing)} score(s)); every panel member "
                           "needs N_EFF for the sqrt_n_eff rule")
    panel_records = [records[s] for s in score_ids]
    n_eff = np.array([r.n_eff for r in panel_records], dtype=float)
    if not np.all(np.isfinite(n_eff) & (n_eff > 0.0)):
        raise SystemExit("some panel scores have no usable N_EFF; the Catalog "
                         "records none for them and this benchmark will not "
                         "invent one")
    # The same-trait assumption is meta_pgs's central precondition, so it is
    # checked against the machine-readable EFO identifiers. The Catalog's free
    # text disagrees with itself routinely — "Height" and "Adult standing
    # height" are one trait — and warning on that would train the reader to
    # ignore the warning that matters.
    traits = sorted({r.trait_reported for r in panel_records if r.trait_reported})
    efo_sets = {r.efo_ids for r in panel_records if r.efo_ids}
    if len(efo_sets) > 1:
        print(f"WARNING: the panel spans {len(efo_sets)} distinct EFO trait "
              "sets; meta_pgs assumes the scores estimate one genetic value "
              "and these rules are wrong for a heterogeneous panel")
    elif not efo_sets:
        print("WARNING: no score declares an EFO trait, so the same-trait "
              f"precondition is unchecked; reported labels are {traits}")

    raw = _read_gwas(args.gwas, spec,
                     set(np.asarray(variants.id, dtype=object).tolist()),
                     args.gwas_n_eff)
    z, gwas_af, covered, gwas_log = _harmonize_gwas(raw, variants, reference_af)
    print(f"gwas: {gwas_log['n_rows']:,} rows read, "
          f"{gwas_log['n_standardized']:,} standardized onto the reference "
          f"({100.0 * gwas_log['reference_coverage']:.1f}% of its {m:,} "
          f"variants; the rest enter c as z = 0)")

    # W_gwas and W_ld are the same raw scores on two different genotype scales.
    # Rescaling by the ratio of the two dosage SDs is exact and avoids parsing
    # every scoring file a second time; a monomorphic reference variant carries
    # no weight on either scale, so the 0/0 there is 0 rather than undefined.
    sd_reference = np.sqrt(2.0 * reference_af * (1.0 - reference_af))
    sd_gwas = np.sqrt(2.0 * gwas_af * (1.0 - gwas_af))
    ratio = np.divide(sd_gwas, sd_reference, out=np.zeros(m),
                      where=sd_reference > 0.0)
    pairs_gwas = [(idx, w * ratio[idx]) for idx, w in pairs_ld]

    t0 = time.perf_counter()
    c, gram, score_var = score_moments(pairs_ld, z, blocks,
                                       weights_gwas=pairs_gwas,
                                       n_variants_ld=m)
    moments_seconds = time.perf_counter() - t0
    k = gram.shape[0]
    score_sd = np.sqrt(np.maximum(score_var, 0.0))
    alive = score_sd > 0.0
    print(f"moments: {k} x {k} Gram in {moments_seconds:.1f}s, "
          f"{int(np.count_nonzero(~alive))} score(s) with no variance")
    if np.count_nonzero(alive) < 2:
        raise SystemExit("fewer than two scores have any variance under this "
                         "reference; nothing to combine")

    # How much of each score actually survived to the two moments. Both losses
    # attenuate a score's accuracy in a way indistinguishable from the score
    # being poor, so both are reported rather than absorbed.
    n_aligned = np.array([idx.size for idx, _ in pairs_ld])
    variant_coverage = np.array(
        [float(np.mean(covered[idx])) if idx.size else float("nan")
         for idx, _ in pairs_ld])
    weight_coverage = np.array(
        [float(np.sum(np.abs(w[covered[idx]])) / np.sum(np.abs(w)))
         if idx.size and np.sum(np.abs(w)) > 0.0 else float("nan")
         for idx, w in pairs_ld])
    thin = alive & (n_aligned < args.min_score_variants)
    uncovered = alive & (variant_coverage == variant_coverage) \
        & (weight_coverage <= args.min_gwas_coverage)
    for j in np.flatnonzero(~alive):
        print(f"  WARNING: {score_ids[j]} has no variance under this reference "
              f"({n_aligned[j]} variant(s) aligned); it is excluded from every "
              "reported correlation and carries weight 0 in every rule")
    for j in np.flatnonzero(thin):
        print(f"  WARNING: {score_ids[j]} rests on {n_aligned[j]} reference "
              f"variant(s) (< --min-score-variants {args.min_score_variants}); "
              "its correlations and accuracy are noise, not a measurement")
    for j in np.flatnonzero(uncovered):
        print(f"  WARNING: the target GWAS covers only "
              f"{100.0 * weight_coverage[j]:.1f}% of {score_ids[j]}'s aligned "
              "weight; the rest enters c as z = 0, which attenuates its "
              "accuracy for a reason that is not the score")

    # The observed shared-information structure: what `decorrelated` inverts.
    #
    # A score the reference leaves with no variance has no correlation with
    # anything — that is undefined, not zero, and writing zero would put a
    # fabricated number into the very distribution this benchmark exists to
    # measure. `meta_pgs` and the eigendecomposition need a finite PSD matrix,
    # so `correlation` keeps the zero fill (a score with sd 0 is dead in
    # meta_pgs and gets weight 0 there regardless); every *reported* correlation
    # is read from `observed`, which is nan wherever nothing was measured.
    safe_sd = np.where(alive, score_sd, 1.0)
    correlation = gram / np.outer(safe_sd, safe_sd)
    correlation[~alive, :] = 0.0
    correlation[:, ~alive] = 0.0
    correlation[np.arange(k), np.arange(k)] = 1.0
    observed = np.array(correlation, dtype=float)
    observed[~alive, :] = np.nan
    observed[:, ~alive] = np.nan
    # Pruning happens after the moments and before anything is fitted, so
    # every rule, every correlation report and the cohort cross-tab all see the
    # same panel. Doing it later would let the reported correlations describe a
    # panel the rules never saw.
    ids_before = list(np.asarray(score_ids, dtype=object))
    n_pruned = 0
    pruned_detail = {}
    if args.prune_correlation is not None:
        keep, pruned_detail = _prune_redundant(
            correlation, n_eff, alive, args.prune_correlation)
        n_pruned = k - len(keep)
        if len(keep) < 2:
            raise SystemExit(
                f"--prune-correlation {args.prune_correlation} leaves "
                f"{len(keep)} score(s); nothing to combine")
        if n_pruned:
            index = np.array(keep, dtype=int)
            c, gram = c[index], gram[np.ix_(index, index)]
            score_sd, n_eff, alive = score_sd[index], n_eff[index], alive[index]
            score_ids = np.asarray(score_ids, dtype=object)[index]
            panel_records = [panel_records[j] for j in keep]
            k = len(keep)
            safe_sd = np.where(alive, score_sd, 1.0)
            correlation = gram / np.outer(safe_sd, safe_sd)
            correlation[~alive, :] = 0.0
            correlation[:, ~alive] = 0.0
            correlation[np.arange(k), np.arange(k)] = 1.0
            observed = np.array(correlation, dtype=float)
            observed[~alive, :] = np.nan
            observed[:, ~alive] = np.nan
            eigenvalues = np.linalg.eigvalsh(0.5 * (correlation + correlation.T))
            print(f"prune: |r| > {args.prune_correlation} dropped {n_pruned} "
                  f"of {n_pruned + k} score(s), leaving {k}")
            for j, (kept, r) in sorted(pruned_detail.items()):
                print(f"    dropped {ids_before[j]} (|r| {abs(r):.4f} with "
                      f"{ids_before[kept]})")

    surrogate, clipped_mass, psd_correlation = _surrogate_cohort(correlation)
    with np.errstate(invalid="ignore", divide="ignore"):
        # A dead score's surrogate column is constant once meta_pgs zeroes it,
        # and np.corrcoef warns on the 0/0 before its own isfinite guard.
        surrogate_error = (float(np.max(np.abs(
            np.corrcoef(surrogate, rowvar=False) - psd_correlation)))
            if k > 1 else 0.0)
    eigenvalues = np.linalg.eigvalsh(0.5 * (correlation + correlation.T))

    # Architecture is a genome-wide property, so --m-causal is spent over the
    # whole reference even when --chrom restricts everything else. daetwyler_r2
    # depends on p and n_variants only through their product, so this does not
    # move a number; it keeps p meaning the genome-wide polygenicity fraction
    # rather than a per-chromosome one that would have to be read differently
    # under --chrom than without it.
    h2_used, h2_ldsc, auto = args.h2, None, None
    h2_source = "declared" if args.h2 is not None else None
    p_used = (args.m_causal / meta["n_variants_total"]
              if args.m_causal is not None else None)
    p_source = "declared" if args.m_causal is not None else None
    auto_daetwyler_bound = None
    if args.h2_ldsc:
        from ldpred3 import ld_scores as _ld_scores
        h2_ldsc = _ldsc_h2_from_gwas(
            z, covered, _ld_scores(blocks),
            float(args.gwas_n_eff if args.gwas_n_eff is not None
                  else np.median(raw[-1][raw[-1] > 0])),
            meta["n_variants_total"])
        estimate = float(getattr(h2_ldsc, "h2", h2_ldsc))
        print(f"h2 by LD Score regression: {estimate:.4f}"
              + (f" (declared {args.h2})" if args.h2 is not None else ""))
        if not 0.0 < estimate <= 1.0:
            print("  WARNING: the LDSC estimate is outside (0, 1], so it "
                  "cannot drive daetwyler_r2; falling back to --h2 and "
                  "recording both")
        else:
            h2_used = estimate
            h2_source = "ldsc"

    if args.h2_auto:
        t_auto = time.perf_counter()
        auto = _auto_h2_and_polygenicity(
            blocks, z, float(args.gwas_n_eff), args.h2_auto_chains,
            args.h2_auto_iter, args.h2_auto_cores)
        print(f"h2 by LDpred3-auto: {auto.h2_est:.4f} "
              f"[{auto.h2_ci[0]:.4f}, {auto.h2_ci[1]:.4f}], "
              f"p {auto.p_est:.3g} [{auto.p_ci[0]:.3g}, {auto.p_ci[1]:.3g}], "
              f"{auto.n_chains_kept}/{auto.n_chains} chains kept "
              f"({time.perf_counter() - t_auto:.0f}s)")
        # The closed form checks the sampler, not the other way round.
        # daetwyler_r2 at auto's OWN inferred h2 and p is what a predictor
        # achieves knowing which variants are causal and facing no LD noise, so
        # it is an upper reference: a sampled r2_est materially above it is not
        # a better score, it is a fit to distrust -- unconverged chains, an LD
        # reference that does not match the GWAS, or QC that dropped the wrong
        # variants. Well below it is ordinary, since real LD and finite
        # reference cost accuracy the bound never pays.
        #
        # This is the only check available on this panel. r2_est is the
        # accuracy of an LDpred3 fit to one GWAS, so it exists for the target
        # and not per score: the Catalog distributes weights, not summary
        # statistics, and daetwyler_r2 is used precisely because n_eff is all a
        # Catalog score supplies. Where a panel is built by
        # multipgs.panel_from_sumstats instead, every component has its own
        # auto fit and expected_r2 should come from r2_est directly.
        closed_form = float(daetwyler_r2(
            float(auto.h2_est), float(auto.p_est), float(args.gwas_n_eff),
            meta["n_variants_total"]))
        auto_daetwyler_bound = closed_form
        print(f"predictive r2: LDpred3-auto {auto.r2_est:.4f} "
              f"[{auto.r2_ci[0]:.4f}, {auto.r2_ci[1]:.4f}] against the "
              f"daetwyler_r2 bound {closed_form:.4f} at auto's own h2 and p")
        if auto.r2_est > closed_form * 1.05:
            print("  WARNING: the sampled predictive r2 exceeds the bound that "
                  "assumes causal variants known and no LD noise. Treat this "
                  "fit as suspect: check chain convergence, whether the LD "
                  "reference matches the GWAS, and n_chains_kept.")
        # LDpred3-auto is the more accurate of the two where there is signal,
        # and the first to fail where there is not. The threshold is on the
        # estimate itself rather than on chain agreement because a sampler
        # given nothing to condition on can agree precisely and wrongly.
        if auto.h2_est >= args.h2_near_zero:
            h2_used = float(auto.h2_est)
            p_used = float(auto.p_est)
            h2_source = p_source = "ldpred3_auto"
        else:
            print(f"  h2 below --h2-near-zero {args.h2_near_zero}: keeping "
                  "the LD Score regression estimate, which has no sampler to "
                  "fail")

    architecture = {
        "declared": {"h2": args.h2, "m_causal": args.m_causal},
        "used_for_expected_r2": {
            "h2": h2_used,
            "h2_source": h2_source,
            "polygenicity": p_used,
            "polygenicity_source": p_source,
            "m_causal": (None if p_used is None else
                         float(p_used * meta["n_variants_total"])),
        },
        "ldsc": (None if h2_ldsc is None else {
            "h2": float(getattr(h2_ldsc, "h2", h2_ldsc)),
            "h2_se": getattr(h2_ldsc, "h2_se", None),
            "h2_ci": list(getattr(h2_ldsc, "h2_ci", ())),
            "intercept": getattr(h2_ldsc, "intercept", None),
            "intercept_se": getattr(h2_ldsc, "intercept_se", None),
            "mean_chisq": getattr(h2_ldsc, "mean_chisq", None),
            "ratio": getattr(h2_ldsc, "ratio", None),
        }),
        "ldpred3_auto": (None if auto is None else {
            "h2": float(auto.h2_est),
            "h2_ci": [float(x) for x in auto.h2_ci],
            "polygenicity": float(auto.p_est),
            "polygenicity_ci": [float(x) for x in auto.p_ci],
            "m_causal": float(auto.p_est * meta["n_variants_total"]),
            "predictive_r2": float(auto.r2_est),
            "predictive_r2_ci": [float(x) for x in auto.r2_ci],
            "daetwyler_r2_bound": auto_daetwyler_bound,
            "n_chains": int(auto.n_chains),
            "n_chains_kept": int(auto.n_chains_kept),
        }),
    }

    expected_r2 = None
    if h2_used is not None and p_used is not None:
        total_variants = meta["n_variants_total"]
        expected_r2 = daetwyler_r2(h2_used, p_used, n_eff, total_variants)

    center = np.zeros(k)
    combinations = []
    # The n_eff baseline. Catalog scores derived from one meta-analysis share an
    # n_eff to the individual, so a tie here is common and argmax then resolves
    # it by filename order; n_scores_tied_max_n_eff says so, because the
    # docstring points at this baseline as the yardstick for the others.
    masked_n_eff = np.where(alive, n_eff, -np.inf)
    n_tied_max = int(np.count_nonzero(masked_n_eff == masked_n_eff.max()))
    combinations.append(("best_single_max_n_eff", args.regime,
                         _single_score(int(np.argmax(masked_n_eff)),
                                       score_sd), {}))
    if n_tied_max > 1:
        print(f"  WARNING: {n_tied_max} scores tie at the largest n_eff "
              f"({masked_n_eff.max():,.0f}); best_single_max_n_eff took "
              f"{score_ids[int(np.argmax(masked_n_eff))]} by filename order, "
              "so that baseline is arbitrary among the tied scores")
    with np.errstate(invalid="ignore", divide="ignore"):
        # meta_pgs zeroes a dead score's column before np.corrcoef, which warns
        # on the resulting 0/0 ahead of its own isfinite guard. The weights are
        # unaffected: a dead score gets weight 0 by construction.
        equal = meta_pgs(surrogate, n_eff=np.ones(k), method="sqrt_n_eff",
                         score_ids=score_ids, center=center, scale=score_sd)
        combinations.append(("equal_weight", args.regime, equal.beta,
                             equal.log))
        for name, method, rho in (("sqrt_n_eff", "sqrt_n_eff", "n_eff"),
                                  ("expected_r2", "expected_r2",
                                   "expected_r2"),
                                  ("decorrelated_n_eff", "decorrelated",
                                   "n_eff"),
                                  ("decorrelated_expected_r2", "decorrelated",
                                   "expected_r2")):
            if rho == "expected_r2" and expected_r2 is None:
                continue
            kwargs = ({"n_eff": n_eff} if rho == "n_eff"
                      else {"expected_r2": expected_r2})
            fit = meta_pgs(surrogate, method=method, score_ids=score_ids,
                           center=center, scale=score_sd, ridge=args.ridge,
                           **kwargs)
            combinations.append((name, args.regime, fit.beta, fit.log))

    # Every single score's own accuracy, which also supplies the oracle
    # baseline. Choosing the maximum uses the evaluation moments, so that row
    # is regime C no matter what the target GWAS's provenance is.
    single_r2 = np.array([
        _evaluate(_single_score(j, score_sd), c, gram, args.regime, 1.0)[0].r2
        if alive[j] else float("nan") for j in range(k)])
    oracle = int(np.nanargmax(single_r2)) if np.any(np.isfinite(single_r2)) \
        else int(np.argmax(alive))
    combinations.append(("best_single_oracle", "C",
                         _single_score(oracle, score_sd), {}))

    rule_rows, weights_by_rule = [], {}
    for name, regime, beta, log in combinations:
        evaluation, sd = _evaluate(beta, c, gram, regime, 1.0)
        weights_by_rule[name] = beta * score_sd
        order = np.argsort(-np.abs(beta * score_sd))
        # R2 squares beta'c, so a combination anticorrelated with the trait is
        # indistinguishable from a good one at this column alone. beta'c keeps
        # the sign, and the rescaling above makes it the whole of the MSE.
        beta_c = float(evaluation.log["beta_c"])
        rule_rows.append({
            "rule": name, "regime": evaluation.regime,
            "r2": evaluation.r2, "beta_c": beta_c, "mse": evaluation.mse,
            "n_nonzero_weights": int(np.count_nonzero(beta)),
            "n_negative_weights": int(log.get("negative_weights", 0)),
            "condition_number": log.get("condition_number"),
            "combined_score_sd_reference": sd,
            "discarded_c_null_fraction":
                evaluation.log["discarded_c_null_fraction"],
            "top_score": str(score_ids[order[0]]),
            "top_weight": float((beta * score_sd)[order[0]]),
        })
        print(f"  {name:<26s} regime {evaluation.regime}  "
              f"R2 {evaluation.r2:.5f}"
              + ("  (beta'c < 0: anticorrelated with the trait, so this R2 is "
                 "not accuracy)" if beta_c < 0.0 else ""))

    overlap, overlap_ids = cohort_overlap(panel_records)
    if [str(s) for s in overlap_ids] != [str(s) for s in score_ids]:
        raise SystemExit("cohort_overlap returned a different score order than "
                         "the panel; refusing to cross-tabulate")
    # Cohort short names are matched case-insensitively: the target's cohorts
    # are typed on a command line and the panel's come from the Catalog, and a
    # case mismatch would silently answer "no overlap" — which under --regime A
    # is the one wrong answer that costs nothing to produce and everything to
    # believe. Names that match nothing in the panel are reported for the same
    # reason: a typo and a genuinely absent cohort look identical otherwise.
    gwas_cohorts = {c.strip() for c in args.gwas_cohorts.split(",") if c.strip()}
    folded = {c.casefold() for c in gwas_cohorts}
    declared_anywhere = {c.casefold() for r in panel_records for c in r.cohorts}
    unmatched_gwas_cohorts = sorted(c for c in gwas_cohorts
                                    if c.casefold() not in declared_anywhere)
    shares_gwas = np.array([bool({c.casefold() for c in r.cohorts} & folded)
                            for r in panel_records])

    upper = np.triu_indices(k, 1)
    measurable_pair = alive[upper[0]] & alive[upper[1]]
    pair_rows = []
    for position, (i, j) in enumerate(zip(*upper)):
        shared = sorted(panel_records[i].cohorts & panel_records[j].cohorts)
        pair_rows.append({
            "score_i": str(score_ids[i]), "score_j": str(score_ids[j]),
            "correlation_measured": bool(measurable_pair[position]),
            "score_correlation": float(observed[i, j]),
            "cohort_jaccard": float(overlap[i, j]),
            "n_shared_cohorts": len(shared),
            "shared_cohorts": ";".join(shared),
            "n_eff_i": float(n_eff[i]), "n_eff_j": float(n_eff[j]),
        })

    jaccard = overlap[upper]
    signed_corr = observed[upper]
    pair_corr = np.abs(signed_corr)
    measured = np.isfinite(signed_corr)
    off = signed_corr[measured]
    declared = np.isfinite(jaccard)
    # The cross-tab needs both halves of a row, so it runs on pairs that have
    # both; the plain "how many pairs declare a shared cohort" count is a
    # statement about metadata alone and stays over every declared pair.
    crosstab = declared & measured
    declared_sharing = declared & (jaccard > 0.0)
    sharing = crosstab & (jaccard > 0.0)
    disjoint = crosstab & (jaccard <= 0.0)
    high = measured & (pair_corr > args.high_correlation)

    def _mean(values):
        return float(np.mean(values)) if values.size else float("nan")

    # Regime A asks a question about people that only --gwas-cohorts can even
    # begin to answer. Not asking it is not the same as asking and getting a
    # clean answer, and only this column distinguishes the two.
    if args.regime != "A":
        cohort_check = "not_applicable"
    elif not gwas_cohorts:
        cohort_check = "not_checked"
    elif np.any(shares_gwas):
        cohort_check = "contested"
    else:
        cohort_check = "no_declared_overlap"

    summary = {
        "n_scores": k,
        "n_scoring_files_found": n_files_found,
        "n_scores_requested": len(files),
        "n_scores_failed_alignment": int(align_log.get("n_failed", 0)),
        "n_scores_dead": int(np.count_nonzero(~alive)),
        "n_scores_below_min_variants": int(np.count_nonzero(thin)),
        "min_score_variants_aligned": int(np.min(n_aligned)),
        "n_distinct_efo_trait_sets": len(efo_sets),
        "n_distinct_trait_labels": len(traits),
        "chrom": args.chrom if args.chrom is not None else "all",
        "n_reference_variants": m,
        "n_reference_variants_genome_wide": meta["n_variants_total"],
        "n_blocks": meta["n_blocks"],
        "n_weight_entries": int(sum(len(idx) for idx, _ in pairs_ld)),
        "gwas_n_rows": gwas_log["n_rows"],
        "gwas_n_kept_by_id": gwas_log["n_kept_by_id"],
        "gwas_n_unparsable": gwas_log["n_unparsable"],
        "gwas_n_standardized": gwas_log["n_standardized"],
        "gwas_reference_coverage": gwas_log["reference_coverage"],
        "gwas_n_dropped_ambiguous": gwas_log["harmonize_n_dropped_ambiguous"],
        "gwas_n_flipped": gwas_log["harmonize_n_flipped"],
        "gwas_af_column_used": bool(gwas_log["has_af_column"]),
        "min_score_gwas_weight_coverage": float(np.nanmin(weight_coverage))
            if np.any(np.isfinite(weight_coverage)) else float("nan"),
        "n_scores_below_min_gwas_coverage": int(np.count_nonzero(uncovered)),
        "offdiag_corr_min": float(np.min(off)),
        "offdiag_corr_q25": float(np.percentile(off, 25)),
        "offdiag_corr_median": float(np.median(off)),
        "offdiag_corr_q75": float(np.percentile(off, 75)),
        "offdiag_corr_max": float(np.max(off)),
        "offdiag_abs_corr_mean": float(np.mean(np.abs(off))),
        "frac_pairs_above_threshold":
            float(np.count_nonzero(high) / np.count_nonzero(measured))
            if np.any(measured) else float("nan"),
        "correlation_min_eigenvalue": float(eigenvalues[0]),
        "correlation_condition_number":
            float(eigenvalues[-1] / eigenvalues[0]) if eigenvalues[0] > 0
            else float("inf"),
        "surrogate_clipped_eigenvalue_mass": clipped_mass,
        "surrogate_max_correlation_error": surrogate_error,
        "n_pairs": int(jaccard.size),
        "n_pairs_correlation_measured": int(np.count_nonzero(measured)),
        "n_pairs_declaring_cohorts": int(np.count_nonzero(declared)),
        "n_pairs_sharing_cohort": int(np.count_nonzero(declared_sharing)),
        "n_pairs_cross_tabulated": int(np.count_nonzero(crosstab)),
        "mean_abs_corr_sharing_cohort": _mean(pair_corr[sharing]),
        "mean_abs_corr_disjoint_cohorts": _mean(pair_corr[disjoint]),
        "corr_of_jaccard_and_score_correlation":
            float(np.corrcoef(jaccard[crosstab], pair_corr[crosstab])[0, 1])
            if np.count_nonzero(crosstab) > 1
            and np.std(jaccard[crosstab]) > 0 else float("nan"),
        "n_sharing_and_high": int(np.count_nonzero(sharing & high)),
        "n_sharing_and_low": int(np.count_nonzero(sharing & ~high)),
        "n_disjoint_and_high": int(np.count_nonzero(disjoint & high)),
        "n_disjoint_and_low": int(np.count_nonzero(disjoint & ~high)),
        "declared_regime": args.regime,
        "n_gwas_cohorts_declared": len(gwas_cohorts),
        "n_gwas_cohorts_unmatched": len(unmatched_gwas_cohorts),
        "n_scores_sharing_gwas_cohort": int(np.count_nonzero(shares_gwas)),
        "regime_a_cohort_check": cohort_check,
        "regime_a_contested": cohort_check == "contested",
        "n_scores_tied_max_n_eff": n_tied_max,
        "expected_r2_rules_run": expected_r2 is not None,
        "score_moments_seconds": moments_seconds,
    }
    used_architecture = architecture["used_for_expected_r2"]
    ldsc_architecture = architecture["ldsc"] or {}
    auto_architecture = architecture["ldpred3_auto"] or {}
    auto_h2_ci = auto_architecture.get("h2_ci") or [None, None]
    summary.update({
        "architecture_h2_used": used_architecture["h2"],
        "architecture_h2_source": used_architecture["h2_source"],
        "architecture_polygenicity_used": used_architecture["polygenicity"],
        "architecture_polygenicity_source":
            used_architecture["polygenicity_source"],
        "architecture_m_causal_used": used_architecture["m_causal"],
        "h2_ldsc": ldsc_architecture.get("h2"),
        "h2_ldsc_se": ldsc_architecture.get("h2_se"),
        "h2_auto": auto_architecture.get("h2"),
        "h2_auto_ci_low": auto_h2_ci[0],
        "h2_auto_ci_high": auto_h2_ci[1],
        "polygenicity_auto": auto_architecture.get("polygenicity"),
        "m_causal_auto": auto_architecture.get("m_causal"),
        "predictive_r2_auto": auto_architecture.get("predictive_r2"),
        "daetwyler_r2_bound_auto":
            auto_architecture.get("daetwyler_r2_bound"),
        "auto_n_chains_kept": auto_architecture.get("n_chains_kept"),
        "auto_n_chains": auto_architecture.get("n_chains"),
    })
    for row in rule_rows:
        summary[f"r2_{row['rule']}"] = row["r2"]
    if cohort_check == "contested":
        print(f"WARNING: {summary['n_scores_sharing_gwas_cohort']} panel "
              "score(s) declare a discovery cohort that also contributed to "
              "the target GWAS. Regime A was requested; these numbers are "
              "inflated by an amount no statistic here can measure.")
    elif cohort_check == "not_checked":
        print("WARNING: regime A was requested but --gwas-cohorts is empty, so "
              "no score was checked against the target GWAS's cohorts. This is "
              "an unperformed check, not a passed one; regime_a_cohort_check "
              "records it as not_checked and the A on every row is the "
              "caller's assertion alone.")
    elif cohort_check == "no_declared_overlap":
        print(f"note: no panel score declares one of the {len(gwas_cohorts)} "
              "cohorts given for the target GWAS. Declared cohorts are a lower "
              "bound, so this supports regime A without establishing it.")
    if unmatched_gwas_cohorts:
        print("  WARNING: --gwas-cohorts named "
              + ", ".join(unmatched_gwas_cohorts)
              + ", which no panel score declares. Either the panel genuinely "
                "excludes them or they are spelled differently from the "
                "Catalog's short names, and the two are indistinguishable "
                "here.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rules_path = args.output_dir / "real_meta_rules.csv"
    with rules_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RULE_FIELDS,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rule_rows)

    scores_path = args.output_dir / "real_meta_rules_scores.csv"
    score_fields = ("score_id", "n_eff", "expected_r2", "n_variants_aligned",
                    "has_variance_in_reference", "gwas_variant_coverage",
                    "gwas_weight_coverage", "score_sd_reference",
                    "single_score_r2", "single_score_r2_regime",
                    "mean_abs_corr_to_panel", "max_abs_corr_to_panel",
                    "n_measured_pairs", "n_declared_cohorts",
                    "shares_gwas_cohort",
                    *(f"weight_{name}" for name in weights_by_rule))
    off_mask = ~np.eye(k, dtype=bool)
    with scores_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=score_fields,
                                lineterminator="\n")
        writer.writeheader()
        for j in range(k):
            row = np.abs(observed[j][off_mask[j]])
            neighbours = row[np.isfinite(row)]
            writer.writerow({
                "score_id": str(score_ids[j]), "n_eff": float(n_eff[j]),
                "expected_r2": (float(expected_r2[j])
                                if expected_r2 is not None else ""),
                "n_variants_aligned": int(n_aligned[j]),
                "has_variance_in_reference": bool(alive[j]),
                "gwas_variant_coverage": float(variant_coverage[j]),
                "gwas_weight_coverage": float(weight_coverage[j]),
                "score_sd_reference": float(score_sd[j]),
                "single_score_r2": float(single_r2[j]),
                # This score alone against the target GWAS: nothing was fitted
                # or selected, so it carries the declared regime. Only the
                # maximum over this column (best_single_oracle) is regime C.
                "single_score_r2_regime": args.regime,
                "mean_abs_corr_to_panel": _mean(neighbours),
                "max_abs_corr_to_panel": (float(np.max(neighbours))
                                          if neighbours.size else float("nan")),
                "n_measured_pairs": int(neighbours.size),
                "n_declared_cohorts": len(panel_records[j].cohorts),
                "shares_gwas_cohort": bool(shares_gwas[j]),
                **{f"weight_{name}": float(w[j])
                   for name, w in weights_by_rule.items()}})

    pairs_path = args.output_dir / "real_meta_rules_pairs.csv"
    with pairs_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(pair_rows[0]),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(pair_rows)

    summary_path = args.output_dir / "real_meta_rules_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(summary),
                                lineterminator="\n")
        writer.writeheader()
        writer.writerow(summary)

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
                         "n_ref": meta["n_ref"],
                         "n_variants_total": meta["n_variants_total"]},
        "gwas": {"path": str(args.gwas), "sha256": _sha256(args.gwas),
                 "format": args.gwas_format, "columns": spec,
                 "declared_cohorts": sorted(gwas_cohorts), **gwas_log},
        "panel": {"directory": str(args.scores),
                  "metadata": str(args.metadata),
                  "metadata_sha256": _sha256(args.metadata),
                  "scoring_files": [
                      {"path": str(path), "sha256": _sha256(path)}
                      for path in files
                  ],
                  "score_ids": [str(s) for s in score_ids],
                  "trait_labels": traits,
                  "efo_trait_sets": sorted(list(s) for s in efo_sets),
                  **align_log},
        "architecture": architecture,
        "parameters": {key: (str(value) if isinstance(value, Path) else value)
                       for key, value in vars(args).items()},
        "note": ("Meta-PGS weighting rules on a real same-trait panel, with "
                 "the shared-information structure measured from a real LD "
                 "reference rather than simulated. One panel, one trait, one "
                 "reference and one target GWAS: this is a single draw and "
                 "not a rule ranking. The declared regime applies to every "
                 "rule except best_single_oracle, which selects on the "
                 "evaluation moments and is regime C. Regime A additionally "
                 "requires that no score's discovery cohorts contributed to "
                 "the target GWAS; regime_a_cohort_check says whether that was "
                 "checked at all, and a check that ran reports only the "
                 "declared lower bound, which cohort metadata cannot make "
                 "tight. The regime labels the target GWAS and not the LD "
                 "reference: the decorrelated rules invert the correlation "
                 "implied by the same Gram matrix that forms every R2 "
                 "denominator, and every rule standardizes by its diagonal, so "
                 "no rule here is independent of this one reference. The "
                 "architecture object records declared, estimated, and "
                 "actually used h2 and polygenicity values; expected_r2 is "
                 "conditional on used_for_expected_r2. Scores with no "
                 "variance under the reference are excluded from every "
                 "reported correlation rather than recorded as uncorrelated; "
                 "reference variants the GWAS does not carry enter c as z = 0, "
                 "and gwas_weight_coverage per score is how much of each score "
                 "that costs."),
    }
    with (args.output_dir / "real_meta_rules_provenance.json").open(
            "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, indent=2)
        handle.write("\n")

    undefined = summary["n_pairs"] - summary["n_pairs_correlation_measured"]
    print(f"off-diagonal score correlation over "
          f"{summary['n_pairs_correlation_measured']} of {summary['n_pairs']} "
          f"pairs: median {summary['offdiag_corr_median']:.3f}, "
          f"range [{summary['offdiag_corr_min']:.3f}, "
          f"{summary['offdiag_corr_max']:.3f}]"
          + (f" ({undefined} pair(s) undefined: a score with no variance under "
             "this reference)" if undefined else "")
          + f"; {summary['n_pairs_sharing_cohort']} of "
          f"{summary['n_pairs_declaring_cohorts']} declared pairs share a "
          "cohort")
    print(rules_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
