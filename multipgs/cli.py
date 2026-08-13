"""Command-line interface: ``multipgs <command>``.

Commands, matching the steps of the analysis::

    multipgs fetch     get scoring files and metadata from the PGS Catalog
    multipgs panel     build the n x K score matrix from a target cohort
    multipgs fit       learn a combination from a training phenotype
    multipgs meta      combine same-trait scores with no phenotype at all
    multipgs combine   collapse a panel and a fit into one weight file
    multipgs score     score a cohort from that file (frozen scaling)
    multipgs evaluate  measure a score in held-out individuals

Every command that reads individuals matches them on ``FID:IID`` and reports
how many were dropped, rather than assuming two files are in the same order —
silently mismatched rows are the single easiest way to produce an accuracy
number that means nothing.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np


def _read_table(path, *, value_columns=None, name="file",
                require_single_id=False):
    """Read a whitespace/TSV table keyed by ``FID IID`` or ``IID``.

    Returns ``(keys, values, columns)``. A header is detected by its first
    field being FID, IID, ID, SCORE or SCORE_ID (case-insensitive); without
    one, two leading non-numeric columns are read as FID/IID and one as IID.
    """
    rows = []
    header = None
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("##"):
                continue
            parts = line.split("\t") if "\t" in line else line.split()
            if header is None and parts[0].lstrip("#").upper() in (
                    "FID", "IID", "ID", "SCORE", "SCORE_ID"):
                header = [p.lstrip("#") for p in parts]
                continue
            rows.append(parts)
    if not rows:
        raise SystemExit(f"{name}: {path} has no data rows")

    width = len(header) if header is not None else len(rows[0])
    if header is not None:
        has_fid = header[0].upper() == "FID"
    else:
        has_fid = width > 2 and not _is_number(rows[0][1])
    if require_single_id and has_fid:
        raise SystemExit(f"{name}: {path} must use one SCORE/ID column, not "
                         "FID IID")
    key_cols = 2 if has_fid else 1
    if width <= key_cols:
        raise SystemExit(f"{name}: {path} has no value columns")

    keys, values = [], []
    for parts in rows:
        if len(parts) != width:
            raise SystemExit(f"{name}: {path} has a row with {len(parts)} "
                             f"fields; expected {width}")
        keys.append((parts[0], parts[1]) if has_fid
                    else (parts[0], parts[0]))
        values.append(parts[key_cols:width])
    columns = (header[key_cols:width] if header
               else [f"col{i}" for i in range(width - key_cols)])

    duplicate_columns = _duplicates(columns)
    if duplicate_columns:
        raise SystemExit(f"{name}: {path} has duplicate value column(s): "
                         f"{', '.join(duplicate_columns)}")
    duplicate_keys = _duplicates(keys)
    if duplicate_keys:
        raise SystemExit(f"{name}: {path} has duplicate identifier(s): "
                         f"{', '.join(duplicate_keys[:3])}")

    if value_columns is not None:
        want = list(value_columns)
        missing = [c for c in want if c not in columns]
        if missing:
            raise SystemExit(f"{name}: {path} has no column(s) "
                             f"{', '.join(missing)}; found {columns}")
        idx = [columns.index(c) for c in want]
        values = [[row[i] for i in idx] for row in values]
        columns = want

    out = np.full((len(values), len(columns)), np.nan)
    for i, row in enumerate(values):
        for j, v in enumerate(row):
            try:
                out[i, j] = float(v)
            except ValueError:
                out[i, j] = np.nan
    return _object_vector(keys), out, columns


def _duplicates(values):
    """Unique duplicate string values, preserving first duplicate order."""
    seen, reported, out = set(), set(), []
    for value in values:
        value = str(value)
        if value in seen and value not in reported:
            out.append(value)
            reported.add(value)
        seen.add(value)
    return out


def _object_vector(values):
    """A one-dimensional object array, including when values are tuples."""
    out = np.empty(len(values), dtype=object)
    out[:] = values
    return out


def _is_number(text):
    try:
        float(text)
    except (TypeError, ValueError):
        return False
    return True


def _align(*tables):
    """Intersect keyed tables, preserving the first one's order."""
    keys = [t[0] for t in tables]
    common = set(keys[0])
    for k in keys[1:]:
        common &= set(k)
    if not common:
        raise SystemExit("the input files share no individuals (matched on "
                         "FID:IID); check that the identifier columns agree")
    order = [k for k in keys[0] if k in common]
    picked = []
    for k, values in ((t[0], t[1]) for t in tables):
        lookup = {key: i for i, key in enumerate(k)}
        picked.append(values[[lookup[key] for key in order]])
    return _object_vector(order), picked


def _drop_missing(keys, arrays):
    ok = np.ones(len(keys), dtype=bool)
    for a in arrays:
        if a.size:
            ok &= np.all(np.isfinite(a), axis=1)
    return keys[ok], [a[ok] for a in arrays], int(np.sum(~ok))


def _read_score_vector(path, score_ids, *, name):
    """Read one value per score and return it in ``score_ids`` order.

    The first column must be a single score identifier (``SCORE``, ``ID`` or
    ``IID``); unlike individual-level tables, a two-column ``FID IID`` key has
    no meaning here. Exact key matching prevents a reordered metadata file from
    silently assigning one score's sample size or penalty to another score.
    """
    keys, values, _ = _read_table(path, name=name, require_single_id=True)
    if values.shape[1] != 1:
        raise SystemExit(f"{name}: {path} has {values.shape[1]} value columns; "
                         "expected exactly one")

    ids = []
    for left, right in keys:
        assert left == right
        ids.append(str(left))

    want = [str(s) for s in np.asarray(score_ids, dtype=object)]
    duplicate_want = _duplicates(want)
    if duplicate_want:
        raise SystemExit(f"--scores has duplicate score column(s): "
                         f"{', '.join(duplicate_want)}")
    lookup = {sid: i for i, sid in enumerate(ids)}
    wanted = set(want)
    missing = [sid for sid in want if sid not in lookup]
    extra = [sid for sid in ids if sid not in wanted]
    if missing or extra:
        detail = (f"; missing {missing[:3]}" if missing else "") + \
            (f"; unknown {extra[:3]}" if extra else "")
        raise SystemExit(f"{name}: score ids do not match --scores{detail}")
    return values[[lookup[sid] for sid in want], 0]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _cmd_fetch(args):
    import os

    from .fetch import (cohort_overlap, download_scores, search_scores,
                        write_score_metadata)

    def note(text):
        if not args.quiet:
            print(text, file=sys.stderr, flush=True)

    selector = {}
    if args.trait:
        selector["trait_id"] = args.trait
    elif args.pgs:
        selector["pgs_ids"] = args.pgs
    elif args.pmid:
        selector["pmid"] = args.pmid
    else:
        selector["publication_id"] = args.publication

    try:
        records = search_scores(include_children=args.include_children,
                                cache_dir=args.cache, **selector)
    except ValueError as exc:
        raise SystemExit(f"fetch: {exc}") from exc
    note(f"{len(records)} score(s) found")

    os.makedirs(args.out, exist_ok=True)
    if args.metadata_only:
        paths, log = [], {"n_downloaded": 0, "n_failed": 0}
    else:
        def progress(i, n, pgs_id):
            note(f"[{i + 1}/{n}] {pgs_id}")

        paths, log = download_scores(
            records, args.out, build=args.build, overwrite=args.overwrite,
            on_error="skip" if args.skip_errors else "raise",
            progress=progress)
        # Metadata must describe what is actually on disk, or a downstream
        # --n-eff file will name scores the panel does not contain.
        records = [r for r, p in zip(records, paths) if p is not None]

    meta_path = os.path.join(args.out, "metadata.tsv")
    n_eff_path = os.path.join(args.out, "n_eff.tsv")
    write_score_metadata(records, meta_path)
    write_score_metadata(records, n_eff_path, columns=["N_EFF"])

    n_missing = sum(1 for r in records if not np.isfinite(r.n_eff))
    print(f"{len(records)} score(s) in {args.out}")
    print(f"  downloaded {log['n_downloaded']}, failed {log['n_failed']}")
    print(f"  wrote {meta_path} and {n_eff_path}")
    if n_missing:
        print(f"  {n_missing} score(s) have no sample size in the Catalog and "
              f"are NA in n_eff.tsv; screen(min_n_eff=...) will reject them")

    if args.cohort_overlap and len(records) > 1:
        overlap, ids = cohort_overlap(records)
        pairs = [(overlap[j, i], ids[j], ids[i])
                 for j in range(len(ids)) for i in range(j + 1, len(ids))
                 if np.isfinite(overlap[j, i]) and overlap[j, i] > 0]
        pairs.sort(reverse=True)
        print(f"  {len(pairs)} score pair(s) share a named discovery cohort")
        for value, left, right in pairs[:10]:
            print(f"    {left} {right}  jaccard {value:.2f}")
        if pairs:
            print("  shared discovery samples are invisible to "
                  "cross-validation; see docs/theory.md")
    return 0


def _cmd_panel(args):
    from .panel import (panel_from_catalog, panel_from_sumstats,
                        panel_from_weights, save_panel, write_panel)

    def progress(i, n, sid):
        if not args.quiet:
            print(f"[{i + 1}/{n}] {sid}", file=sys.stderr, flush=True)

    extra = {}
    if args.method:
        extra["method"] = args.method
    if args.infer:
        extra["infer"] = True
    if args.auto_chains is not None:
        extra["auto_chains"] = args.auto_chains

    if args.catalog:
        panel = panel_from_catalog(
            args.catalog, args.plink, sample_path=args.sample,
            standardize=args.standardize, min_matched=args.min_matched,
            on_error="skip" if args.skip_errors else "raise",
            progress=progress, metadata=args.metadata)
    elif args.weights:
        panel = panel_from_weights(
            args.weights, args.plink, sample_path=args.sample,
            min_matched=args.min_matched,
            on_error="skip" if args.skip_errors else "raise",
            progress=progress)
    else:
        # --sumstats or --traits
        if args.ld_cache is None and args.ld_prefix is None:
            raise SystemExit(
                "panel --sumstats needs --ld-prefix or --ld-cache so LD is "
                "not built from the target cohort")
        if (args.ld_cache is not None and args.ld_prefix is None
                and not os.path.exists(args.ld_cache)):
            raise SystemExit(
                f"--ld-cache {args.ld_cache} does not exist; pass --ld-prefix "
                "to build it from an external LD reference")
        panel = panel_from_sumstats(
            args.sumstats, args.plink, ld_cache=args.ld_cache,
            ld_prefix=args.ld_prefix, sample_path=args.sample,
            on_error="skip" if args.skip_errors else "raise",
            progress=progress, n_jobs=args.n_jobs,
            weights_dir=args.weights_dir, preflight=not args.no_preflight,
            traits=args.traits, **extra)
    if str(args.out).endswith(".npz"):
        save_panel(panel, args.out)
    else:
        write_panel(panel, args.out)
        if args.out_panel:
            save_panel(panel, args.out_panel)
    print(panel.summary())
    print(f"wrote {args.out}")
    return 0


def _load_scores(args):
    if args.panel:
        from .panel import read_panel
        panel = read_panel(args.panel)
        fid = np.asarray(panel.sample_fid).astype(str)
        iid = np.asarray(panel.sample_iid).astype(str)
        pairs = list(zip(fid, iid))
        duplicate_samples = []
        seen, reported = set(), set()
        for pair in pairs:
            if pair in seen and pair not in reported:
                duplicate_samples.append(pair)
                reported.add(pair)
            seen.add(pair)
        if duplicate_samples:
            shown = ", ".join(f"{f}:{i}" for f, i in duplicate_samples[:3])
            raise SystemExit(f"--panel has duplicate FID:IID(s): {shown}")
        score_ids = [str(s) for s in panel.score_ids]
        duplicate_scores = _duplicates(score_ids)
        if duplicate_scores:
            raise SystemExit("--panel has duplicate score id(s): "
                             + ", ".join(duplicate_scores[:3]))
        keys = _object_vector(pairs)
        return keys, np.asarray(panel.scores, dtype=float), score_ids
    return _read_table(args.scores, name="--scores")


def _cmd_fit(args):
    from .metrics import evaluate
    from .stack import multi_pgs_fit

    scores = _load_scores(args)
    pheno = _read_table(args.pheno, value_columns=[args.pheno_name]
                        if args.pheno_name else None, name="--pheno")
    tables = [scores, pheno]
    if args.covar:
        tables.append(_read_table(args.covar, name="--covar"))
    keys, arrays = _align(*tables)
    keys, arrays, n_dropped = _drop_missing(keys, arrays)
    S, Y = arrays[0], arrays[1]
    C = arrays[2] if args.covar else None
    if Y.shape[1] != 1:
        raise SystemExit(f"--pheno has {Y.shape[1]} value columns; name one "
                         f"with --pheno-name")
    y = Y[:, 0]
    if not args.quiet:
        print(f"{len(keys)} individuals, {S.shape[1]} scores"
              + (f", {C.shape[1]} covariates" if C is not None else "")
              + (f" ({n_dropped} dropped for missing values)" if n_dropped
                 else ""), file=sys.stderr)

    penalty = None
    if args.penalty_factor:
        penalty = _read_score_vector(args.penalty_factor, scores[2],
                                     name="--penalty-factor")

    fit = multi_pgs_fit(
        S, y, covar=C, family=args.family, alpha=args.alpha,
        n_folds=args.folds, assessment_folds=args.assessment_folds,
        n_lambda=args.n_lambda, seed=args.seed,
        penalty_factor=penalty, score_ids=scores[2],
        unpenalized_scores=args.unpenalized,
        covar_ids=tables[2][2] if args.covar else None)
    print(fit.summary())

    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("SCORE\tBETA\tBETA_STD\n")
        for sid, b_std, b in zip(fit.score_ids, fit.beta_std, fit.beta):
            fh.write(f"{sid}\t{b:.8g}\t{b_std:.8g}\n")
    print(f"wrote coefficients to {args.out}")

    if args.out_score:
        combined = fit.multi_pgs(S)
        with open(args.out_score, "w", encoding="utf-8") as fh:
            fh.write("FID\tIID\tMULTI_PGS\n")
            for (fid, iid), value in zip(keys, combined):
                fh.write(f"{fid}\t{iid}\t{value:.8g}\n")
        print(f"wrote in-sample combined score to {args.out_score}")
        print("  (in-sample: these individuals trained the combination, so "
              "their accuracy here is optimistic)")
        res = evaluate(y, combined, covar=C, family=args.family, n_boot=0)
        print(res)
    return 0


class _CoefFit:
    """Minimal fit object for :func:`combine_weights`."""

    def __init__(self, score_ids, beta):
        self.score_ids = np.asarray(score_ids, dtype=object)
        self.beta = np.asarray(beta, dtype=float)

    def multi_pgs(self, panel):
        scores = panel.scores if hasattr(panel, "scores") else panel
        return np.asarray(scores, dtype=float) @ self.beta


def _cmd_combine(args):
    from .panel import check_weights, combine_weights, load_panel, read_panel

    panel = (load_panel(args.panel) if str(args.panel).endswith(".npz")
             else read_panel(args.panel))
    keys, values, columns = _read_table(args.fit, name="--fit",
                                        require_single_id=True)
    if "BETA" not in [c.upper() for c in columns]:
        raise SystemExit("--fit needs a BETA column (the coefficients TSV "
                         "from multipgs fit)")
    beta_col = next(c for c in columns if c.upper() == "BETA")
    ids = [str(left) for left, _ in keys]
    lookup = {sid: i for i, sid in enumerate(ids)}
    want = [str(s) for s in panel.score_ids]
    duplicate_panel_ids = _duplicates(want)
    if duplicate_panel_ids:
        raise SystemExit("--panel has duplicate score id(s): "
                         + ", ".join(duplicate_panel_ids[:3]))
    missing = [sid for sid in want if sid not in lookup]
    wanted = set(want)
    extra = [sid for sid in ids if sid not in wanted]
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing {missing[:3]}")
        if extra:
            details.append(f"unexpected {extra[:3]}")
        raise SystemExit("--fit score IDs do not exactly match --panel: "
                         + "; ".join(details))
    beta = values[[lookup[sid] for sid in want], columns.index(beta_col)]
    fit = _CoefFit(want, beta)
    combine_weights(panel, fit, path=args.out)
    print(f"wrote {args.out}")
    if args.check:
        if not args.plink:
            raise SystemExit("combine --check needs --plink")
        info = check_weights(panel, fit, args.out, args.plink)
        print(f"frozen scoring reproduces the fit: corr = {info['corr']:.10f}")
    return 0


def _cmd_score(args):
    from ldpred3 import score_from_weights

    result = score_from_weights(args.weights, args.plink,
                                sample_path=args.sample, scaling="frozen")
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("FID\tIID\tSCORE\n")
        for fid, iid, value in zip(result.sample_fid, result.sample_iid,
                                   result.scores):
            fh.write(f"{fid}\t{iid}\t{value:.8g}\n")
    print(f"wrote {args.out} ({len(result.scores)} individuals, frozen scaling)")
    return 0


def _cmd_meta(args):
    from .meta import meta_pgs

    scores = _read_table(args.scores, name="--scores")
    keys, S, ids = scores[0], scores[1], scores[2]
    n_eff = (_read_score_vector(args.n_eff, ids, name="--n-eff")
             if args.n_eff else None)
    expected_r2 = (_read_score_vector(args.expected_r2, ids,
                                      name="--expected-r2")
                   if args.expected_r2 else None)
    if args.method == "sqrt_n_eff" and n_eff is None:
        raise SystemExit("meta --method sqrt_n_eff needs --n-eff")
    if args.method == "expected_r2" and expected_r2 is None:
        raise SystemExit("meta --method expected_r2 needs --expected-r2")
    if args.method == "decorrelated" and n_eff is None and expected_r2 is None:
        raise SystemExit("meta --method decorrelated needs --expected-r2 "
                         "(preferred) or --n-eff")
    res = meta_pgs(S, n_eff=n_eff, expected_r2=expected_r2,
                   method=args.method, score_ids=ids)
    print(res.summary())
    combined = res.multi_pgs(S)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("FID\tIID\tMETA_PGS\n")
        for (fid, iid), value in zip(keys, combined):
            fh.write(f"{fid}\t{iid}\t{value:.8g}\n")
    print(f"wrote {args.out}")
    return 0


def _cmd_evaluate(args):
    from .metrics import evaluate

    scores = _read_table(args.scores, value_columns=[args.score_name]
                         if args.score_name else None, name="--scores")
    pheno = _read_table(args.pheno, value_columns=[args.pheno_name]
                        if args.pheno_name else None, name="--pheno")
    tables = [scores, pheno]
    if args.covar:
        tables.append(_read_table(args.covar, name="--covar"))
    keys, arrays = _align(*tables)
    keys, arrays, n_dropped = _drop_missing(keys, arrays)
    if arrays[0].shape[1] != 1:
        raise SystemExit(f"--scores has {arrays[0].shape[1]} value columns; "
                         f"name one with --score-name")
    if arrays[1].shape[1] != 1:
        raise SystemExit("--pheno has more than one value column; name one "
                         "with --pheno-name")
    res = evaluate(arrays[1][:, 0], arrays[0][:, 0],
                   covar=arrays[2] if args.covar else None,
                   family=args.family, prevalence=args.prevalence,
                   n_boot=args.n_boot, seed=args.seed)
    if n_dropped and not args.quiet:
        print(f"({n_dropped} individuals dropped for missing values)",
              file=sys.stderr)
    print(res)
    return 0


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(
        prog="multipgs",
        description="Multivariate polygenic scoring: combine many polygenic "
                    "scores into one.")
    from . import __version__
    p.add_argument("--version", action="version",
                   version=f"multipgs {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="get scoring files and metadata from "
                                         "the PGS Catalog")
    who = fetch.add_mutually_exclusive_group(required=True)
    who.add_argument("--trait", metavar="ID",
                     help="Catalog trait identifier, e.g. MONDO_0004989. "
                          "These are migrating from EFO to MONDO; a retired "
                          "identifier is an error, not an empty panel")
    who.add_argument("--pgs", nargs="+", metavar="PGS",
                     help="explicit score identifiers")
    who.add_argument("--pmid", help="every score from one publication's PMID")
    who.add_argument("--publication", metavar="PGP",
                     help="every score from one Catalog publication")
    fetch.add_argument("--out", required=True,
                       help="output directory for scoring files, metadata.tsv "
                            "and n_eff.tsv")
    fetch.add_argument("--build", default="GRCh37",
                       choices=("GRCh37", "GRCh38"),
                       help="must match the build of your target genotypes")
    fetch.add_argument("--include-children", action="store_true",
                       help="with --trait, also take scores registered against "
                            "child traits (the Catalog keeps them separate, so "
                            "without this they are silently omitted)")
    fetch.add_argument("--cache", help="directory for raw API responses, so a "
                                       "re-run costs no requests")
    fetch.add_argument("--metadata-only", action="store_true",
                       help="write the tables without downloading any file")
    fetch.add_argument("--overwrite", action="store_true",
                       help="re-download files already present")
    fetch.add_argument("--cohort-overlap", action="store_true",
                       help="report score pairs sharing a named discovery "
                            "cohort")
    fetch.add_argument("--skip-errors", action="store_true",
                       help="skip scores that fail to download instead of "
                            "stopping")
    fetch.add_argument("--quiet", action="store_true")
    fetch.set_defaults(func=_cmd_fetch)

    panel = sub.add_parser("panel", help="build the n x K score matrix")
    src = panel.add_mutually_exclusive_group(required=True)
    src.add_argument("--catalog", nargs="+", metavar="PATH",
                     help="PGS Catalog scoring file(s), or a directory of "
                          "them")
    src.add_argument("--sumstats", nargs="+", metavar="PATH",
                     help="GWAS summary-statistics file(s), or a directory "
                          "of them (each fitted with LDpred3)")
    src.add_argument("--weights", nargs="+", metavar="PATH",
                     help="ldpred3 weight file(s), or a directory of them")
    src.add_argument("--traits", metavar="PATH",
                     help="per-trait table (TRAIT PATH [N_EFF ...]); each "
                          "GWAS is fitted with LDpred3")
    panel.add_argument("--plink", required=True,
                       help="target PLINK prefix or .bgen path")
    panel.add_argument("--out", required=True,
                       help="output score matrix TSV, or .npz for a full "
                            "panel with weights")
    panel.add_argument("--out-panel", help="also write a full .npz panel "
                                           "when --out is a TSV")
    panel.add_argument("--sample", help="BGEN .sample file")
    panel.add_argument("--metadata", help="Catalog metadata.tsv to attach")
    panel.add_argument("--ld-cache", help="LD cache path, built once and "
                                          "reused across traits (--sumstats)")
    panel.add_argument("--ld-prefix", help="external LD-reference PLINK "
                                           "prefix (--sumstats/--traits)")
    panel.add_argument("--weights-dir", help="write each LDpred3 weight file "
                                             "here (--sumstats)")
    panel.add_argument("--n-jobs", type=int, default=1,
                       help="concurrent traits after the LD cache exists")
    panel.add_argument("--method", help="LDpred3 model (--sumstats)")
    panel.add_argument("--infer", action="store_true",
                       help="request LDpred3 architecture inference")
    panel.add_argument("--auto-chains", type=int,
                       help="LDpred3 auto chain count (--sumstats)")
    panel.add_argument("--no-preflight", action="store_true",
                       help="skip ldpred3.preflight_prs before fitting")
    panel.add_argument("--standardize", action="store_true",
                       help="z-score genotypes before applying catalog "
                            "weights (default: raw allele counts, the PGS "
                            "Catalog convention)")
    panel.add_argument("--min-matched", type=int, default=1,
                       help="drop a score matching fewer target variants")
    panel.add_argument("--skip-errors", action="store_true",
                       help="skip unreadable or unmatched inputs instead of "
                            "stopping")
    panel.add_argument("--quiet", action="store_true")
    panel.set_defaults(func=_cmd_panel)

    fit = sub.add_parser("fit", help="learn a combination from a phenotype")
    src = fit.add_mutually_exclusive_group(required=True)
    src.add_argument("--scores", help="score matrix TSV")
    src.add_argument("--panel", help="full panel .npz from panel --out")
    fit.add_argument("--pheno", required=True, help="phenotype table")
    fit.add_argument("--pheno-name", help="phenotype column to use")
    fit.add_argument("--covar", help="covariate table")
    fit.add_argument("--out", required=True, help="output coefficients TSV")
    fit.add_argument("--out-score", help="also write the combined score")
    fit.add_argument("--family", default="gaussian",
                     choices=("gaussian", "binomial"))
    fit.add_argument("--alpha", type=float, default=1.0,
                     help="elastic-net mixing (1 = lasso)")
    fit.add_argument("--folds", type=int, default=10, help="CMSA folds")
    fit.add_argument("--assessment-folds", type=int, default=5,
                     help="outer folds for nested assessment and the signal "
                          "gate")
    fit.add_argument("--n-lambda", type=int, default=100)
    fit.add_argument("--penalty-factor", help="per-score penalty factors")
    fit.add_argument("--unpenalized", nargs="+", metavar="SCORE",
                     help="score ids left in the unpenalized baseline")
    fit.add_argument("--seed", type=int)
    fit.add_argument("--quiet", action="store_true")
    fit.set_defaults(func=_cmd_fit)

    comb = sub.add_parser("combine", help="collapse a panel and a fit into "
                                          "one ldpred3 weight file")
    comb.add_argument("--panel", required=True,
                      help="full panel .npz (must carry per-variant weights)")
    comb.add_argument("--fit", required=True,
                      help="coefficients TSV from multipgs fit")
    comb.add_argument("--out", required=True, help="output weight file")
    comb.add_argument("--check", action="store_true",
                      help="require frozen scoring to reproduce the fit")
    comb.add_argument("--plink", help="target used for --check")
    comb.set_defaults(func=_cmd_combine)

    sc = sub.add_parser("score", help="score a cohort from a combined "
                                      "weight file (frozen scaling)")
    sc.add_argument("--weights", required=True)
    sc.add_argument("--plink", required=True)
    sc.add_argument("--out", required=True)
    sc.add_argument("--sample", help="BGEN .sample file")
    sc.set_defaults(func=_cmd_score)

    meta = sub.add_parser("meta", help="combine same-trait scores, no "
                                       "phenotype needed")
    meta.add_argument("--scores", required=True, help="score matrix TSV")
    meta.add_argument("--n-eff",
                      help="one effective sample size per score, keyed by "
                           "score id (sqrt_n_eff; decorrelated fallback)")
    meta.add_argument("--expected-r2",
                      help="one expected r2 per score, keyed by score id "
                           "(expected_r2; preferred for decorrelated)")
    meta.add_argument("--out", required=True)
    meta.add_argument("--method", default="sqrt_n_eff",
                      choices=("sqrt_n_eff", "expected_r2", "decorrelated"))
    meta.set_defaults(func=_cmd_meta)

    ev = sub.add_parser("evaluate", help="measure a score against a phenotype")
    ev.add_argument("--scores", required=True)
    ev.add_argument("--score-name", help="score column to evaluate")
    ev.add_argument("--pheno", required=True)
    ev.add_argument("--pheno-name")
    ev.add_argument("--covar")
    ev.add_argument("--family", default="gaussian",
                    choices=("gaussian", "binomial"))
    ev.add_argument("--prevalence", type=float,
                    help="population prevalence, for a liability-scale R2")
    ev.add_argument("--n-boot", type=int, default=1000)
    ev.add_argument("--seed", type=int)
    ev.add_argument("--quiet", action="store_true")
    ev.set_defaults(func=_cmd_evaluate)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":     # pragma: no cover
    raise SystemExit(main())
