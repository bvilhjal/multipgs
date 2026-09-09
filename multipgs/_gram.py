"""Streaming score Gram ``W^T D W`` from an LD reference."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from ._validate import _nonnegative_integer, _positive_integer


def _integer_indices(values, label):
    """Return integer-valued indices without truncating malformed input."""
    raw = np.asarray(values)
    if raw.dtype.kind == "b" or raw.dtype.kind not in "iuf":
        raise ValueError(f"{label} must contain integer variant indices")
    flat = raw.ravel()
    if raw.dtype.kind == "f":
        limit = np.iinfo(np.int64).max
        if (not np.all(np.isfinite(flat))
                or np.any(flat != np.floor(flat))
                or np.any(flat < -limit - 1) or np.any(flat > limit)):
            raise ValueError(f"{label} must contain integer variant indices")
    elif raw.dtype.kind == "u" and np.any(flat > np.iinfo(np.int64).max):
        raise ValueError(f"{label} contains an index outside int64 range")
    return flat.astype(np.int64, copy=False)


def _as_blocks(ld, n_variants):
    """Normalize an LD argument to ``(corr_block, idx)`` pairs.

    Accepts ldpred3's native block list, a single dense correlation matrix, or
    anything ``ldpred3.ld_matmul`` understands paired with explicit indices.
    Non-dense inputs are consumed lazily and must tile ``0..m-1`` exactly once.
    """
    if isinstance(ld, np.ndarray):
        if ld.ndim != 2 or ld.shape[0] != ld.shape[1]:
            raise ValueError(f"a dense LD matrix must be square, got {ld.shape}")
        if ld.shape[0] != n_variants:
            raise ValueError(f"LD matrix is {ld.shape[0]} x {ld.shape[0]} but "
                             f"the weights cover {n_variants} variants")
        return [(ld, np.arange(n_variants))]

    def validated():
        expected = 0
        seen = False
        for b, block in enumerate(ld):
            if not (isinstance(block, tuple) and len(block) == 2):
                raise ValueError(
                    f"LD block {b} is not a (corr_block, idx) pair; this is the "
                    "layout ldpred3.compute_ld_blocks and "
                    "ldpred3.load_ld_blocks return, got "
                    f"{type(block).__name__}")
            corr, idx = block
            idx = _integer_indices(idx, f"LD block {b} indices")
            if idx.size == 0:
                continue
            seen = True
            if np.any(np.diff(idx) != 1):
                raise ValueError(
                    f"LD block {b} does not cover a contiguous run of variants. "
                    "The streaming Gram slices weights by block range; use "
                    "ldpred3.compute_ld_blocks, whose blocks tile 0..m-1.")
            if int(idx[0]) != expected:
                kind = "overlaps an earlier block" if int(idx[0]) < expected \
                    else "leaves a gap before it"
                raise ValueError(
                    f"LD block {b} starts at variant {int(idx[0])}, expected "
                    f"{expected}; it {kind}. Blocks must tile 0..m-1 exactly "
                    "once and in order.")
            expected = int(idx[-1]) + 1
            yield corr, idx
        if not seen:
            raise ValueError("the LD reference has no blocks")
        if expected != n_variants:
            raise ValueError(f"the LD blocks cover variants 0..{expected - 1} "
                             f"but the weights cover 0..{n_variants - 1}")

    return validated()


@dataclass(frozen=True)
class _DenseWeights:
    """Validated dense weights kept dense throughout the moment calculation."""

    values: np.ndarray

    @property
    def m(self):
        return int(self.values.shape[0])

    @property
    def k(self):
        return int(self.values.shape[1])


def _dense_weights_finite(weights):
    """Check finite values with at most about one million Boolean temporaries."""
    rows = max(1, 1_000_000 // max(int(weights.shape[1]), 1))
    return all(np.all(np.isfinite(weights[start:start + rows]))
               for start in range(0, weights.shape[0], rows))


def _weight_columns(weights, n_variants=None):
    """Normalize weights to a dense view or canonical sparse COO tuple.

    Dense ``(m, K)`` input remains dense.  Turning it into COO first costs three
    extra arrays per non-zero (24 bytes with int64 indices and float64 values),
    which is catastrophic precisely when a matrix is dense. A panel of catalog
    scores is usually sparse, so a list of per-score ``(index, weight)`` pairs
    remains the representation that scales for that use case.
    """
    if isinstance(weights, np.ndarray):
        if weights.ndim != 2:
            raise ValueError(f"dense weights must be 2-D (m, K), got "
                             f"{weights.shape}")
        if not _dense_weights_finite(weights):
            raise ValueError("dense weights contain non-finite values")
        m, k = weights.shape
        if k == 0:
            raise ValueError("dense weights contain no score columns")
        if n_variants is not None:
            requested = _nonnegative_integer(n_variants, "n_variants")
            if requested != m:
                raise ValueError(
                    f"n_variants={requested} but dense weights have {m} rows")
        values = np.asarray(weights)
        if not np.issubdtype(values.dtype, np.floating):
            values = values.astype(float)
        return _DenseWeights(values)

    pairs = list(weights)
    if not pairs:
        raise ValueError("no score weights given")
    idx_parts, col_parts, val_parts = [], [], []
    largest = -1
    for k, pair in enumerate(pairs):
        if not (isinstance(pair, tuple) and len(pair) == 2):
            raise ValueError(f"score {k} must be an (index, weight) pair, got "
                             f"{type(pair).__name__}")
        idx = _integer_indices(pair[0], f"score {k} indices")
        val = np.asarray(pair[1], dtype=float).ravel()
        if idx.shape != val.shape:
            raise ValueError(f"score {k} has {idx.size} indices and {val.size} "
                             "weights")
        if idx.size and not np.all(np.isfinite(val)):
            raise ValueError(f"score {k} has non-finite weights")
        if idx.size:
            if int(idx.min()) < 0:
                raise ValueError(f"score {k} has a negative variant index")
            # A sparse score is a mathematical vector, not an insertion log.
            # Coalescing here gives W'DW and W'z the same interpretation.
            unique, inverse = np.unique(idx, return_inverse=True)
            if unique.size != idx.size:
                combined = np.zeros(unique.size, dtype=float)
                np.add.at(combined, inverse, val)
                keep = combined != 0.0
                idx, val = unique[keep], combined[keep]
            else:
                # Sparse input is a mathematical vector, so an explicitly
                # stored zero is not an entry and must not alter memory logs or
                # the inferred reference extent.
                keep = val != 0.0
                idx, val = idx[keep], val[keep]
            if idx.size:
                largest = max(largest, int(idx.max()))
        idx_parts.append(idx)
        col_parts.append(np.full(idx.size, k, dtype=np.int64))
        val_parts.append(val)

    m = (int(largest + 1) if n_variants is None else
         _nonnegative_integer(n_variants, "n_variants"))
    if m <= largest:
        raise ValueError(f"a weight indexes variant {largest} but the reference "
                         f"has {m}")
    return (np.concatenate(idx_parts) if idx_parts else np.zeros(0, np.int64),
            np.concatenate(col_parts) if col_parts else np.zeros(0, np.int64),
            np.concatenate(val_parts) if val_parts else np.zeros(0),
            m, len(pairs))


def score_gram(weights, ld, *, n_variants=None, progress=None):
    """The ``K x K`` score covariance ``W^T D W`` from an LD reference.

    Streams the reference one LD block at a time, densifying only that block's
    slice of the weights, so peak memory is ``O(block_size * K)`` rather than
    ``O(m * K)``. For a 900-score panel and 500-variant blocks that is a few
    megabytes instead of tens of gigabytes.

    Parameters
    ----------
    weights : ndarray or sequence of (index, weight)
        Per-variant weights for each score, on the **standardized** genotype
        scale, aligned to the LD reference's variants. PGS Catalog weights count
        raw alleles and must be converted first — see :func:`align_to_reference`.
    ld : sequence of (corr_block, idx), or ndarray
        ldpred3 LD blocks, or one dense correlation matrix.
    n_variants : int, optional
        Reference size, needed only when the weights are sparse and no score
        touches the last variant.
    progress : callable, optional
        Called as ``progress(done, total)`` after each LD block is contracted,
        with ``total=None`` when ``ld`` is a lazy stream of unknown length.
        Streaming the reference is where a genome-wide Gram spends its time,
        so this is the fraction of the real work.

    Returns
    -------
    (gram, score_var) : (ndarray, ndarray)
        ``gram`` is ``W^T D W`` (``K x K``); ``score_var`` is its diagonal, the
        variance of each score under the reference's LD.
    """
    return _score_gram_from_coo(_weight_columns(weights, n_variants), ld,
                                progress=progress)


def _block_quadform(corr, block_w):
    """``W_b^T D_b W_b`` for one LD block, using its own representation.

    ldpred3 stores a large block as a low-rank factor (LR8):
    ``D = U U^T + diag(residual)``. Going through :func:`ldpred3.ld_matmul`
    computes ``U (U^T W)`` — projecting back up to the block's full variant
    dimension — only for this function to immediately contract it back down
    again. Keeping the factor instead,

        W^T D W = (U^T W)^T (U^T W) + (residual * W)^T W,

    skips that back-projection and shrinks the second product from ``O(k A^2)``
    to ``O(r A^2)``. In the bigsnpr HapMap3+ reference the low-rank blocks hold
    the bulk of the variants — median 3,120 variants at median rank 890, so
    ``r/k`` is about 0.29 — and this is where a genome-wide Gram spends its
    time.

    LDpred3 owns bounded decoding and factor contractions for every supported
    representation, so compact storage is not followed by whole-block widening.
    """
    from ldpred3.interop import ld_crossproducts

    return ld_crossproducts(corr, block_w)


def _score_gram_from_coo(parsed, ld, *, progress=None):
    """:func:`score_gram` on already-parsed dense or sparse weights.

    Parsing a sparse weight set materializes three arrays over every non-zero
    entry, which for a genome-wide panel is the largest allocation in the whole
    fit. A caller that already holds the parse passes it here instead of
    handing the raw weights back to be parsed a second time.

    ``progress(done, total)`` is invoked from this thread after each block, so
    a caller driving a genome-wide reference can report the LD stream rather
    than appear to hang. ``total`` is ``None`` for a lazy block stream.
    """
    # A generator has no length and is not consumed to acquire one, so its
    # total is reported as None rather than guessed. A concrete block list is
    # counted over its index arrays only -- no LD payload is touched -- and
    # skips the empty blocks ``_as_blocks`` drops, so ``done`` reaches ``total``
    # instead of stalling one short of it. A malformed block list falls back to
    # the raw length; ``_as_blocks`` raises the real diagnostic immediately.
    total = None
    if isinstance(ld, np.ndarray):
        total = 1
    elif hasattr(ld, "__len__"):
        # ``__getitem__`` as well, so only a re-iterable sequence is walked
        # twice; ldpred3's cache views subclass ``list``. Anything else keeps
        # its raw length.
        if hasattr(ld, "__getitem__"):
            try:
                total = sum(1 for _corr, idx in ld if np.size(idx))
            except (TypeError, ValueError):
                total = len(ld)
        else:
            total = len(ld)
    if isinstance(parsed, _DenseWeights):
        matrix = parsed.values
        blocks = _as_blocks(ld, parsed.m)
        gram = np.zeros((parsed.k, parsed.k), dtype=float)
        for done, (corr, idx) in enumerate(blocks, start=1):
            lo, hi = int(idx[0]), int(idx[-1]) + 1
            gram += _block_quadform(corr, matrix[lo:hi])
            if progress is not None:
                progress(done, total)
        gram = 0.5 * (gram + gram.T)
        return gram, np.diag(gram).copy()

    rows, cols, vals, m, k = parsed
    blocks = _as_blocks(ld, m)

    order = np.argsort(rows, kind="stable")
    rows, cols, vals = rows[order], cols[order], vals[order]

    gram = np.zeros((k, k), dtype=float)
    for done, (corr, idx) in enumerate(blocks, start=1):
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        start, stop = np.searchsorted(rows, (lo, hi))
        if start == stop:
            # The block was still read and skipped; it counts as work done.
            if progress is not None:
                progress(done, total)
            continue
        # Catalog scores are sparse across both variants and blocks. Work only
        # on the scores touching this block; forming a B x K matrix and a full
        # K x K product here can waste two orders of magnitude of work.
        active = np.unique(cols[start:stop])
        local_cols = np.searchsorted(active, cols[start:stop])
        block_w = np.zeros((idx.size, active.size), dtype=float)
        block_w[rows[start:stop] - lo, local_cols] = vals[start:stop]
        gram[np.ix_(active, active)] += _block_quadform(corr, block_w)
        if progress is not None:
            progress(done, total)

    # W^T D W is symmetric in exact arithmetic; the accumulation is not, and an
    # asymmetric Gram makes the coordinate descent's covariance updates drift.
    gram = 0.5 * (gram + gram.T)
    return gram, np.diag(gram).copy()


def accuracy_blocks(weights, z, ld, *, n_variants=None, chrom=None,
                    progress=None):
    """Per-block ``(u, v)`` for one collapsed variant-weight vector.

    ``u_b = w_b' z_b`` and ``v_b = w_b' D_b w_b``, so the genome-wide plug-in
    accuracy is ``(sum u)^2 / (sum v * var_y)`` -- the same identity
    :func:`pseudo_r2` computes in score space, decomposed by LD block.

    The decomposition is what turns that point estimate into an interval and a
    null. ``D`` is block-diagonal, so both sums are exactly additive over
    blocks: a delete-one-block jackknife gives a standard error, and flipping
    the sign of every weight in a block negates ``u_b`` while leaving ``v_b``
    identical, which is a negative control costing no extra pass.
    :mod:`ppb` consumes these arrays directly (``ppb.r2_block_jackknife``,
    ``ppb.sign_flip_null``).

    Parameters
    ----------
    weights : array_like, shape (m,)
        The combined per-variant weights on the LD reference's standardized
        basis and in its variant order -- what
        :meth:`multipgs.SumstatFit.frozen_variant_weights` returns.
    z : array_like, shape (m,)
        Target-trait standardized marginal effects in that same order, zero
        where the trait has no usable variant.
    ld : sequence of (corr_block, idx), or ndarray
        The same LD reference the weights are aligned to.
    chrom : array_like, shape (m,), optional
        Per-variant chromosome. When given, the returned ``groups`` labels each
        block by the chromosome it falls in, which is the more conservative
        jackknife unit when block sizes are very uneven. A block spanning two
        chromosomes is labelled by its first variant and counted, since the
        reference is supposed to tile within chromosomes.

    Returns
    -------
    (u, v, groups) : (ndarray, ndarray, ndarray or None)
    """
    weights = np.asarray(weights, dtype=float).ravel()
    z = np.asarray(z, dtype=float).ravel()
    if weights.size != z.size:
        raise ValueError(f"weights cover {weights.size} variants but z covers "
                         f"{z.size}; both must be in the reference's order")
    if weights.size == 0:
        raise ValueError("weights are empty")
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(z)):
        raise ValueError("weights and z must be finite")
    m = _nonnegative_integer(n_variants, "n_variants") \
        if n_variants is not None else weights.size
    if m != weights.size:
        raise ValueError(f"n_variants={m} but weights cover {weights.size}")
    if chrom is not None:
        chrom = np.asarray(chrom).ravel()
        if chrom.size != m:
            raise ValueError(f"chrom covers {chrom.size} variants, expected {m}")

    total = None
    if isinstance(ld, np.ndarray):
        total = 1
    elif hasattr(ld, "__len__") and hasattr(ld, "__getitem__"):
        try:
            total = sum(1 for _corr, idx in ld if np.size(idx))
        except (TypeError, ValueError):
            total = len(ld)
    u, v, groups, n_split = [], [], [], 0
    for done, (corr, idx) in enumerate(_as_blocks(ld, m), start=1):
        lo, hi = int(idx[0]), int(idx[-1]) + 1
        block_w = weights[lo:hi]
        u.append(float(block_w @ z[lo:hi]))
        # A one-column quadratic form through the same bounded contraction the
        # Gram uses, so every LD representation is handled in its own form.
        v.append(float(_block_quadform(corr, block_w[:, None])[0, 0]))
        if chrom is not None:
            labels = chrom[lo:hi]
            groups.append(labels[0])
            if labels.size and not np.all(labels == labels[0]):
                n_split += 1
        if progress is not None:
            progress(done, total)
    if n_split:
        warnings.warn(
            f"{n_split} LD block(s) span more than one chromosome; each was "
            "labelled by its first variant. Chromosome jackknife groups are "
            "only as clean as the reference's blocking", stacklevel=2)
    return (np.asarray(u, dtype=float), np.asarray(v, dtype=float),
            np.asarray(groups) if chrom is not None else None)


def _weight_digest(rows, cols, vals, n_variants, n_scores):
    """Canonical digest of the exact aligned score matrix used by a fit."""
    import hashlib

    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    vals = np.asarray(vals, dtype=np.float64)
    keep = vals != 0.0
    rows, cols, vals = rows[keep], cols[keep], vals[keep]
    order = np.lexsort((cols, rows))
    digest = hashlib.sha256()
    digest.update(np.asarray([n_variants, n_scores], dtype="<i8").tobytes())
    digest.update(rows[order].astype("<i8", copy=False).tobytes())
    digest.update(cols[order].astype("<i8", copy=False).tobytes())
    digest.update(vals[order].astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def _parsed_weight_info(parsed):
    """Return ``(m, K, number of non-zero entries)`` without expanding dense W."""
    if isinstance(parsed, _DenseWeights):
        return parsed.m, parsed.k, int(np.count_nonzero(parsed.values))
    _, _, vals, m, k = parsed
    return int(m), int(k), int(vals.size)


def _parsed_weight_digest(parsed):
    """Canonical sparse-matrix digest without a dense-to-COO allocation.

    The byte stream matches :func:`_weight_digest`: non-zero row indices, then
    column indices, then values, all in row-major order. Dense inputs are walked
    in bounded row chunks and never materialize genome-wide coordinate arrays.
    """
    if not isinstance(parsed, _DenseWeights):
        return _weight_digest(*parsed)

    import hashlib

    matrix = parsed.values
    digest = hashlib.sha256()
    digest.update(np.asarray([parsed.m, parsed.k], dtype="<i8").tobytes())
    chunk_rows = max(1, min(parsed.m, 1_000_000 // max(parsed.k, 1)))
    for part in ("row", "col", "value"):
        for start in range(0, parsed.m, chunk_rows):
            block = matrix[start:start + chunk_rows]
            row, col = np.nonzero(block)
            if part == "row":
                value = row.astype(np.int64, copy=False) + start
                digest.update(value.astype("<i8", copy=False).tobytes())
            elif part == "col":
                digest.update(col.astype("<i8", copy=False).tobytes())
            else:
                value = block[row, col]
                digest.update(value.astype("<f8", copy=False).tobytes())
    return digest.hexdigest()


def _collapse_parsed_weights(parsed, coefficients):
    """Collapse component scores without changing the stored weight basis."""
    coefficients = np.asarray(coefficients, dtype=float)
    if isinstance(parsed, _DenseWeights):
        return parsed.values @ coefficients
    rows, cols, vals, m, _ = parsed
    out = np.zeros(m, dtype=float)
    np.add.at(out, rows, vals * coefficients[cols])
    return out


def _ld_variant_count(weights_ld, ld, explicit, label):
    """Infer one LD source's variant count without borrowing a GWAS length."""
    if explicit is not None:
        count = _positive_integer(explicit, label)
        if isinstance(weights_ld, np.ndarray):
            if weights_ld.ndim != 2:
                raise ValueError(f"{label} weights must be two-dimensional")
            if weights_ld.shape[0] != count:
                raise ValueError(
                    f"{label}={count} but dense LD weights have "
                    f"{weights_ld.shape[0]} rows")
        if isinstance(ld, np.ndarray):
            if ld.ndim != 2 or ld.shape[0] != ld.shape[1]:
                raise ValueError(f"{label} LD matrix must be square")
            if ld.shape[0] != count:
                raise ValueError(
                    f"{label}={count} but dense LD has {ld.shape[0]} rows")
        return count
    if isinstance(weights_ld, np.ndarray):
        if weights_ld.ndim != 2:
            raise ValueError(f"{label} weights must be two-dimensional")
        return int(weights_ld.shape[0])
    if isinstance(ld, np.ndarray):
        if ld.ndim != 2 or ld.shape[0] != ld.shape[1]:
            raise ValueError(f"{label} LD matrix must be square")
        return int(ld.shape[0])
    if isinstance(ld, (list, tuple)) and ld:
        last = _integer_indices(ld[-1][1], f"{label} LD block indices")
        if last.size:
            return int(last[-1]) + 1
    return None


def _score_cross_moment(weights_gwas, z, n_scores, label):
    """Compute ``W_gwas' z`` on that GWAS's own standardized-genotype basis."""
    parsed = _weight_columns(weights_gwas, int(z.size))
    return _score_cross_moment_parsed(parsed, z, n_scores, label)


def _score_cross_moment_parsed(parsed, z, n_scores, label, *, n_entries=None):
    """Cross-moment from an existing parse, preserving one-shot iterables."""
    if isinstance(parsed, _DenseWeights):
        m, k = parsed.m, parsed.k
        if m != z.size:
            raise ValueError(f"{label} weights cover {m} variants but z covers "
                             f"{z.size}")
        if k != n_scores:
            raise ValueError(f"{label} weights describe {k} scores but the LD "
                             f"weights describe {n_scores}; score identity and "
                             "column order must agree")
        if n_entries is None:
            n_entries = int(np.count_nonzero(parsed.values))
        return parsed.values.T @ z, n_entries, m

    rows, cols, vals, m, k = parsed
    if m != z.size:
        raise ValueError(f"{label} weights cover {m} variants but z covers "
                         f"{z.size}")
    if k != n_scores:
        raise ValueError(f"{label} weights describe {k} scores but the LD "
                         f"weights describe {n_scores}; score identity and "
                         "column order must agree")
    c = np.zeros(k, dtype=float)
    np.add.at(c, cols, vals * z[rows])
    return c, int(vals.size) if n_entries is None else int(n_entries), m


def score_moments(weights_ld, z, ld, *, weights_gwas=None,
                  n_variants_ld=None, progress=None):
    """The score-space moments ``(c, G)`` for one set of summary statistics.

    The pair that :func:`evaluate_sumstat` scores against, and the same pair
    :func:`multi_pgs_sumstats` fits from. Building them for an *evaluation* GWAS
    is how a combination gets an honest regime A number. ``weights_gwas`` and
    ``weights_ld`` represent the same raw component scores, but each is
    multiplied by the empirical genotype SD of its own dataset:
    ``c = W_gwas.T @ z`` and ``G = W_ld.T @ D @ W_ld``. They may cover different
    variant sets; only their score columns must agree. Equality with
    individual-level Gaussian regression moments is exact for unadjusted data,
    or when genotypes and phenotype were jointly residualized on the identical
    covariate design—not for arbitrary adjusted marginal GWAS coefficients.

    ``progress`` is forwarded to :func:`score_gram` as ``progress(done, total)``
    over the LD blocks.
    """
    if weights_gwas is None:
        raise ValueError(
            "weights_gwas is required separately from weights_ld; pass the "
            "same matrix explicitly only when GWAS and LD genotype scales are "
            "genuinely identical")
    z = np.asarray(z, dtype=float).ravel()
    if not np.all(np.isfinite(z)):
        raise ValueError("z contains non-finite values")
    n_variants_ld = _ld_variant_count(
        weights_ld, ld, n_variants_ld, "n_variants_ld")
    parsed = _weight_columns(weights_ld, n_variants_ld)
    gram, var = _score_gram_from_coo(parsed, ld, progress=progress)
    if weights_gwas is weights_ld:
        c, _, _ = _score_cross_moment_parsed(
            parsed, z, gram.shape[0], "weights_gwas")
    else:
        c, _, _ = _score_cross_moment(
            weights_gwas, z, gram.shape[0], "weights_gwas")
    return c, gram, var
