"""Training-free combination of several scores for the **same** trait.

:mod:`multipgs.stack` learns its weights from a training cohort. Often there is
no such cohort — and often there does not need to be one. When the ``K`` scores
all estimate the *same* genetic value, differing only in which discovery GWAS
produced them (PGC, deCODE, a biobank, an internal meta-analysis), the weights
follow from the discovery sample sizes and no phenotype is required.

The rule used in the Hansen et al. pipeline (``code/meta_prs.R`` in
`PGS-pipeline <https://github.com/olex2148/PGS-pipeline>`_) is to standardize
each score and weight it by :math:`\\sqrt{n_\\mathrm{eff}}`::

    meta_i = sum_k  z_ik * sqrt(n_eff_k)

The justification is that in the power-limited regime a score's correlation with
the trait grows as :math:`\\sqrt{N}` — from the Daetwyler expression,
:math:`r \\approx h\\sqrt{Nh^2/M}` while :math:`Nh^2 \\ll M` — so
:math:`\\sqrt{n_\\mathrm{eff}}` is proportional to each score's expected
accuracy.

Weighting by accuracy is *not*, however, the exact inverse-variance combination,
and it is worth being precise about the gap. Under the same independent-error
model, with :math:`R_k` the accuracy of score :math:`k`, the optimal weights are
:math:`w_k \\propto R_k / (1 - R_k^2)` — that is the GLS solution, obtained from
:math:`C^{-1}\\rho` by Sherman–Morrison. Weighting by :math:`R_k` alone drops
the :math:`1/(1-R_k^2)` factor, and the two agree only as every
:math:`R_k \\to 0`. That limit *is* the power-limited regime this rule is for,
so the approximation is sound exactly where the :math:`\\sqrt{N}` justification
is, and degrades in the same place — see :mod:`multipgs.architecture` and
``docs/theory.md``.

Two refinements are available here:

* ``method="expected_r2"`` replaces :math:`\\sqrt{n_\\mathrm{eff}}` with
  :math:`\\sqrt{R^2_k}` from each score's *fitted* architecture
  (:func:`multipgs.daetwyler_r2`), which accounts for heritability and
  polygenicity rather than assuming sample size stands in for them. Note this
  does **not** by itself make it the better of the two ``C = I`` rules.
* ``method="decorrelated"`` drops the independence assumption. The two simple
  rules approximate the score-correlation matrix by :math:`C=I`, while
  discovery GWAS are frequently not independent. With :math:`C` estimated
  from the scores in the target cohort and :math:`\\rho` the vector of expected
  accuracies, the model-based weights are
  :math:`w \\propto C^{-1}\\rho`, which discounts a score for the information
  it shares with the others.

The reproducible comparison is ``benchmarks/meta_rules.py``. It reports every
replicate, 30-seed means and standard deviations, and runtime provenance under
``benchmarks/results/``. Its ``shared`` parameter is an **error-term
correlation**, a stylized consequence of shared discovery information; it is
not a literal fraction of overlapping samples. The committed simulation shows
that decorrelation with ``expected_r2`` is robust as this correlation rises,
whereas putting the :math:`\\sqrt{N}` approximation inside :math:`C^{-1}` is
poor. Those results are a regression reference for this model, not evidence
about real cohorts. ``docs/theory.md`` derives all three rules.

**This module assumes the scores target one trait.** Weighting scores of
*different* traits by their own sample sizes is wrong: what a score contributes
to a different target then depends on the genetic correlation between them,
which none of these rules use. For a heterogeneous panel of traits, use
:func:`multipgs.multi_pgs_fit`, which learns the relevance instead of assuming
it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


__all__ = ["meta_pgs", "MetaPGS"]

_METHODS = ("sqrt_n_eff", "expected_r2", "decorrelated")


@dataclass
class MetaPGS:
    """A fixed-weight score combination.

    Deliberately shares :meth:`multi_pgs` with
    :class:`~multipgs.stack.MultiPGSFit`, so the same evaluation code takes
    either. ``beta`` is on the raw score scale; ``weight`` is on the
    standardized scale, which is the one the method reasons in and the one to
    read when asking what the combination is doing.
    """

    beta: np.ndarray
    weight: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    score_ids: np.ndarray
    method: str
    log: dict = field(default_factory=dict)

    def multi_pgs(self, scores, *, score_ids=None):
        """The combined score for new individuals: ``scores @ beta``.

        Scale and location are arbitrary — the weights are only defined up to a
        positive constant — which is why this returns the combination itself and
        no intercept. R², AUC and ranking are unaffected; if you need a
        calibrated predictor, regress this on the phenotype in a training set,
        or use :func:`multipgs.multi_pgs_fit`.

        Pass a :class:`~multipgs.panel.ScorePanel` or ``score_ids=`` when the
        scores were built separately. The identifiers are then checked so a
        reordered panel cannot silently receive the wrong weights.
        """
        if hasattr(scores, "scores") and hasattr(scores, "score_ids"):
            if score_ids is None:
                score_ids = scores.score_ids
            scores = scores.scores
        if score_ids is not None:
            got = [str(s) for s in np.asarray(score_ids, dtype=object).ravel()]
            want = [str(s) for s in np.asarray(self.score_ids,
                                               dtype=object).ravel()]
            if got != want:
                raise ValueError(
                    "these scores are not the ones this meta-PGS was built "
                    "from, or are in a different order. Realign with "
                    "panel.select(list(meta.score_ids)).")
        s = np.asarray(scores, dtype=float)
        if s.ndim == 1:
            s = s[None, :]
        if s.ndim != 2 or s.shape[1] != self.beta.size:
            raise ValueError(f"scores must have {self.beta.size} columns, got "
                             f"{s.shape}")
        return s @ self.beta

    def selected(self, top=None):
        """Scores ordered by ``|weight|`` — the standardized-scale weights."""
        order = np.argsort(-np.abs(self.weight))
        if top is not None:
            order = order[:int(top)]
        return [(self.score_ids[j], float(self.weight[j]),
                 float(self.beta[j])) for j in order]

    def summary(self):
        lines = [f"meta-PGS ({self.method}): {self.weight.size} scores"]
        top = self.selected(top=6)
        lines.append("  weights: " + ", ".join(f"{sid} {w:+.3g}"
                                               for sid, w, _ in top))
        for key in ("condition_number", "negative_weights"):
            if key in self.log:
                lines.append(f"  {key.replace('_', ' ')}: {self.log[key]}")
        return "\n".join(lines)


def meta_pgs(scores, *, n_eff=None, expected_r2=None, method="sqrt_n_eff",
             score_ids=None, ridge=1e-3, center=None, scale=None):
    """Combine same-trait scores with weights that need no phenotype.

    Parameters
    ----------
    scores : array ``(n, K)`` or :class:`~multipgs.panel.ScorePanel`
        The scores in the cohort to be scored. They are used only for their
        means and standard deviations (and, for ``"decorrelated"``, their
        correlation matrix) — no phenotype is read.
    n_eff : array ``(K,)``, optional
        Effective sample size of each discovery GWAS. Required for
        ``method="sqrt_n_eff"``. For case/control studies use
        :func:`ldpred3.n_eff_case_control`, i.e.
        ``4 / (1/n_case + 1/n_control)``.
    expected_r2 : array ``(K,)``, optional
        Expected r² of each score for its own trait, e.g. from
        :func:`multipgs.daetwyler_r2`. Required for ``"expected_r2"``; used by
        ``"decorrelated"`` when given, which otherwise falls back to
        ``n_eff``.
    method : {"sqrt_n_eff", "expected_r2", "decorrelated"}
    ridge : float
        Ridge added to the correlation matrix's diagonal before inversion in
        ``"decorrelated"``. Near-duplicate scores make ``C`` ill-conditioned,
        and an unregularised inverse answers with two enormous weights of
        opposite sign that cancel to noise. It does not rescue a mis-specified
        ``expected_r2``: raising it only drags the answer back toward the
        undecorrelated weighted sum, which it never quite reaches.

        ``"decorrelated"`` requires accuracies that are right per score, not
        merely correctly ordered, because ``C**-1`` amplifies their error along
        the directions where the panel's scores differ least -- which in a
        same-trait panel is noise. Accuracies derived from ``n_eff`` or
        :func:`multipgs.daetwyler_r2` do not meet that bar on real panels: see
        ``docs/algorithm.md``, where both are measured against the truth and
        decorrelating comes out about fifty times worse than not. Use it when
        each component carries its own fitted accuracy, such as LDpred3-auto's
        ``r2_est`` from :func:`multipgs.panel_from_sumstats`; otherwise prefer
        ``method="expected_r2"``.
    center, scale : array ``(K,)``, optional
        Standardization to apply instead of this cohort's own. Pass the
        training cohort's values to score a second cohort on the same scale.

    Returns
    -------
    MetaPGS
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if hasattr(scores, "scores"):          # a ScorePanel
        panel_ids = np.asarray(scores.score_ids, dtype=object)
        if score_ids is None:
            score_ids = panel_ids
        elif ([str(s) for s in np.asarray(score_ids, dtype=object).ravel()]
              != [str(s) for s in panel_ids.ravel()]):
            raise ValueError("score_ids do not match the ScorePanel columns")
        scores = scores.scores
    S = np.asarray(scores, dtype=float)
    if S.ndim != 2:
        raise ValueError("scores must be 2-dimensional")
    if not np.all(np.isfinite(S)):
        raise ValueError("scores contain non-finite values")
    n, K = S.shape
    if K == 0:
        raise ValueError("scores must contain at least one score column")
    if score_ids is None:
        score_ids = np.array([f"score_{j}" for j in range(K)], dtype=object)
    score_ids = np.asarray(score_ids, dtype=object)
    if score_ids.shape != (K,):
        raise ValueError(f"score_ids has {score_ids.size} entries for {K} "
                         f"scores")
    string_ids = [str(s) for s in score_ids]
    if len(set(string_ids)) != K:
        raise ValueError("score_ids must be unique")

    center = (S.mean(axis=0) if center is None
              else np.asarray(center, dtype=float))
    if center.shape != (K,):
        raise ValueError(f"center must have shape ({K},), got {center.shape}")
    if not np.all(np.isfinite(center)):
        raise ValueError("center must contain only finite values")
    if scale is None:
        scale = S.std(axis=0)
    else:
        scale = np.asarray(scale, dtype=float)
    if scale.shape != (K,):
        raise ValueError(f"scale must have shape ({K},), got {scale.shape}")
    if not np.all(np.isfinite(scale)):
        raise ValueError("scale must contain only finite values")
    if np.any(scale < 0):
        raise ValueError("scale must be non-negative")
    try:
        ridge_value = float(ridge)
    except (TypeError, ValueError):
        raise ValueError("ridge must be finite and non-negative") from None
    if not np.isfinite(ridge_value) or ridge_value < 0:
        raise ValueError("ridge must be finite and non-negative")
    dead = scale <= 1e-12
    scale = np.where(dead, 1.0, scale)
    if dead.any():
        # A constant score carries nothing; give it weight 0 rather than let it
        # divide the combination by ~0.
        pass

    rho = _accuracy_vector(method, n_eff, expected_r2, K)
    rho = np.where(dead, 0.0, rho)

    log = {"method": method, "n": int(n), "n_scores": int(K),
           "dead_scores": int(dead.sum())}

    if method == "decorrelated":
        # Do the centering and scaling in place on one copy; ``(S-center)/scale``
        # otherwise holds two n-by-K temporaries at peak.
        Z = S.copy()
        Z -= center
        Z /= scale
        Z[:, dead] = 0.0
        C = (np.ones((1, 1), dtype=float) if K == 1
             else np.corrcoef(Z, rowvar=False))
        C = np.where(np.isfinite(C), C, 0.0)
        C = (C + C.T) * 0.5
        np.fill_diagonal(C, 1.0)
        C[dead, :] = 0.0
        C[:, dead] = 0.0
        C[dead, dead] = 1.0
        A = C + ridge_value * np.eye(K)
        log["condition_number"] = float(np.linalg.cond(A))
        try:
            w = np.linalg.solve(A, rho)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, rho, rcond=None)[0]
        # How far the inverse moved the answer, as description only. This is
        # deliberately *not* a failure detector, and the attempt to make one is
        # worth recording so it is not retried: on a real 24-score panel where
        # this rule returned R2 0.00001 against 0.156 for the same accuracies
        # undecorrelated, the alignment was 0.67 -- higher than the 0.40 of a
        # configuration that scored three hundred times better. Alignment,
        # negative-weight count and condition number all fail to separate those
        # cases, and they must: the four differ only in rho, and it is rho's
        # accuracy that decides the outcome. That is unobservable here, so no
        # in-sample statistic can stand in for it. The precondition is
        # documented instead, in docs/algorithm.md.
        norm_product = float(np.linalg.norm(w) * np.linalg.norm(rho))
        log["rho_alignment"] = (float(w @ rho) / norm_product
                                if norm_product > 0 else float("nan"))
        n_neg = int(np.count_nonzero(w < 0))
        if n_neg:
            # Not an error: a negative weight is the correct answer when a
            # score is a noisier copy of another. It is worth surfacing,
            # because it is also what an ill-conditioned C produces.
            log["negative_weights"] = n_neg
    else:
        w = rho.copy()

    norm = float(np.linalg.norm(w))
    if norm <= 0:
        raise ValueError("all weights came out zero; check n_eff/expected_r2")
    w = w / norm
    beta = np.where(dead, 0.0, w / scale)
    return MetaPGS(beta=beta, weight=w, center=center, scale=scale,
                   score_ids=score_ids, method=method, log=log)


def _accuracy_vector(method, n_eff, expected_r2, K):
    """Expected accuracy per score, on an arbitrary common scale."""
    if method == "sqrt_n_eff" or (method == "decorrelated"
                                  and expected_r2 is None):
        if n_eff is None:
            raise ValueError(f"method={method!r} needs n_eff")
        v = np.asarray(n_eff, dtype=float).ravel()
        if v.size != K:
            raise ValueError(f"n_eff has {v.size} entries for {K} scores")
        if np.any(v <= 0) or not np.all(np.isfinite(v)):
            raise ValueError("n_eff must be finite and positive")
        return np.sqrt(v)
    if expected_r2 is None:
        raise ValueError(f"method={method!r} needs expected_r2")
    v = np.asarray(expected_r2, dtype=float).ravel()
    if v.size != K:
        raise ValueError(f"expected_r2 has {v.size} entries for {K} scores")
    v = np.where(np.isfinite(v), v, 0.0)
    if np.all(v <= 0):
        raise ValueError("every expected_r2 is non-positive; nothing to weight")
    return np.sqrt(np.maximum(v, 0.0))
