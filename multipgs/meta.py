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
  does **not** make it the better of the two ``C = I`` rules — see below.
* ``method="decorrelated"`` drops the independence assumption. The
  :math:`\\sqrt{N}` rule is the inverse-variance weighting of *independent*
  estimates, and discovery GWAS are frequently not independent — a consortium
  meta-analysis usually contains the cohort you are also using separately.
  With :math:`C` the correlation matrix of the scores in the target cohort
  (estimable with no phenotype, straight off the panel) and :math:`\\rho` the
  vector of expected accuracies, the optimal weights are
  :math:`w \\propto C^{-1}\\rho`, which discounts a score for the information
  it shares with the others.

**Use ``"decorrelated"``, and give it ``expected_r2`` rather than ``n_eff``.**
Three same-trait scores (:math:`n_\\mathrm{eff}` 150k/60k/20k,
:math:`h^2 = 0.4`, 5,000 causal variants, so :math:`Nh^2/M` = 12, 4.8, 1.6),
r² against a simulated phenotype as the discovery cohorts are made to overlap
(``simulate_same_trait_panel``, n = 40,000):

===============  =============  ==============  =============  =============  =============
overlap          best single    ``sqrt_n_eff``  ``expected_r2``  decorr(n_eff)  decorr(r²)
===============  =============  ==============  =============  =============  =============
none             0.364          0.373           0.366          0.190          **0.375**
moderate (0.3)   0.364          0.361           0.352          0.179          **0.368**
strong (0.6)     0.364          0.350           0.339          0.173          **0.367**
severe (0.8)     0.364          0.343           0.330          0.170          **0.374**
===============  =============  ==============  =============  =============  =============

Three readings, and the second is counter-intuitive:

1. **Decorrelation wins everywhere**, and past mild overlap it is the *only*
   rule still beating the best single score. Both ``C = I`` rules fall below
   it once the discovery cohorts share individuals, because they keep adding
   information they have already counted.
2. **``sqrt_n_eff`` beats ``expected_r2``**, which is not what "use the better
   estimate of accuracy" would suggest. The optimal weight is
   :math:`R_k/(1-R_k^2) = R_k(1+x_k) \\propto \\sqrt{x_k}\\sqrt{1+x_k}`, while
   :math:`\\sqrt{n_\\mathrm{eff}} \\propto \\sqrt{x_k}` and
   :math:`\\sqrt{R^2_k} \\propto \\sqrt{x_k/(1+x_k)}`. Accuracy *saturates* as
   power grows and the optimal weight does not, so weighting by accuracy
   under-weights the large GWAS — by a factor 5 across this panel, against
   2.2 for :math:`\\sqrt{N}`. Sample size is the cruder statistic and the
   better-shaped one.
3. **``decorrelated`` with ``n_eff`` is much worse than doing nothing.**
   :math:`C^{-1}` amplifies any error in :math:`\\rho`, and
   :math:`\\sqrt{n_\\mathrm{eff}}` is a mis-specified :math:`\\rho` — fine as a
   direct weight, ruinous inside a matrix inverse. The combination is permitted
   because :math:`h^2` and polygenicity are not always available, but prefer
   ``sqrt_n_eff`` over ``decorrelated`` if ``expected_r2`` is out of reach.

``docs/theory.md`` derives all three.

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

    def multi_pgs(self, scores):
        """The combined score for new individuals: ``scores @ beta``.

        Scale and location are arbitrary — the weights are only defined up to a
        positive constant — which is why this returns the combination itself and
        no intercept. R², AUC and ranking are unaffected; if you need a
        calibrated predictor, regress this on the phenotype in a training set,
        or use :func:`multipgs.multi_pgs_fit`.
        """
        s = np.asarray(scores, dtype=float)
        if s.ndim == 1:
            s = s[None, :]
        if s.shape[1] != self.beta.size:
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
        opposite sign that cancel to noise.
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
        if score_ids is None:
            score_ids = np.asarray(scores.score_ids, dtype=object)
        scores = scores.scores
    S = np.asarray(scores, dtype=float)
    if S.ndim != 2:
        raise ValueError("scores must be 2-dimensional")
    if not np.all(np.isfinite(S)):
        raise ValueError("scores contain non-finite values")
    n, K = S.shape
    if score_ids is None:
        score_ids = np.array([f"score_{j}" for j in range(K)], dtype=object)
    score_ids = np.asarray(score_ids, dtype=object)
    if score_ids.size != K:
        raise ValueError(f"score_ids has {score_ids.size} entries for {K} "
                         f"scores")

    center = S.mean(axis=0) if center is None else np.asarray(center,
                                                              dtype=float)
    if scale is None:
        scale = S.std(axis=0)
    else:
        scale = np.asarray(scale, dtype=float)
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
        Z = (S - center) / scale
        Z[:, dead] = 0.0
        C = np.corrcoef(Z, rowvar=False)
        C = np.where(np.isfinite(C), C, 0.0)
        C = (C + C.T) * 0.5
        np.fill_diagonal(C, 1.0)
        C[dead, :] = 0.0
        C[:, dead] = 0.0
        C[dead, dead] = 1.0
        A = C + float(ridge) * np.eye(K)
        log["condition_number"] = float(np.linalg.cond(A))
        try:
            w = np.linalg.solve(A, rho)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, rho, rcond=None)[0]
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
