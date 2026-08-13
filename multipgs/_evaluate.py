"""Summary-statistic evaluation of a fixed combination."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ._moments import _directional_score_moments, _project_c_to_gram_range


#: What the evaluation statistics were computed against, and what each is worth.
#:
#: The distinction is the whole difference between a publishable number and a
#: meaningless one, and it is not visible in the number itself — a regime C R²
#: looks exactly like a regime A R², only larger. Every evaluation this module
#: returns carries its regime, and :func:`evaluate_sumstat` determines it from
#: the inputs rather than taking the caller's word for it.
REGIMES = {
    "A": "fixed after all model selection, then evaluated on an untouched "
         "independent GWAS (clean external assessment)",
    "B": "used to choose the model, either an independent tuning GWAS or a "
         "PUMAS pseudo-validation split (tuning, not clean assessment)",
    "C": "fitted and evaluated on the same unsplit GWAS "
         "(optimistically biased; not a validation)",
}


@dataclass
class SumstatEval:
    """Accuracy of a fixed combination, measured from summary statistics.

    ``regime`` is one of the keys of :data:`REGIMES` and is the first thing to
    read: ``"C"`` means the combination was scored against the statistics it was
    fitted on and the number is an upper bound, not an estimate.
    """

    r2: float
    mse: float
    regime: str
    var_y: float
    n_scores: int
    log: dict = field(default_factory=dict)

    @property
    def is_validation(self):
        """Whether this is clean validation after every model choice was fixed."""
        return self.regime == "A"

    @property
    def is_assessment(self):
        """Whether this is clean assessment after every model choice was fixed."""
        return self.regime == "A"

    def summary(self):
        lines = [f"summary-statistic evaluation (regime {self.regime}): "
                 f"R2 {self.r2:.4f}, MSE {self.mse:.4f}",
                 f"  {REGIMES.get(self.regime, 'unknown provenance')}"]
        if self.regime == "B":
            lines.append("  this was used for tuning — do not report it as a "
                         "clean assessment")
        elif not self.is_validation:
            lines.append("  this is not a validation — do not report it as one")
        if self.log.get("warning"):
            lines.append(f"  {self.log['warning']}")
        for key in ("moment_warning", "mse_moment_warning"):
            if self.log.get(key):
                lines.append(f"  {self.log[key]}")
        return "\n".join(lines)


def evaluate_sumstat(beta, c_eval, gram_eval, *, var_y=1.0, regime=None,
                     fitted_on=None):
    """Accuracy of a fixed combination ``beta`` against evaluation moments.

    Implements the two summary-statistic identities, applied in score space to
    the combined weight vector ``W beta``:

    .. math::

        R^2(a) = \\frac{(a^\\top c)^2}{(a^\\top G a)\\,\\mathrm{var}(y)},
        \\qquad
        \\mathrm{MSE}(a) = \\mathrm{var}(y) - 2 a^\\top c + a^\\top G a.

    Parameters
    ----------
    beta : array_like, shape (K,) or SumstatFit
        The combination to score, on the raw score scale.
    c_eval, gram_eval : array_like
        Score-space moments from the **evaluation** data: ``W^T z_eval`` and
        ``W^T D_eval W``, on the same score scaling ``beta`` is on.
        The metric projects ``c_eval`` onto the scale-invariant positive range
        of ``gram_eval`` and logs the discarded finite-reference null component.
    var_y : float
        Phenotype variance on the scale ``c_eval`` was formed on. The MSE is
        meaningless if this is wrong, and the R² is off by a constant factor.
    regime : {"A", "B", "C"}, optional
        Declare the provenance. Omitted, it is inferred: identical to the
        fitting moments means regime C.
    fitted_on : array_like, optional
        The observed or identifiable ``c`` the combination was fitted on, so
        that regime C can be detected rather than trusted. A
        :class:`SumstatFit` automatically recognizes both its ``c_raw`` and
        projected ``r``.

    Returns
    -------
    SumstatEval
    """
    if not isinstance(beta, np.ndarray):
        from .sumstat import SumstatFit
        if isinstance(beta, SumstatFit):
            if fitted_on is None:
                c_candidate = np.asarray(c_eval, dtype=float)
                fitted_on = beta.c_raw
                if (c_candidate.shape == np.asarray(beta.r).shape
                        and np.array_equal(c_candidate, np.asarray(beta.r))):
                    fitted_on = beta.r
            beta = beta.beta
    beta = np.asarray(beta, dtype=float).ravel()
    c_eval = np.asarray(c_eval, dtype=float).ravel()
    gram_eval = np.asarray(gram_eval, dtype=float)
    k = beta.size
    if c_eval.size != k or gram_eval.shape != (k, k):
        raise ValueError(f"beta is length {k}, so c_eval must be ({k},) and "
                         f"gram_eval ({k}, {k}); got {c_eval.shape} and "
                         f"{gram_eval.shape}")
    if not (np.all(np.isfinite(beta)) and np.all(np.isfinite(c_eval))
            and np.all(np.isfinite(gram_eval))):
        raise ValueError("beta, c_eval and gram_eval must be finite")
    var_y = float(var_y)
    if not np.isfinite(var_y) or var_y <= 0.0:
        raise ValueError("var_y must be finite and strictly positive")

    same = (fitted_on is not None
            and np.asarray(fitted_on).shape == c_eval.shape
            and np.array_equal(np.asarray(fitted_on, dtype=float), c_eval))
    if regime is None:
        # Only regime C is inferable: it is the one with an observable
        # signature, namely evaluation moments identical to the fitting ones.
        # "Not C" does NOT imply A — a PUMAS pseudo-validation split is also
        # not identical to its pseudo-training split, and defaulting to A there
        # would stamp "clean external validation" on a regime B number. An
        # unprovable label is worse than none, so this refuses instead.
        if not same:
            raise ValueError(
                "cannot infer the evaluation regime: these moments differ from "
                "the ones the combination was fitted on, which rules out C but "
                "does not distinguish A (an untouched assessment GWAS) from B "
                "(tuning or pseudo-validation moments). Pass regime='A' or "
                "regime='B' "
                "explicitly — the two are not interchangeable and the "
                "difference is invisible in the number.")
        regime = "C"
    regime = str(regime).upper()
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {sorted(REGIMES)}, got "
                         f"{regime!r}")
    if same and regime in ("A", "B"):
        raise ValueError(
            f"regime {regime} is incompatible with evaluation moments equal "
            "to the fitting moments; this is regime C same-data evaluation")
    if fitted_on is not None and not same and regime == "C":
        raise ValueError(
            "regime C was declared, but c_eval differs from the fitting "
            "moments; declare A or B according to its actual role")

    # A fixed vector uses only one scalar direction. An unused noisy/null score
    # or an indefinite unused subspace must not invalidate that direction.
    observed_beta_c = float(beta @ c_eval)
    c_identifiable, discarded_c_norm, discarded_c_fraction = (
        _project_c_to_gram_range(c_eval, gram_eval))
    num, quad, quad_tol, gram_eval, gram_asymmetry = (
        _directional_score_moments(
            beta, gram_eval, c_identifiable, var_y, "gram_eval"))
    r2 = (float("nan") if quad <= quad_tol
          else (num * num) / (quad * var_y))
    mse = var_y - 2.0 * num + quad

    log = {"n_nonzero": int(np.sum(beta != 0.0)),
           "beta_c": num, "beta_G_beta": quad,
           "observed_beta_c": observed_beta_c,
           "discarded_beta_c_null": observed_beta_c - num,
           "discarded_c_null_norm": discarded_c_norm,
           "discarded_c_null_fraction": discarded_c_fraction,
           "gram_asymmetry": gram_asymmetry,
           "regime_detail": REGIMES[regime]}
    if regime == "C":
        log["warning"] = ("evaluated on the same summary statistics it was "
                          "fitted on; this is an upper bound, not a validation")
    if np.isfinite(r2) and r2 > 1.0:
        log["moment_warning"] = (
            f"plug-in evaluation R2 is {r2:.6g}, above 1; external moments "
            "are noisy, but check scaling, alignment, var_y, and the LD source")
    if mse < 0.0:
        log["mse_moment_warning"] = (
            f"plug-in MSE is {mse:.6g}, below zero; retain it only as a noisy "
            "summary-moment diagnostic")
    if mse > var_y:
        log["mse_warning"] = ("MSE exceeds var(y): this combination predicts "
                              "worse than the mean")
    return SumstatEval(r2=r2, mse=mse, regime=regime, var_y=var_y, n_scores=k,
                       log=log)
