"""End-to-end multi-PGS on simulated data: ``python -m examples.minimal``.

Simulates a cohort and a set of PGS Catalog-format scoring files, builds the
score panel, fits the combination, measures it in held-out individuals against
the best single score, and writes the deployable weight file. Everything runs in
a temporary directory in a few seconds; nothing here needs real data.
"""

import os
import tempfile

import numpy as np

from multipgs import (combine_weights, evaluate, meta_pgs, multi_pgs_fit,
                      panel_from_catalog, r2, simulate_target)


def main():
    rng = np.random.default_rng(0)
    n, n_scores = 3000, 12

    with tempfile.TemporaryDirectory() as work:
        # ------------------------------------------------------------------
        # 1. A target cohort and some scoring files. In a real analysis these
        #    come from your genotypes and the PGS Catalog.
        # ------------------------------------------------------------------
        print("simulating a cohort and PGS Catalog scoring files ...")
        sim = simulate_target(os.path.join(work, "cohort"), n=n,
                              n_variants=2000, n_scores=n_scores,
                              n_per_score=300, seed=1)

        # A phenotype driven by three of the scores, plus covariates.
        truth = sim["true_scores"]
        z = (truth - truth.mean(0)) / truth.std(0)
        g = z[:, 0] * 1.0 - z[:, 3] * 0.7 + z[:, 7] * 0.5
        g = g / g.std()
        covar = rng.normal(size=(n, 2))
        y = 0.6 * g + 0.5 * covar[:, 0] + rng.normal(size=n) * 0.8

        train = np.arange(n) < 2000
        test = ~train

        # ------------------------------------------------------------------
        # 2. Build the n x K score matrix. One pass over the genotypes.
        # ------------------------------------------------------------------
        panel = panel_from_catalog(sim["scoring_files"], sim["prefix"])
        print()
        print(panel.summary())

        # ------------------------------------------------------------------
        # 3. Fit the combination on the training individuals only.
        # ------------------------------------------------------------------
        fit = multi_pgs_fit(panel.scores[train], y[train], covar=covar[train],
                            score_ids=panel.score_ids, n_folds=5, seed=1)
        print()
        print(fit.summary())
        print("  selected:", ", ".join(
            f"{sid} ({b:+.2f})" for sid, b, _ in fit.selected(top=5)))

        # ------------------------------------------------------------------
        # 4. Measure it where it was not trained, against the best single
        #    score chosen on the same training data.
        # ------------------------------------------------------------------
        best = max(range(panel.n_scores),
                   key=lambda k: r2(y[train], panel.scores[train, k]))
        print()
        print("held-out accuracy")
        print(f"  best single score ({panel.score_ids[best]}):"
              f" r2 = {r2(y[test], panel.scores[test, best]):.4f}")
        print(f"  multi-PGS:                     "
              f" r2 = {r2(y[test], fit.multi_pgs(panel.scores[test])):.4f}")
        print(f"  (oracle, the simulated truth):  "
              f"r2 = {r2(y[test], g[test]):.4f}")
        print()
        print(evaluate(y[test], fit.multi_pgs(panel.scores[test]),
                       covar=covar[test], n_boot=500, seed=1))

        # ------------------------------------------------------------------
        # 5. Collapse it into one weight file. This is the artefact you ship:
        #    a new cohort is scored from it with no reference to the K inputs.
        # ------------------------------------------------------------------
        path = os.path.join(work, "multi.weights")
        table = combine_weights(panel, fit, path=path)
        print()
        print(f"combined {fit.n_selected} scores into {table['weight'].size} "
              f"per-variant weights")

        from ldpred3 import score_from_weights
        redone = score_from_weights(path, sim["prefix"], scaling="frozen")
        agreement = np.corrcoef(redone.scores,
                                fit.multi_pgs(panel.scores))[0, 1]
        print(f"  scoring from that file reproduces the model: "
              f"corr = {agreement:.10f}")

        # ------------------------------------------------------------------
        # 6. The training-free alternative, for comparison. It assumes every
        #    score targets one trait -- which is false here, so it does badly.
        #    That contrast is the point: use meta_pgs only when it applies.
        # ------------------------------------------------------------------
        n_eff = np.full(panel.n_scores, 50_000.0)
        combined = meta_pgs(panel, n_eff=n_eff)
        print()
        print("training-free meta_pgs on the same (heterogeneous) panel:"
              f" r2 = {r2(y[test], combined.multi_pgs(panel.scores[test])):.4f}"
              "\n  -- lower by design: sqrt(n_eff) weighting assumes every "
              "score\n     targets the same trait, and here they do not.")


if __name__ == "__main__":
    main()
