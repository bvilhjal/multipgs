"""Synthetic-data contracts that downstream tests rely on."""

import numpy as np

from multipgs import simulate_panel


def test_beta_true_reconstructs_the_reported_genetic_value():
    sim = simulate_panel(n=500, n_scores=12, n_causal=4, seed=11)
    standardized = ((sim.scores - sim.scores.mean(axis=0))
                    / sim.scores.std(axis=0))
    assert np.allclose(standardized @ sim.beta_true, sim.genetic_value,
                       atol=1e-12)
    np.testing.assert_allclose(np.std(sim.genetic_value), 1.0, atol=1e-12)
