"""Optional end-to-end checks across the sibling package weight seam."""

import numpy as np
import pytest

from multipgs import panel_from_weights, simulate_target


def test_bipred_and_gwfm_weights_roundtrip_through_multipgs(tmp_path):
    """Real sibling fits export scores identical to direct LDpred3 scoring."""
    bipred = pytest.importorskip("bipred")
    gwfm = pytest.importorskip("gwfm")
    from ldpred3 import run_ldpred3_prs, score_from_weights
    from ldpred3.interop import prepare_ld_cache

    made = simulate_target(
        str(tmp_path / "target"), n=60, n_variants=24, n_scores=1,
        missing=0.0, seed=95)
    variants = made["variants"]
    rng = np.random.default_rng(96)
    beta1 = rng.normal(0.0, 0.015, len(variants))
    beta2 = 0.5 * beta1 + rng.normal(0.0, 0.01, len(variants))

    def write_gwas(path, beta):
        rows = ["SNP\tCHR\tBP\tA1\tA2\tBETA\tSE\tN"]
        for j in range(len(variants)):
            rows.append(
                f"{variants.id[j]}\t{variants.chrom[j]}\t{variants.pos[j]}\t"
                f"{variants.a1[j]}\t{variants.a2[j]}\t"
                f"{beta[j]:.8g}\t0.02\t5000")
        path.write_text("\n".join(rows) + "\n", encoding="utf-8")

    gwas1, gwas2 = tmp_path / "trait1.tsv", tmp_path / "trait2.tsv"
    write_gwas(gwas1, beta1)
    write_gwas(gwas2, beta2)
    cache = tmp_path / "shared.npz"
    run_ldpred3_prs(
        gwas1, made["prefix"], method="inf", h2=0.2,
        subset_to_sumstats=False, ld_out=cache, score=False, qc=False,
        sd_check=False, af_check=False, ld_int8=False, block_size=12)

    bipred_path = tmp_path / "bipred.weights"
    gwfm_path = tmp_path / "gwfm.weights"
    with prepare_ld_cache(cache) as prepared:
        with bipred.prepare_bivariate_sumstats(
                prepared, gwas1, gwas2, n_eff1=5000, n_eff2=5000,
                qc=False) as inputs:
            bivariate = bipred.ldpred3_auto_bivariate_blocks(
                inputs.blocks, inputs.beta_hat1, inputs.beta_hat2,
                inputs.n_eff1, inputs.n_eff2,
                burn_in=2, num_iter=4, seed=97)
            source = {value: j for j, value in enumerate(variants.id)}
            index = np.array([source[value] for value in inputs.id])
            dosage = made["dosage"][:, index].astype(float)
            bivariate.write_weights(
                bipred_path, trait=1, id=inputs.id, chrom=inputs.chrom,
                pos=inputs.pos, effect_allele=inputs.effect_allele,
                other_allele=inputs.other_allele,
                af=dosage.mean(axis=0) / 2.0, sd=dosage.std(axis=0))

        functional = gwfm.fit_from_ldpred3(
            prepared, gwas1, scale=0.002, qc=False,
            burn_in=2, num_iter=4, seed=98)
        functional.write_weights(gwfm_path)

    paths = [bipred_path, gwfm_path]
    panel = panel_from_weights(paths, made["prefix"])
    for column, (path, scaling) in enumerate(
            zip(paths, ("frozen", "target"))):
        direct = score_from_weights(path, made["prefix"], scaling=scaling)
        np.testing.assert_array_equal(panel.sample_iid, direct.sample_iid)
        # Frozen scaling re-reads %.8g AF_REF/SD_REF from the weights file; see
        # FROZEN_ROUNDTRIP_ATOL in test_panel for the bound this implies.
        np.testing.assert_allclose(
            panel.scores[:, column], direct.scores, atol=1e-7)
