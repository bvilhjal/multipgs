"""The genotype-free service entry point: an LD cache, a prepared trait, weights."""

import numpy as np
import pytest

from multipgs import fit_prepared_panel

_NUC = ("A", "C", "G", "T")
_AMBIGUOUS = ({"A", "T"}, {"C", "G"})


@pytest.fixture(scope="module")
def reference(tmp_path_factory):
    """A real ldpred3 LD cache, its HWE scale, and K component weight files.

    Built the way a service's own reference is: dosages, ``compute_ld_blocks``,
    ``save_ld_blocks`` with the allele/coordinate/frequency provenance a
    multi-PGS panel needs. The component weight files stand for earlier
    univariate runs against this same cache.
    """
    from ldpred3 import compute_ld_blocks, save_ld_blocks
    from ldpred3.interop import write_weights

    rng = np.random.default_rng(0)
    m, n, k = 300, 800, 4
    directory = tmp_path_factory.mktemp("reference")
    pos = np.cumsum(rng.exponential(2000.0, size=m)).astype(np.int64) + 1
    chrom = np.where(np.arange(m) < m // 2, "1", "2")
    ids = np.array([f"rs{1_000_000 + i}" for i in range(m)])
    pairs = []
    while len(pairs) < m:
        pair = (rng.choice(_NUC), rng.choice(_NUC))
        if pair[0] != pair[1] and set(pair) not in _AMBIGUOUS:
            pairs.append(pair)
    ea = np.array([p[0] for p in pairs])
    oa = np.array([p[1] for p in pairs])
    frequency = rng.uniform(0.05, 0.5, size=m)
    dosage = ((rng.random((n, m)) < frequency).astype(np.int8)
              + (rng.random((n, m)) < frequency).astype(np.int8))
    af = dosage.mean(axis=0) / 2.0
    blocks = compute_ld_blocks(np.ascontiguousarray(dosage), chrom=chrom,
                               block_size=100)
    cache = directory / "ref.ld.npz"
    save_ld_blocks(cache, blocks, ids, reference_af=af, n_ref=n,
                   counted_allele=ea, other_allele=oa, chrom=chrom, pos=pos)
    sd = np.sqrt(2.0 * af * (1.0 - af))

    components, truth = [], []
    for j in range(k):
        sel = np.sort(rng.choice(m, size=80, replace=False))
        weight = rng.normal(scale=0.03, size=sel.size)
        path = directory / f"job{j}.weights.tsv"
        write_weights(str(path), id=ids[sel], chrom=chrom[sel], pos=pos[sel],
                      effect_allele=ea[sel], other_allele=oa[sel],
                      weight=weight, af=af[sel], sd=sd[sel], sd_source="hwe")
        components.append(str(path))
        truth.append((sel, weight))
    return {"cache": str(cache), "m": m, "af": af, "sd": sd,
            "components": components, "truth": truth, "n_blocks": len(blocks),
            "score_ids": [f"job{j}" for j in range(k)], "dir": directory}


def _trait(reference, rng, *, loadings=(0.6, 0.3), noise=0.02, n_eff=1.2e5,
           coverage=260):
    """A prepared target trait whose signal is a known mix of the components."""
    from ldpred3.prepare import PreparedTrait

    m = reference["m"]
    signal = np.zeros(m)
    for weight, (sel, values) in zip(loadings, reference["truth"]):
        signal[sel] += weight * values
    indices = np.sort(rng.choice(m, size=coverage, replace=False))
    z = signal[indices] * 6.0 + rng.normal(scale=noise, size=indices.size)
    return PreparedTrait(indices=indices, beta_hat=z.copy(),
                         n_eff=np.full(indices.size, n_eff), z=z,
                         eaf=reference["af"][indices], n_cache=m)


def test_fit_prepared_panel_recovers_the_contributing_scores(reference):
    """The whole genotype-free route, and it has to find the real signal.

    Two of the four components carry the trait; the other two are noise. A
    combination that cannot rank them is not doing its job, whatever the
    plumbing reports.
    """
    stages = []
    result = fit_prepared_panel(
        reference["cache"], _trait(reference, np.random.default_rng(1)),
        reference["components"], score_ids=reference["score_ids"],
        tune="pumas", weights_independent_of_z=True, rng=0,
        progress=lambda done, total, stage: stages.append(stage))

    ranked = [row["score_id"] for row in result.coefficient_table()]
    assert set(ranked[:2]) == {"job0", "job1"}, ranked
    coefficients = {row["score_id"]: row["beta_std"]
                    for row in result.coefficient_table()}
    # The loadings were 0.6 and 0.3, so the first should carry clearly more.
    assert coefficients["job0"] > coefficients["job1"] > 0.0
    assert abs(coefficients["job2"]) < coefficients["job1"]
    assert abs(coefficients["job3"]) < coefficients["job1"]

    assert result.regime == "B"
    assert {"align", "ld"} <= set(stages)
    assert result.cache_provenance["n_variants"] == reference["m"]
    assert result.cache_provenance["n_ref"] == 800
    assert result.n_eff_policy["policy"] == "trait_median"
    assert result.n_eff == pytest.approx(1.2e5)
    assert 0.0 < result.trait_log["coverage"] <= 1.0
    assert "regime B" in result.summary()


def test_fit_prepared_panel_writes_a_deployable_weight_file(reference, tmp_path):
    """The artefact is an ldpred3 weight file on the reference's own scale."""
    from ldpred3.interop import read_weights

    result = fit_prepared_panel(
        reference["cache"], _trait(reference, np.random.default_rng(2)),
        reference["components"], score_ids=reference["score_ids"],
        tune="none")
    path = result.write_weights(tmp_path / "multi.weights.tsv")
    table = read_weights(path)
    assert np.asarray(table.weight).size == result.log["n_combined_variants"]
    assert np.all(np.isfinite(np.asarray(table.weight, dtype=float)))
    # SD_REF is the reference HWE approximation, and the file must say so:
    # a reader has to know it is not a target dosage SD.
    assert table.sd_source == "hwe"
    assert result.regime == "C"


def test_fit_prepared_panel_requires_the_leakage_acknowledgement(reference):
    """PUMAS holds the component weights fixed, so it cannot remove leakage."""
    with pytest.raises(ValueError, match="weights_independent_of_z"):
        fit_prepared_panel(
            reference["cache"], _trait(reference, np.random.default_rng(3)),
            reference["components"], tune="pumas")


def test_fit_prepared_panel_refuses_a_trait_from_another_reference(reference):
    """A prepared trait carries the reference length it was built against."""
    from ldpred3.prepare import PreparedTrait

    trait = _trait(reference, np.random.default_rng(4))
    foreign = PreparedTrait(indices=trait.indices, beta_hat=trait.beta_hat,
                            n_eff=trait.n_eff, z=trait.z, eaf=trait.eaf,
                            n_cache=reference["m"] + 1)
    with pytest.raises(ValueError, match="not the same LD reference"):
        fit_prepared_panel(reference["cache"], foreign,
                           reference["components"], tune="none")


def test_fit_prepared_panel_refuses_an_independent_tuning_gwas(reference):
    """That route needs its own aligned panel and LD, so it is not offered."""
    with pytest.raises(ValueError, match="tune must be"):
        fit_prepared_panel(reference["cache"],
                           _trait(reference, np.random.default_rng(5)),
                           reference["components"], tune="independent")


def test_fit_prepared_panel_needs_a_component(reference):
    with pytest.raises(ValueError, match="at least one component"):
        fit_prepared_panel(reference["cache"],
                           _trait(reference, np.random.default_rng(6)), [])


def test_fit_prepared_panel_borrows_a_caller_owned_cache(reference):
    """A service opens the cache once and fits several panels from it."""
    from ldpred3.interop import prepare_ld_cache

    context = prepare_ld_cache(reference["cache"])
    try:
        first = fit_prepared_panel(
            context, _trait(reference, np.random.default_rng(7)),
            reference["components"], tune="none")
        second = fit_prepared_panel(
            context, _trait(reference, np.random.default_rng(7)),
            reference["components"], tune="none")
        assert np.allclose(first.beta, second.beta)
        assert not context.closed          # borrowed, not consumed
    finally:
        context.close()
    with pytest.raises(ValueError, match="closed"):
        fit_prepared_panel(context,
                           _trait(reference, np.random.default_rng(8)),
                           reference["components"], tune="none")


def _catalog_scoring_file(path, reference, rng, size=70):
    """A PGS Catalog-format file: raw allele-count weights, no reference scale."""
    m = reference["m"]
    sel = np.sort(rng.choice(m, size=size, replace=False))
    weight = rng.normal(scale=0.05, size=sel.size)
    from ldpred3 import load_ld_blocks
    blocks, ids, meta = load_ld_blocks(reference["cache"], return_metadata=True)
    close = getattr(blocks, "close", None)
    ids = np.asarray(ids)
    ea = np.asarray(meta["counted_allele"])
    oa = np.asarray(meta["other_allele"])
    if close:
        close()
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("#pgs_id=PGS000001\n#genome_build=GRCh37\n")
        fh.write("rsID\teffect_allele\tother_allele\teffect_weight\n")
        for i, value in zip(sel, weight):
            fh.write(f"{ids[i]}\t{ea[i]}\t{oa[i]}\t{value:.8g}\n")
    return str(path)


def test_a_mixed_panel_records_each_component_s_scale(reference, tmp_path):
    """An HWE-approximated component must never look like a verified one.

    A Catalog scoring file counts raw alleles and carries no AF_REF/SD_REF, so
    it can only reach the standardized scale through the reference's HWE
    approximation. The folded weight file inherits that weaker assumption, so
    which components carry which has to stay visible.
    """
    from multipgs import Component

    rng = np.random.default_rng(21)
    catalog = _catalog_scoring_file(tmp_path / "PGS000001.txt", reference, rng)
    components = [Component(reference["components"][0], score_id="job0"),
                  Component(reference["components"][1], score_id="job1"),
                  Component(catalog, kind="scoring_file",
                            score_id="PGS000001")]
    result = fit_prepared_panel(
        reference["cache"], _trait(reference, np.random.default_rng(22)),
        components, tune="none")

    by_id = {row["score_id"]: row for row in result.coefficient_table()}
    assert by_id["job0"]["scale_source"] == "ldpred3_weight_file"
    assert by_id["job0"]["scale_verified"] is True
    assert by_id["PGS000001"]["scale_source"] == "hwe_from_af"
    assert by_id["PGS000001"]["scale_verified"] is False
    assert by_id["PGS000001"]["kind"] == "scoring_file"
    assert result.align_log["n_hwe_approximated"] == 1
    assert "HWE" in result.align_log["warning"]
    assert "PGS000001" in result.align_log["warning"]
    assert "HWE-approximated" in result.summary()
    # It still folds into one deployable file on the reference's scale.
    assert result.log["n_combined_variants"] > 0


def test_component_kind_is_validated():
    from multipgs import Component

    with pytest.raises(ValueError, match="component kind must be"):
        Component("x", kind="sumstats")


def test_duplicate_component_ids_are_refused(reference):
    from multipgs import Component

    same = reference["components"][0]
    with pytest.raises(ValueError, match="unique across the whole panel"):
        fit_prepared_panel(
            reference["cache"], _trait(reference, np.random.default_rng(23)),
            [Component(same, score_id="dup"), Component(same, score_id="dup")],
            tune="none")


def test_the_fit_carries_a_per_block_accuracy_decomposition(reference):
    """The plug-in R2 is a point estimate with no interval and no null.

    Keeping the per-block sums is what lets a caller attach both, so they are
    computed while the cache is open and checked against the identity here.
    """
    result = fit_prepared_panel(
        reference["cache"], _trait(reference, np.random.default_rng(24)),
        reference["components"], score_ids=reference["score_ids"],
        tune="none")
    assert result.log["accuracy_blocks_error"] is None
    u, v = np.asarray(result.accuracy_u), np.asarray(result.accuracy_v)
    assert u.size == v.size == reference["n_blocks"]
    numerator, denominator, n_blocks = result.accuracy_totals()
    assert n_blocks == reference["n_blocks"]
    assert numerator == pytest.approx(u.sum())
    assert denominator == pytest.approx(v.sum())
    # Chromosome labels, the more conservative jackknife unit.
    assert result.accuracy_groups is not None
    assert set(np.asarray(result.accuracy_groups).tolist()) <= {"1", "2"}
    # ppb consumes these directly; check the three readouts it provides.
    ppb = pytest.importorskip("ppb")
    raw, corrected, se = ppb.corrected_r2(numerator, denominator, result.n_eff)
    assert np.isfinite(raw) and np.isfinite(corrected) and se >= 0
    assert ppb.r2_block_jackknife(u, v).se >= 0
    assert ppb.r2_block_jackknife(u, v, groups=result.accuracy_groups).se >= 0
    null = ppb.sign_flip_null(u, v)
    # Cauchy-Schwarz bounds the coherence statistic by sqrt(n_blocks).
    assert abs(null.z) <= np.sqrt(u.size) + 1e-9
