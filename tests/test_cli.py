"""The command line, end to end."""

import numpy as np
import pytest

from multipgs import simulate_target
from multipgs.cli import _read_score_vector, main


def _write_pheno(path, iid, values, name="PHENO"):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"FID\tIID\t{name}\n")
        for i, v in zip(iid, values):
            fh.write(f"{i}\t{i}\t{v:.6g}\n")
    return str(path)


def _write_covar(path, iid, covar):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("FID\tIID\t" + "\t".join(f"C{j}" for j in
                                          range(covar.shape[1])) + "\n")
        for i, row in zip(iid, covar):
            fh.write(f"{i}\t{i}\t" + "\t".join(f"{v:.6g}" for v in row) + "\n")
    return str(path)


def _write_score_vector(path, rows, value_name="VALUE"):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"SCORE\t{value_name}\n")
        for score_id, value in rows:
            fh.write(f"{score_id}\t{value}\n")
    return str(path)


@pytest.fixture
def cohort(tmp_path):
    target = simulate_target(str(tmp_path / "sim"), n=200, n_variants=250,
                             n_scores=4, seed=5)
    rng = np.random.default_rng(0)
    y = 2 * target["true_scores"][:, 0] + rng.normal(size=200)
    target["pheno"] = _write_pheno(tmp_path / "pheno.tsv",
                                   target["sample_iid"], y)
    target["covar"] = _write_covar(tmp_path / "covar.tsv",
                                   target["sample_iid"],
                                   rng.normal(size=(200, 2)))
    target["y"] = y
    return target


def test_panel_then_fit_then_evaluate(cohort, tmp_path, capsys):
    scores = str(tmp_path / "scores.tsv")
    assert main(["panel", "--catalog", *cohort["scoring_files"],
                 "--plink", cohort["prefix"], "--out", scores, "--quiet"]) == 0
    capsys.readouterr()

    coefs = str(tmp_path / "coefs.tsv")
    combined = str(tmp_path / "combined.tsv")
    assert main(["fit", "--scores", scores, "--pheno", cohort["pheno"],
                 "--covar", cohort["covar"], "--out", coefs,
                 "--out-score", combined, "--folds", "4", "--n-lambda", "20",
                 "--seed", "0", "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "multi-PGS" in out
    header = open(coefs, encoding="utf-8").readline().split()
    assert header == ["SCORE", "BETA", "BETA_STD"]
    assert sum(1 for _ in open(coefs, encoding="utf-8")) == 5

    assert main(["evaluate", "--scores", combined, "--score-name", "MULTI_PGS",
                 "--pheno", cohort["pheno"], "--covar", cohort["covar"],
                 "--n-boot", "50", "--seed", "0"]) == 0
    out = capsys.readouterr().out
    assert "r2" in out and "incremental_r2" in out


def test_panel_npz_fit_combine_and_score(cohort, tmp_path, capsys):
    panel_path = str(tmp_path / "panel.npz")
    assert main(["panel", "--catalog", *cohort["scoring_files"],
                 "--plink", cohort["prefix"], "--out", panel_path,
                 "--quiet"]) == 0
    capsys.readouterr()
    coefs = str(tmp_path / "coefs.tsv")
    assert main(["fit", "--panel", panel_path, "--pheno", cohort["pheno"],
                 "--covar", cohort["covar"], "--out", coefs,
                 "--folds", "4", "--n-lambda", "20", "--seed", "0",
                 "--quiet"]) == 0
    capsys.readouterr()
    weights = str(tmp_path / "multi.weights")
    assert main(["combine", "--panel", panel_path, "--fit", coefs,
                 "--out", weights, "--check", "--plink",
                 cohort["prefix"]]) == 0
    out = capsys.readouterr().out
    assert "reproduces the fit" in out
    scored = str(tmp_path / "scored.tsv")
    assert main(["score", "--weights", weights, "--plink", cohort["prefix"],
                 "--out", scored]) == 0
    assert "frozen scaling" in capsys.readouterr().out
    header = open(scored, encoding="utf-8").readline().split()
    assert header == ["FID", "IID", "SCORE"]


def test_panel_sumstats_without_ld_is_refused(cohort, tmp_path):
    with pytest.raises(SystemExit, match="ld-prefix or --ld-cache"):
        main(["panel", "--sumstats", cohort["scoring_files"][0],
              "--plink", cohort["prefix"], "--out", str(tmp_path / "x.tsv")])


def test_panel_accepts_a_directory(cohort, tmp_path, capsys):
    import os
    import shutil
    d = tmp_path / "cat"
    d.mkdir()
    for p in cohort["scoring_files"]:
        shutil.copy(p, d / os.path.basename(p))
    out_path = str(tmp_path / "panel.tsv")
    assert main(["panel", "--catalog", str(d), "--plink", cohort["prefix"],
                 "--out", out_path, "--quiet"]) == 0
    assert "4 scores" in capsys.readouterr().out
    assert sum(1 for _ in open(out_path, encoding="utf-8")) == 201


def test_meta_command(cohort, tmp_path, capsys):
    scores = str(tmp_path / "scores.tsv")
    main(["panel", "--catalog", cohort["scoring_files"][0], "--plink",
          cohort["prefix"], "--out", scores, "--quiet"])
    capsys.readouterr()
    n_eff = str(tmp_path / "neff.tsv")
    with open(n_eff, "w", encoding="utf-8") as fh:
        fh.write("IID\tN_EFF\nPGS000001\t50000\n")
    out_path = str(tmp_path / "meta.tsv")
    assert main(["meta", "--scores", scores, "--n-eff", n_eff,
                 "--out", out_path]) == 0
    assert "meta-PGS" in capsys.readouterr().out
    assert open(out_path, encoding="utf-8").readline().split() == \
        ["FID", "IID", "META_PGS"]


@pytest.mark.parametrize(
    ("method", "option", "values"),
    [("sqrt_n_eff", "--n-eff", [("alpha", 100), ("beta", 400)]),
     ("expected_r2", "--expected-r2", [("alpha", 0.1), ("beta", 0.4)]),
     ("decorrelated", "--expected-r2", [("alpha", 0.1), ("beta", 0.4)]),
     ("decorrelated", "--n-eff", [("alpha", 100), ("beta", 400)])])
def test_meta_inputs_are_method_specific_and_aligned_by_score_id(
        tmp_path, monkeypatch, method, option, values):
    scores = tmp_path / "scores.tsv"
    scores.write_text("FID\tIID\tbeta\talpha\n"
                      "f1\ti1\t1\t2\n"
                      "f2\ti2\t3\t5\n", encoding="utf-8")
    metadata = _write_score_vector(tmp_path / "metadata.tsv", values)
    captured = {}

    class _Result:
        def summary(self):
            return "meta-PGS test"

        def multi_pgs(self, matrix):
            return np.zeros(matrix.shape[0])

    def fake_meta(matrix, **kwargs):
        captured.update(kwargs)
        return _Result()

    monkeypatch.setattr("multipgs.meta.meta_pgs", fake_meta)
    assert main(["meta", "--scores", str(scores), option, metadata,
                 "--method", method, "--out", str(tmp_path / "out.tsv")]) == 0
    supplied = captured["n_eff" if option == "--n-eff" else "expected_r2"]
    assert supplied.tolist() == [values[1][1], values[0][1]]
    assert list(captured["score_ids"]) == ["beta", "alpha"]


@pytest.mark.parametrize(
    ("method", "message"),
    [("sqrt_n_eff", "needs --n-eff"),
     ("expected_r2", "needs --expected-r2"),
     ("decorrelated", "needs --expected-r2")])
def test_meta_reports_its_missing_method_input(tmp_path, method, message):
    scores = tmp_path / "scores.tsv"
    scores.write_text("FID\tIID\ts\n1\t1\t0.1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match=message):
        main(["meta", "--scores", str(scores), "--method", method,
              "--out", str(tmp_path / "out.tsv")])


def test_score_metadata_alignment_and_id_contract(tmp_path):
    values = _write_score_vector(
        tmp_path / "values.tsv", [("score:two", 2), ("score_one", 1)])
    got = _read_score_vector(values, ["score_one", "score:two"],
                             name="--n-eff")
    assert got.tolist() == [1, 2]

    duplicate = _write_score_vector(
        tmp_path / "duplicate.tsv", [("score_one", 1), ("score_one", 2)])
    with pytest.raises(SystemExit, match="duplicate identifier"):
        _read_score_vector(duplicate, ["score_one"], name="--n-eff")

    mismatch = _write_score_vector(
        tmp_path / "mismatch.tsv", [("score_one", 1), ("unknown", 2)])
    with pytest.raises(SystemExit, match="score ids do not match"):
        _read_score_vector(mismatch, ["score_one", "score_two"],
                           name="--n-eff")

    two_ids = tmp_path / "two-ids.tsv"
    two_ids.write_text("FID\tIID\tN\nf\ti\t1\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="one SCORE/ID column"):
        _read_score_vector(str(two_ids), ["i"], name="--n-eff")


def test_fit_penalty_factors_are_aligned_by_score_id(tmp_path, monkeypatch):
    scores = tmp_path / "scores.tsv"
    scores.write_text("FID\tIID\tbeta\talpha\n"
                      "f1\ti1\t1\t2\n"
                      "f2\ti2\t3\t5\n", encoding="utf-8")
    pheno = tmp_path / "pheno.tsv"
    pheno.write_text("FID\tIID\tY\nf1\ti1\t0\nf2\ti2\t1\n",
                     encoding="utf-8")
    penalty = _write_score_vector(
        tmp_path / "penalty.tsv", [("alpha", 2), ("beta", 3)])
    captured = {}

    class _Fit:
        score_ids = np.array(["beta", "alpha"], dtype=object)
        beta_std = np.zeros(2)
        beta = np.zeros(2)

        def summary(self):
            return "multi-PGS test"

    def fake_fit(matrix, outcome, **kwargs):
        captured.update(kwargs)
        return _Fit()

    monkeypatch.setattr("multipgs.stack.multi_pgs_fit", fake_fit)
    assert main(["fit", "--scores", str(scores), "--pheno", str(pheno),
                 "--penalty-factor", penalty, "--out",
                 str(tmp_path / "coefs.tsv"), "--assessment-folds", "3",
                 "--quiet"]) == 0
    assert captured["penalty_factor"].tolist() == [3, 2]
    assert captured["assessment_folds"] == 3


def test_duplicate_individual_and_score_ids_are_rejected(tmp_path):
    duplicate_individual = tmp_path / "duplicate-individual.tsv"
    duplicate_individual.write_text(
        "FID\tIID\tS\nf\ti\t1\nf\ti\t2\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="duplicate identifier"):
        main(["evaluate", "--scores", str(duplicate_individual),
              "--pheno", "not-read.tsv", "--n-boot", "0"])

    duplicate_score = tmp_path / "duplicate-score.tsv"
    duplicate_score.write_text(
        "FID\tIID\tS\tS\nf\ti\t1\t2\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="duplicate value column"):
        main(["evaluate", "--scores", str(duplicate_score),
              "--pheno", "not-read.tsv", "--n-boot", "0"])


def test_binomial_fit_runs(cohort, tmp_path, capsys):
    scores = str(tmp_path / "scores.tsv")
    main(["panel", "--catalog", cohort["scoring_files"][0], "--plink",
          cohort["prefix"], "--out", scores, "--quiet"])
    binary = _write_pheno(tmp_path / "bin.tsv", cohort["sample_iid"],
                          (cohort["y"] > np.median(cohort["y"])).astype(float))
    capsys.readouterr()
    assert main(["fit", "--scores", scores, "--pheno", binary, "--out",
                 str(tmp_path / "b.tsv"), "--family", "binomial",
                 "--folds", "4", "--n-lambda", "15", "--seed", "0",
                 "--quiet"]) == 0
    assert "binomial" in capsys.readouterr().out


def test_individuals_are_matched_not_assumed_in_order(cohort, tmp_path,
                                                      capsys):
    """A shuffled, partly-overlapping phenotype file must still line up."""
    from multipgs import panel_from_catalog, write_panel, r2
    panel = panel_from_catalog(cohort["scoring_files"], cohort["prefix"])
    scores = str(tmp_path / "s.tsv")
    write_panel(panel, scores)

    order = np.random.default_rng(1).permutation(200)[:150]
    shuffled = _write_pheno(tmp_path / "shuf.tsv",
                            np.asarray(cohort["sample_iid"])[order],
                            cohort["y"][order])
    combined = str(tmp_path / "c.tsv")
    capsys.readouterr()
    assert main(["fit", "--scores", scores, "--pheno", shuffled, "--out",
                 str(tmp_path / "co.tsv"), "--out-score", combined,
                 "--folds", "4", "--n-lambda", "20", "--seed", "0"]) == 0
    # If the rows had been paired by position the fit would be noise.
    rows = [l.split() for l in open(combined, encoding="utf-8")][1:]
    assert len(rows) == 150
    got = {r[1]: float(r[2]) for r in rows}
    iid = np.asarray(cohort["sample_iid"])
    pred = np.array([got[str(i)] for i in iid[order]])
    assert r2(cohort["y"][order], pred) > 0.3


def test_missing_column_is_reported_by_name(cohort, tmp_path):
    scores = str(tmp_path / "scores.tsv")
    main(["panel", "--catalog", cohort["scoring_files"][0], "--plink",
          cohort["prefix"], "--out", scores, "--quiet"])
    with pytest.raises(SystemExit, match="NOT_THERE"):
        main(["evaluate", "--scores", scores, "--score-name", "NOT_THERE",
              "--pheno", cohort["pheno"], "--n-boot", "0"])


def test_non_overlapping_files_are_an_error(cohort, tmp_path):
    scores = str(tmp_path / "scores.tsv")
    main(["panel", "--catalog", cohort["scoring_files"][0], "--plink",
          cohort["prefix"], "--out", scores, "--quiet"])
    other = _write_pheno(tmp_path / "other.tsv",
                         [f"XX{i}" for i in range(10)], np.zeros(10))
    with pytest.raises(SystemExit, match="share no individuals"):
        main(["fit", "--scores", scores, "--pheno", other, "--out",
              str(tmp_path / "o.tsv")])


def test_version_and_help_are_wired():
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    with pytest.raises(SystemExit):
        main([])


# ---------------------------------------------------------------------------
# fetch
# ---------------------------------------------------------------------------

def _fake_catalog(monkeypatch, records, trait=None):
    """Route multipgs.fetch's single network seam at a fixed payload."""
    from multipgs import fetch

    def fake_json(url, *, timeout=30):
        if trait is not None and "/trait/" in url:
            return trait
        return {"size": len(records), "count": len(records), "next": None,
                "previous": None, "results": records}

    monkeypatch.setattr(fetch, "_fetch_json", fake_json)
    monkeypatch.setattr(fetch, "_download_file",
                        lambda *a, **k: pytest.fail("no download expected"))


def _catalog_record(pgs_id, *, n=50000, cohorts=()):
    return {
        "id": pgs_id, "name": f"n_{pgs_id}",
        "ftp_scoring_file": "https://ftp.example/x.txt.gz",
        "ftp_harmonized_scoring_files": {
            "GRCh37": {"positions": f"https://ftp.example/{pgs_id}.txt.gz"}},
        "publication": {"PMID": 1, "doi": "10.1/x"},
        "samples_variants": [{"sample_number": n, "sample_cases": None,
                              "sample_controls": None,
                              "cohorts": [{"name_short": c} for c in cohorts]}],
        "samples_training": [], "trait_reported": "Trait",
        "trait_efo": [{"id": "MONDO_0000001", "label": "t"}],
        "method_name": "m", "variants_number": 10, "weight_type": "beta",
        "ancestry_distribution": {"gwas": {"dist": {"EUR": 100.0}}},
    }


def test_fetch_writes_metadata_that_the_other_commands_can_read(
        tmp_path, monkeypatch, capsys):
    _fake_catalog(monkeypatch, [_catalog_record("PGS000001", n=20000),
                                _catalog_record("PGS000002", n=80000)])
    out = str(tmp_path / "panel")
    assert main(["fetch", "--pgs", "PGS000001", "PGS000002", "--out", out,
                 "--metadata-only", "--quiet"]) == 0

    values = _read_score_vector(f"{out}/n_eff.tsv",
                                ["PGS000001", "PGS000002"], name="n_eff")
    assert values.tolist() == [20000.0, 80000.0]
    header = open(f"{out}/metadata.tsv", encoding="utf-8").readline()
    assert header.startswith("SCORE\t") and "N_EFF" in header


def test_fetch_reports_shared_discovery_cohorts(tmp_path, monkeypatch, capsys):
    _fake_catalog(monkeypatch, [
        _catalog_record("PGS000001", cohorts=("UKB", "WHI")),
        _catalog_record("PGS000002", cohorts=("UKB", "WHI"))])
    assert main(["fetch", "--pgs", "PGS000001", "PGS000002", "--out",
                 str(tmp_path / "p"), "--metadata-only", "--cohort-overlap",
                 "--quiet"]) == 0
    printed = capsys.readouterr().out
    assert "share a named discovery cohort" in printed
    assert "jaccard 1.00" in printed
    assert "invisible to cross-validation" in printed


def test_fetch_reports_scores_with_no_sample_size(tmp_path, monkeypatch,
                                                  capsys):
    record = _catalog_record("PGS000001")
    record["samples_variants"] = []
    _fake_catalog(monkeypatch, [record])
    main(["fetch", "--pgs", "PGS000001", "--out", str(tmp_path / "p"),
          "--metadata-only", "--quiet"])
    assert "no sample size in the Catalog" in capsys.readouterr().out


def test_fetch_can_include_child_traits(tmp_path, monkeypatch):
    _fake_catalog(monkeypatch,
                  [_catalog_record("PGS000001"), _catalog_record("PGS000002")],
                  trait={"id": "MONDO_0000001",
                         "associated_pgs_ids": ["PGS000001"],
                         "child_associated_pgs_ids": ["PGS000002"]})
    out = str(tmp_path / "p")
    main(["fetch", "--trait", "MONDO_0000001", "--out", out,
          "--metadata-only", "--include-children", "--quiet"])
    ids = [ln.split("\t")[0] for ln in
           open(f"{out}/n_eff.tsv", encoding="utf-8").read().splitlines()[1:]]
    assert ids == ["PGS000001", "PGS000002"]


def test_fetch_turns_a_retired_trait_id_into_a_clean_exit(tmp_path,
                                                          monkeypatch):
    _fake_catalog(monkeypatch, [], trait={})
    with pytest.raises(SystemExit, match="MONDO"):
        main(["fetch", "--trait", "EFO_0000305", "--out", str(tmp_path / "p"),
              "--metadata-only", "--quiet"])


def test_fetch_requires_exactly_one_selector(tmp_path):
    with pytest.raises(SystemExit):
        main(["fetch", "--out", str(tmp_path / "p")])
