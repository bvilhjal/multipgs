"""The command line, end to end."""

import numpy as np
import pytest

from multipgs import simulate_target
from multipgs.cli import main


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
