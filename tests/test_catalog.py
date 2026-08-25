"""PGS Catalog scoring-file parsing and allele alignment."""

import gzip

import numpy as np
import pytest

from multipgs import read_scoring_file
from multipgs.catalog import harmonize_scoring_file, scoring_file_id


HEADER = """###PGS CATALOG SCORING FILE
#format_version=2.0
#pgs_id=PGS000042
#trait_reported=Test trait
#genome_build=GRCh37
#weight_type={weight_type}
"""


def _write(tmp_path, rows, columns, weight_type="beta", name="PGS000042.txt",
           gz=False):
    text = HEADER.format(weight_type=weight_type)
    text += "\t".join(columns) + "\n"
    for row in rows:
        text += "\t".join(str(v) for v in row) + "\n"
    path = tmp_path / name
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as fh:
            fh.write(text)
    else:
        path.write_text(text, encoding="utf-8")
    return str(path)


def test_reads_metadata_and_rows(tmp_path):
    path = _write(tmp_path,
                  [("rs1", 1, 100, "A", "G", 0.5),
                   ("rs2", 2, 200, "C", "T", -0.25)],
                  ["rsID", "chr_name", "chr_position", "effect_allele",
                   "other_allele", "effect_weight"])
    sf = read_scoring_file(path)
    assert len(sf) == 2
    assert sf.pgs_id == "PGS000042"
    assert sf.trait == "Test trait"
    assert sf.genome_build == "GRCh37"
    assert list(sf.id) == ["rs1", "rs2"]
    assert list(sf.pos) == [100, 200]
    assert np.allclose(sf.weight, [0.5, -0.25])
    assert sf.log["n_kept"] == 2
    assert "PGS000042" in str(sf)


def test_gzipped_file_reads_identically(tmp_path):
    rows = [("rs1", 1, 100, "A", "G", 0.5)]
    cols = ["rsID", "chr_name", "chr_position", "effect_allele",
            "other_allele", "effect_weight"]
    plain = read_scoring_file(_write(tmp_path, rows, cols, name="a.txt"))
    zipped = read_scoring_file(_write(tmp_path, rows, cols, name="b.txt.gz",
                                      gz=True))
    assert np.allclose(plain.weight, zipped.weight)


def test_the_header_fixes_the_delimiter_for_the_whole_file(tmp_path):
    """A data row without tabs must not switch to whitespace splitting."""
    cols = ["rsID", "chr_name", "chr_position", "effect_allele",
            "other_allele", "effect_weight"]
    path = _write(tmp_path,
                  [("rs1", 1, 100, "A", "G", 0.5),
                   ("rs2", 2, 200, "C", "T", -0.25)],
                  cols)
    sf = read_scoring_file(path)
    assert len(sf) == 2
    assert np.allclose(sf.weight, [0.5, -0.25])

    # Same tabbed header, but one row joined by spaces: it cannot be split on
    # the file's delimiter, so it is a bad row, not six misaligned fields.
    text = HEADER.format(weight_type="beta")
    text += "\t".join(cols) + "\n"
    text += "\t".join(["rs1", "1", "100", "A", "G", "0.5"]) + "\n"
    text += " ".join(["rs2", "2", "200", "C", "T", "-0.25"]) + "\n"
    mixed = tmp_path / "mixed.txt"
    mixed.write_text(text, encoding="utf-8")
    sf = read_scoring_file(str(mixed))
    assert len(sf) == 1
    assert sf.log["n_unparsable_weight"] == 1


def test_odds_ratios_are_log_transformed(tmp_path):
    path = _write(tmp_path, [("rs1", 1, 100, "A", "G", 2.0),
                             ("rs2", 2, 200, "C", "T", 0.5)],
                  ["rsID", "chr_name", "chr_position", "effect_allele",
                   "other_allele", "effect_weight"], weight_type="OR")
    sf = read_scoring_file(path)
    assert np.allclose(sf.weight, [np.log(2.0), np.log(0.5)])
    assert sf.log["log_transformed"] is True


def test_non_positive_odds_ratio_is_an_error_not_a_nan(tmp_path):
    path = _write(tmp_path, [("rs1", 1, 100, "A", "G", -0.3)],
                  ["rsID", "chr_name", "chr_position", "effect_allele",
                   "other_allele", "effect_weight"], weight_type="OR")
    with pytest.raises(ValueError, match="cannot be"):
        read_scoring_file(path)


def test_non_additive_rows_are_dropped_and_counted(tmp_path):
    path = _write(tmp_path,
                  [("rs1", 1, 100, "A", "G", 0.5, "False"),
                   ("rs2", 2, 200, "C", "T", 0.3, "True")],
                  ["rsID", "chr_name", "chr_position", "effect_allele",
                   "other_allele", "effect_weight", "is_recessive"])
    sf = read_scoring_file(path)
    assert len(sf) == 1
    assert sf.log["n_non_additive"] == 1
    kept = read_scoring_file(path, drop_non_additive=False)
    assert len(kept) == 2


def test_harmonized_columns_are_preferred(tmp_path):
    path = _write(tmp_path, [("rs1", 9, 999, "rs1", 1, 100, "A", "G", 0.5)],
                  ["rsID", "chr_name", "chr_position", "hm_rsID", "hm_chr",
                   "hm_pos", "effect_allele", "other_allele", "effect_weight"])
    sf = read_scoring_file(path)
    assert sf.pos[0] == 100                    # hm_pos, not chr_position
    assert sf.log["harmonized_columns"] is True
    raw = read_scoring_file(path, prefer_harmonized=False)
    assert raw.pos[0] == 999


def test_inferred_other_allele_is_a_row_wise_fallback(tmp_path):
    path = _write(
        tmp_path,
        [("rs1", 1, 100, "A", "G", "T", 0.5),
         ("rs2", 1, 200, "C", "", "A", 0.3),
         ("rs3", 1, 300, "G", "NA", "C", -0.2)],
        ["rsID", "chr_name", "chr_position", "effect_allele",
         "other_allele", "hm_inferOtherAllele", "effect_weight"])
    sf = read_scoring_file(path)
    assert list(sf.oa) == ["G", "A", "C"]
    assert sf.log["n_inferred_other_allele"] == 2
    assert sf.log["columns_used"]["oa_fallback"] == "hm_inferotherallele"


def test_unparsable_weights_are_skipped(tmp_path):
    path = _write(tmp_path, [("rs1", 1, 100, "A", "G", "NA"),
                             ("rs2", 2, 200, "C", "T", 0.3)],
                  ["rsID", "chr_name", "chr_position", "effect_allele",
                   "other_allele", "effect_weight"])
    sf = read_scoring_file(path)
    assert len(sf) == 1
    assert sf.log["n_unparsable_weight"] == 1


def test_missing_required_columns_name_the_problem(tmp_path):
    path = _write(tmp_path, [("rs1", "A", "G")],
                  ["rsID", "effect_allele", "other_allele"])
    with pytest.raises(ValueError, match="effect_weight"):
        read_scoring_file(path)


def test_scoring_file_id_from_filename():
    assert scoring_file_id("/x/PGS000123_hmPOS_GRCh37.txt.gz") == "PGS000123"
    assert scoring_file_id("my_scores.txt") == "my_scores"


class _Variants:
    """Minimal stand-in for ldpred3's VariantTable."""

    def __init__(self, id, chrom, pos, a1, a2):
        self.id = np.array(id, dtype=object)
        self.chrom = np.array(chrom, dtype=object)
        self.pos = np.array(pos, dtype=np.int64)
        self.a1 = np.array(a1, dtype=object)
        self.a2 = np.array(a2, dtype=object)

    def __len__(self):
        return self.id.size


def test_harmonize_flips_swapped_alleles_and_drops_palindromes(tmp_path):
    path = _write(tmp_path,
                  [("rs1", 1, 100, "A", "G", 0.5),     # same order  -> +0.5
                   ("rs2", 1, 200, "C", "T", 0.3),     # swapped     -> -0.3
                   ("rs3", 1, 300, "A", "T", 0.7),     # palindromic -> dropped
                   ("rs9", 1, 900, "A", "G", 0.1)],    # unmatched   -> dropped
                  ["rsID", "chr_name", "chr_position", "effect_allele",
                   "other_allele", "effect_weight"])
    sf = read_scoring_file(path)
    variants = _Variants(["rs1", "rs2", "rs3"], ["1", "1", "1"],
                         [100, 200, 300], ["A", "T", "A"], ["G", "C", "T"])
    idx, w, log = harmonize_scoring_file(sf, variants)
    assert list(idx) == [0, 1]
    assert np.allclose(w, [0.5, -0.3])
    assert log["n_weights"] == 4
    assert log["n_matched"] == 2
    assert 0.0 < log["weight_mass_matched"] < 1.0


def test_harmonize_of_an_empty_file_is_not_an_error(tmp_path):
    path = _write(tmp_path, [], ["rsID", "chr_name", "chr_position",
                                 "effect_allele", "other_allele",
                                 "effect_weight"])
    sf = read_scoring_file(path)
    idx, w, log = harmonize_scoring_file(
        sf, _Variants(["rs1"], ["1"], [100], ["A"], ["G"]))
    assert idx.size == 0 and w.size == 0 and log["n_matched"] == 0
