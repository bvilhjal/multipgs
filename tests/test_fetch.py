"""Acquiring scores from the PGS Catalog, without touching the network.

Every test here patches :func:`multipgs.fetch._fetch_json` or
:func:`multipgs.fetch._download_file`, which are the only two functions in the
package that open a socket. The payloads are trimmed copies of real API
responses, so the field names and the shapes they arrive in are the Catalog's.
"""

import gzip
import json

import numpy as np
import pytest

from multipgs import fetch


def _score(pgs_id, *, samples=None, harmonized=("GRCh37", "GRCh38"),
           variants=100, trait="Breast cancer", efo="MONDO_0004989"):
    return {
        "id": pgs_id,
        "name": f"name_{pgs_id}",
        "ftp_scoring_file":
            f"https://ftp.ebi.ac.uk/.../{pgs_id}/ScoringFiles/{pgs_id}.txt.gz",
        "ftp_harmonized_scoring_files": {
            build: {"positions": f"https://ftp.ebi.ac.uk/.../{pgs_id}_hmPOS_"
                                 f"{build}.txt.gz"}
            for build in harmonized},
        "publication": {"id": "PGP000001", "doi": "10.1093/jnci/djv036",
                        "PMID": 25855707, "firstauthor": "Mavaddat N",
                        "date_publication": "2015-04-08"},
        "samples_variants": [] if samples is None else samples,
        "samples_training": [],
        "trait_reported": trait,
        "trait_efo": [{"id": efo, "label": "breast carcinoma"}],
        "method_name": "SNPs passing genome-wide significance",
        "variants_number": variants,
        "weight_type": "beta",
        "ancestry_distribution": {"gwas": {"dist": {"EUR": 95.8, "EAS": 4.2},
                                           "count": 125478}},
    }


def _sample(number=None, cases=None, controls=None, cohorts=()):
    return {"sample_number": number, "sample_cases": cases,
            "sample_controls": controls, "ancestry_broad": "European",
            "cohorts": [{"name_short": c, "name_full": c} for c in cohorts]}


def _page(results, next_url=None):
    return {"size": len(results), "count": len(results), "next": next_url,
            "previous": None, "results": results}


@pytest.fixture
def api(monkeypatch):
    """Route every request through a dict of url-substring -> payload."""
    routes = {}
    calls = []

    def fake(url, *, timeout=30):
        calls.append(url)
        for key, payload in routes.items():
            if key in url:
                if isinstance(payload, Exception):
                    raise payload
                return payload
        raise AssertionError(f"no fixture matches {url}")

    monkeypatch.setattr(fetch, "_fetch_json", fake)
    return type("Api", (), {"routes": routes, "calls": calls})()


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def test_n_eff_prefers_cases_and_controls_over_the_total():
    """A set reporting both must use 4/(1/cases + 1/controls), not the total."""
    record = fetch.ScoreRecord.from_api(_score(
        "PGS000004", samples=[_sample(158648, cases=88916, controls=69732)]))
    expected = 4.0 / (1 / 88916 + 1 / 69732)
    assert record.n_eff == pytest.approx(expected)
    assert record.n_total == pytest.approx(158648)
    assert record.n_eff < record.n_total


def test_n_eff_sums_over_discovery_sample_sets():
    record = fetch.ScoreRecord.from_api(_score(
        "PGS000765", samples=[_sample(120184), _sample(5294)]))
    assert record.n_eff == pytest.approx(125478)
    assert record.n_sample_sets == 2


def test_a_score_with_no_samples_reports_nan_not_zero():
    """Zero is a number a screening gate would happily compare against."""
    record = fetch.ScoreRecord.from_api(_score("PGS000001", samples=[]))
    assert np.isnan(record.n_eff)
    assert np.isnan(record.n_total)


def test_training_samples_are_used_when_there_are_no_discovery_samples():
    payload = _score("PGS000900", samples=[])
    payload["samples_training"] = [_sample(40000)]
    assert fetch.ScoreRecord.from_api(payload).n_eff == pytest.approx(40000)


def test_an_empty_body_is_rejected_rather_than_parsed():
    """A missing score is HTTP 200 with {}, not a 404."""
    with pytest.raises(ValueError, match="no 'id'"):
        fetch.ScoreRecord.from_api({})


def test_record_exposes_ancestry_and_harmonized_urls():
    record = fetch.ScoreRecord.from_api(_score("PGS000001"))
    assert record.top_ancestry == "EUR"
    assert record.ancestry_percent("EAS") == pytest.approx(4.2)
    assert np.isnan(record.ancestry_percent("AFR"))
    assert record.harmonized_url("GRCh38").endswith("_hmPOS_GRCh38.txt.gz")
    assert record.harmonized_url("GRCh36") is None


def test_a_score_with_no_ancestry_reports_nan_not_zero():
    """Zero is a number a downstream EUR_PERCENT gate would compare against."""
    payload = _score("PGS000001")
    payload["ancestry_distribution"] = {}
    record = fetch.ScoreRecord.from_api(payload)
    assert record.top_ancestry == ""
    assert np.isnan(record.ancestry_percent("EUR"))


def test_recorded_zero_ancestry_share_is_zero_not_missing():
    payload = _score("PGS000001")
    payload["ancestry_distribution"] = {
        "gwas": {"dist": {"EUR": 0.0, "EAS": 100.0}, "count": 1}}
    record = fetch.ScoreRecord.from_api(payload)
    assert record.ancestry_percent("EUR") == 0.0
    assert record.ancestry_percent("EAS") == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def test_search_by_id_returns_records_in_the_requested_order(api):
    api.routes["score/search"] = _page([_score("PGS000765"),
                                        _score("PGS000001")])
    records = fetch.search_scores(pgs_ids=["PGS000001", "PGS000765"])
    assert [r.pgs_id for r in records] == ["PGS000001", "PGS000765"]


def test_a_score_the_catalog_does_not_have_is_an_error(api):
    """The trap: an unknown id comes back as an empty page, not a 404."""
    api.routes["score/search"] = _page([_score("PGS000001")])
    with pytest.raises(ValueError, match="did not return 1 of the 2"):
        fetch.search_scores(pgs_ids=["PGS000001", "PGS999999"])


def test_search_follows_pagination(api):
    api.routes["offset=1"] = _page([_score("PGS000002")])
    api.routes["score/search"] = _page([_score("PGS000001")],
                                       next_url="https://x/rest/score/search"
                                                "?offset=1")
    records = fetch.search_scores(pgs_ids=["PGS000001", "PGS000002"])
    assert [r.pgs_id for r in records] == ["PGS000001", "PGS000002"]


def test_trait_search_can_include_child_traits(api):
    api.routes["trait/MONDO_0004989"] = {
        "id": "MONDO_0004989", "label": "breast carcinoma",
        "associated_pgs_ids": ["PGS000001"],
        "child_associated_pgs_ids": ["PGS000002"]}
    api.routes["score/search"] = _page([_score("PGS000001"),
                                        _score("PGS000002")])

    parent = fetch.search_scores(trait_id="MONDO_0004989")
    assert [r.pgs_id for r in parent] == ["PGS000001"]

    both = fetch.search_scores(trait_id="MONDO_0004989", include_children=True)
    assert [r.pgs_id for r in both] == ["PGS000001", "PGS000002"]


def test_a_retired_trait_id_is_an_error_naming_the_migration(api):
    api.routes["trait/EFO_0000305"] = {}
    with pytest.raises(ValueError, match="EFO_0000305.*MONDO"):
        fetch.search_scores(trait_id="EFO_0000305")


def test_exactly_one_selector_is_required(api):
    with pytest.raises(ValueError, match="exactly one"):
        fetch.search_scores()
    with pytest.raises(ValueError, match="exactly one"):
        fetch.search_scores(trait_id="MONDO_0004989", pmid=123)


def test_the_cache_makes_a_second_search_free(tmp_path, api):
    api.routes["score/search"] = _page([_score("PGS000001")])
    cache = str(tmp_path / "cache")
    first = fetch.search_scores(pgs_ids=["PGS000001"], cache_dir=cache)
    n_calls = len(api.calls)
    second = fetch.search_scores(pgs_ids=["PGS000001"], cache_dir=cache)
    assert len(api.calls) == n_calls          # no further network requests
    assert first[0].pgs_id == second[0].pgs_id


def test_a_transient_failure_is_retried(monkeypatch):
    import urllib.error

    attempts = []

    def flaky(url, *, timeout=30):
        attempts.append(url)
        if len(attempts) < 3:
            raise urllib.error.HTTPError(url, 503, "busy", None, None)
        return _page([_score("PGS000001")])

    monkeypatch.setattr(fetch, "_fetch_json", flaky)
    monkeypatch.setattr(fetch.time, "sleep", lambda _s: None)
    records = fetch.search_scores(pgs_ids=["PGS000001"])
    assert len(attempts) == 3 and records[0].pgs_id == "PGS000001"


def test_a_client_error_is_not_retried(monkeypatch):
    import urllib.error

    attempts = []

    def broken(url, *, timeout=30):
        attempts.append(url)
        raise urllib.error.HTTPError(url, 400, "bad", None, None)

    monkeypatch.setattr(fetch, "_fetch_json", broken)
    monkeypatch.setattr(fetch.time, "sleep", lambda _s: None)
    with pytest.raises(RuntimeError, match=r"\(400\)"):
        fetch.search_scores(pgs_ids=["PGS000001"])
    assert len(attempts) == 1


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _write_scoring_file(path, pgs_id):
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("###PGS CATALOG SCORING FILE\n")
        fh.write(f"#pgs_id={pgs_id}\n#genome_build=GRCh37\n")
        fh.write("rsID\tchr_name\tchr_position\teffect_allele\tother_allele"
                 "\teffect_weight\n")
        fh.write("rs1\t1\t1000\tA\tG\t0.1\n")


def test_download_writes_verifies_and_resumes(tmp_path, monkeypatch):
    records = [fetch.ScoreRecord.from_api(_score("PGS000001")),
               fetch.ScoreRecord.from_api(_score("PGS000002"))]
    urls = []

    def fake_download(url, dest, *, timeout=120):
        urls.append(url)
        _write_scoring_file(dest, url.split("/")[-1].split("_")[0])

    monkeypatch.setattr(fetch, "_download_file", fake_download)
    dest = str(tmp_path / "scores")

    paths, log = fetch.download_scores(records, dest, build="GRCh37")
    assert log["n_downloaded"] == 2 and log["n_failed"] == 0
    assert all(p.endswith("_hmPOS_GRCh37.txt.gz") for p in paths)

    # A second call must not re-fetch what is already on disk.
    _, log2 = fetch.download_scores(records, dest, build="GRCh37")
    assert log2["n_cached"] == 2 and log2["n_downloaded"] == 0
    assert len(urls) == 2


def test_a_truncated_download_is_caught_by_verification(tmp_path, monkeypatch):
    records = [fetch.ScoreRecord.from_api(_score("PGS000001"))]

    def truncating(url, dest, *, timeout=120):
        with open(dest, "wb") as fh:
            fh.write(b"\x1f\x8b\x08\x00truncated")

    monkeypatch.setattr(fetch, "_download_file", truncating)
    with pytest.raises(ValueError, match="truncated"):
        fetch.download_scores(records, str(tmp_path / "s"))

    paths, log = fetch.download_scores(records, str(tmp_path / "s2"),
                                       on_error="skip")
    assert paths == [None] and log["n_failed"] == 1


@pytest.mark.parametrize("body, message", [
    ("", "no pgs_id"),
    ("#pgs_id=PGS000001\n", "no scoring-table column header"),
    ("#pgs_id=PGS000001\nnot_a_table\nstill_not_a_variant\n",
     "no valid scoring-table header"),
    ("#pgs_id=PGS000001\n"
     "rsID\tchr_name\tchr_position\teffect_allele\tother_allele\t"
     "effect_weight\n", "no variant rows"),
])
def test_empty_scoring_downloads_are_rejected(tmp_path, body, message):
    path = tmp_path / "empty.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(body)
    with pytest.raises(ValueError, match=message):
        fetch._verify_scoring_file(str(path), "PGS000001")


@pytest.mark.parametrize("row", [
    "rs1\t1\t1000\tA\tG",                       # too few fields
    "rs1\t1\t1000\tA\tG\t0.1\textra",        # too many fields
    "rs1\t1\t1000\tA\tG\tnot-a-number",
    "rs1\t1\t1000\tA\tG\tnan",
    "rs1\t1\t1000\tA\tG\tinf",
    ".\t.\t0\tA\tG\t0.1",                    # no usable identity
    "\t1\tnot-a-position\tA\tG\t0.1",
    "rs1\t1\t1000\t\tG\t0.1",               # no effect allele
])
def test_scoring_download_needs_a_structurally_valid_variant_row(
        tmp_path, row):
    path = tmp_path / "malformed.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("#pgs_id=PGS000001\n")
        fh.write("rsID\tchr_name\tchr_position\teffect_allele\t"
                 "other_allele\teffect_weight\n")
        fh.write(row + "\n")
    with pytest.raises(ValueError, match="no structurally valid variant row"):
        fetch._verify_scoring_file(str(path), "PGS000001")


@pytest.mark.parametrize("row", [
    "rs1\t\t\tA\tG\t0.1",       # usable ID; coordinates may be absent
    "\t1\t1000\tA\tG\t0.1",     # usable coordinates; ID may be absent
])
def test_scoring_download_accepts_either_variant_identity(tmp_path, row):
    path = tmp_path / "valid.txt.gz"
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("#pgs_id=PGS000001\n")
        fh.write("rsID\tchr_name\tchr_position\teffect_allele\t"
                 "other_allele\teffect_weight\n")
        fh.write("garbage\twith\tthe\tright\tfield\tcount\n")
        fh.write(row + "\n")
    fetch._verify_scoring_file(str(path), "PGS000001")


def test_a_bad_cached_download_is_removed_so_resume_can_retry(
        tmp_path, monkeypatch):
    record = fetch.ScoreRecord.from_api(_score("PGS000001"))
    calls = []

    def download(_url, dest, *, timeout=120):
        calls.append(dest)
        if len(calls) == 1:
            with gzip.open(dest, "wt", encoding="utf-8"):
                pass
        else:
            _write_scoring_file(dest, "PGS000001")

    monkeypatch.setattr(fetch, "_download_file", download)
    dest = str(tmp_path / "scores")
    paths, log = fetch.download_scores([record], dest, on_error="skip")
    assert paths == [None] and log["n_failed"] == 1
    assert log["n_downloaded"] == 0 and log["n_cached"] == 0
    assert not (tmp_path / "scores" /
                "PGS000001_hmPOS_GRCh37.txt.gz").exists()

    paths, log = fetch.download_scores([record], dest)
    assert paths[0] is not None and log["n_downloaded"] == 1
    assert len(calls) == 2


def test_a_score_not_harmonized_on_the_build_is_an_error(tmp_path, monkeypatch):
    records = [fetch.ScoreRecord.from_api(
        _score("PGS000001", harmonized=("GRCh38",)))]
    monkeypatch.setattr(fetch, "_download_file",
                        lambda *a, **k: pytest.fail("must not download"))
    with pytest.raises(ValueError, match="no harmonized file on GRCh37"):
        fetch.download_scores(records, str(tmp_path / "s"), build="GRCh37")


def test_download_rejects_an_unknown_build(tmp_path):
    with pytest.raises(ValueError, match="build must be one of"):
        fetch.download_scores([], str(tmp_path), build="hg19")


# ---------------------------------------------------------------------------
# Metadata and overlap
# ---------------------------------------------------------------------------

def test_metadata_table_is_readable_by_the_score_vector_reader(tmp_path):
    from multipgs.cli import _read_score_vector

    records = [
        fetch.ScoreRecord.from_api(_score("PGS000001",
                                          samples=[_sample(22627)])),
        fetch.ScoreRecord.from_api(_score("PGS000002",
                                          samples=[_sample(cases=1000,
                                                           controls=3000)])),
    ]
    path = str(tmp_path / "n_eff.tsv")
    fetch.write_score_metadata(records, path, columns=["N_EFF"])
    values = _read_score_vector(path, ["PGS000001", "PGS000002"], name="n_eff")
    assert values[0] == pytest.approx(22627)
    assert values[1] == pytest.approx(4.0 / (1 / 1000 + 1 / 3000))


def test_read_score_metadata_round_trips_na_and_n_eff(tmp_path):
    payload = _score("PGS000001")
    payload["ancestry_distribution"] = {}
    path = str(tmp_path / "meta.tsv")
    fetch.write_score_metadata([fetch.ScoreRecord.from_api(payload)], path)
    table = fetch.read_score_metadata(path)
    assert np.isnan(table["PGS000001"]["EUR_PERCENT"])
    assert "N_EFF" in table["PGS000001"]


def test_an_unknown_sample_size_is_written_NA_not_zero(tmp_path):
    records = [fetch.ScoreRecord.from_api(_score("PGS000001", samples=[]))]
    path = str(tmp_path / "meta.tsv")
    fetch.write_score_metadata(records, path)
    body = open(path, encoding="utf-8").read()
    assert "\tNA\t" in body


def test_missing_ancestry_is_written_NA_not_zero(tmp_path):
    payload = _score("PGS000001")
    payload["ancestry_distribution"] = {}
    path = str(tmp_path / "meta.tsv")
    fetch.write_score_metadata([fetch.ScoreRecord.from_api(payload)], path)
    header, row = [ln.split("\t") for ln in
                   open(path, encoding="utf-8").read().splitlines() if ln]
    assert row[header.index("EUR_PERCENT")] == "NA"
    assert row[header.index("ANCESTRY")] == ""


def test_metadata_rejects_duplicates_and_unknown_columns(tmp_path):
    records = [fetch.ScoreRecord.from_api(_score("PGS000001"))] * 2
    with pytest.raises(ValueError, match="duplicate score id"):
        fetch.write_score_metadata(records, str(tmp_path / "a.tsv"))
    with pytest.raises(ValueError, match="unknown metadata column"):
        fetch.write_score_metadata(records[:1], str(tmp_path / "b.tsv"),
                                   columns=["NOPE"])


def test_free_text_cannot_add_a_column(tmp_path):
    payload = _score("PGS000001")
    payload["trait_reported"] = "breast\tcancer\nstage II"
    path = str(tmp_path / "meta.tsv")
    fetch.write_score_metadata([fetch.ScoreRecord.from_api(payload)], path)
    lines = [ln for ln in open(path, encoding="utf-8").read().splitlines() if ln]
    assert len(lines) == 2
    assert len(lines[0].split("\t")) == len(lines[1].split("\t"))


def test_cohort_overlap_flags_shared_discovery_cohorts():
    records = [
        fetch.ScoreRecord.from_api(_score("PGS000001", samples=[
            _sample(1000, cohorts=("UKB", "WHI"))])),
        fetch.ScoreRecord.from_api(_score("PGS000002", samples=[
            _sample(1000, cohorts=("UKB", "WHI"))])),
        fetch.ScoreRecord.from_api(_score("PGS000003", samples=[
            _sample(1000, cohorts=("MEC",))])),
        fetch.ScoreRecord.from_api(_score("PGS000004", samples=[
            _sample(1000)])),
    ]
    overlap, ids = fetch.cohort_overlap(records)
    assert list(ids) == ["PGS000001", "PGS000002", "PGS000003", "PGS000004"]
    assert overlap[0, 1] == pytest.approx(1.0)      # identical cohort sets
    assert overlap[0, 2] == pytest.approx(0.0)      # disjoint
    # A score naming no cohorts is unknown, not disjoint.
    assert np.all(np.isnan(overlap[3, :]))
    assert np.all(np.isnan(overlap[:, 3]))


def test_cache_file_names_are_readable_and_unique(tmp_path):
    a = fetch._cache_path(str(tmp_path), fetch.REST_BASE + "/score/PGS000001")
    b = fetch._cache_path(str(tmp_path), fetch.REST_BASE + "/score/PGS000002")
    assert a != b
    assert "score_PGS000001" in a
    assert a.endswith(".json")


def test_the_only_network_calls_are_the_two_named_seams():
    """Anything else opening a socket must be a deliberate, reviewed change."""
    import inspect

    source = inspect.getsource(fetch)
    for token in ("urlopen(", "urlretrieve"):
        assert source.count(token) <= 2, (
            f"{token} appears more than twice; every network call belongs in "
            "_fetch_json or _download_file")


def test_json_payloads_are_not_evaluated():
    """A cached response is data; nothing may execute it."""
    import inspect

    source = inspect.getsource(fetch)
    assert "eval(" not in source and "exec(" not in source
    assert "json.load" in source


def test_search_records_round_trip_through_the_cache(tmp_path, api):
    api.routes["score/search"] = _page([_score("PGS000001",
                                               samples=[_sample(22627)])])
    cache = str(tmp_path / "c")
    fetch.search_scores(pgs_ids=["PGS000001"], cache_dir=cache)
    files = list((tmp_path / "c").iterdir())
    assert len(files) == 1
    payload = json.loads(files[0].read_text())
    assert payload["results"][0]["id"] == "PGS000001"
