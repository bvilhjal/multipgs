"""Find and download PGS Catalog scores, with the metadata multipgs needs.

Everything else in this package starts from *scoring files you already have*.
Getting them is its own problem: Albiñana et al. combined 937 of them, and
assembling 937 files plus their discovery sample sizes by hand is not a
pipeline. This module is the acquisition step. It turns a trait identifier into
a directory of harmonized scoring files and a metadata table keyed by score id,
which is what :func:`multipgs.panel.panel_from_catalog`, :func:`multipgs.meta_pgs`
and :func:`multipgs.architecture.screen` consume.

The `PGS Catalog REST API <https://www.pgscatalog.org/rest/>`_ is read-only,
unauthenticated and paginated. Four of its behaviours cost you data silently if
they are not handled, and all four are handled here:

* **A missing score is HTTP 200.** ``/rest/score/PGS999999`` returns ``200``
  with an empty JSON object rather than ``404``. Checking the status code is
  not enough, so :func:`search_scores` checks that every requested identifier
  came back and raises naming the ones that did not.
* **An unrecognised filter returns zero results, not an error.** A misspelled
  query parameter yields ``count: 0``, which is indistinguishable from a trait
  that genuinely has no scores. Only documented parameters are sent from here.
* **Trait identifiers move.** Breast cancer was ``EFO_0000305`` and is now
  ``MONDO_0004989``; the retired identifier returns ``{}``. Identifiers are
  passed through verbatim rather than rewritten, and an empty trait response is
  an error naming the identifier rather than an empty panel.
* **Child traits are a separate list.** ``/rest/trait/{id}`` reports
  ``associated_pgs_ids`` and ``child_associated_pgs_ids`` disjointly.
  ``include_children=True`` unions them; without it a query for breast carcinoma
  omits the scores registered against its subtypes.

**Sample sizes.** ``n_eff`` is summed over a score's discovery sample sets:
``4/(1/cases + 1/controls)`` for case/control sets, the plain sample count
otherwise. It is ``nan`` when the Catalog records no sample information, which
is honest rather than convenient — :func:`multipgs.architecture.screen` then
reports the score as failing the effective-sample-size gate instead of passing
it on a fabricated number.

**Cohort overlap.** :func:`cohort_overlap` reports shared discovery cohorts
between every pair of scores. Overlap between discovery GWAS is the failure mode
cross-validation cannot see (``docs/theory.md``), and named cohorts are the only
machine-readable evidence of it the Catalog carries. It is a lower bound and a
flag, not a measurement: cohort lists are incomplete, ``name_short`` is free
text, and two scores naming one cohort may still use disjoint individuals from
it.
"""

from __future__ import annotations

import gzip
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from dataclasses import dataclass, field

import numpy as np


__all__ = ["search_scores", "ScoreRecord", "download_scores",
           "write_score_metadata", "read_score_metadata",
           "cohort_overlap", "REST_BASE"]

REST_BASE = "https://www.pgscatalog.org/rest"

#: Builds the Catalog publishes harmonized position files on.
BUILDS = ("GRCh37", "GRCh38")

# The Catalog asks for considerate use of the API. One request in flight, a
# short pause between them, and a cache so a re-run of the same panel costs
# nothing. Page size is the API maximum; fewer, larger pages is both faster and
# politer than many small ones.
_PAGE_SIZE = 250
_ID_CHUNK = 50
_PAUSE = 0.2

_METADATA_COLUMNS = (
    "N_EFF", "N_TOTAL", "N_CASES", "N_CONTROLS", "N_SAMPLE_SETS",
    "N_VARIANTS", "TRAIT", "EFO", "ANCESTRY", "EUR_PERCENT", "WEIGHT_TYPE",
    "METHOD", "PMID", "DOI", "N_COHORTS", "COHORTS",
)


# ---------------------------------------------------------------------------
# Transport
# ---------------------------------------------------------------------------

def _user_agent():
    from . import __version__
    return f"multipgs/{__version__} (https://github.com/bvilhjal/multipgs)"


def _fetch_json(url, *, timeout=30):
    """One HTTP GET returning parsed JSON. The only network call in multipgs.

    Isolated so that tests, and anyone auditing what this package talks to, have
    exactly one function to look at.
    """
    request = urllib.request.Request(
        url, headers={"Accept": "application/json",
                      "User-Agent": _user_agent()})
    with urllib.request.urlopen(request, timeout=timeout) as fh:
        return json.loads(fh.read().decode("utf-8"))


def _cache_path(cache_dir, url):
    """A readable, collision-free cache file name for ``url``.

    Readable matters: a cache you cannot inspect by listing the directory is a
    cache you cannot debug, and hand-writing a fixture into it is how the tests
    stay offline.
    """
    import hashlib

    tail = url[len(REST_BASE):] if url.startswith(REST_BASE) else url
    slug = "".join(c if c.isalnum() else "_" for c in tail).strip("_")[:80]
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
    return os.path.join(cache_dir, f"{slug}__{digest}.json")


def _request_json(url, *, cache_dir=None, timeout=30, retries=3, pause=_PAUSE):
    """``_fetch_json`` with an on-disk cache and backoff on transient failures.

    Retries 429 and 5xx and transport errors; a 4xx other than 429 is the
    caller's mistake and is raised immediately rather than hammered.
    """
    retries = int(retries)
    if retries < 1:
        raise ValueError("retries must be at least 1")
    path = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        path = _cache_path(cache_dir, url)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)

    delay = pause
    for attempt in range(retries):
        try:
            payload = _fetch_json(url, timeout=timeout)
            break
        except urllib.error.HTTPError as exc:
            transient = exc.code == 429 or 500 <= exc.code < 600
            if not transient or attempt == retries - 1:
                raise RuntimeError(
                    f"PGS Catalog request failed ({exc.code}) for {url}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt == retries - 1:
                raise RuntimeError(
                    f"PGS Catalog request failed for {url}: {exc}") from exc
        time.sleep(delay)
        delay *= 2

    if path:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)
    return payload


def _paged(path, params, *, cache_dir=None, timeout=30, progress=None):
    """Collect every result across a paginated endpoint."""
    query = dict(params)
    query["limit"] = _PAGE_SIZE
    url = f"{REST_BASE}/{path}?" + urllib.parse.urlencode(query)
    out = []
    expected = None
    while url:
        payload = _request_json(url, cache_dir=cache_dir, timeout=timeout)
        if expected is None:
            expected = payload.get("count")
        out.extend(payload.get("results") or [])
        url = payload.get("next")
        if progress is not None and expected:
            progress(len(out), int(expected))
        if url:
            time.sleep(_PAUSE)
    return out, expected


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

def _n_eff_from_samples(samples):
    """Effective sample size summed over a score's discovery sample sets.

    Case/control sets contribute ``4/(1/cases + 1/controls)``; sets reporting
    only a total contribute that total. A set reporting only cases, only
    controls, or neither contributes nothing, and a score with no usable set
    at all gets ``nan`` rather than zero — zero is a number a downstream gate
    would happily compare against.
    """
    from ldpred3 import n_eff_case_control

    total, used = 0.0, 0
    for entry in samples or ():
        cases = entry.get("sample_cases")
        controls = entry.get("sample_controls")
        if cases and controls:
            total += float(n_eff_case_control(float(cases), float(controls)))
            used += 1
            continue
        number = entry.get("sample_number")
        if number:
            total += float(number)
            used += 1
    return total if used else float("nan")


def _sum_field(samples, key):
    values = [entry.get(key) for entry in samples or ()]
    present = [float(v) for v in values if v]
    return float(sum(present)) if present else float("nan")


def _cohorts(samples):
    names = set()
    for entry in samples or ():
        for cohort in entry.get("cohorts") or ():
            short = (cohort.get("name_short") or "").strip()
            if short:
                names.add(short)
    return frozenset(names)


@dataclass(frozen=True)
class ScoreRecord:
    """One PGS Catalog score's metadata, in the units multipgs uses.

    ``n_eff`` is the discovery effective sample size (``nan`` when the Catalog
    records no samples); ``ancestry`` is the discovery ancestry distribution as
    percentages keyed by the Catalog's broad codes (``EUR``, ``EAS``, ...);
    :meth:`ancestry_percent` is ``nan`` when that code is absent, not ``0``.
    ``cohorts`` is the set of named discovery cohorts. ``raw`` keeps the
    unmodified API record, because this dataclass deliberately does not try to
    represent every field the Catalog has.
    """

    pgs_id: str
    name: str = ""
    trait_reported: str = ""
    efo_ids: tuple = ()
    efo_labels: tuple = ()
    n_eff: float = float("nan")
    n_total: float = float("nan")
    n_cases: float = float("nan")
    n_controls: float = float("nan")
    n_sample_sets: int = 0
    n_variants: int = 0
    weight_type: str = ""
    method_name: str = ""
    ancestry: dict = field(default_factory=dict)
    cohorts: frozenset = frozenset()
    pmid: str = ""
    doi: str = ""
    first_author: str = ""
    publication_date: str = ""
    scoring_file_url: str = ""
    harmonized_urls: dict = field(default_factory=dict)
    raw: dict = field(default_factory=dict, repr=False, compare=False)

    @classmethod
    def from_api(cls, payload):
        """Build a record from one ``/rest/score`` JSON object."""
        if not isinstance(payload, dict) or not payload.get("id"):
            raise ValueError(
                "PGS Catalog returned a record with no 'id'. A request for a "
                "score that does not exist comes back as HTTP 200 with an "
                f"empty body, which is what this looks like: {payload!r:.200}")

        samples = payload.get("samples_variants") or []
        # A score developed on a training set with no discovery sample block
        # would otherwise report nan for everything; the training samples are
        # the only sample information it has.
        if not samples:
            samples = payload.get("samples_training") or []

        harmonized = {}
        for build, entry in (payload.get("ftp_harmonized_scoring_files")
                             or {}).items():
            url = (entry or {}).get("positions")
            if url:
                harmonized[str(build)] = str(url)

        efo = payload.get("trait_efo") or []
        publication = payload.get("publication") or {}
        gwas_ancestry = ((payload.get("ancestry_distribution") or {})
                         .get("gwas") or {})

        return cls(
            pgs_id=str(payload["id"]),
            name=str(payload.get("name") or ""),
            trait_reported=str(payload.get("trait_reported") or ""),
            efo_ids=tuple(str(t.get("id")) for t in efo if t.get("id")),
            efo_labels=tuple(str(t.get("label")) for t in efo if t.get("label")),
            n_eff=_n_eff_from_samples(samples),
            n_total=_sum_field(samples, "sample_number"),
            n_cases=_sum_field(samples, "sample_cases"),
            n_controls=_sum_field(samples, "sample_controls"),
            n_sample_sets=len(samples),
            n_variants=int(payload.get("variants_number") or 0),
            weight_type=str(payload.get("weight_type") or ""),
            method_name=str(payload.get("method_name") or ""),
            ancestry={str(k): float(v)
                      for k, v in (gwas_ancestry.get("dist") or {}).items()},
            cohorts=_cohorts(samples),
            pmid=str(publication.get("PMID") or ""),
            doi=str(publication.get("doi") or ""),
            first_author=str(publication.get("firstauthor") or ""),
            publication_date=str(publication.get("date_publication") or ""),
            scoring_file_url=str(payload.get("ftp_scoring_file") or ""),
            harmonized_urls=harmonized,
            raw=payload)

    def harmonized_url(self, build):
        """Harmonized-positions URL on ``build``, or ``None`` if absent."""
        return self.harmonized_urls.get(str(build))

    @property
    def top_ancestry(self):
        """Largest discovery ancestry group, or ``""`` if none is recorded."""
        if not self.ancestry:
            return ""
        return max(self.ancestry.items(), key=lambda kv: kv[1])[0]

    def ancestry_percent(self, code="EUR"):
        """Discovery share for ``code``, or ``nan`` if the Catalog omitted it.

        Missing is not zero: a gate on ``EUR_PERCENT`` would treat an
        unrecorded ancestry block as non-European.
        """
        value = self.ancestry.get(str(code))
        return float("nan") if value is None else float(value)

    def __str__(self):
        n = "n_eff unknown" if not np.isfinite(self.n_eff) \
            else f"n_eff {self.n_eff:,.0f}"
        return (f"{self.pgs_id}: {self.trait_reported or self.name}, "
                f"{self.n_variants:,} variants, {n}")


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def _trait_score_ids(trait_id, *, include_children, cache_dir, timeout):
    payload = _request_json(f"{REST_BASE}/trait/{urllib.parse.quote(trait_id)}"
                            "?include_children=1",
                            cache_dir=cache_dir, timeout=timeout)
    if not payload:
        raise ValueError(
            f"the PGS Catalog has no trait {trait_id!r}. Trait identifiers are "
            "migrating from EFO to MONDO — breast cancer, for instance, moved "
            "from EFO_0000305 to MONDO_0004989, and the retired identifier "
            "returns an empty response rather than an error. Look the current "
            f"one up at https://www.pgscatalog.org/trait/{trait_id}/")
    ids = list(payload.get("associated_pgs_ids") or [])
    if include_children:
        ids += list(payload.get("child_associated_pgs_ids") or [])
    # Order is the Catalog's; dedupe without losing it, since a score can be
    # associated with both the parent trait and one of its children.
    return list(dict.fromkeys(str(i) for i in ids)), payload


def _scores_by_id(pgs_ids, *, cache_dir, timeout, progress):
    """Fetch records for explicit identifiers, in chunks.

    The Catalog has no bulk endpoint that reports which of the requested
    identifiers it did not recognise, so that check is done here.
    """
    wanted = list(dict.fromkeys(str(i).strip() for i in pgs_ids if str(i).strip()))
    if not wanted:
        raise ValueError("no score identifiers given")

    found = {}
    for start in range(0, len(wanted), _ID_CHUNK):
        chunk = wanted[start:start + _ID_CHUNK]
        results, _ = _paged("score/all", {"filter_ids": ",".join(chunk)},
                            cache_dir=cache_dir, timeout=timeout)
        for payload in results:
            record = ScoreRecord.from_api(payload)
            found[record.pgs_id] = record
        if progress is not None:
            progress(min(start + _ID_CHUNK, len(wanted)), len(wanted))

    missing = [i for i in wanted if i not in found]
    if missing:
        raise ValueError(
            f"the PGS Catalog did not return {len(missing)} of the "
            f"{len(wanted)} requested score(s): {', '.join(missing[:5])}"
            + (" ..." if len(missing) > 5 else "")
            + ". A request for a score that does not exist returns HTTP 200 "
              "with an empty body, so a typo looks exactly like a real score "
              "until this check.")
    return [found[i] for i in wanted]


def search_scores(*, trait_id=None, pgs_ids=None, pmid=None,
                  publication_id=None, include_children=False,
                  cache_dir=None, timeout=30, progress=None):
    """Look up PGS Catalog scores and return them as :class:`ScoreRecord`.

    Exactly one selector must be given.

    Parameters
    ----------
    trait_id : str, optional
        Catalog trait identifier, e.g. ``"MONDO_0004989"``. Passed through
        verbatim; a retired identifier raises rather than returning nothing.
    pgs_ids : sequence of str, optional
        Explicit score identifiers, e.g. ``["PGS000001", "PGS000765"]``. Every
        one must come back or the call raises.
    pmid : str or int, optional
        Every score from one publication's PubMed identifier.
    publication_id : str, optional
        Every score from one Catalog publication, e.g. ``"PGP000001"``.
    include_children : bool
        With ``trait_id``, also take scores registered against the trait's child
        traits. The Catalog keeps those in a separate list, so leaving this off
        silently omits subtype-specific scores.
    cache_dir : str, optional
        Directory for the raw JSON responses. A second run with the same
        arguments makes no network requests at all, which matters when you are
        iterating on a panel of several hundred scores.
    progress : callable, optional
        Called as ``progress(done, total)``.

    Returns
    -------
    list of ScoreRecord
        In the Catalog's order.
    """
    selectors = {"trait_id": trait_id, "pgs_ids": pgs_ids, "pmid": pmid,
                 "publication_id": publication_id}
    given = [name for name, value in selectors.items() if value is not None]
    if len(given) != 1:
        raise ValueError(
            "give exactly one of trait_id, pgs_ids, pmid or publication_id; "
            + (f"got {', '.join(given)}" if given else "got none"))

    if pgs_ids is not None:
        return _scores_by_id(pgs_ids, cache_dir=cache_dir, timeout=timeout,
                             progress=progress)

    if trait_id is not None:
        ids, _ = _trait_score_ids(str(trait_id), include_children=include_children,
                                  cache_dir=cache_dir, timeout=timeout)
        if not ids:
            raise ValueError(
                f"trait {trait_id!r} exists in the PGS Catalog but has no "
                "associated scores"
                + ("" if include_children else
                   "; it may have them on its child traits, which need "
                   "include_children=True"))
        return _scores_by_id(ids, cache_dir=cache_dir, timeout=timeout,
                             progress=progress)

    # An unrecognised query parameter comes back as an empty page rather than an
    # error, so only these two documented names are ever sent.
    params = {"pmid": str(pmid)} if pmid is not None \
        else {"pgp_id": str(publication_id)}
    results, _ = _paged("score/search", params, cache_dir=cache_dir,
                        timeout=timeout, progress=progress)
    if not results:
        label = f"PMID {pmid}" if pmid is not None else str(publication_id)
        raise ValueError(
            f"the PGS Catalog returned no scores for {label}. Note that an "
            "unrecognised query returns an empty result set rather than an "
            "error, so this may equally mean the identifier is not one the "
            "Catalog knows.")
    return [ScoreRecord.from_api(payload) for payload in results]


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------

def _download_file(url, dest, *, timeout=120):
    """Stream one URL to ``dest``. The second and last network call."""
    request = urllib.request.Request(url, headers={"User-Agent": _user_agent()})
    tmp = dest + ".part"
    with urllib.request.urlopen(request, timeout=timeout) as fh, \
            open(tmp, "wb") as out:
        while True:
            chunk = fh.read(1 << 16)
            if not chunk:
                break
            out.write(chunk)
    os.replace(tmp, dest)


def _verify_scoring_file(path, pgs_id):
    """Decompress the file and confirm it is this score's scoring file.

    A truncated gzip download is the failure this catches, and it is a common
    one over a long panel build. Reading the whole stream is the only way to
    know the tail arrived; these files are a few megabytes, so the cost is
    worth the certainty.
    """
    from .catalog import _COLUMNS

    header = {}
    column_header = None
    column_index = None
    tab_separated = False
    schema_valid = False
    n_rows = 0
    has_valid_row = False

    def present(value):
        return value.strip().upper() not in {"", ".", "NA", "N/A", "NULL"}

    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("#"):
                    stripped = line.lstrip("#").strip()
                    if "=" in stripped:
                        key, _, value = stripped.partition("=")
                        header.setdefault(key.strip().lower(), value.strip())
                    continue
                raw = line.rstrip("\n").rstrip("\r")
                if not raw.strip():
                    continue
                if column_header is None:
                    tab_separated = "\t" in raw
                    fields = raw.split("\t") if tab_separated else raw.split()
                    column_header = [field.strip().lower()
                                     for field in fields]
                    lookup = {name: i for i, name in enumerate(column_header)}
                    column_index = {
                        key: next((lookup[name] for name in names
                                   if name in lookup), None)
                        for key, names in _COLUMNS.items()
                    }
                    schema_valid = (
                        column_index["weight"] is not None
                        and column_index["ea"] is not None
                        and (column_index["id"] is not None
                             or (column_index["chrom"] is not None
                                 and column_index["pos"] is not None)))
                    continue

                n_rows += 1
                # One valid row proves this is a scoring table. Continue to EOF
                # without parsing the remaining rows so gzip tail verification
                # remains complete and the added work stays constant.
                if has_valid_row:
                    continue
                if not schema_valid:
                    continue
                fields = raw.split("\t") if tab_separated else raw.split()
                if len(fields) != len(column_header):
                    continue
                try:
                    weight = float(fields[column_index["weight"]])
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(weight):
                    continue

                if not present(fields[column_index["ea"]]):
                    continue
                has_id = (column_index["id"] is not None
                          and present(fields[column_index["id"]]))
                has_position = False
                if (column_index["chrom"] is not None
                        and column_index["pos"] is not None
                        and present(fields[column_index["chrom"]])):
                    position = fields[column_index["pos"]].strip()
                    has_position = (position.isdigit()
                                    and bool(position.lstrip("0")))
                has_valid_row = has_id or has_position
                # Keep reading to EOF: an intact header proves nothing about
                # an interrupted body.
    except (OSError, EOFError, zlib.error) as exc:
        # zlib.error covers a corrupt (not merely truncated) deflate stream;
        # BadGzipFile and EOFError cover bad or short containers. OSError
        # subsumes the former, but naming it documents the intent.
        raise ValueError(f"{path} is not a readable gzip file — the download "
                         f"was probably truncated or corrupted: {exc}") from exc

    got = header.get("pgs_id", "")
    if not got:
        raise ValueError(f"{path} has no pgs_id metadata and cannot be "
                         f"verified as {pgs_id}")
    if got != pgs_id:
        raise ValueError(f"{path} declares pgs_id={got!r} but was downloaded "
                         f"as {pgs_id!r}")
    if column_header is None:
        raise ValueError(f"{path} has no scoring-table column header")
    has_weight = column_index["weight"] is not None
    has_allele = column_index["ea"] is not None
    has_id_column = column_index["id"] is not None
    has_position_columns = (column_index["chrom"] is not None
                            and column_index["pos"] is not None)
    if not (has_weight and has_allele
            and (has_id_column or has_position_columns)):
        raise ValueError(f"{path} has no valid scoring-table header")
    if n_rows == 0:
        raise ValueError(f"{path} has no variant rows")
    if not has_valid_row:
        raise ValueError(
            f"{path} has no structurally valid variant row: one row must "
            f"have exactly {len(column_header)} fields, a finite numeric "
            "effect weight, a non-missing effect allele, and either a variant "
            "ID or a chromosome with a positive integer position")


def download_scores(records, dest, *, build="GRCh37", overwrite=False,
                    verify=True, on_error="raise", progress=None, timeout=120):
    """Download harmonized scoring files for ``records`` into ``dest``.

    Harmonized (``hmPOS``) files are the ones taken, because they are the only
    ones guaranteed to be on the build their name advertises — see
    :mod:`multipgs.catalog`. A score the Catalog has not harmonized onto
    ``build`` has no file to take, and is an error rather than a silent gap.

    Parameters
    ----------
    records : sequence of ScoreRecord
    dest : str
        Directory, created if absent.
    build : {"GRCh37", "GRCh38"}
        Must match the build of the target genotypes you will score.
    overwrite : bool
        Re-download files that are already present. Off by default, so an
        interrupted panel build resumes instead of starting over.
    verify : bool
        Decompress each file and check its header. Catches truncated downloads.
    on_error : {"raise", "skip"}
        ``"skip"`` records the failure in the log and carries on, which is what
        you want across several hundred scores.
    progress : callable, optional
        Called as ``progress(index, total, pgs_id)``.

    Returns
    -------
    (paths, log) : (list of str, dict)
        ``paths`` is one path per input record, ``None`` where a download was
        skipped after an error. ``log`` counts what happened.
    """
    if str(build) not in BUILDS:
        raise ValueError(f"build must be one of {BUILDS}, got {build!r}")
    if on_error not in ("raise", "skip"):
        raise ValueError(f"on_error must be 'raise' or 'skip', got {on_error!r}")

    records = list(records)
    os.makedirs(dest, exist_ok=True)
    paths, errors = [], {}
    n_cached = n_downloaded = 0

    for i, record in enumerate(records):
        if progress is not None:
            progress(i, len(records), record.pgs_id)
        url = record.harmonized_url(build)
        path = os.path.join(dest, f"{record.pgs_id}_hmPOS_{build}.txt.gz")
        try:
            if url is None:
                raise ValueError(
                    f"{record.pgs_id} has no harmonized file on {build}; the "
                    f"Catalog has it on {sorted(record.harmonized_urls) or 'no build'}")
            downloaded = overwrite or not os.path.exists(path)
            if downloaded:
                _download_file(url, path, timeout=timeout)
            if verify:
                try:
                    _verify_scoring_file(path, record.pgs_id)
                except Exception:
                    # Do not strand a bad cache entry: without removal every
                    # resumed run would trust the path, fail verification, and
                    # never attempt the download again.
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
                    raise
            if downloaded:
                n_downloaded += 1
            else:
                n_cached += 1
        except Exception as exc:                      # noqa: BLE001
            if on_error == "raise":
                raise
            errors[record.pgs_id] = str(exc)
            paths.append(None)
            continue
        paths.append(path)

    log = {"n_requested": len(records), "n_downloaded": n_downloaded,
           "n_cached": n_cached, "n_failed": len(errors), "build": str(build),
           "verified": bool(verify)}
    if errors:
        log["errors"] = errors
    return paths, log


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def _metadata_value(record, column):
    if column == "N_EFF":
        return record.n_eff
    if column == "N_TOTAL":
        return record.n_total
    if column == "N_CASES":
        return record.n_cases
    if column == "N_CONTROLS":
        return record.n_controls
    if column == "N_SAMPLE_SETS":
        return record.n_sample_sets
    if column == "N_VARIANTS":
        return record.n_variants
    if column == "TRAIT":
        return record.trait_reported or record.name
    if column == "EFO":
        return ",".join(record.efo_ids)
    if column == "ANCESTRY":
        return record.top_ancestry
    if column == "EUR_PERCENT":
        return record.ancestry_percent("EUR")
    if column == "WEIGHT_TYPE":
        return record.weight_type
    if column == "METHOD":
        return record.method_name
    if column == "PMID":
        return record.pmid
    if column == "DOI":
        return record.doi
    if column == "N_COHORTS":
        return len(record.cohorts)
    if column == "COHORTS":
        return ",".join(sorted(record.cohorts))
    raise ValueError(f"unknown metadata column {column!r}; known columns are "
                     + ", ".join(_METADATA_COLUMNS))


def _format(value):
    if isinstance(value, float):
        if not np.isfinite(value):
            return "NA"
        return f"{value:.6g}"
    text = str(value)
    # A tab or newline inside free text would silently add a column.
    return text.replace("\t", " ").replace("\n", " ").replace("\r", " ")


def write_score_metadata(records, path, *, columns=None):
    """Write a ``SCORE``-keyed metadata table for ``records``.

    The first column is ``SCORE``, which is what
    :func:`multipgs.cli._read_score_vector` keys on, so a single-column table
    such as ``columns=["N_EFF"]`` drops straight into ``multipgs meta --n-eff``
    or ``multipgs fit --penalty-factor``. The full table is for reading and for
    deciding what belongs in the panel.

    Missing numbers are written ``NA`` rather than ``0``: a score whose sample
    size the Catalog does not record must not be silently treated as one with
    no samples.
    """
    columns = list(_METADATA_COLUMNS if columns is None else columns)
    unknown = [c for c in columns if c not in _METADATA_COLUMNS]
    if unknown:
        raise ValueError(f"unknown metadata column(s) {unknown}; known columns "
                         "are " + ", ".join(_METADATA_COLUMNS))
    records = list(records)
    ids = [r.pgs_id for r in records]
    if len(set(ids)) != len(ids):
        seen, duplicates = set(), []
        for i in ids:
            if i in seen:
                duplicates.append(i)
            seen.add(i)
        raise ValueError("records contain duplicate score id(s): "
                         + ", ".join(sorted(set(duplicates))[:5]))

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("SCORE\t" + "\t".join(columns) + "\n")
        for record in records:
            row = [_format(_metadata_value(record, c)) for c in columns]
            fh.write(record.pgs_id + "\t" + "\t".join(row) + "\n")
    return path


def read_score_metadata(path):
    """Read a table written by :func:`write_score_metadata`.

    Returns ``{score_id: {column: value}}``. Numeric columns become floats;
    ``NA`` is ``nan``.
    """
    numeric = {"N_EFF", "N_TOTAL", "N_CASES", "N_CONTROLS", "N_SAMPLE_SETS",
               "N_VARIANTS", "EUR_PERCENT", "N_COHORTS"}
    with open(path, "r", encoding="utf-8") as fh:
        header = fh.readline().rstrip("\n").split("\t")
    if not header or header[0].upper() != "SCORE":
        raise ValueError(f"{path}: expected a SCORE-keyed metadata table")
    columns = header[1:]
    out = {}
    with open(path, "r", encoding="utf-8") as fh:
        next(fh)
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != len(header):
                raise ValueError(f"{path}: expected {len(header)} columns")
            row = {}
            for name, raw in zip(columns, parts[1:]):
                if name in numeric:
                    row[name] = (float("nan") if raw in ("", "NA", "N/A", ".")
                                 else float(raw))
                else:
                    row[name] = raw
            out[str(parts[0])] = row
    return out


def cohort_overlap(records):
    """Shared discovery cohorts between every pair of scores.

    Returns ``(overlap, score_ids)`` where ``overlap[j, k]`` is the Jaccard
    index of the two scores' named cohort sets — 1.0 when they name exactly the
    same cohorts, 0.0 when they share none, and ``nan`` on any row or column
    whose score names no cohorts at all.

    This is a flag, not a measurement. Two scores naming one cohort may use
    disjoint individuals from it; a score naming none may still share every
    individual with its neighbour. Overlap is invisible to cross-validation on
    the target cohort — the stacking weights will look fine and the accuracy
    will not survive to a new cohort — so a cheap lower bound on it is worth
    having before a panel is trusted. ``docs/theory.md`` covers what it does to
    the fit.
    """
    records = list(records)
    k = len(records)
    ids = np.array([r.pgs_id for r in records], dtype=object)
    overlap = np.full((k, k), np.nan)
    sets = [r.cohorts for r in records]
    for j in range(k):
        if not sets[j]:
            continue
        for i in range(k):
            if not sets[i]:
                continue
            union = sets[j] | sets[i]
            overlap[j, i] = len(sets[j] & sets[i]) / len(union) if union else np.nan
    return overlap, ids
