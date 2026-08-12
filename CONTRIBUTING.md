# Contributing

Keep changes small, evidence-backed, and explicit about what was actually
tested. Run commands below from the repository root unless a section says
otherwise.

## Development environment

`ldpred3` is developed beside this repository and is not published on PyPI:

```bash
python -m pip install -e "../ldpred3[fast]"
python -m pip install -e ".[fast,test,lint]"
```

CI installs the exact ldpred3 revision in `.github/workflows/ci.yml`. Check
compatibility with that revision before changing the declared dependency
range.

## Routine verification

```bash
python -m ruff check .
python -m pytest -q
python -m examples.minimal
python benchmarks/check_results.py
```

The example uses only simulated data. The benchmark checker recomputes
committed summaries from raw rows and checks the headline benchmark prose; it
does not rerun expensive or externally sourced experiments.

## Benchmark evidence

Run only the producer whose method, default, or evidence changed; every script
under `benchmarks/` documents its command and writes to `benchmarks/results/`
by default. Then run:

```bash
python benchmarks/check_results.py
git diff -- benchmarks/results benchmarks/README.md
```

Commit raw rows, derived summaries, and provenance together. Never rewrite a
historical provenance record merely to make it look current. Large LD
references, scoring files, scratch runs, and logs stay outside the package as
specified in `.gitignore`. Peak-RSS benchmarks use `resource.getrusage` on
POSIX and `GetProcessMemoryInfo` on Windows; absolute memory and timing values
are platform-specific and should be compared only within a documented run.

## Methods report

If methods, checked result claims, or the reported test environment change,
refresh the checked benchmark and report evidence from the repository root:

```bash
python benchmarks/check_results.py
python report/generate_evidence.py
```

The generator runs the full test suite and writes `report/evidence.json` plus
the LaTeX macros in `report/generated_evidence.tex`. Update narrative TeX when
the interpretation changed. Then change into `report/`, regenerate once more
so that edit is included in the input digest, and build:

```bash
cd report
python generate_evidence.py
pdflatex -interaction=nonstopmode -halt-on-error multipgs_methods.tex
pdflatex -interaction=nonstopmode -halt-on-error multipgs_methods.tex
cd ..
python report/generate_evidence.py --check
```

Inspect every page of `multipgs_methods.pdf` before committing both source and
PDF. The final check is read-only and catches stale inputs or generated files;
Appendix A names the benchmark artifacts behind quantitative statements.

## Distribution and release refresh

```bash
python -m pip install --upgrade build twine
python -m build
python -m twine check dist/*
```

Before tagging a release:

1. Start from a clean worktree and update the version and `CHANGELOG.md`.
2. Refresh affected benchmarks and the methods report; do not manufacture
   missing historical inputs or provenance.
3. Run the routine verification and build commands above.
4. Check the wheel from outside the checkout so local source cannot shadow the
   installed package. CI performs this check and verifies the shipped report,
   guide, API map, benchmark evidence, and ignored-input exclusions.
5. Tag only the commit that passed CI.

User-facing material is split by purpose: `docs/guide.md` for workflows,
`docs/api.md` for the public surface, `docs/theory.md` for estimands and
interpretation, `docs/algorithm.md` for implementation details, and
`benchmarks/README.md` for executable evidence. A wheel installs the report,
these Markdown documents, and this contributor guide below
`<sys.prefix>/share/doc/multipgs`.
