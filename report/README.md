# Technical report

`multipgs_methods.pdf` is a methods note for colleagues and PhD
students who already use polygenic scores and GWAS summary
statistics. It fixes the combination estimand, the
selection-index algebra, the three information routes, and which
simulation claims survive their design.

Rebuild from this directory (requires
[Tectonic](https://tectonic-typesetting.github.io/) or
`pdflatex`):

```bash
python generate_evidence.py
tectonic -X compile multipgs_methods.tex
python generate_evidence.py --record-pdf
python generate_evidence.py --check
```

`generate_evidence.py` derives checked report values and
provenance. `evidence.json` records the declared input digest,
test environment, and benchmark headlines. `generated_evidence.tex`
supplies those values to the LaTeX source.

The generator runs the full test suite, derives headline and
plotted values from the checked benchmark CSVs, and hashes the
declared report inputs. The source revision is generation
context; the input SHA-256 is the freshness authority. The PDF
embeds that input digest in its Subject metadata; `--record-pdf`
verifies the binding before recording a separate PDF SHA-256.
This prevents an old PDF from being blessed after the evidence
changed. Tectonic's compressed metadata requires `pypdf`,
included in the `test` extra.

The PDF is documentation in the repository, not a runtime
dependency of the installed package.
