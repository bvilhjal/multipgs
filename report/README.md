# multipgs methods and validation report

This directory contains the packaged technical report for multipgs:

- `multipgs_methods.tex` is the editable LaTeX source.
- `generate_evidence.py` derives checked report values and provenance.
- `evidence.json` records the declared input digest, test environment, and
  benchmark evidence.
- `generated_evidence.tex` supplies those values to the LaTeX source.
- `multipgs_methods.pdf` is the compiled and visually verified report.

Build from this directory with:

    python generate_evidence.py
    pdflatex -interaction=nonstopmode -halt-on-error multipgs_methods.tex
    pdflatex -interaction=nonstopmode -halt-on-error multipgs_methods.tex
    python generate_evidence.py --check

The generator runs the full test suite, derives headline and plotted values
from the checked benchmark CSVs, and hashes the declared report inputs. The
source revision is generation context; the input SHA-256 is the freshness
authority. Regenerate and visually inspect the PDF when those inputs change.
