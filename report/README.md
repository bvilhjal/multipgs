# multipgs methods and validation report

This directory contains the packaged technical report for multipgs:

- `multipgs_methods.tex` is the editable LaTeX source.
- `multipgs_methods.pdf` is the compiled and visually verified report.

Build from this directory with:

    pdflatex -interaction=nonstopmode -halt-on-error multipgs_methods.tex
    pdflatex -interaction=nonstopmode -halt-on-error multipgs_methods.tex

The figures are drawn in LaTeX from values copied from the checked benchmark
summary CSV files named in Appendix A. Regenerate and visually inspect the PDF
when those artifacts, methods, or documented test results change.
