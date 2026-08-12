# multipgs technical review

This directory contains the versioned technical review of multipgs 0.3.0 at
commit 0ce9ad8:

- multipgs_review.tex is the editable LaTeX source.
- multipgs_review.pdf is the compiled and visually verified report.

Build from this directory with:

    pdflatex -interaction=nonstopmode -halt-on-error multipgs_review.tex
    pdflatex -interaction=nonstopmode -halt-on-error multipgs_review.tex

The figures are drawn in LaTeX from values copied from the checked benchmark
summary CSV files named in Appendix A. Regenerate or re-audit the report when
those artifacts or the reviewed source commit change.
