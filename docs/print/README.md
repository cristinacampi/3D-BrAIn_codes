# Printable Manual

The publication-oriented manual is maintained in
`3dbrain_user_scientific_manual.tex`. It is intentionally generic: final
dataset reports should add the actual sample counts, experimental design, run
parameters, and results.

Build from the repository root:

```bash
mkdir -p output/pdf
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -output-directory=output/pdf \
  docs/print/3dbrain_user_scientific_manual.tex
```

The result is `output/pdf/3dbrain_user_scientific_manual.pdf`. The `output/`
directory and generated PDFs are ignored by Git. Attach the reviewed PDF and a
checksum to a GitHub release instead of committing build artifacts.

Before release, render every page and inspect the cover, table of contents,
tables, code blocks, equations, links, headers, and page breaks. Update the
software version, edition date, author list, and citation metadata together.
