# Paper: Randomness Quality in AI/ML and Stochastic Optimisation

LaTeX manuscript for a critical literature review and research agenda.

## Build

```bash
cd paper
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Or with `latexmk`:

```bash
latexmk -pdf main.tex
```

## Layout

| Path | Content |
|------|---------|
| `main.tex` | Root document |
| `refs.bib` | Bibliography |
| `sections/` | Introduction, background, ML review, optimisation review, critique, gaps, agenda |
| `figures/` | Placeholder for future figures |

The current draft emphasises the **critical literature review** and **research gaps**. Experimental sections can be added after the planned studies land.
