# AGENTS.md

## Cursor Cloud specific instructions

This is an offline research repository with two independent products; there is no
web app, server, or database to run.

1. Python random-bitstream toolkit — `scripts/generate_pilot_bits.py`
2. LaTeX critical-review paper — `paper/`

### Python toolkit
- Standard-library only: there is **no** dependency manifest (`requirements.txt`,
  `pyproject.toml`, etc.) and nothing to `pip install`. Runs on the system `python3`.
- Run it with `python3 scripts/generate_pilot_bits.py` (see `README.md`). By default
  it overwrites the committed pilot `.bin` files and `MANIFEST.md` in
  `data/random_bits/`; pass `--out-dir` to write elsewhere.
- Four of the five generators (`weak_lcg`, `xorshift32`, `mt19937_python`,
  `biased_p60`) are deterministic and reproduce the committed files byte-for-byte;
  `os_urandom_*` is nondeterministic by design, so its sha256 changes every run.
- There are no automated tests and no configured linter. `python3 -m py_compile
  scripts/generate_pilot_bits.py` is a quick syntax check.

### LaTeX paper
- Build with the sequence in `paper/README.md`:
  `cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main`
  (produces a 19-page `paper/main.pdf`). Build artifacts, including `main.pdf`,
  are gitignored.
- The full LaTeX toolchain (`texlive-latex-base/recommended/extra`,
  `texlive-fonts-recommended`, `texlive-bibtex-extra`) plus the Debian **`lmodern`**
  package are system dependencies preinstalled in the environment snapshot. They are
  intentionally NOT in the update script (system packages, not per-startup refresh).
  Non-obvious gotcha: `lmodern.sty` (used by `main.tex`) is not part of the
  `texlive-latex-*` packages on Debian/Ubuntu — the separate `lmodern` package is
  required, otherwise the first `pdflatex` pass fails with `File 'lmodern.sty' not found`.
