# Randomness quality in AI/ML experiments

Toolkit and plan for comparing **random bitstream quality** across AI/ML stochastic components (init, dropout, shuffling, augmentation).

## Quick start

```bash
# Regenerate small pilot bit files (64 KiB each)
python3 scripts/generate_pilot_bits.py

# Register your large archives in sources.yaml (paths under data/random_bits/large/)
```

Pilot files live in `data/random_bits/` (see `MANIFEST.md`). Large archives should go in `data/random_bits/large/` (gitignored).

## Paper (literature review)

See **[`paper/`](paper/)** for a LaTeX critical review covering RNG quality in ML, SGD/Adam noise, and metaheuristics, plus research gaps and agenda.

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Experiment plan

See **[docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md)** for:

1. Literature-backed hypotheses and quality tiers
2. Two-stage design: statistical characterization → ML ablations
3. Recommended first experiment (dropout on Fashion-MNIST / MLP)

## Layout

```
paper/                    # LaTeX review manuscript
docs/EXPERIMENT_PLAN.md   # operational experiment plan
data/random_bits/         # pilot .bin files + MANIFEST
scripts/                  # generators / (soon) RNG + stats runners
sources.yaml              # source registry
```
