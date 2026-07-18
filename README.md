# Randomness quality in AI/ML experiments

Toolkit and plan for comparing **random bitstream quality** across AI/ML stochastic components (init, dropout, shuffling, augmentation).

## Quick start

```bash
# Regenerate small pilot bit files (64 KiB each)
python3 scripts/generate_pilot_bits.py

# Register your large archives in sources.yaml (paths under data/random_bits/large/)
```

Pilot files live in `data/random_bits/` (see `MANIFEST.md`). Large archives should go in `data/random_bits/large/` (gitignored).

## Plan

See **[docs/EXPERIMENT_PLAN.md](docs/EXPERIMENT_PLAN.md)** for:

1. Literature summary (Soft Computing, ESWA, MLST, NIST/TestU01, MLSys, …)
2. Hypotheses and quality tiers
3. Two-stage design: statistical characterization → ML ablations
4. Recommended first experiment (dropout on Fashion-MNIST / MLP)

## Layout

```
docs/EXPERIMENT_PLAN.md   # full experiment plan
data/random_bits/         # pilot .bin files + MANIFEST
scripts/                  # generators / (soon) RNG + stats runners
sources.yaml              # source registry
```
