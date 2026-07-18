# Experiment Plan: Randomness Quality in AI/ML Tasks

## Goal

Compare whether **binary random bitstreams of different statistical quality** change outcomes in common AI/ML stochastic components (weight init, dropout, shuffling, data augmentation), using your large bit archives plus small local pilot files.

This plan is grounded in peer-reviewed / preprint literature (see §References). Prior work shows: (1) RNG choice can change learning dynamics, but effects are often **dataset- and task-dependent**; (2) framework PRNG implementations differ even when algorithm names match; (3) valid comparisons need **many seeds** and non-parametric tests.

---

## Research landscape (what journals say)

### A. Does RNG quality change ML performance?

| Paper | Venue / source | Finding relevant to us |
|---|---|---|
| Bird et al., *On the effects of pseudorandom and quantum-random number generators in soft computing* | Soft Computing (2019) | QRNG vs PRNG weight init changes learning curves on DNNs/CNNs; gains are small and **data-dependent** (e.g. EEG +2.8%, CIFAR-10 +0.9%, MNIST ~0). |
| Koivu et al., *Quality of randomness and node dropout regularization for fitting neural networks* | Expert Systems with Applications (2022) | Dropout driven by different RNGs (incl. TRNG) can help **or** hurt overfitting depending on task; recommend RNG as an experimental factor. |
| Heese et al. (cited in Koivu) | 2021 | Often **no reproducible statistical difference** for QRNG vs PRNG init — motivates careful stats. |
| Huk / related contextual NN work | ACIIDS / related | PRNG choice can affect training of specific NN architectures. |

### B. Randomness as variability / reproducibility (not quality per se)

| Paper | Finding |
|---|---|
| Summers & Dinneen, *Nondeterminism and Instability in Neural Network Optimization* (2021) | Tiny init perturbations can rival all other noise sources; SGD is highly unstable. |
| Pham et al., *Quantifying Inherent Randomness in ML Algorithms* (2022) | Init, shuffle, dropout, split all contribute; data splits often dominate. |
| Bouthillier et al. / model-comparison guidance; ACM REP-style work | For small effects (<~2.5%), use **≥25 seeds** and non-parametric tests (e.g. Brunner–Munzel). |
| Zhuang et al., *Randomness in NN Training: Impact of Tooling* (MLSys 2022) | Algorithmic vs implementation nondeterminism both matter. |

### C. Characterizing RNG quality (and ML-as-a-test)

| Paper / standard | Role |
|---|---|
| NIST SP 800-22, Dieharder, TestU01 (SmallCrush/Crush/BigCrush) | Classical statistical batteries for bitstreams. |
| Valle et al., *Assessing the quality of RNGs through neural networks* | Machine Learning: Science and Technology (2024) | CNN/LSTM distinguish weak PRNGs from a “golden” RNG; strong PRNG/QRNG often indistinguishable. |
| Li & Zeng, *Transformers in PRNG* | AAAI (2026) | Transformers can emulate LCG/MT; NIST used as quality check. |
| arXiv:2507.03007, *Statistical Quality and Reproducibility of PRNGs in ML technologies* | Framework PRNGs (PyTorch/TF/NumPy) can diverge from reference C implementations under TestU01. |

**Takeaway for experiment design:** Treat this as a **two-stage** study: (1) quantify bitstream quality; (2) inject those streams into isolated ML stochastic sites and measure downstream metrics with enough replicates to detect small effects.

---

## Hypotheses

1. **H1 (quality ladder):** Lower-quality streams (biased / weak LCG) produce worse or more unstable ML metrics than high-quality streams (OS CSPRNG / your best archive) when randomness is heavily used (dropout, shuffle).
2. **H2 (site sensitivity):** Effect size is largest for **dropout + data shuffle**, smaller for **one-shot weight init**, smallest when randomness is used only once.
3. **H3 (task dependence):** Absolute accuracy deltas are small on easy tasks (MNIST) and larger / more detectable on noisier or smaller-data tasks — consistent with Bird et al. and Koivu et al.
4. **H4 (null possible):** For cryptographically strong streams that already pass NIST/TestU01, ML metrics may be **statistically indistinguishable** (aligns with Heese / Valle “cannot distinguish” results).

---

## Random sources (inputs)

### Your large archives (primary)

Place large binary files under `data/random_bits/large/` (gitignored). Recommended naming:

```
data/random_bits/large/<source_id>/<shard>.bin
```

Each file: raw bytes, bit order documented (MSB-first vs LSB-first). Prefer ≥100–500 MiB per source for full Crush-scale stats; ML runs can stream with a cursor.

### Pilot files (already generated)

Under `data/random_bits/` (64 KiB each):

| File | Role |
|---|---|
| `biased_p60_64kib.bin` | Negative control (biased bits) |
| `weak_lcg_64kib.bin` | Weak classical PRNG |
| `xorshift32_64kib.bin` | Intermediate PRNG |
| `mt19937_python_64kib.bin` | Common ML-style PRNG |
| `os_urandom_64kib.bin` | High-quality OS CSPRNG reference |

Regenerate / extend with `scripts/generate_pilot_bits.py`.

### Quality tiers (labels)

Assign each source a tier after Stage 1:

- **T0** fails basic frequency / runs
- **T1** fails Dieharder / SmallCrush subset
- **T2** passes SmallCrush, fails Crush
- **T3** passes Crush / NIST suite (practical “good”)
- **T4** your external high-entropy archive (candidate golden standard)

---

## Experimental architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────────┐
│ Bitstream store │────▶│ BitstreamRNG     │────▶│ ML stochastic site │
│ (files / shards)│     │ (byte→float API) │     │ init/dropout/shuf  │
└─────────────────┘     └──────────────────┘     └─────────┬──────────┘
         │                                                    │
         ▼                                                    ▼
┌──────────────────┐                               ┌────────────────────┐
│ Stats battery    │                               │ Metrics + curves   │
│ NIST/Dieharder/  │                               │ multi-seed tables  │
│ TestU01 + entropy│                               └────────────────────┘
└──────────────────┘
```

### Core abstraction: `BitstreamRNG`

A drop-in RNG that:

1. Memory-maps or streams a binary file.
2. Exposes `random()`, `randint()`, `choice()`, `shuffle()`, `bernoulli(p)`.
3. Uses a **consumed-byte cursor** (not a seedable PRNG), so quality comes from the file.
4. Logs bytes consumed per experiment for auditability.
5. Optionally wraps framework hooks (PyTorch: custom dropout mask generator; NumPy: `Generator` with `BitGenerator` from bits).

**Critical control:** Keep **all other seeds fixed** when ablating one site (e.g. fix init PRNG while varying dropout bitstream).

---

## Stage 1 — Randomness quality characterization

**Purpose:** Independent variable = measured quality, not just “source name.”

### Tests (lightweight → heavy)

1. **Cheap local suite (always):** bit frequency, block frequency, runs, serial correlation, Shannon entropy, χ² on bytes.
2. **NIST SP 800-22** (when installed): frequency, block frequency, runs, longest run, FFT, approximate entropy, serial, cumulative sums (subset OK for pilots).
3. **Dieharder** or **TestU01 SmallCrush** for medium files; Crush/BigCrush only when archives are large enough.
4. **Optional ML distinguishability** (Valle et al.): train a small CNN/LSTM to classify windows of bits vs `os_urandom` / golden source; report AUC / accuracy. If AUC≈0.5, streams are operationally indistinguishable for that model class.

### Outputs

- `results/quality/<source_id>.json` — p-values, pass/fail, entropy, bytes tested
- Tier label T0–T4
- Correlation later: tier vs ML delta

---

## Stage 2 — ML task experiments

Isolate **one stochastic site per experiment series**. Hold architecture, LR, epochs, batch size, data split fixed.

### Sites to ablate

| ID | Stochastic site | Why |
|---|---|---|
| S1 | Weight initialization | Classic Bird et al. setup |
| S2 | Dropout masks | Direct Koivu et al. setup |
| S3 | Per-epoch data shuffle / batch order | Often large variance source |
| S4 | Data augmentation randomness | Strong on vision |
| S5 | (Optional) Bootstrap / bagging / RF feature subsample | Classical ML baseline |

### Tasks (start small, then scale)

**Pilot (CPU-friendly, validate pipeline):**

1. Logistic / MLP on Fashion-MNIST or sklearn digits
2. Small CNN on MNIST (≤5 epochs)

**Main:**

3. CNN on CIFAR-10 (Bird et al. saw clearer gaps)
4. Small tabular MLP on a noisy UCI / OpenML set (or EEG-like if available)
5. Optional: fine-tune a tiny transformer for 1 epoch on a toy LM task (randomness in dropout + sampling)

### Metrics

- Primary: test accuracy / F1 / NLL
- Secondary: train–test gap (overfitting), best-epoch, learning-curve AUC
- Stability: std / IQR across seeds; pairwise disagreement rate (prediction churn)
- Efficiency: wall time, bytes of entropy consumed

### Replicates & statistics

Following model-comparison guidance for small effects:

- Pilot: **n = 10** runs per (source × site × task)
- Confirmatory: **n ≥ 25** when expected Δ < 2.5%
- Predefine seed indices `{0…n−1}` for any residual PRNG needs
- Test: Brunner–Munzel or Mann–Whitney + report effect size (Cliff’s δ / Cohen’s d)
- Multiple comparisons: Holm–Bonferroni across sources within a site
- Report full distributions (violin / ECDFs), not only means

### Controls

- **Same-bits control:** two disjoint shards from the same high-quality archive → expect null.
- **Negative control:** `biased_p60` → expect detectable degradation or instability.
- **Framework baseline:** native `torch.Generator` / NumPy default vs bitstream injection.

---

## Stage 3 — Analysis & reporting

1. Quality–performance plots: NIST fail count or ML-distinguishability score vs mean test accuracy.
2. Site ranking: which ML site is most sensitive to quality tier.
3. Interaction: quality × dataset difficulty.
4. Practical recommendation: when to bother using high-entropy archives vs framework PRNG.

Artifact layout:

```
results/
  quality/
  ml/<task>/<site>/<source_id>/run_*.json
  figures/
  summary.md
```

---

## Implementation roadmap (engineering)

### Milestone M0 — Scaffold (done / next)

- [x] Literature-informed plan
- [x] Pilot bit files + manifest
- [ ] `BitstreamRNG` + unit tests
- [ ] Cheap statistical suite script
- [ ] One end-to-end MNIST MLP dropout ablation on pilot files

### Milestone M1 — Quality pipeline

- Wire NIST / Dieharder if available; else ship pure-Python subset
- Ingest large user archives via config YAML (`sources.yaml`)
- Produce tier labels

### Milestone M2 — ML ablations

- PyTorch trainers with injectable RNG for S1–S4
- Multi-seed runner + CSV/JSON aggregator
- Statistical report notebook or script

### Milestone M3 — Scale & paper-style writeup

- CIFAR-10 + tabular tasks
- Optional NN-based RNG distinguishability
- Final comparison tables matching Soft Computing / ESWA style

---

## Threats to validity

1. **Confounded “quality”:** Different sources may differ in bias *and* correlation structure; always report Stage 1 metrics.
2. **Byte exhaustion / wraparound:** Looping a short file reintroduces periodicity — forbid wrap for confirmatory runs or document period.
3. **Framework nondeterminism:** Disable cuDNN nondeterminism or run CPU-only for confirmatory studies (Zhuang et al.).
4. **Underpowered n:** Small Δ on MNIST needs many seeds.
5. **Data leakage of randomness:** Ensure bitstream shards used for quality testing are disjoint from those consumed in training when claiming independence.

---

## Minimal first experiment (recommended start)

1. Sources: `biased_p60`, `weak_lcg`, `mt19937`, `os_urandom` (+ one shard of your large archive).
2. Site: **S2 dropout** on a 2-hidden-layer MLP, Fashion-MNIST, dropout p=0.5.
3. Fix: init + shuffle use a locked PRNG seed; only dropout masks read from bitstream.
4. n=10 runs; compare test accuracy and train–test gap.
5. Parallel: run cheap Stage 1 stats on the same files.

Success criterion for the pipeline: negative control (`biased_p60`) differs from `os_urandom` with p < 0.05 or clearly higher variance; if not, fix injection / underpowering before scaling.

---

## References (selected)

1. Bird, J.J., et al. (2019). On the effects of pseudorandom and quantum-random number generators in soft computing. *Soft Computing*. https://doi.org/10.1007/s00500-019-04450-0  
2. Koivu, A., et al. (2022). Quality of randomness and node dropout regularization for fitting neural networks. *Expert Systems with Applications*, 207, 117938. https://doi.org/10.1016/j.eswa.2022.117938  
3. Valle, C., et al. (2024). Assessing the quality of random number generators through neural networks. *Machine Learning: Science and Technology*. https://doi.org/10.1088/2632-2153/ad56fb  
4. Summers, C., & Dinneen, M.J. (2021). Nondeterminism and Instability in Neural Network Optimization. https://arxiv.org/abs/2103.04514  
5. Pham, H.V., et al. / related tooling papers; Zhuang et al. (2022). Randomness in Neural Network Training: Characterizing the Impact of Tooling. *MLSys*.  
6. L’Ecuyer, P., & Simard, R. (2007). TestU01. *ACM TOMS*.  
7. Bassham, L., et al. (2010). NIST SP 800-22 Rev. 1a.  
8. Statistical Quality and Reproducibility of PRNGs in ML technologies (2025). https://arxiv.org/abs/2507.03007  
9. Li, R., & Zeng, L. (2026). Transformers in Pseudo-Random Number Generation. *AAAI*.  

---

## Next decision points (for implementation)

1. Preferred stack: **PyTorch** vs TensorFlow vs JAX.
2. Which large bit archives you will mount (paths + provenance).
3. Whether confirmatory runs must be CPU-deterministic only.
4. Priority order of sites: recommend **S2 → S3 → S1 → S4**.
