# Methodology Focus: What to Prioritise and What Can Be Done

This note extracts a **concrete methodology** from the literature review and experiment plan: which design choices matter scientifically, which are feasible with large binary archives + pilot files, and a phased build order.

---

## 1. The methodological question (sharpened)

Not: “Does a random seed change accuracy?”  
Not: “Is QRNG better than PRNG by brand name?”

**Focus question:**

> Holding architecture, data split, hyperparameters, and all other stochastic sites fixed, does **measured bitstream quality** (of the RNG that drives *one* site) change ML / optimiser outcomes?

That forces three methodological pillars:

| Pillar | Meaning | Literature anchor |
|--------|---------|-------------------|
| **P1 Quality measurement** | Independent variable = quality tier / scores, not source marketing name | NIST, TestU01, Valle et al. |
| **P2 Site isolation** | Vary one stochastic site; freeze the rest | Summers & Dinneen; Zhuang et al. |
| **P3 Powered comparison** | Predefined multi-run protocol + non-parametric tests | Åkesson et al. (ACM REP) |

If any pillar is missing, results are hard to interpret (this is the main failure mode in prior QRNG/PRNG papers).

---

## 2. Methodology stack (what to focus on)

### 2.1 Stage A — Characterise bitstreams (do first, keep cheap)

**Must do (always runnable locally):**

1. Bit frequency / balance  
2. Runs test (or simple run-length histogram)  
3. Serial / lag-1 autocorrelation on bits or bytes  
4. Byte χ² uniformity  
5. Shannon entropy (block-wise)  
6. Manifest: SHA-256, length, bytes later consumed in ML runs  

**Should do when tools/data allow:**

- NIST SP 800-22 subset (frequency, block frequency, runs, FFT, approx. entropy)  
- TestU01 SmallCrush (needs more bits than 64 KiB pilots — use large archives or regenerate longer pilots)

**Optional / later:**

- Neural distinguisher vs `os_urandom` (Valle-style) — useful paper novelty, not required for the first decisive experiment  
- Full Crush/BigCrush — only when archives are large enough  

**Output:** each source → `{tier: T0–T4, scores..., hash}` in `results/quality/`.

**Feasibility now:** ✅ Cheap suite + tier labels on pilot files. ⚠️ NIST/TestU01 need install + more bits. ❌ BigCrush not for 64 KiB pilots.

---

### 2.2 Stage B — Inject via `BitstreamRNG` (core engineering)

**Focus API (minimal):**

```text
random() → U[0,1) from next 64 bits (or 53 for float)
bernoulli(p) → for dropout masks
shuffle(indices) → for data order
normal(mean, std) → for init (Box–Muller / inverse CDF from uniforms)
bytes_consumed() → audit
```

**Hard rules (methodology, not style):**

1. **Cursor consumption** — quality comes from the file; do not re-seed a PRNG that ignores the archive.  
2. **No silent wraparound** in confirmatory runs (raise if exhausted). Pilots may allow wrap with a logged warning only for smoke tests.  
3. **Disjoint shards** for Stage A testing vs Stage B training when claiming independence.  
4. **Bit order documented** (MSB-first default).

**Feasibility now:** ✅ Pure-Python `BitstreamRNG` over pilot `.bin` files is the highest-leverage next code drop.

---

### 2.3 Site-isolation protocol (the experimental design)

Adopt Summers & Dinneen’s control logic, adapted from *seeds* to *bitstreams*:

```text
For site i under test:
  - Site i  ← BitstreamRNG(source_s)     # factor of interest
  - Site j≠i ← fixed PRNG with seed 0    # frozen
  - Data split ← fixed indices (no random split per run)
  - Determinism ← cudnn.deterministic / CPU-only for confirmatory
  - Repeat for sources s ∈ {biased, weak_lcg, mt, urandom, archive…}
  - Within each (source, site): n independent runs with disjoint byte ranges
    OR n runs that only differ by starting offset / shard id
```

**Site priority (focus order):**

| Priority | Site | Why focus here | Feasibility |
|----------|------|----------------|-------------|
| **1** | **S2 Dropout** | Highest bytes/epoch; Koivu et al. already saw quality effects; easy to custom-mask | ✅ High |
| **2** | **S3 Shuffle** | Directly shapes SGD/Adam gradient noise (theory link) | ✅ High |
| **3** | **S1 Init** | Classic Bird/Heese debate; one-shot randomness → smaller effects | ✅ Medium (need careful float mapping) |
| **4** | **S4 Augmentation** | Strong on vision; more engineering | ⚠️ Later |
| **5** | Metaheuristics (PSO/DE) | Clear prior “quality floor”; good secondary paper thread | ⚠️ Parallel track |

**Do not start with “all randomness from the archive.”** That confounds attribution (the exact mistake to avoid).

---

### 2.4 Task & model choices (keep small until pipeline is honest)

| Phase | Task | Model | Optimiser | Goal |
|-------|------|-------|-----------|------|
| Smoke | Fashion-MNIST | 2-layer MLP + dropout 0.5 | SGD | Prove injection works (biased vs urandom) |
| Core | Fashion-MNIST + one noisy tabular / subset CIFAR-10 | MLP / small CNN | **SGD and Adam** | Site × quality × optimiser |
| Scale | Full CIFAR-10, longer schedules | ResNet-20-class | SGD/Adam | Paper-grade effects |

**Why Fashion-MNIST first:** cheap, standard, hard enough that dropout matters; MNIST alone is often too saturated to see gaps (Bird et al.).

---

### 2.5 Statistical methodology (non-negotiable for claims)

| Item | Spec |
|------|------|
| Replicates | Pilot \(n=10\); confirmatory \(n\geq 25\) if \(\Delta < 2.5\%\) |
| Seeds for frozen sites | Predefined `{0,…,n−1}` only where a PRNG remains |
| Primary metrics | Test accuracy (or F1), train–test gap, optional NLL |
| Secondary | Learning-curve AUC, run-to-run IQR, prediction churn |
| Tests | Brunner–Munzel (default) or Mann–Whitney; Holm correction across sources |
| Effect size | Cliff’s \(\delta\) or stochastic superiority \(P(X<Y)+0.5P(X=Y)\) |
| Plots | KDE / ECDF of the \(n\) runs — not mean±error bar alone |
| Logging | source id, tier, site, optimiser, bytes consumed, git hash, hyperparams |

**Feasibility now:** ✅ All of this is scriptable in Python (`scipy` has Brunner–Munzel in recent versions).

---

### 2.6 Optimiser methodology (where the review adds novelty)

Literature gap G3: almost nobody crosses **RNG quality × optimiser**.

**Focused design (2×2 to start):**

- Sites: dropout **or** shuffle (one at a time)  
- Optimisers: SGD (fixed LR) vs Adam (fixed defaults / fixed LR)  
- Sources: biased (neg. control), weak LCG, urandom, one archive  

**What you can claim if done well:**

- Whether adaptive methods are more/less sensitive to shuffle-bitstream defects than SGD  
- Whether dropout-mask quality interacts with optimiser choice  

**Defer:** Langevin / SGLD, full hyperparameter sweeps, large-batch regimes — interesting but not needed for the first paper contribution.

---

## 3. What can be done *now* (capability cut)

### Ready immediately (no new deps beyond Python stdlib + optional numpy/torch)

| Work item | Status | Effort |
|-----------|--------|--------|
| Cheap Stage A stats on pilot `.bin` | Ready to implement | Small |
| `BitstreamRNG` + unit tests | Ready to implement | Small |
| Fashion-MNIST MLP dropout ablation (CPU) | Ready once RNG exists | Medium |
| Multi-run JSON logger + Brunner–Munzel table | Ready | Small |
| Negative-control validation (biased vs urandom) | Ready | Medium |

### Needs extra setup

| Work item | Needs |
|-----------|-------|
| NIST / Dieharder / TestU01 | System packages + longer bit files |
| CIFAR / GPU confirmatory | CUDA env, determinism flags |
| Neural distinguisher Stage A | Training budget + design choices |
| User large archives | Mount under `data/random_bits/large/`, register in `sources.yaml` |

### Explicitly out of scope for v1 methodology

- Claiming QRNG superiority without Stage A scores  
- Jointly randomising init+shuffle+dropout from the archive  
- BigCrush on 64 KiB pilots  
- Full metaheuristic suite before the NN pipeline is validated  

---

## 4. Recommended focus (decision)

**Primary methodology to own:**

> **Isolated-site bitstream substitution + quality tier covariates + powered non-parametric comparison**, starting with **dropout**, then **shuffle × {SGD, Adam}**.

That combination:

1. Fixes the main confound in prior work (P2)  
2. Uses your large binaries as a real asset (P1, scale)  
3. Connects to optimiser theory (shuffle → gradient noise)  
4. Is implementable with pilot files before archives are wired in  

**Secondary methodology (paper enrichment, not blocker):**

- Stage A neural distinguisher  
- Quasi-random vs pseudo-random shuffle schedules (Sobol index permutations)  
- PSO/DE quality-floor replication on a short benchmark  

---

## 5. Phased delivery (methodology → code)

### Phase M0 — Honesty checks (do first)

1. Implement `BitstreamRNG`  
2. Cheap stats → tier labels for pilots  
3. Dropout-only MLP: `biased_p60` vs `os_urandom`, \(n=10\)  
4. **Gate:** negative control must differ (accuracy, gap, or variance). If not, fix injection.

### Phase M1 — Quality ladder

5. Add `weak_lcg`, `mt19937`, archive shard  
6. Same dropout protocol, \(n=10\) then \(n=25\) if gaps small  
7. Report quality–utility plot (tier/score vs metric)

### Phase M2 — Optimiser × shuffle

8. Isolate shuffle site; SGD vs Adam  
9. Same sources; same stats protocol  

### Phase M3 — Breadth

10. Init-only ablation (Bird/Heese replication style)  
11. Second dataset (tabular or CIFAR subset)  
12. Optional distinguisher + NIST on large archives  

---

## 6. Validity checklist (use before every claim)

- [ ] Only one site reads the bitstream  
- [ ] Frozen sites use fixed seeds  
- [ ] Split is fixed across compared runs  
- [ ] Wraparound policy stated  
- [ ] Bytes consumed logged and sufficient for the schedule  
- [ ] Stage A scores attached to each source  
- [ ] \(n\) and test pre-specified  
- [ ] Multiplicity correction applied  
- [ ] Determinism mode documented (CPU / deterministic cuDNN)

---

## 7. Bottom line

**Focus methodology:** two-stage *characterise → isolated inject*, with dropout-first then shuffle×optimiser, and Brunner–Munzel multi-run statistics.

**What can be done next in this repo:** implement `BitstreamRNG` + cheap stats + Fashion-MNIST dropout negative-control ladder — that is the methodology-critical path; everything else builds on it.
