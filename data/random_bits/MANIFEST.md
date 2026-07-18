# Sample random bitstreams (pilot)

Each file is 64 KiB of raw bytes for local pilot experiments.
Replace / extend with your large binary random bit sources for full runs.

| file | bytes | ones_ratio | sha256 |
|---|---:|---:|---|
| `weak_lcg_64kib.bin` | 65536 | 0.500000 | `668146c7d1ea251b17c91b6915247639511446ebf1e0cfa5ed81fd9eb8a6dc52` |
| `xorshift32_64kib.bin` | 65536 | 0.501062 | `1175a06efa2ebcf6369b8626a09737043925a60214786f0b7b1570033ae125c6` |
| `mt19937_python_64kib.bin` | 65536 | 0.499956 | `322ff7f9a802adc75e114310b63e261b4922b39b1d1db306b5b2dd4972026721` |
| `os_urandom_64kib.bin` | 65536 | 0.500505 | `fc44f8a603347b3a05df765207decfcffb1c146b1507c8650b3e9309010bf7d1` |
| `biased_p60_64kib.bin` | 65536 | 0.600979 | `61165cde589d45ddbf9ec74141b8e669bf2057ca89320153cd96e52ec7abda5e` |

## Intended quality tiers

- `biased_p60_*`: deliberately non-uniform (negative control)
- `weak_lcg_*`: classic weak PRNG (expected to fail many statistical tests)
- `xorshift32_*`: lightweight PRNG (intermediate)
- `mt19937_python_*`: common ML-framework-style PRNG
- `os_urandom_*`: OS CSPRNG reference (high quality on typical hosts)
