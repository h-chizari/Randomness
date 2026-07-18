#!/usr/bin/env python3
"""Generate small pilot random bitstreams of graded statistical quality."""

from __future__ import annotations

import argparse
import hashlib
import os
import random
from pathlib import Path


def lcg_bytes(n: int, seed: int = 1) -> bytes:
    a, c, m = 1664525, 1013904223, 2**32
    x = seed
    buf = bytearray()
    while len(buf) < n:
        x = (a * x + c) % m
        buf.append(x & 0xFF)
    return bytes(buf[:n])


def xorshift32_bytes(n: int, seed: int = 0xDEADBEEF) -> bytes:
    x = seed & 0xFFFFFFFF
    buf = bytearray()
    while len(buf) < n:
        x ^= (x << 13) & 0xFFFFFFFF
        x ^= (x >> 17) & 0xFFFFFFFF
        x ^= (x << 5) & 0xFFFFFFFF
        buf.append(x & 0xFF)
    return bytes(buf[:n])


def mt_bytes(n: int, seed: int = 42) -> bytes:
    r = random.Random(seed)
    return bytes(r.getrandbits(8) for _ in range(n))


def biased_bytes(n: int, p: float = 0.6, seed: int = 7) -> bytes:
    r = random.Random(seed)
    outb = bytearray()
    for _ in range(n):
        b = 0
        for _ in range(8):
            b = (b << 1) | (1 if r.random() < p else 0)
        outb.append(b)
    return bytes(outb)


def ones_ratio(data: bytes) -> float:
    ones = sum(bin(b).count("1") for b in data)
    return ones / (len(data) * 8)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "random_bits",
    )
    parser.add_argument("--nbytes", type=int, default=64 * 1024)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        f"weak_lcg_{args.nbytes // 1024}kib.bin": lcg_bytes(args.nbytes, seed=1),
        f"xorshift32_{args.nbytes // 1024}kib.bin": xorshift32_bytes(
            args.nbytes, seed=0xDEADBEEF
        ),
        f"mt19937_python_{args.nbytes // 1024}kib.bin": mt_bytes(args.nbytes, seed=42),
        f"os_urandom_{args.nbytes // 1024}kib.bin": os.urandom(args.nbytes),
        f"biased_p60_{args.nbytes // 1024}kib.bin": biased_bytes(
            args.nbytes, p=0.6, seed=7
        ),
    }

    lines = [
        "# Sample random bitstreams (pilot)",
        "",
        f"Each file is {args.nbytes} bytes of raw bits for local pilot experiments.",
        "Replace / extend with your large binary random bit sources for full runs.",
        "",
        "| file | bytes | ones_ratio | sha256 |",
        "|---|---:|---:|---|",
    ]
    for name, data in sources.items():
        path = args.out_dir / name
        path.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        ratio = ones_ratio(data)
        lines.append(f"| `{name}` | {len(data)} | {ratio:.6f} | `{digest}` |")
        print(f"wrote {path} ones_ratio={ratio:.4f}")

    lines.extend(
        [
            "",
            "## Intended quality tiers",
            "",
            "- `biased_p60_*`: deliberately non-uniform (negative control)",
            "- `weak_lcg_*`: classic weak PRNG (expected to fail many statistical tests)",
            "- `xorshift32_*`: lightweight PRNG (intermediate)",
            "- `mt19937_python_*`: common ML-framework-style PRNG",
            "- `os_urandom_*`: OS CSPRNG reference (high quality on typical hosts)",
            "",
        ]
    )
    (args.out_dir / "MANIFEST.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
