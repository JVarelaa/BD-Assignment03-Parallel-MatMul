#!/usr/bin/env python3
"""Generate the shared dense input matrices (standard library only).

These are the Assignment 1 and 2 inputs, produced by the same code and seeds, so
the files are byte-identical (compare the SHA-256 values in data/MANIFEST.csv).
Format: raw little-endian float64, row-major, n*n values, no header.
Generator: Python's random.Random (Mersenne Twister, MT19937), values uniform in
[-1, 1), seeds A: 20261012 + n, B: 20261013 + n.

Usage: python3 scripts/generate_inputs.py 512 1024 2048
"""
import hashlib
import os
import random
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
LOW, HIGH = -1.0, 1.0


def dense_name(m, n):
    return f"{m}_{n}.bin"


def write(name, n, seed):
    rng = random.Random(seed)
    data = struct.pack("<%dd" % (n * n), *(LOW + (HIGH - LOW) * rng.random() for _ in range(n * n)))
    with open(os.path.join(DATA, name), "wb") as f:
        f.write(data)
    print(f"wrote {name} (seed={seed}, sha256={hashlib.sha256(data).hexdigest()})", file=sys.stderr)


def main():
    os.makedirs(DATA, exist_ok=True)
    for n in map(int, sys.argv[1:]):
        write(dense_name("A", n), n, 20261012 + n)
        write(dense_name("B", n), n, 20261013 + n)


if __name__ == "__main__":
    main()
