#!/usr/bin/env python3
"""Correctness check against NumPy (trusted reference) on the shared inputs.

For n in SIZES, seq and par (every thread count of the benchmark) are compared with
NumPy's A @ B using the forward error bound of a length-n dot product (as in
Assignments 1-2):  |C_ij - Cref_ij| <= 2 * gamma_n * (|A||B|)_ij,
gamma_n = n u / (1 - n u), u = 2^-53.  Every par result must also be bitwise
identical to seq (same tile code and k order, no FMA). par is run 3 times per
thread count, because a data race would make results differ between runs.
Writes results/raw/correctness.csv. Usage: python scripts/verify.py
"""
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_inputs import DATA, ROOT, dense_name  # noqa: E402
from run_benchmarks import BIN, PLACES, THREADS  # noqa: E402

U = 2.0 ** -53
SIZES = [512, 1024, 2048]
ENV = dict(os.environ, OMP_DYNAMIC="false", OMP_PROC_BIND="close", OMP_PLACES=PLACES)


def main():
    rows, ok = [], True
    with tempfile.TemporaryDirectory() as tmp:
        for n in SIZES:
            fa, fb = os.path.join(DATA, dense_name("A", n)), os.path.join(DATA, dense_name("B", n))
            A = np.fromfile(fa, dtype="<f8").reshape(n, n)
            B = np.fromfile(fb, dtype="<f8").reshape(n, n)
            ref = A @ B
            bound = 2 * (n * U / (1 - n * U)) * (np.abs(A) @ np.abs(B))
            seq = None
            for method, p, run in [("seq", 1, 0)] + [("par", p, r) for p in THREADS for r in range(3)]:
                out = os.path.join(tmp, "C.bin")
                subprocess.run([BIN, method, str(n), fa, fb, str(p), "0", "1", out], check=True,
                               capture_output=True, env=ENV)
                C = np.fromfile(out, dtype="<f8").reshape(n, n)
                err = np.abs(C - ref)
                within = bool(np.all(err <= bound))
                if seq is None:
                    seq = C
                same = bool(np.array_equal(C.view(np.uint64), seq.view(np.uint64)))
                ok &= within and same
                rows.append([n, method, p, run, f"{err.max():.3e}", f"{(err / bound).max():.3e}", within, same])
            print(f"n={n}: {'OK' if ok else 'FAIL'}")
    with open(os.path.join(ROOT, "results", "raw", "correctness.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["n", "method", "threads", "run", "max_abs_err_vs_numpy", "max_err_over_bound",
                    "within_bound", "bitwise_equal_to_seq"])
        w.writerows(rows)
    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
