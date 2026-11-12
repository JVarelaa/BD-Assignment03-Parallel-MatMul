#!/usr/bin/env python3
"""Summary table and figure from the raw CSV files (requires numpy, matplotlib).

Inputs:  results/raw/timings.csv, results/raw/runs.csv
Outputs: results/summary/summary.csv, results/figures/fig1_scaling.png
Statistics: median and interquartile range (numpy percentile, linear interpolation)
of the measured samples only (warm-up excluded).
Speedup S(p) = median T_seq(n) / median T_p(n); efficiency E(p) = S(p) / p.
Usage: python scripts/analyze.py
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_inputs import ROOT  # noqa: E402

RAW, SUM, FIG = (os.path.join(ROOT, "results", d) for d in ("raw", "summary", "figures"))
INK, MUTED, GRID = "#1f1e1c", "#6b6a66", "#e4e2dc"
COLORS = {512: "#2a78d6", 1024: "#eb6834", 2048: "#1baf7a"}
MARKERS = {512: "o", 1024: "s", 2048: "^"}


def read(name):
    with open(os.path.join(RAW, name)) as f:
        return list(csv.DictReader(f))


def style_axes(ax):
    ax.grid(True, which="major", color=GRID, linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(colors=MUTED, labelcolor=INK)


def main():
    T, R = read("timings.csv"), read("runs.csv")
    os.makedirs(SUM, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)
    rows = []
    for r in R:
        n, p, m = int(r["n"]), int(r["threads"]), r["method"]
        v = np.array([float(t["time_s"]) for t in T if t["method"] == m and int(t["n"]) == n
                      and int(t["threads"]) == p and t["phase"] == "measured"])
        sums = {t["checksum"] for t in T if int(t["n"]) == n}
        assert len(sums) == 1, f"checksums differ for n={n}"          # every run gave the same C
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        rows.append(dict(method=m, n=n, threads=p, status=r["status"], reps=len(v), median=med, q1=q1, q3=q3,
                         iqr=q3 - q1, min=v.min(), max=v.max(), gflops=2 * n ** 3 / med / 1e9,
                         hwm_kernel_kb=int(r["hwm_kernel_kb"]), hwm_process_kb=int(r["hwm_process_kb"])))
    for row in rows:
        tseq = next(x["median"] for x in rows if x["method"] == "seq" and x["n"] == row["n"])
        row["speedup"] = tseq / row["median"]
        row["efficiency"] = row["speedup"] / row["threads"]
    cols = ["method", "n", "threads", "status", "reps", "median", "q1", "q3", "iqr", "min", "max", "gflops",
            "speedup", "efficiency", "hwm_kernel_kb", "hwm_process_kb"]
    with open(os.path.join(SUM, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for row in rows:
            w.writerow([f"{row[c]:.6f}" if isinstance(row[c], float) else row[c] for c in cols])

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    for n in sorted({r["n"] for r in rows}):
        par = sorted((r for r in rows if r["method"] == "par" and r["n"] == n), key=lambda r: r["threads"])
        tseq = next(r["median"] for r in rows if r["method"] == "seq" and r["n"] == n)
        ps = [r["threads"] for r in par]
        s = [r["speedup"] for r in par]
        lo = [tseq / r["q3"] for r in par]
        hi = [tseq / r["q1"] for r in par]
        kw = dict(color=COLORS[n], marker=MARKERS[n], label=f"n = {n}", linewidth=1.6, markersize=5)
        a.errorbar(ps, s, yerr=[np.subtract(s, lo), np.subtract(hi, s)], capsize=2, **kw)
        b.plot(ps, [r["efficiency"] for r in par], **kw)
    ps = [1, 2, 4, 6, 8, 10, 12]
    a.plot(ps, ps, "--", color=MUTED, linewidth=1, label="ideal S = p")
    for ax in (a, b):
        ax.axvspan(10.5, 12.5, color=GRID, alpha=0.6, linewidth=0)
        ax.set_xticks(ps)
        ax.set_xlabel("threads p")
        style_axes(ax)
    a.set_ylabel("speedup S(p) = T_seq / T_p")
    a.set_title("(a) Speedup (median; bars from IQR of T_p)", fontsize=10, color=INK)
    b.set_ylabel("efficiency E(p) = S(p) / p")
    b.set_ylim(0, 1.15)
    b.set_title("(b) Efficiency", fontsize=10, color=INK)
    a.legend(frameon=False, fontsize=8)
    for ax in (a, b):
        ax.text(1.5, ax.get_ylim()[1] * 0.97, "2 P", fontsize=7, color=MUTED, ha="center", va="top")
        ax.text(6.5, ax.get_ylim()[1] * 0.97, "+ 8 E-cores", fontsize=7, color=MUTED, ha="center", va="top")
        ax.text(11.5, ax.get_ylim()[1] * 0.97, "+ HT", fontsize=7, color=MUTED, ha="center", va="top")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig1_scaling.png"), dpi=200)
    for r in rows:
        print(f"{r['method']} n={r['n']:5d} p={r['threads']:2d} med={r['median']:.4f} iqr={r['iqr'] / r['median'] * 100:4.1f}% "
              f"S={r['speedup']:.2f} E={r['efficiency']:.2f} GF={r['gflops']:.1f} rss={r['hwm_kernel_kb'] / 1024:.1f}MiB")


if __name__ == "__main__":
    main()
