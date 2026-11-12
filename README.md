# Assignment 3 — Parallel Matrix Multiplication (OpenMP)

Big Data, Grado en Ciencia e Ingeniería de Datos, ULPGC — academic year 2026-2027.

**Author:** Jesus Varela Vasquez

**Research question.** How does the number of threads affect the run time, speedup and
efficiency of a shared-memory (OpenMP) blocked matrix multiplication, and when does adding
threads help?

The report is in [`report/Assignment03_Report.pdf`](report/Assignment03_Report.pdf).

## Reused from Assignment 2

* Kernel: the Assignment 2 `blocked` kernel (BS = 64, i-k-j inside the tiles), with the j and k
  tile loops swapped so that each 64 x 64 tile of C is an independent task.
* Inputs: the Assignment 1/2 dense files `A_n.bin`, `B_n.bin` for n = 512, 1024, 2048
  (same generator and seeds; SHA-256 in `data/MANIFEST.csv`).
* Same machine, compiler, flags (+ `-fopenmp`), timing method and memory metric.

## Layout

```
c/matmul3.c                 seq (sequential baseline) and par (OpenMP) in one program
scripts/generate_inputs.py  regenerates the inputs (stdlib only, seeded)
scripts/verify.py           correctness vs NumPy + bitwise seq == par for every thread count
scripts/run_benchmarks.py   benchmark harness (stdlib only): affinity, budgets, raw CSV
scripts/analyze.py          summary table and figure
scripts/make_report.py      builds the PDF report
scripts/env_info.sh         records the environment
results/raw/                timings.csv (every sample), runs.csv (one row per process),
                            correctness.csv, supporting_checks.csv (seq on an E-core; seq/par(1) alternated)
results/summary/summary.csv median, IQR, speedup, efficiency, GFLOP/s, peak RSS per configuration
results/figures/            figure used in the report
results/affinity_check.txt  thread -> CPU binding reported by OMP_DISPLAY_AFFINITY
results/environment.txt     environment of the benchmark session
```

Program interface:

```
build/matmul3 <seq|par> <n> <A.bin> <B.bin> <threads> <warmup> <reps> [C_out.bin]
build/matmul3 --selftest
stdout CSV: method,n,threads,phase,rep,time_s,checksum      (phase: warmup | measured)
            meta,n,threads,omp_max_threads,hwm_process_kb,hwm_kernel_kb
```

## Requirements (versions used)

| Component | Version |
|---|---|
| OS | Fedora Linux 44, kernel 7.1.7, glibc 2.43 |
| CPU | Intel Core i5-1335U: 2 P-cores x 2 HT (CPUs 0-3) + 8 E-cores (CPUs 4-11); 10 cores, 12 logical |
| C compiler | GCC 16.2.1 + libgomp, `-O2 -std=c17 -Wall -Wextra -ffp-contract=off -fopenmp`, `BS=64` |
| Python | CPython 3.14.6 (inputs and harness: standard library only) |
| Analysis | numpy 2.5.4, matplotlib 3.11.2, reportlab 5.0.1 (`requirements.txt`) |

## Reproducing everything

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
make test                                        # self-test: seq and par (p = 1..16) on known products
python3 scripts/generate_inputs.py 512 1024 2048
.venv/bin/python scripts/verify.py               # NumPy reference + bitwise checks
bash scripts/env_info.sh > results/environment.txt
rm -f results/raw/timings.csv results/raw/runs.csv
python3 scripts/run_benchmarks.py                # ≈ 2 min; needs AC power
# supporting checks (results/raw/supporting_checks.csv), see the report section 5:
#   taskset -c 4 build/matmul3 seq <n> data/A_<n>.bin data/B_<n>.bin 1 1 5        (n = 1024, 2048)
#   3 rounds of: OMP_PLACES="{0}" OMP_PROC_BIND=close taskset -c 0 build/matmul3 <seq|par> 2048 ... 1 1 3
.venv/bin/python scripts/analyze.py
.venv/bin/python scripts/make_report.py "Jesus Varela Vasquez" "https://github.com/JVarelaa/BD-Assignment03-Parallel-MatMul"
```

## Thread configurations

* **Thread counts:** p = 1, 2, 4, 6, 8, 10, 12 (no oversubscription: 12 logical CPUs).
* **Affinity (par):** `OMP_PROC_BIND=close`, `OMP_PLACES={0},{2},{4},{5},{6},{7},{8},{9},{10},{11},{1},{3}`,
  `OMP_DYNAMIC=false`: p = 1 → P-core 0, p = 2 → both P-cores, p = 3-10 → + E-cores,
  p = 11-12 → + hyper-thread siblings of the P-cores.
* **seq:** pinned to CPU 0 with `taskset -c 0`; no OpenMP region is entered.
* **Scheduling:** `collapse(2) schedule(dynamic, 1)` over the 64 x 64 tiles of C.
* No numerical library is linked, so there are no hidden internal threads.

## Fixed experimental settings

* **Numeric type:** float64, no FMA contraction; par is bitwise identical to seq for every p.
* **Sizes:** n = 512, 1024, 2048.
* **Repetitions:** 1 warm-up (excluded, also creates the OpenMP thread pool) + 10 measured runs
  for n = 512, 5 for n ≥ 1024.
* **Timer:** `clock_gettime(CLOCK_MONOTONIC)` around the kernel only (zeroing of C included).
* **Statistics:** median and IQR; S(p) = median T_seq / median T_p, E(p) = S(p) / p.
* **Memory:** peak RSS (`VmHWM`) of the kernel phase, reset through `/proc/self/clear_refs`.
* **Budgets:** 300 s per configuration; 1.5 GB peak RSS.

## Use of generative AI

Claude Code (Anthropic) was used to help write the code, scripts and the first draft of
the report. All measurements come from actual runs on the machine described in
`results/environment.txt`. The author is responsible for checking and defending the code,
the results and the conclusions.
