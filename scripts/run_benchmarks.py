#!/usr/bin/env python3
"""Benchmark harness for Assignment 3 (standard library only).

Every configuration runs as a separate process of build/matmul3:
  seq        sequential baseline, pinned to logical CPU 0 (a P-core) with taskset
  par, p     OpenMP kernel with p threads, p in THREADS; threads are bound to the
             places in PLACES (OMP_PROC_BIND=close): first one context per physical
             core (P-cores 0 and 2, then E-cores 4-11), then the hyper-thread
             siblings 1 and 3. So p = 1 runs on CPU 0 like seq, p <= 10 uses one
             context per physical core and p = 12 fills all logical CPUs.
OMP_DYNAMIC=false, no numerical library is linked, so p is the real thread count.
All samples are appended to results/raw/timings.csv (one row per timed run) and
results/raw/runs.csv (one row per process). Budgets: 300 s per configuration and
1.5 GB peak RSS. Measurement is refused while the CPU frequency is capped.
Usage: python3 scripts/run_benchmarks.py      (resumable: done configurations are skipped)
"""
import csv
import datetime
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_inputs import DATA, ROOT, dense_name  # noqa: E402

RAW = os.path.join(ROOT, "results", "raw")
BIN = os.path.join(ROOT, "build", "matmul3")
SEQ_CPU = "0"
PLACES = "{0},{2},{4},{5},{6},{7},{8},{9},{10},{11},{1},{3}"
SIZES = [512, 1024, 2048]
THREADS = [1, 2, 4, 6, 8, 10, 12]
WARMUP = 1
CONFIG_TIMEOUT_S = 300
MEMORY_BUDGET_KB = 1_500_000

TIMINGS_HEADER = ["method", "n", "threads", "phase", "rep", "time_s", "checksum"]
RUNS_HEADER = ["method", "n", "threads", "warmup", "reps", "status", "exit_code", "wall_s",
               "omp_max_threads", "hwm_process_kb", "hwm_kernel_kb", "loadavg_1m", "started"]


def reps_for(n):
    return 10 if n <= 512 else 5


def append(name, header, rows):
    path = os.path.join(RAW, name)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerows(rows)


def read_sys(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return ""


def check_power():
    cpufreq = "/sys/devices/system/cpu/cpu0/cpufreq/"
    state = dict(ac=read_sys("/sys/class/power_supply/ADP1/online"),
                 profile=read_sys("/sys/firmware/acpi/platform_profile"),
                 no_turbo=read_sys("/sys/devices/system/cpu/intel_pstate/no_turbo"),
                 max_khz=read_sys(cpufreq + "scaling_max_freq"),
                 hw_max_khz=read_sys(cpufreq + "cpuinfo_max_freq"))
    if state["ac"] != "1" or state["no_turbo"] == "1" or state["max_khz"] != state["hw_max_khz"] \
            or state["profile"] == "low-power":
        sys.exit(f"ABORT: CPU frequency is capped {state}; plug in the charger and rerun.")


def done():
    path = os.path.join(RAW, "runs.csv")
    if not os.path.exists(path):
        return set()
    with open(path) as f:
        return {(r["method"], r["n"], r["threads"]) for r in csv.DictReader(f)}


def run(method, n, p):
    check_power()
    reps = reps_for(n)
    args = [BIN, method, str(n), os.path.join(DATA, dense_name("A", n)), os.path.join(DATA, dense_name("B", n)),
            str(p), str(WARMUP), str(reps)]
    env = dict(os.environ, OMP_DYNAMIC="false")
    if method == "seq":                               # no parallel region: taskset pins the process
        cmd = ["taskset", "-c", SEQ_CPU] + args
    else:
        env.update(OMP_PROC_BIND="close", OMP_PLACES=PLACES)
        cmd = args
    started = datetime.datetime.now().isoformat(timespec="seconds")
    load1 = os.getloadavg()[0]
    time.sleep(2)                                     # short pause between processes
    t0 = time.monotonic()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, env=env)
    try:
        stdout, _ = proc.communicate(timeout=CONFIG_TIMEOUT_S)
        status = "ok" if proc.returncode == 0 else "error"
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, _ = proc.communicate()
        status = "timeout"
    wall = time.monotonic() - t0
    samples, meta = [], ["", "", ""]
    for line in stdout.splitlines():
        f = line.split(",")
        if f[0] == "meta":
            meta = f[3:]
        else:
            samples.append(f)
    if meta[1] and int(meta[1]) > MEMORY_BUDGET_KB:
        status = "over_memory_budget"
    append("timings.csv", TIMINGS_HEADER, samples)
    append("runs.csv", RUNS_HEADER, [[method, n, p, WARMUP, reps, status, proc.returncode, f"{wall:.3f}"]
                                     + meta + [f"{load1:.2f}", started]])
    print(f"{method} n={n:5d} p={p:2d} {status} wall={wall:6.1f}s", flush=True)


def main():
    os.makedirs(RAW, exist_ok=True)
    subprocess.run(["make", "-C", ROOT, "-s", "all"], check=True)
    finished = done()
    for n in SIZES:
        if not os.path.exists(os.path.join(DATA, dense_name("B", n))):
            subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "generate_inputs.py"), str(n)], check=True)
        for method, p in [("seq", 1)] + [("par", p) for p in THREADS]:
            if (method, str(n), str(p)) not in finished:
                run(method, n, p)


if __name__ == "__main__":
    main()
