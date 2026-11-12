#!/usr/bin/env bash
# Record the hardware/software environment of a benchmark session.
# Usage: bash scripts/env_info.sh > results/environment.txt
set -u
section() { printf '\n===== %s =====\n' "$1"; }
section "date";             date --iso-8601=seconds
section "os";               grep -E '^(NAME|VERSION)=' /etc/os-release; uname -srm
section "cpu";              lscpu | grep -E 'Model name|^CPU\(s\)|Thread|Core|Socket|MHz|L1d|L1i|L2|L3|NUMA'
section "hybrid cores";     echo "P-cores (cpu_core): $(cat /sys/devices/cpu_core/cpus 2>/dev/null)"; echo "E-cores (cpu_atom): $(cat /sys/devices/cpu_atom/cpus 2>/dev/null)"
section "frequency policy"; echo "governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor)"; echo "platform_profile: $(cat /sys/firmware/acpi/platform_profile)"; echo "no_turbo: $(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"; echo "AC online: $(cat /sys/class/power_supply/ADP1/online)"
section "memory";           free -h
section "load";             cat /proc/loadavg
section "gcc";              gcc --version | head -1; echo "CFLAGS: -O2 -std=c17 -Wall -Wextra -ffp-contract=off -fopenmp (BS=64)"; echo "libgomp: $(ls /usr/lib64/libgomp.so.* 2>/dev/null | tail -1)"
section "glibc";            ldd --version | head -1
section "python";           python3 --version
section "analysis venv";    .venv/bin/python -c 'import numpy, matplotlib, reportlab; print("numpy", numpy.__version__); print("matplotlib", matplotlib.__version__); print("reportlab", reportlab.Version)' 2>/dev/null
