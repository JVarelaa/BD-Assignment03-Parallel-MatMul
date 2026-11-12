#!/usr/bin/env python3
"""Build report/Assignment03_Report.pdf from the summary CSV and figure.

Numbers quoted in the text are read from results/summary/summary.csv,
results/raw/correctness.csv and results/raw/supporting_checks.csv.
Usage: .venv/bin/python scripts/make_report.py ["Author Name"] ["repository URL"]
"""
import csv
import os
import statistics
import sys
from datetime import date

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUM, FIG, RAW = (os.path.join(ROOT, "results", d) for d in ("summary", "figures", "raw"))
OUT = os.path.join(ROOT, "report", "Assignment03_Report.pdf")
AUTHOR = sys.argv[1] if len(sys.argv) > 1 else "[Student name]"
REPO = sys.argv[2] if len(sys.argv) > 2 else ""


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


S = load(os.path.join(SUM, "summary.csv"))
CK = load(os.path.join(RAW, "correctness.csv"))
SC = load(os.path.join(RAW, "supporting_checks.csv"))


def v(method, n, p, col="median"):
    return float(next(r[col] for r in S if r["method"] == method and int(r["n"]) == n and int(r["threads"]) == p))


def sc(check, method, n):
    return statistics.median(float(r["time_s"]) for r in SC if r["check"].startswith(check)
                             and r["method"] == method and int(r["n"]) == n)


SIZES = sorted({int(r["n"]) for r in S})
THREADS = sorted({int(r["threads"]) for r in S if r["method"] == "par"})
sp = lambda n, p: v("par", n, p, "speedup")
ef = lambda n, p: v("par", n, p, "efficiency")
iqr_pct = lambda m, n, p: v(m, n, p, "iqr") / v(m, n, p) * 100
MiB = lambda kb: float(kb) / 1024

# ------------------------------------------------------------------ styles (as Assignment 2)
ss = getSampleStyleSheet()
BODY = ParagraphStyle("body", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.6, leading=13.2,
                      alignment=TA_JUSTIFY, spaceAfter=5)
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=13, spaceBefore=10,
                    spaceAfter=5)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=10.5, spaceBefore=6,
                    spaceAfter=3)
CAP = ParagraphStyle("cap", parent=BODY, fontSize=8.4, leading=11, textColor=colors.HexColor("#52514e"),
                     spaceAfter=9)
TITLE = ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold", fontSize=16, leading=20,
                       spaceAfter=4)
META = ParagraphStyle("meta", parent=BODY, alignment=1, fontSize=9.5, textColor=colors.HexColor("#52514e"))
HDR = ParagraphStyle("hdr", parent=ss["BodyText"], fontName="Helvetica-Bold", fontSize=7.6, leading=9,
                     alignment=2)
ABS = ParagraphStyle("abs", parent=BODY, leftIndent=0.8 * cm, rightIndent=0.8 * cm, fontSize=9.2, leading=12.6)

story = []
P = lambda text, style=BODY: story.append(Paragraph(text, style))
code = lambda t: f"<font face='Courier'>{t}</font>"


def table(data, widths, caption):
    data = [[Paragraph(str(c), HDR) for c in data[0]]] + data[1:]
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8),
        ("FONT", (0, 1), (-1, -1), "Helvetica", 8),
        ("LINEABOVE", (0, 0), (-1, 0), 0.8, colors.black),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
        ("LINEBELOW", (0, -1), (-1, -1), 0.8, colors.black),
        ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 1.6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f3f0")]),
    ]))
    story.append(KeepTogether([t, Spacer(1, 3), Paragraph(caption, CAP)]))


def figure(name, width_cm, caption):
    path = os.path.join(FIG, name)
    iw, ih = ImageReader(path).getSize()
    w = width_cm * cm
    story.append(KeepTogether([Image(path, width=w, height=w * ih / iw), Paragraph(caption, CAP)]))


# ------------------------------------------------------------------ numbers used in the text
pmax = max(THREADS)
best = {n: max(THREADS, key=lambda p: sp(n, p)) for n in SIZES}
p1_gain = [(1 - v("par", n, 1) / v("seq", n, 1)) * 100 for n in SIZES]
ecore = sc("seq_on_ecore", "seq", 2048)
ratio_e = v("seq", 2048, 1) / ecore            # E-core speed relative to the P-core
ratio_e1024 = v("seq", 1024, 1) / sc("seq_on_ecore", "seq", 1024)
cap = lambda p: min(p, 2) + max(0, min(p, 10) - 2) * ratio_e     # P-core equivalents (one thread per core)
alt_seq, alt_par = sc("alternating", "seq", 2048), sc("alternating", "par", 2048)
max_bound = max(float(r["max_err_over_bound"]) for r in CK)
n_checks = len(CK)
max_iqr = max(iqr_pct(r["method"], int(r["n"]), int(r["threads"])) for r in S)
ht_gain = {n: (1 - v("par", n, 12) / v("par", n, 10)) * 100 for n in SIZES}
rss = {n: MiB(v("seq", n, 1, "hwm_kernel_kb")) for n in SIZES}
rss_spread = max(max(MiB(v("par", n, p, "hwm_kernel_kb")) for p in THREADS) - rss[n] for n in SIZES)

# ------------------------------------------------------------------ title + abstract
P("Thread scaling of a blocked matrix multiplication with OpenMP on a hybrid (P-core / E-core) laptop CPU",
  TITLE)
P(f"{AUTHOR} &nbsp;·&nbsp; Big Data, Individual Assignment 3 &nbsp;·&nbsp; Grado en Ciencia e Ingeniería de "
  f"Datos, ULPGC &nbsp;·&nbsp; {date.today().strftime('%B %Y')}", META)
if REPO:
    P(f"Code and data: <link href='{REPO}' color='#2a5db0'>{REPO}</link>", META)
story.append(Spacer(1, 6))
P("<b>Abstract.</b> We ask how the number of threads affects the run time of a shared-memory matrix "
  "multiplication and when parallel execution helps. The cache-blocked float64 kernel of Assignment 2 was "
  "parallelised with OpenMP by distributing independent 64 × 64 output tiles over threads. It was compared "
  f"with the same tile code run sequentially, for n = {', '.join(map(str, SIZES))} and "
  f"p = {', '.join(map(str, THREADS))} threads on an Intel Core i5-1335U (2 P-cores with hyper-threading and "
  "8 E-cores). Results were bitwise identical to the sequential baseline for every thread count. Two threads "
  f"on the two P-cores gave a speedup of {min(sp(n, 2) for n in SIZES):.2f}-{max(sp(n, 2) for n in SIZES):.2f} "
  f"(efficiency ≥ {min(ef(n, 2) for n in SIZES):.2f}). Adding E-cores kept reducing the time, but efficiency "
  f"fell to {min(ef(n, 10) for n in SIZES):.2f}-{max(ef(n, 10) for n in SIZES):.2f} at 10 threads, partly "
  f"because one E-core ran this kernel at only {ratio_e:.2f}× the speed of a P-core. The best speedup was "
  f"{sp(1024, pmax):.1f} (n = 1024, 12 threads). The parallel version with one thread was not slower than the "
  "sequential one, so the OpenMP overhead was not measurable at these sizes. The conclusions are limited to "
  "this kernel, these sizes and one laptop.", ABS)

# ------------------------------------------------------------------ 1 introduction
P("1. Introduction and objective", H1)
P("Assignment 2 showed that cache blocking gives a sequential C kernel a throughput of about 5.7 GFLOP/s that "
  "does not depend on n. This study keeps that kernel and asks: <b>how does the number of threads p affect "
  "the run time, speedup and efficiency of a shared-memory parallel version, and when does adding threads "
  "help?</b> The factor changed is p (and n as a secondary factor); the outcomes are the kernel time, "
  "speedup, efficiency and peak memory.")
P("<b>Hypotheses</b> (stated before measuring). H1: with one thread the parallel version will be about as "
  "fast as the sequential one, because the only extra work is creating the parallel region. H2: speedup will "
  "be close to p on the two P-cores, but efficiency will fall when E-cores are added, because they are slower "
  "and S(p) is measured against a P-core. H3: the two hyper-threads (p = 11-12) will add little, since they "
  "share a core with a thread that already runs. H4: memory will not depend on p, because all threads share "
  "the same three matrices.")

# ------------------------------------------------------------------ 2 background
P("2. Background", H1)
P("In C = AB every output element is independent, so the work can be split without synchronisation if each "
  "thread writes a disjoint part of C. <b>OpenMP</b> is a compiler-directive API for shared-memory "
  f"parallelism: {code('#pragma omp parallel for')} splits the iterations of a loop among a team of threads, "
  "with a choice of static or dynamic scheduling. <b>Speedup</b> S(p) = T<sub>seq</sub>(n) / T<sub>p</sub>(n) "
  "and <b>efficiency</b> E(p) = S(p) / p measure how well p threads are used. Amdahl's law limits S(p) when "
  "part of the work is serial. Here the serial part is very small, so the main limits are hardware: shared "
  "caches and memory bandwidth, the turbo frequency, which drops when more cores are active, and, on hybrid "
  "CPUs, cores of different speed.")

# ------------------------------------------------------------------ 3 methodology
P("3. Methodology", H1)
P("3.1 Implementations", H2)
P(f"Both methods are in one C program ({code('c/matmul3.c')}) and run the same function {code('tile()')}, "
  "which computes one 64 × 64 block of C: it zeroes the block and adds A[i][k]·B[k][j] over the k blocks in "
  "increasing order, with the i-k-j loop order inside each block (the inner code of the Assignment 2 blocked "
  f"kernel). <b>seq</b> calls {code('tile()')} for every block in row-major order without OpenMP. <b>par</b> "
  f"runs the same two loops under {code('#pragma omp parallel for collapse(2) schedule(dynamic, 1)')}, so "
  "each block is a task taken by the next free thread (64, 256 and 1024 tasks for n = 512, 1024, 2048). "
  "Dynamic scheduling was chosen because the cores have different speeds; a static split would make the "
  "P-cores wait for the E-cores. Each task writes only its own block of C and only reads A and B, so there "
  "are no shared writes, locks or reductions. The only difference from the Assignment 2 kernel is the order "
  "of the j and k tile loops, which makes each output block an independent task. The sequential baseline ran "
  f"at {v('seq', 2048, 1):.2f} s for n = 2048, against 3.00 s for the Assignment 2 blocked kernel. GCC 16.2.1 "
  f"was used with {code('-O2 -std=c17 -ffp-contract=off -fopenmp')} (libgomp), the flags of Assignments 1-2 "
  "plus OpenMP. GCC's vectorisation report shows that the kernel is not vectorised (scalar code), and the "
  "optional vectorisation extension was not done.")
P("3.2 Inputs, environment and thread configuration", H2)
P("The inputs are the Assignment 1-2 files for n = 512, 1024 and 2048: float64, uniform in [−1, 1), "
  "Mersenne Twister (Python random.Random) with seeds 20261012 + n (A) and 20261013 + n (B), byte-identical "
  "to the earlier assignments (SHA-256 in data/MANIFEST.csv). The machine is an Intel Core i5-1335U laptop: "
  "2 P-cores with 2 hardware threads each (logical CPUs 0-3, up to 4.6 GHz, 48 KiB L1d and 1.25 MiB L2 each) "
  "and 8 E-cores without hyper-threading (CPUs 4-11, up to 3.4 GHz, 2 MiB L2 per group of 4), with a shared "
  "12 MiB L3. That gives 10 physical cores and 12 logical processors. It has 7.4 GiB RAM and runs Fedora 44 "
  "(kernel 7.1.7), on AC power with the performance profile.")
P(f"Thread counts: p = {', '.join(map(str, THREADS))}, all ≤ 12 logical processors, so there is no "
  f"oversubscription. Affinity: {code('OMP_PROC_BIND=close')} with {code('OMP_PLACES={0},{2},{4},...,{11},{1},{3}')}, "
  "so thread t is bound to the t-th CPU in that list. p = 1 uses P-core CPU 0, p = 2 both P-cores, "
  "p = 3-10 add E-cores, and p = 11-12 add the hyper-thread siblings of the P-cores. The binding was checked "
  f"with {code('OMP_DISPLAY_AFFINITY')} (results/affinity_check.txt). seq is pinned to CPU 0 with "
  f"{code('taskset')}. {code('OMP_DYNAMIC=false')}, and no numerical library is linked, so p is the actual "
  "number of computing threads.")
P("3.3 Measurement protocol", H2)
P("Each configuration runs as a separate process: 1 warm-up run (excluded) and 10 measured runs for n = 512 or "
  f"5 for n ≥ 1024. The timer is {code('clock_gettime(CLOCK_MONOTONIC)')} around the kernel only. Inputs are "
  "read and C is allocated and touched before timing. Zeroing C is inside the timed region, as in "
  "Assignment 2. The warm-up run creates the OpenMP thread pool, so the measured parallel times include the "
  "fork/join and scheduling cost of every multiplication but not thread creation. A checksum of C is computed "
  "after every run, outside the timer. We report the median and IQR, and S(p) and E(p) are computed from the "
  "<b>medians</b>. <b>Memory:</b> peak RSS (VmHWM) of the kernel phase, reset with "
  f"{code('/proc/self/clear_refs')} after the inputs are ready, as in Assignment 2. <b>Budgets:</b> 300 s per "
  "configuration and 1.5 GB peak RSS. All configurations completed within both, so the largest tested size "
  "is n = 2048 for every p.")
P("3.4 Correctness and safe access to shared data", H2)
P("A self-test checks seq and par with p = 1, 2, 3, 4, 7, 12 and 16 on hand-computed 2 × 2 and 3 × 3 "
  "products, A·I = A, I·A = A, A·0 = 0 and 0·B = 0, and on a random 200 × 200 product that must equal a plain "
  "triple loop bitwise (200 is not a multiple of 64, so partial tiles are tested, and C is filled with "
  "garbage first). The verification script compares seq and par for every benchmarked p (par three times "
  "each) with NumPy's A @ B for all three sizes. The tolerance is the forward error bound of a length-n dot "
  "product, |C − C<sub>ref</sub>| ≤ 2γ<sub>n</sub>(|A||B|), γ<sub>n</sub> = nu/(1 − nu), "
  f"u = 2<super>−53</super>, because NumPy's BLAS sums in a different order. All {n_checks} checks passed "
  f"(maximum error {max_bound:.1e} of the bound), and every parallel result was bitwise identical to seq. "
  "This is expected, because each element is summed in the same k order by exactly one thread. A data race "
  "would show up as run-to-run differences. The checksums of all benchmark runs for a given n were also identical.")

# ------------------------------------------------------------------ 4 results
P("4. Results", H1)
hdr = ["config"]
for n in SIZES:
    hdr += [f"n = {n}: median s (IQR %)", "S", "E"]
rows = [hdr]
for m, p in [("seq", 1)] + [("par", p) for p in THREADS]:
    row = ["seq" if m == "seq" else f"par p = {p}"]
    for n in SIZES:
        row += [f"{v(m, n, p):.4g} ({iqr_pct(m, n, p):.1f})", f"{v(m, n, p, 'speedup'):.2f}",
                f"{v(m, n, p, 'efficiency'):.2f}"]
    rows.append(row)
w = [1.9 * cm] + [2.6 * cm, 0.95 * cm, 0.95 * cm] * 3
table(rows, w, "<b>Table 1.</b> Kernel time per multiplication: median in seconds and IQR as a percentage of "
      "the median (10 runs for n = 512, 5 for n ≥ 1024, after 1 warm-up). S = median T<sub>seq</sub> / median "
      "T<sub>p</sub>; E = S / p.")
figure("fig1_scaling.png", 16.5, "<b>Figure 1.</b> (a) Speedup against p. The error bars use the IQR of "
       "T<sub>p</sub> (T<sub>seq</sub> / Q3 to T<sub>seq</sub> / Q1), and the dashed line is ideal S = p. "
       "(b) Efficiency. p = 1-2 run on P-cores, p = 3-10 add E-cores, and the shaded band (p = 11-12) adds the "
       "hyper-thread siblings of the P-cores.")
P(f"The time fell steadily with p at every size (Table 1, Figure 1). With p = 2 the speedup was "
  f"{min(sp(n, 2) for n in SIZES):.2f}-{max(sp(n, 2) for n in SIZES):.2f}. Adding the first two E-cores "
  f"(p = 4) raised it only to {min(sp(n, 4) for n in SIZES):.2f}-{max(sp(n, 4) for n in SIZES):.2f}, and "
  f"efficiency fell to {min(ef(n, 4) for n in SIZES):.2f}-{max(ef(n, 4) for n in SIZES):.2f}. From there to "
  f"p = 10 the speedup rose by roughly 0.3-0.4 per added E-core for n = 512 and 1024. For n = 2048 it "
  f"flattened between p = 8 ({sp(2048, 8):.2f}) and p = 10 ({sp(2048, 10):.2f}). Adding the two hyper-threads "
  f"(p = 10 → 12) reduced the time by {min(ht_gain.values()):.0f}-{max(ht_gain.values()):.0f} %. The best "
  f"configuration was p = 12 for every n: S = {sp(512, 12):.2f}, {sp(1024, 12):.2f} and {sp(2048, 12):.2f} "
  f"for n = 512, 1024 and 2048 ({v('par', 1024, 12, 'gflops'):.0f} GFLOP/s at n = 1024 against "
  f"{v('seq', 1024, 1, 'gflops'):.1f} sequentially). With one thread, par was "
  f"{min(p1_gain):.0f}-{max(p1_gain):.0f} % <i>faster</i> than seq. The IQR was below 5 % of the median "
  f"except for n = 512 with p = 4-8 (up to {max_iqr:.1f} %), where one run takes only 11-17 ms and small "
  "disturbances count more. Peak RSS of the kernel phase was "
  f"{rss[512]:.1f}, {rss[1024]:.1f} and {rss[2048]:.1f} MiB for the three sizes, which is the 24n² bytes of A, "
  f"B and C plus about 2 MiB, and it changed by at most {rss_spread:.1f} MiB with p.")

# ------------------------------------------------------------------ 5 discussion
P("5. Discussion", H1)
P("<b>Overhead and the one-thread case (H1).</b> The OpenMP version with one thread was not slower. Whatever the "
  "fork/join and dynamic scheduling cost per multiplication is, it was smaller than the run-to-run variation "
  "for runs of 41 ms or more, so it could not be measured here. The small advantage of par at p = 1 is not a parallel effect. "
  f"It persisted when seq and par were alternated three times on CPU 0 (medians {alt_seq:.3f} s and "
  f"{alt_par:.3f} s at n = 2048; results/raw/supporting_checks.csv), so drift over time is not the cause. "
  "The same C source is compiled into two different machine-code functions (GCC moves the parallel loop "
  "into a separate function), and a difference in code generation is the plausible cause. We did not check "
  "this further. For this reason S(p) is computed against the sequential baseline, as required, and the "
  "par(1) row is shown separately.")
P(f"<b>Hybrid cores (H2).</b> Two P-cores gave almost ideal scaling (E ≈ {statistics.mean(ef(n, 2) for n in SIZES):.2f}). "
  "Efficiency then fell as soon as E-cores joined, because E(p) assumes p equal cores. To check this, the "
  f"sequential kernel was also run on E-core CPU 4: it took {ecore:.2f} s for n = 2048, so one E-core ran at "
  f"{ratio_e:.2f}× the speed of a P-core ({ratio_e1024:.2f}× at n = 1024). With that ratio, 2 P-cores plus "
  f"8 E-cores are worth about {cap(10):.1f} P-cores, and the ideal speedup at p = 10 is about {cap(10):.1f}, "
  f"not 10. We measured {min(sp(n, 10) for n in SIZES):.2f}-{max(sp(n, 10) for n in SIZES):.2f}, or "
  f"{min(sp(n, 10) for n in SIZES) / cap(10) * 100:.0f}-{max(sp(n, 10) for n in SIZES) / cap(10) * 100:.0f} % "
  "of that capacity. Dynamic scheduling prevents the P-cores from idling, so the rest of the loss is "
  "plausibly shared resources. First, the 15 W laptop chip cannot hold its single-core turbo frequency "
  "with all cores busy. Second, each group of 4 E-cores shares one L2 cache, and all cores share the L3 and "
  "the memory bandwidth. The flat part at n = 2048 between p = 8 and 10, where B (32 MiB) is larger than "
  "the L3, agrees with a memory-traffic limit. We did not measure frequencies or hardware counters, so "
  "these explanations are not confirmed.")
P(f"<b>Hyper-threading (H3)</b> was rejected in part. The two extra threads gave a {min(ht_gain.values()):.0f}-"
  f"{max(ht_gain.values()):.0f} % lower time. A plausible reason is that the scalar kernel, with one multiply and one add per loaded "
  "element, leaves execution units of a P-core idle, and a second hardware thread can use them. Because tasks are dynamic, the hyper-threads only take the tiles that are left over. "
  "<b>Memory (H4)</b> was confirmed: memory use did not depend on p, because all threads share the same A, "
  "B and C and use no private buffers.")
P("<b>When does parallel execution help?</b> On this machine it always helped from p = 2 on, for every tested "
  "n ≥ 512. The benefit per thread was highest on the two P-cores and much lower afterwards. Using all 12 "
  f"logical processors gave a {min(sp(n, 12) for n in SIZES):.1f}-{max(sp(n, 12) for n in SIZES):.1f}× "
  "speedup, with less than half of the ideal efficiency. Where the time of the machine matters more than the "
  "time of one product, 2 threads on the P-cores are the most efficient choice. Where only the shortest time "
  "matters, all 12 threads are best.")
P("<b>Limitations.</b> (1) One laptop, one kernel (scalar, -O2, BS = 64 not tuned for the parallel case) and "
  "three sizes. Very small n, where overhead would dominate, was not studied. (2) The CPU frequency, "
  "temperature and power were not recorded during the runs, so the turbo and memory-bandwidth explanations "
  "are plausible but not confirmed. (3) The desktop session was running during the measurements, which "
  "affects multi-threaded runs more than single-core ones. The low IQRs suggest the effect was small. (4) E(p) "
  "counts P-cores and E-cores as equal processors, as the assignment defines it. The capacity-adjusted view "
  "above is an extra interpretation based on one additional measurement.")

# ------------------------------------------------------------------ 6 conclusions
P("6. Conclusions", H1)
P(f"Distributing independent 64 × 64 output tiles of the Assignment 2 blocked kernel over OpenMP threads gave "
  f"results bitwise identical to the sequential version and reduced the time at every tested size. Two P-core "
  f"threads were almost ideally efficient (S ≈ {statistics.mean(sp(n, 2) for n in SIZES):.1f}). The maximum "
  f"speedup with all 12 logical processors was {min(sp(n, 12) for n in SIZES):.1f}-"
  f"{max(sp(n, 12) for n in SIZES):.1f}, limited mainly by the slower E-cores (≈ {ratio_e:.2f}× a P-core) and "
  "probably by shared power and memory resources. The OpenMP overhead was too small to measure. Two "
  "questions remain open: how much of the remaining efficiency loss is due to frequency and how much to "
  "memory bandwidth, and how vectorisation would change the balance between compute and memory. Assignment 4 "
  "will use these sequential and parallel versions as the reference for the distributed implementation.")

# ------------------------------------------------------------------ AI disclosure (required by the course policy)
P("Use of generative AI", H1)
P("Claude Code (Anthropic) was used to help write the code, the benchmark and analysis scripts and a first "
  "draft of this report. Every measurement comes from runs on the machine described above. The author "
  "checked the code and the results and is responsible for the conclusions.")


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#52514e"))
    canvas.drawString(2 * cm, 1.2 * cm, f"Big Data · Assignment 3 · {AUTHOR}")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, str(doc.page))
    canvas.restoreState()


os.makedirs(os.path.dirname(OUT), exist_ok=True)
SimpleDocTemplate(OUT, pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm,
                  bottomMargin=1.8 * cm, title="Assignment 3 - Parallel Matrix Multiplication",
                  author=AUTHOR).build(story, onFirstPage=footer, onLaterPages=footer)
print("wrote", OUT)
