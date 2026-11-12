/*
 * Assignment 3: sequential and OpenMP-parallel blocked matrix multiplication, C = A * B.
 *
 * Both methods run exactly the same tile code (tile()). A tile is one BS x BS block
 * C[ii.., jj..]; it is zeroed and then accumulated over the k blocks in increasing
 * order, with the i-k-j loop order inside (as the Assignment 2 blocked kernel).
 *   seq  loops over all tiles in row-major order, no OpenMP runtime involved
 *   par  distributes the same tiles over p OpenMP threads
 *        (#pragma omp parallel for collapse(2) schedule(dynamic, 1))
 * Different tiles write disjoint parts of C and only read A and B, so there is no
 * shared write and no synchronisation inside the kernel. For every C[i][j] the
 * products are added in increasing k order in both methods (no FMA), so the
 * results are bitwise identical for any thread count.
 *
 * Usage:
 *   matmul3 <seq|par> <n> <A.bin> <B.bin> <threads> <warmup> <reps> [C_out.bin]
 *   matmul3 --selftest
 * Inputs: raw little-endian float64, row-major, n*n values (Assignment 1/2 files).
 *
 * Output (stdout, CSV, no header):
 *   <method>,<n>,<threads>,<phase>,<rep>,<time_s>,<checksum>   phase = warmup | measured
 *   meta,<n>,<threads>,<omp_max_threads>,<hwm_process_kb>,<hwm_kernel_kb>
 * Timed region: the kernel only (zeroing of C included, as in Assignment 2). Inputs
 * are read and C is allocated and touched before timing. For par, the OpenMP thread
 * pool is created in the warm-up run (a first parallel region), so the measured runs
 * contain the fork/join and scheduling cost of each multiplication but not thread
 * creation. The checksum of C is computed outside the timer.
 */
#define _GNU_SOURCE
#include <malloc.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef BS
#define BS 64
#endif

static double now_seconds(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static void *xmalloc(size_t bytes)
{
    void *p = malloc(bytes ? bytes : 1);
    if (!p) { fprintf(stderr, "out of memory (%zu bytes)\n", bytes); exit(3); }
    return p;
}

/* One output tile C[ii..ie) x [jj..je): zero it, then add A[i][k] * B[k][j] over all
 * k blocks in increasing order (i-k-j order inside each block). */
static inline void tile(const double *A, const double *B, double *C, int n, int ii, int jj)
{
    const int ie = ii + BS < n ? ii + BS : n;
    const int je = jj + BS < n ? jj + BS : n;
    for (int i = ii; i < ie; i++)
        memset(C + (size_t)i * n + jj, 0, (size_t)(je - jj) * sizeof(double));
    for (int kk = 0; kk < n; kk += BS) {
        const int ke = kk + BS < n ? kk + BS : n;
        for (int i = ii; i < ie; i++) {
            double *c = C + (size_t)i * n;
            for (int k = kk; k < ke; k++) {
                const double a = A[(size_t)i * n + k];
                const double *b = B + (size_t)k * n;
                for (int j = jj; j < je; j++)
                    c[j] += a * b[j];
            }
        }
    }
}

static void mm_seq(const double *A, const double *B, double *C, int n)
{
    for (int ii = 0; ii < n; ii += BS)
        for (int jj = 0; jj < n; jj += BS)
            tile(A, B, C, n, ii, jj);
}

static void mm_par(const double *A, const double *B, double *C, int n, int p)
{
#pragma omp parallel for collapse(2) schedule(dynamic, 1) num_threads(p)
    for (int ii = 0; ii < n; ii += BS)
        for (int jj = 0; jj < n; jj += BS)
            tile(A, B, C, n, ii, jj);
}

static void run(const char *method, const double *A, const double *B, double *C, int n, int p)
{
    if (strcmp(method, "seq") == 0) mm_seq(A, B, C, n);
    else mm_par(A, B, C, n, p);
}

/* ------------------------------ helpers / I/O ------------------------------ */

static double checksum(const double *C, int n)
{
    double s = 0.0;
    for (size_t i = 0; i < (size_t)n * n; i++) s += C[i];
    return s;
}

static double *read_matrix(const char *path, int n)
{
    double *m = xmalloc((size_t)n * n * sizeof(double));
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(2); }
    size_t count = (size_t)n * n;
    if (fread(m, sizeof(double), count, f) != count) {
        fprintf(stderr, "%s: expected %zu doubles\n", path, count);
        exit(2);
    }
    fclose(f);
    return m;
}

static long read_hwm_kb(void)
{
    FILE *f = fopen("/proc/self/status", "r");
    char line[256];
    long kb = -1;
    while (f && fgets(line, sizeof line, f))
        if (sscanf(line, "VmHWM: %ld kB", &kb) == 1) break;
    if (f) fclose(f);
    return kb;
}

static void reset_hwm(void)
{
    FILE *f = fopen("/proc/self/clear_refs", "w");
    if (f) { fputs("5", f); fclose(f); }
}

/* -------------------------------- self tests -------------------------------- */

static int check(const char *name, int p, const double *got, const double *want, int n)
{
    for (int i = 0; i < n * n; i++)
        if (got[i] != want[i]) {
            printf("FAIL p=%-2d %s: element %d got %.17g want %.17g\n", p, name, i, got[i], want[i]);
            return 1;
        }
    printf("PASS p=%-2d %s\n", p, name);
    return 0;
}

static int selftest(void)
{
    double A2[] = {1, 2, 3, 4}, B2[] = {5, 6, 7, 8}, R2[] = {19, 22, 43, 50};
    double A3[] = {1, -2, 0.5, 0, 3, -1, 4, 1, 2};
    double B3[] = {2, 0, 1, -1, 1, 0, 0.5, 2, -3};
    double R3[] = {4.25, -1, -0.5, -3.5, 1, 3, 8, 5, -2};
    double I3[] = {1, 0, 0, 0, 1, 0, 0, 0, 1}, Z3[9] = {0}, C[9];

    /* random 200 x 200: not a multiple of BS (partial tiles), more tiles than threads;
     * the reference is a plain i-j-k triple loop with the same k order */
    const int nr = 200;
    double *Ar = xmalloc(sizeof(double) * nr * nr), *Br = xmalloc(sizeof(double) * nr * nr);
    double *Rr = xmalloc(sizeof(double) * nr * nr), *Cr = xmalloc(sizeof(double) * nr * nr);
    srand(12345);
    for (int i = 0; i < nr * nr; i++) {
        Ar[i] = 2.0 * rand() / RAND_MAX - 1.0;
        Br[i] = 2.0 * rand() / RAND_MAX - 1.0;
    }
    for (int i = 0; i < nr; i++)
        for (int j = 0; j < nr; j++) {
            double s = 0.0;
            for (int k = 0; k < nr; k++) s += Ar[i * nr + k] * Br[k * nr + j];
            Rr[i * nr + j] = s;
        }

    int fails = 0;
    const int ps[] = {0, 1, 2, 3, 4, 7, 12, 16};          /* 0 = seq */
    for (size_t t = 0; t < sizeof ps / sizeof ps[0]; t++) {
        const char *m = ps[t] ? "par" : "seq";
        const int p = ps[t];
        run(m, A2, B2, C, 2, p); fails += check("2x2 known product", p, C, R2, 2);
        run(m, A3, B3, C, 3, p); fails += check("3x3 known product", p, C, R3, 3);
        run(m, A3, I3, C, 3, p); fails += check("A * I = A", p, C, A3, 3);
        run(m, I3, A3, C, 3, p); fails += check("I * A = A", p, C, A3, 3);
        run(m, A3, Z3, C, 3, p); fails += check("A * 0 = 0", p, C, Z3, 3);
        run(m, Z3, B3, C, 3, p); fails += check("0 * B = 0", p, C, Z3, 3);
        for (int i = 0; i < nr * nr; i++) Cr[i] = 1e300;  /* stale values must be overwritten */
        run(m, Ar, Br, Cr, nr, p); fails += check("200x200 random == ijk (bitwise)", p, Cr, Rr, nr);
    }
    free(Ar); free(Br); free(Rr); free(Cr);
    printf(fails ? "SELFTEST FAILED (%d)\n" : "SELFTEST OK\n", fails);
    return fails ? 1 : 0;
}

/* ----------------------------------- main ----------------------------------- */

int main(int argc, char **argv)
{
    mallopt(M_MMAP_THRESHOLD, 64 * 1024);   /* as in Assignment 2 */
    if (argc == 2 && strcmp(argv[1], "--selftest") == 0) return selftest();
    if (argc < 8 || (strcmp(argv[1], "seq") != 0 && strcmp(argv[1], "par") != 0)) {
        fprintf(stderr, "usage: %s <seq|par> <n> <A.bin> <B.bin> <threads> <warmup> <reps> [C_out.bin]\n"
                        "       %s --selftest\n", argv[0], argv[0]);
        return 1;
    }
    const char *method = argv[1];
    const int n = atoi(argv[2]), warmup = atoi(argv[6]), reps = atoi(argv[7]);
    const int p = strcmp(method, "seq") == 0 ? 1 : atoi(argv[5]);
    const char *out = argc >= 9 ? argv[8] : NULL;

    double *A = read_matrix(argv[3], n);
    double *B = read_matrix(argv[4], n);
    double *C = xmalloc((size_t)n * n * sizeof(double));
    memset(C, 0, (size_t)n * n * sizeof(double));          /* touch pages before timing */
    const long hwm_setup = read_hwm_kb();
    reset_hwm();
    for (int r = 0; r < warmup + reps; r++) {
        const double t0 = now_seconds();
        run(method, A, B, C, n, p);
        const double t1 = now_seconds();
        const int measured = r >= warmup;
        printf("%s,%d,%d,%s,%d,%.9f,%.17g\n", method, n, p, measured ? "measured" : "warmup",
               measured ? r - warmup : r, t1 - t0, checksum(C, n));
        fflush(stdout);
    }
    const long hwm_kernel = read_hwm_kb();
    if (out) {
        FILE *f = fopen(out, "wb");
        if (!f) { perror(out); return 2; }
        fwrite(C, sizeof(double), (size_t)n * n, f);
        fclose(f);
    }
    printf("meta,%d,%d,%d,%ld,%ld\n", n, p, omp_get_max_threads(),
           hwm_setup > hwm_kernel ? hwm_setup : hwm_kernel, hwm_kernel);
    free(A); free(B); free(C);
    return 0;
}
