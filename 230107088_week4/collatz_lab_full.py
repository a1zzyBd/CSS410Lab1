"""
Amdahl Reality Gap Lab - FULL PYTHON IMPLEMENTATION
Student ID: 230107088   N = 17,088,000

Runs Phases 2-4 itself (sequential baseline, parallel scaling, false sharing,
scheduling), computes the Amdahl fit, and writes results.csv +
speedup_plot.png automatically. No manual number entry needed.

Requires: pip install matplotlib  (you already have this working)

Run with:
    python collatz_lab_full.py

NOTE: uses multiprocessing (separate OS processes), since Python cannot do
true shared-memory CPU threading. This is the Python-approved substitute
for the OpenMP/C version.
"""

import time
import csv
import ctypes
from multiprocessing import Pool, Array, cpu_count
import matplotlib.pyplot as plt

N = 17_088_000
MOD = 1_000_000_007
THRESHOLD = 100


# ---------------------------------------------------------------------------
# Core kernel
# ---------------------------------------------------------------------------
def collatz_steps(n: int) -> int:
    steps = 0
    while n > 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        steps += 1
    return steps


# ---------------------------------------------------------------------------
# Phase 2: Sequential baseline
# ---------------------------------------------------------------------------
def run_sequential(n_limit: int):
    max_steps = 0
    checksum = 0
    t0 = time.perf_counter()
    for i in range(1, n_limit + 1):
        s = collatz_steps(i)
        if s > max_steps:
            max_steps = s
        checksum = (checksum + s) % MOD
    t1 = time.perf_counter()
    return t1 - t0, max_steps, checksum


def phase2_baseline():
    print("=" * 70)
    print("PHASE 2: Sequential Baseline (3 runs, discard Run 1 as warmup)")
    print("=" * 70)
    times = []
    for run in range(1, 4):
        elapsed, max_steps, checksum = run_sequential(N)
        tag = "(WARMUP - discarded)" if run == 1 else ""
        print(f"  Run {run}: time={elapsed:.4f}s max_steps={max_steps} "
              f"checksum={checksum} {tag}")
        times.append(elapsed)
    t_seq = (times[1] + times[2]) / 2.0
    print(f"\n  T_seq = (Run2 + Run3) / 2 = {t_seq:.4f}s\n")
    return t_seq


# ---------------------------------------------------------------------------
# Phase 3: Parallel scaling
# ---------------------------------------------------------------------------
def chunk_worker(args):
    start, end = args
    max_steps = 0
    checksum = 0
    for i in range(start, end + 1):
        s = collatz_steps(i)
        if s > max_steps:
            max_steps = s
        checksum = (checksum + s) % MOD
    return max_steps, checksum


def make_ranges(n_limit: int, k: int):
    chunk_size = n_limit // k
    ranges = []
    for i in range(k):
        start = i * chunk_size + 1
        end = (i + 1) * chunk_size if i < k - 1 else n_limit
        ranges.append((start, end))
    return ranges


def run_parallel(n_limit: int, k: int):
    ranges = make_ranges(n_limit, k)
    t0 = time.perf_counter()
    with Pool(processes=k) as pool:
        results = pool.map(chunk_worker, ranges)
    t1 = time.perf_counter()
    max_steps = max(r[0] for r in results)
    checksum = sum(r[1] for r in results) % MOD
    return t1 - t0, max_steps, checksum


def phase3_scaling(t_seq: float, thread_counts):
    print("=" * 70)
    print("PHASE 3: Multi-Process Scaling & Empirical Amdahl Fitting")
    print("=" * 70)

    T_k = {}
    for k in thread_counts:
        times_k = []
        for run in range(1, 4):
            elapsed, max_steps, checksum = run_parallel(N, k)
            times_k.append(elapsed)
        avg_tk = (times_k[1] + times_k[2]) / 2.0
        T_k[k] = avg_tk
        print(f"  k={k:2d}: Run1={times_k[0]:.4f}s(warmup) "
              f"Run2={times_k[1]:.4f}s Run3={times_k[2]:.4f}s "
              f"Avg_T_k={avg_tk:.4f}s")

    S_emp = {k: t_seq / T_k[k] for k in thread_counts}

    if 2 in S_emp and S_emp[2] > 0:
        p = 2 * (1 - (1 / S_emp[2]))
    else:
        p = None

    S_theo = {}
    Delta = {}
    for k in thread_counts:
        s_theo = 1 / ((1 - p) + (p / k)) if p is not None else float('nan')
        S_theo[k] = s_theo
        Delta[k] = s_theo - S_emp[k]

    print(f"\n  Derived parallel fraction p (from k=2) = {p:.4f}\n" if p else "\n")
    print(f"  {'k':>4} {'T_k(s)':>10} {'S_emp(k)':>10} {'S_theo(k)':>10} {'Delta(k)':>10}")
    for k in thread_counts:
        print(f"  {k:>4} {T_k[k]:>10.4f} {S_emp[k]:>10.4f} "
              f"{S_theo[k]:>10.4f} {Delta[k]:>10.4f}")
    print()
    return T_k, S_emp, p, S_theo, Delta


# ---------------------------------------------------------------------------
# Phase 4a: False sharing experiment
# ---------------------------------------------------------------------------
def false_sharing_worker(args):
    start, end, shared_arr, idx = args
    for i in range(start, end + 1):
        if collatz_steps(i) > THRESHOLD:
            shared_arr[idx] += 1
    return None


def naive_false_sharing(n_limit: int, k: int):
    shared_arr = Array(ctypes.c_int, k, lock=False)
    ranges = make_ranges(n_limit, k)
    args = [(r[0], r[1], shared_arr, i) for i, r in enumerate(ranges)]
    t0 = time.perf_counter()
    with Pool(processes=k) as pool:
        pool.map(false_sharing_worker, args)
    t1 = time.perf_counter()
    total_hits = sum(shared_arr[:])
    return t1 - t0, total_hits


def reduction_worker(args):
    start, end = args
    count = 0
    for i in range(start, end + 1):
        if collatz_steps(i) > THRESHOLD:
            count += 1
    return count


def reduction_mitigated(n_limit: int, k: int):
    ranges = make_ranges(n_limit, k)
    t0 = time.perf_counter()
    with Pool(processes=k) as pool:
        results = pool.map(reduction_worker, ranges)
    t1 = time.perf_counter()
    return t1 - t0, sum(results)


def phase4a_false_sharing(max_threads: int):
    print("=" * 70)
    print("PHASE 4a: False Sharing Experiment")
    print("=" * 70)

    t_naive, hits_naive = naive_false_sharing(N, max_threads)
    print(f"  Variant 1 (naive shared array):   "
          f"time={t_naive:.4f}s  hits={hits_naive}")

    t_reduction, hits_reduction = reduction_mitigated(N, max_threads)
    print(f"  Variant 2 (reduction/local sum):  "
          f"time={t_reduction:.4f}s  hits={hits_reduction}")

    penalty_ratio = t_naive / t_reduction if t_reduction > 0 else float('nan')
    print(f"\n  Speedup penalty ratio (naive/reduction) = {penalty_ratio:.3f}\n")
    return t_naive, t_reduction, penalty_ratio


# ---------------------------------------------------------------------------
# Phase 4b: Scheduling experiment
# ---------------------------------------------------------------------------
def scheduled_worker(args):
    start, end = args
    max_steps = 0
    for i in range(start, end + 1):
        s = collatz_steps(i)
        if s > max_steps:
            max_steps = s
    return max_steps


def make_static_chunks(n_limit, k, chunk_size=None):
    if chunk_size is None:
        return make_ranges(n_limit, k)
    chunks = []
    i = 1
    while i <= n_limit:
        end = min(i + chunk_size - 1, n_limit)
        chunks.append((i, end))
        i = end + 1
    return chunks


def run_scheduled(n_limit, k, chunks):
    t0 = time.perf_counter()
    with Pool(processes=k) as pool:
        pool.map(scheduled_worker, chunks)
    t1 = time.perf_counter()
    return t1 - t0


def phase4b_scheduling(max_threads: int):
    print("=" * 70)
    print("PHASE 4b: Loop Scheduling & Workload Imbalance")
    print("=" * 70)

    configs = [
        ("static", make_static_chunks(N, max_threads)),
        ("static_chunk", make_static_chunks(N, max_threads, 1000)),
        ("dynamic_100", make_static_chunks(N, None, 100)),
        ("dynamic_10000", make_static_chunks(N, None, 10000)),
        ("guided", make_static_chunks(N, None, max(1000, N // (max_threads * 20)))),
    ]

    sched_times = {}
    for label, chunks in configs:
        elapsed = run_scheduled(N, max_threads, chunks)
        sched_times[label] = elapsed
        print(f"  {label:<20} chunks={len(chunks):>6}  time={elapsed:.4f}s")
    print()
    return sched_times


# ---------------------------------------------------------------------------
# Output: CSV + PNG
# ---------------------------------------------------------------------------
def write_results_csv(T_k, S_emp, S_theo, Delta, t_naive, t_reduction, sched_times,
                       filename="results.csv"):
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Table 1: Amdahl Scaling"])
        writer.writerow(["k", "T_k(s)", "S_emp(k)", "S_theo(k)", "Delta(k)"])
        for k in sorted(T_k):
            writer.writerow([k, f"{T_k[k]:.4f}", f"{S_emp[k]:.4f}",
                              f"{S_theo[k]:.4f}", f"{Delta[k]:.4f}"])
        writer.writerow([])
        writer.writerow(["Table 2: False Sharing Experiment"])
        writer.writerow(["Variant", "Time(s)"])
        writer.writerow(["Naive (false sharing)", f"{t_naive:.4f}"])
        writer.writerow(["Reduction (mitigated)", f"{t_reduction:.4f}"])
        writer.writerow([])
        writer.writerow(["Table 3: Scheduling Comparison"])
        writer.writerow(["Schedule", "Time(s)"])
        for label, t in sched_times.items():
            writer.writerow([label, f"{t:.4f}"])
    print(f"Wrote {filename}")


def plot_speedup(S_emp, S_theo, filename="speedup_plot.png"):
    ks = sorted(S_emp.keys())
    emp_vals = [S_emp[k] for k in ks]
    theo_vals = [S_theo[k] for k in ks]
    ideal_vals = ks

    plt.figure(figsize=(8, 6))
    plt.plot(ks, ideal_vals, linestyle="--", marker="o", label="Ideal Linear S(k)=k", color="gray")
    plt.plot(ks, theo_vals, linestyle="-", marker="s", label="Theoretical S_theo(k) [Amdahl]", color="blue")
    plt.plot(ks, emp_vals, linestyle="-", marker="^", label="Empirical S_emp(k)", color="red")

    plt.xlabel("Number of Threads/Processes (k)")
    plt.ylabel("Speedup S(k)")
    plt.title("Amdahl Reality Gap: Empirical vs Theoretical Speedup\nStudent ID 230107088, N=17,088,000")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Wrote {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    max_threads = cpu_count()
    thread_counts = sorted(set([1, 2, 4, 8, max_threads]))
    print(f"Detected logical CPU count: {max_threads}")
    print(f"Thread counts to test: {thread_counts}\n")

    t_seq = phase2_baseline()
    T_k, S_emp, p, S_theo, Delta = phase3_scaling(t_seq, thread_counts)
    t_naive, t_reduction, penalty_ratio = phase4a_false_sharing(max_threads)
    sched_times = phase4b_scheduling(max_threads)

    write_results_csv(T_k, S_emp, S_theo, Delta, t_naive, t_reduction, sched_times)
    plot_speedup(S_emp, S_theo)

    print("\n" + "=" * 70)
    print("ALL PHASES COMPLETE.")
    print("results.csv and speedup_plot.png are ready for Phase 6 packaging.")
    print("Copy the printed tables above into your worksheet.")
    print("=" * 70)
