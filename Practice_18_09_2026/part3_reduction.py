
import multiprocessing as mp
import random
import time

TOTAL_POINTS = 100_000_000
PROC_COUNTS = [1, 2, 4, 8, 16, 32]


def local_reduction_worker(n, seed):
    rnd = random.Random(seed)
    local_hits = 0  # PRIVATE counter - lives only in this process
    for _ in range(n):
        x = rnd.random()
        y = rnd.random()
        if x * x + y * y <= 1.0:
            local_hits += 1
    return local_hits  # one value returned, once, at the very end


def run_with_procs(nproc):
    points_per_proc = TOTAL_POINTS // nproc
    args = [(points_per_proc, seed) for seed in range(nproc)]

    start = time.perf_counter()
    with mp.Pool(processes=nproc) as pool:
        partials = pool.starmap(local_reduction_worker, args)
    elapsed = time.perf_counter() - start

    # Reduction step: sum partial sums together ONCE, after all processes finish.
    total_hits = sum(partials)
    pi = 4.0 * total_hits / TOTAL_POINTS
    return elapsed, pi


def main():
    print(f"{'Procs':>6}\t{'Runtime(ms)':>12}\t{'Speedup':>8}\t{'Efficiency':>10}\tpi")

    baseline = None
    for n in PROC_COUNTS:
        elapsed, pi = run_with_procs(n)
        if baseline is None:
            baseline = elapsed
        speedup = baseline / elapsed
        efficiency = (speedup / n) * 100.0
        print(f"{n:>6}\t{elapsed*1000:>12.0f}\t{speedup:>7.2f}x\t{efficiency:>9.1f}%\t{pi:.5f}")


if __name__ == "__main__":
    main()
