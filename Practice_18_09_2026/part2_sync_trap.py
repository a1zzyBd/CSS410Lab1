"""
Part 2: The Synchronization Trap (Python version)

Fixes the Part 1 race by wrapping every increment in a `multiprocessing.Lock`
(equivalent to Java's `synchronized`). The answer becomes correct
(pi ~ 3.1416), but now every single increment has to go through the OS-level
lock: acquire a semaphore, do the write, release the semaphore - across
process boundaries this is even heavier than Java's in-process
`synchronized`, because it's real inter-process communication (IPC), not
just a CPU cache-coherency exchange. So it usually ends up *slower* than
Java's synchronized version, relative to its own single-process baseline.
"""
import multiprocessing as mp
import random
import time

TOTAL_POINTS = 50_000_000
NUM_PROCS = 4


def sample_hits(n, seed):
    """Pure computation, no shared state - used for the single-process baseline."""
    rnd = random.Random(seed)
    hits = 0
    for _ in range(n):
        x = rnd.random()
        y = rnd.random()
        if x * x + y * y <= 1.0:
            hits += 1
    return hits


def worker_locked(shared_hits, lock, n, seed):
    rnd = random.Random(seed)
    for _ in range(n):
        x = rnd.random()
        y = rnd.random()
        if x * x + y * y <= 1.0:
            with lock:
                shared_hits.value += 1  # every increment pays for a lock acquire/release


def run_single_threaded():
    start = time.perf_counter()
    hits = sample_hits(TOTAL_POINTS, seed=1)
    elapsed = time.perf_counter() - start
    pi = 4.0 * hits / TOTAL_POINTS
    print(f"[Single-process] hits={hits}  pi≈{pi:.6f}  time={elapsed*1000:.0f}ms")
    return elapsed


def run_locked():
    points_per_proc = TOTAL_POINTS // NUM_PROCS
    shared_hits = mp.Value('l', 0)          # default Value has its own internal Lock
    lock = shared_hits.get_lock()

    start = time.perf_counter()
    procs = [
        mp.Process(target=worker_locked, args=(shared_hits, lock, points_per_proc, i))
        for i in range(NUM_PROCS)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    elapsed = time.perf_counter() - start

    pi = 4.0 * shared_hits.value / TOTAL_POINTS
    print(f"[Locked, {NUM_PROCS} procs] hits={shared_hits.value}  pi≈{pi:.6f}  time={elapsed*1000:.0f}ms")
    return elapsed


def main():
    print("=== Single-process baseline ===")
    t_single = run_single_threaded()

    print("\n=== Multi-process with a Lock around every increment ===")
    t_locked = run_locked()

    print(f"\nLocked version is {t_locked / t_single:.2f}x the single-process time.")


if __name__ == "__main__":
    main()
