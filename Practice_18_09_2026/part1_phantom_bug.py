
import multiprocessing as mp
import random

TOTAL_POINTS = 50_000_000
NUM_PROCS = 4


def worker(shared_hits, n, seed):
    rnd = random.Random(seed)  # each process needs its own RNG stream
    for _ in range(n):
        x = rnd.random()
        y = rnd.random()
        if x * x + y * y <= 1.0:
            # THE BUG: read-modify-write on shared memory, no lock.
            # Directly analogous to Java's totalHits++ inside the loop.
            shared_hits.value += 1


def main():
    points_per_proc = TOTAL_POINTS // NUM_PROCS
    # 'l' = signed long, lock=False -> NO synchronization (the bug)
    shared_hits = mp.Value('l', 0, lock=False)

    procs = [
        mp.Process(target=worker, args=(shared_hits, points_per_proc, i))
        for i in range(NUM_PROCS)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join()

    pi = 4.0 * shared_hits.value / TOTAL_POINTS
    print(f"[Part1] totalHits={shared_hits.value}  pi≈{pi:.6f}")


if __name__ == "__main__":
    main()
