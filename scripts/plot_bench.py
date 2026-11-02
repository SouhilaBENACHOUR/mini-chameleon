"""Plot bench_dgemm.sh CSV files on one graph.
usage: python plot_bench.py file1.csv [file2.csv ...]
Each curve is labelled <file name>:<version>, so the same version measured
at different steps (e.g. seq before/after) stays distinguishable."""
import csv
import os
import sys
import matplotlib.pyplot as plt

PEAK = 40.0  # estimated peak, GFlop/s per core (2 FMA x 4 doubles x 2 flops x 2.5 GHz)

fig, ax = plt.subplots()
for fname in sys.argv[1:]:
    label = os.path.splitext(os.path.basename(fname))[0]
    series = {}
    for row in csv.DictReader(open(fname)):
        if not row["median_gflops"]:
            continue
        series.setdefault(row["version"], []).append(
            (int(row["size"]), float(row["median_gflops"])))
    for version, pts in series.items():
        pts.sort()
        ax.plot(*zip(*pts), "o-", label=f"{label}:{version}")

ax.axhline(PEAK, ls="--", color="gray", label=f"estimated peak ({PEAK:g})")
ax.set_xlabel("N (square matrices)")
ax.set_ylabel("GFlop/s")
ax.grid(True)
ax.legend()
right = ax.secondary_yaxis("right", functions=(lambda y: 100 * y / PEAK,
                                               lambda p: p * PEAK / 100))
right.set_ylabel("% of estimated peak")
plt.savefig("bench.png", dpi=150)
print("bench.png written")
