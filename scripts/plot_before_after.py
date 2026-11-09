"""dgemm before/after step 2, one graph. usage (from the folder holding the CSVs):
python plot_before_after.py baseline_step1.csv step2_opt.csv step2_readme.csv"""
import csv
import sys
import matplotlib.pyplot as plt

PEAK = 40.0  # estimated peak per core, GFlop/s (2 FMA x 4 doubles x 2 flops x 2.5 GHz)

def read(fname):
    series = {}
    for row in csv.DictReader(open(fname)):
        if row["median_gflops"]:
            series.setdefault(row["version"], []).append(
                (int(row["size"]), float(row["median_gflops"])))
    return {v: sorted(p) for v, p in series.items()}

base, opt, readme = (read(f) for f in sys.argv[1:4])

fig, ax = plt.subplots(figsize=(8, 5))
curves = [
    ("step 1: seq (-O3 -march=native)",      base["seq"],    "C0", "o-"),
    ("step 2: seq (README build)",           readme["seq"],  "C2", "s--"),
    ("step 2: seq (-march=native)",          opt["seq"],     "C1", "o-"),
    ("step 2: vecto alone",                  opt["vecto"],   "C3", "^:"),
    ("step 2: bloc",                         opt["bloc"],    "C4", "v:"),
    ("OpenBLAS (vendor)",                    opt["vendor"],  "k",  "o-"),
]
for label, pts, color, style in curves:
    ax.plot(*zip(*pts), style, color=color, label=label)
ax.axhline(PEAK, ls=":", color="gray", label=f"estimated peak ({PEAK:g})")
ax.set_xlabel("N (square matrices; 501 and 999 are not multiples of 4)")
ax.set_ylabel("GFlop/s")
ax.set_title("dgemm, 1 core, median of 5: before and after step 2")
ax.grid(True)
ax.legend(fontsize=8)
right = ax.secondary_yaxis("right", functions=(lambda y: 100 * y / PEAK,
                                               lambda p: p * PEAK / 100))
right.set_ylabel("% of estimated peak")
fig.savefig("before_after.png", dpi=150, bbox_inches="tight")
print("before_after.png written")
