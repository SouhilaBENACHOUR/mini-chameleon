"""One graph per routine: cold (dashed) vs warm (solid), one colour per version.
usage: python plot_cw.py file1.csv [file2.csv ...]
CSV columns: routine,mode,version,size,median_gflops (output of bench_cw.sh)."""
import csv
import sys
import matplotlib.pyplot as plt

PEAK = 40.0  # estimated peak per core, GFlop/s (2 FMA x 4 doubles x 2 flops x 2.5 GHz)
NOTES = {
    "dgetrf": "warm_dgetrf factors the same matrix in place: L underflows (denormals) after "
              "~110 passes, so the warm value at N=500 is biased low (seq and vendor).",
    "ddot":   "perf_ddot times one call with a 1 us clock: cold values at N=1000 are quantised.",
}

data = {}  # routine -> (version, mode) -> [(size, gflops)]
for fname in sys.argv[1:]:
    for row in csv.DictReader(open(fname)):
        if not row.get("median_gflops"):
            continue
        key = (row["version"], row["mode"])
        data.setdefault(row["routine"], {}).setdefault(key, []).append(
            (int(row["size"]), float(row["median_gflops"])))

colors = {}
for routine, series in data.items():
    fig, ax = plt.subplots()
    for (version, mode), pts in sorted(series.items()):
        pts.sort()
        c = colors.setdefault(version, f"C{len(colors)}")
        ax.plot(*zip(*pts), "o--" if mode == "cold" else "o-",
                color=c, label=f"{version} ({mode})")
    if routine != "ddot":
        ax.axhline(PEAK, ls=":", color="gray", label=f"estimated peak ({PEAK:g})")
    if routine == "ddot":
        ax.set_xscale("log")
    ax.set_xlabel("N")
    ax.set_ylabel("GFlop/s")
    ax.set_title(f"{routine}: cold vs warm, 1 core, median of 5")
    ax.grid(True)
    ax.legend()
    if routine in NOTES:
        fig.text(0.01, 0.01, NOTES[routine], fontsize=6)
    fig.savefig(f"cw_{routine}.png", dpi=150, bbox_inches="tight")
    print(f"cw_{routine}.png written")
