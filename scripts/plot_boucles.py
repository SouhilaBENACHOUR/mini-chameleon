"""dgemm loop orders: GFlop/s as a function of N, one curve per version.
usage: python plot_boucles.py resultats_boucles.txt"""
import sys
import matplotlib.pyplot as plt

data = {}                                  # version -> [(N, gflops)]
for line in open(sys.argv[1]):
    w = line.split()
    if not w:
        continue
    version, n = w[0], int(w[1])           # "seq 100 : ..."
    g = float(w[-2])                       # last word is "GFlop/s"
    data.setdefault(version, []).append((n, g))

fig, ax = plt.subplots()
for version, pts in data.items():
    pts.sort()
    ax.plot(*zip(*pts), "o-", label=version)
ax.set_xlabel("N (M = N = K)")
ax.set_ylabel("GFlop/s")
ax.set_title("dgemm loop orders, 1 core")
ax.grid(True)
ax.legend()
fig.savefig("boucles.png", dpi=150, bbox_inches="tight")
print("boucles.png written")