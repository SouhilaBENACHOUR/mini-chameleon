"""ddot_seq performance: GFlop/s as a function of N (log scale).
usage: python plot_ddot.py ddot_courbe.txt"""
import sys
import matplotlib.pyplot as plt

sizes, gflops = [], []
for line in open(sys.argv[1]):
    n, g = line.split()          # "50 2.604826e+00" -> two strings
    sizes.append(int(n))
    gflops.append(float(g))

fig, ax = plt.subplots()
ax.plot(sizes, gflops, "o-", label="ddot_seq")
ax.set_xscale("log")             # N goes from 50 to ~900000
ax.set_xlabel("N")
ax.set_ylabel("GFlop/s")
ax.set_title("ddot_seq, 1 core")
ax.grid(True)
ax.legend()
fig.savefig("ddot.png", dpi=150, bbox_inches="tight")
print("ddot.png written")