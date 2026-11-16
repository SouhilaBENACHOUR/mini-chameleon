import csv
from statistics import median
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

lignes = []
with open("mesures/mpi_miriel009/mpi_brut.csv") as f:
    for r in csv.DictReader(f):
        # on ajoute une colonne : temps total = distribution + calcul + centralisation
        r["t_total"] = str(float(r["t_dist"]) + float(r["t_comp"]) + float(r["t_cent"]))
        lignes.append(r)

def mediane(exp, procs, N, colonne, b=128):
    valeurs = [float(r[colonne]) for r in lignes
               if r["exp"] == exp and r["procs"] == str(procs)
               and r["N"] == str(N) and r["b"] == str(b)]
    return median(valeurs)

procs = [1, 2, 4, 6, 8]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

for N, marque in [(2000, "o"), (4000, "s")]:
    for colonne, trait, nom in [("t_comp", "-", "calcul seul"), ("t_total", "--", "total")]:
        t1 = mediane("echelle", 1, N, colonne)
        speedups = [t1 / mediane("echelle", p, N, colonne) for p in procs]
        efficacites = [100 * s / p for s, p in zip(speedups, procs)]
        ax1.plot(procs, speedups, marque + trait, label=f"{nom}, N={N}")
        ax2.plot(procs, efficacites, marque + trait, label=f"{nom}, N={N}")

ax1.plot(procs, procs, "k:", label="ideal")
ax1.set_title("Speedup")
ax1.set_xlabel("nombre de processus")
ax1.set_ylabel("speedup")
ax1.legend(fontsize=8)

ax2.axhline(100, color="k", linestyle=":")
ax2.set_title("Efficacite")
ax2.set_xlabel("nombre de processus")
ax2.set_ylabel("efficacite (%)")
ax2.legend(fontsize=8)

for ax in (ax1, ax2):
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("mon_speedup.png", dpi=150)
print("courbe ecrite dans mon_speedup.png")

# ------------------------------------------------------------------
# Deuxieme figure : effet de b, effet de N, forme de la grille
# ------------------------------------------------------------------
def med(exp, colonne, **filtres):
    valeurs = [float(r[colonne]) for r in lignes
               if r["exp"] == exp
               and all(r[cle] == str(val) for cle, val in filtres.items())]
    return median(valeurs)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

# 1. taille de tuile (4 processus, N = 3000)
bs = [32, 64, 128, 256, 512]
ax1.plot(bs, [med("tuile", "g_comp", procs=4, N=3000, b=b) for b in bs], "o-", label="calcul seul")
ax1.plot(bs, [med("tuile", "g_total", procs=4, N=3000, b=b) for b in bs], "s--", label="total")
ax1.set_xscale("log", base=2)
ax1.set_xlabel("taille de tuile b")
ax1.set_ylabel("GFlop/s")
ax1.set_title("Effet de b (4 procs, N=3000)")
ax1.legend()

# 2. taille de matrice (4 processus, b = 128)
Ns = [500, 1000, 2000, 3000, 4000]
ax2.plot(Ns, [med("taille", "g_comp", procs=4, N=N, b=128) for N in Ns], "o-", label="calcul seul")
ax2.plot(Ns, [med("taille", "g_total", procs=4, N=N, b=128) for N in Ns], "s--", label="total")
ax2.set_xlabel("N")
ax2.set_ylabel("GFlop/s")
ax2.set_title("Effet de N (4 procs, b=128)")
ax2.legend()

# 3. forme de la grille (N = 3000, b = 128), calcul seul
grilles = [(4, 1), (4, 2), (4, 4), (8, 1), (8, 2), (8, 4), (8, 8)]
noms = [f"{p} procs\n{P}x{p // P}" for p, P in grilles]
valeurs = [med("grille", "g_comp", procs=p, P=P, N=3000) for p, P in grilles]
ax3.bar(range(len(grilles)), valeurs)
ax3.set_xticks(range(len(grilles)))
ax3.set_xticklabels(noms, fontsize=8)
ax3.set_ylabel("GFlop/s (calcul seul)")
ax3.set_title("Forme de la grille (N=3000)")

for ax in (ax1, ax2):
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("mon_autres.png", dpi=150)
print("courbes ecrites dans mon_autres.png")
