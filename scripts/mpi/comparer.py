import csv
from statistics import median
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

dossier = "mesures/mpi_miriel037"

def lire(fichier):
    with open(fichier) as f:
        return list(csv.DictReader(f))

fournie = lire(dossier + "/comparaison_fournie.csv")
mienne = [r for r in lire(dossier + "/mpi_brut.csv") if r["exp"] == "echelle"]

def mediane(lignes, procs, N, colonne):
    valeurs = [float(r[colonne]) for r in lignes
               if r["procs"] == str(procs) and r["N"] == str(N)]
    return median(valeurs)

procs = [1, 2, 4, 6, 8]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

for ax, N in zip(axes, (2000, 4000)):
    gf_f = [mediane(fournie, p, N, "gflops") for p in procs]
    gf_m = [mediane(mienne, p, N, "g_comp") for p in procs]
    ax.plot(procs, gf_f, "s--", label="version fournie (Send/Recv)")
    ax.plot(procs, gf_m, "o-", label="my_pdgemm (Bcast), calcul seul")
    ax.set_title(f"N={N}, b=128")
    ax.set_xlabel("nombre de processus")
    ax.set_ylabel("GFlop/s")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    print(f"N={N}")
    for p, a, b in zip(procs, gf_f, gf_m):
        print(f"  {p} procs : fournie {a:6.1f}   my_pdgemm {b:6.1f}   rapport x{b / a:.1f}")

plt.tight_layout()
plt.savefig("comparaison.png", dpi=150)
print("courbe ecrite dans comparaison.png")
