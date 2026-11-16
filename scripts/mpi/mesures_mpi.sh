#!/bin/bash
# Mesures MPI de my_pdgemm. A lancer sur un noeud reserve, dans le shell [env].
# Usage : bash mesures_mpi.sh [dossier_sortie]
OUT=${1:-mesures/mpi}
ITER=5
BIN=./build/mpi/testings/perf_pdgemm
export OMPI_MCA_pml='^ucx' OMPI_MCA_btl='^openib' OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT"
RAW="$OUT/mpi_brut.csv"
LOG="$OUT/log.txt"
echo "exp,procs,P,M,N,K,b,t_dist,t_comp,t_cent,g_comp,g_total" > "$RAW"
: > "$LOG"
{ date; hostname; uptime; echo "coeurs : $(nproc)"; } > "$OUT/environnement.txt"

# run <experience> <processus> <P> <N> <b> <iterations> [--check]
run() {
  local exp=$1 np=$2 P=$3 N=$4 b=$5 it=$6 chk=${7:-}
  echo "[$exp] procs=$np P=$P N=$N b=$b"
  mpiexec -n $np $BIN -v mpi -P $P -N $N -b $b -i $it $chk 2>&1 \
    | tee -a "$LOG" | grep '^CSV' | sed "s/^CSV,/$exp,/" >> "$RAW"
}

# 0. Verification de la correction sur plusieurs grilles (petite taille)
for cfg in 1:1 2:1 2:2 4:1 4:2 4:4 6:2 6:3 8:2 8:4; do
  run verif ${cfg%%:*} ${cfg##*:} 500 64 1 --check
done

# 1. Passage a l'echelle : N = 2000 et 4000, b = 128
for N in 2000 4000; do
  for cfg in 1:1 2:1 4:2 6:2 8:2; do
    run echelle ${cfg%%:*} ${cfg##*:} $N 128 $ITER
  done
done

# 2. Forme de la grille : N = 3000, b = 128
for cfg in 4:1 4:2 4:4 8:1 8:2 8:4 8:8; do
  run grille ${cfg%%:*} ${cfg##*:} 3000 128 $ITER
done

# 3. Taille de tuile : 4 processus (2x2), N = 3000
for b in 32 64 128 256 512; do
  run tuile 4 2 3000 $b $ITER
done

# 4. Taille de matrice : 4 processus (2x2), b = 128
for N in 500 1000 2000 3000 4000; do
  run taille 4 2 $N 128 $ITER
done

echo "Verifications : $(grep -c SUCCESS "$LOG") SUCCESS, $(grep -c FAIL "$LOG") FAIL"
echo "Resultats dans $RAW"
