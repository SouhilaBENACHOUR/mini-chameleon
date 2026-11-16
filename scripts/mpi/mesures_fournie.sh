#!/bin/bash
# Mesure de la version MPI fournie (perf_dgemm -v mpi), a comparer a my_pdgemm.
# Usage : bash mesures_fournie.sh [dossier_sortie]
OUT=${1:-mesures/mpi_miriel009}
ITER=5
export OMPI_MCA_pml='^ucx' OMPI_MCA_btl='^openib' OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT"
RAW="$OUT/comparaison_fournie.csv"
echo "version,procs,P,N,b,gflops" > "$RAW"

for N in 2000 4000; do
  for cfg in 1:1 2:1 4:2 6:2 8:2; do
    np=${cfg%%:*}; P=${cfg##*:}
    echo "[fournie] procs=$np P=$P N=$N"
    timeout 600 mpiexec -n $np ./build/mpi/testings/perf_dgemm -v mpi -P $P -N $N -b 128 -i $ITER 2>&1 \
      | grep 'GFlop/s' \
      | sed -E "s/.*: ([0-9.eE+-]+) GFlop\/s.*/fournie,$np,$P,$N,128,\1/" >> "$RAW"
  done
done
echo "Resultats dans $RAW"
