#!/bin/bash
# Effet des options du compilateur (-O2 / -O3 / -O3 -march=haswell)
# sur dgemm_seq. A lancer DANS le shell guix, depuis la racine du projet :
#     bash tests_flags.sh                      # -> mesures/reference_avant_blocage/
#     bash tests_flags.sh apres_blocage        # -> mesures/apres_blocage/
DIR="mesures/${1:-reference_avant_blocage}"
BUILDS="o2 o3 o3-haswell"
mkdir -p "$DIR"

# 1 seul thread, sinon la comparaison est faussee
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

# mediane des valeurs "xxx GFlop/s" lues sur l'entree standard
mediane() {
  grep -oE '[0-9.]+e[+-][0-9]+ GFlop/s' | awk '{print $1}' | sort -g \
    | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'
}

# --- 0. verifications de depart -------------------------------------------
for b in $BUILDS; do
  if [ ! -x "build/$b/testings/perf_dgemm" ]; then
    echo "ERREUR : build/$b n'existe pas. Compile d'abord les 3 builds."; exit 1
  fi
done
echo "[A] environnement"
{
  echo "Date    : $(date)"
  echo "Noeud   : $(uname -n)"
  echo "gcc     : $(gcc --version | head -1)"
  echo "Threads : OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS OMP_NUM_THREADS=$OMP_NUM_THREADS"
  echo "Charge  : $(uptime)"
  lscpu | grep -E "Model name|MHz|L1d|L2|L3"
  echo -n "AVX2/FMA: "; lscpu | grep -o -w -E "avx2|fma" | sort -u | tr '\n' ' '; echo
  echo "Code mesure : dgemm_seq.c, md5 = $(md5sum myblas/dgemm_seq.c | cut -d' ' -f1)"
} | tee "$DIR/environnement.txt"
cp myblas/dgemm_seq.c "$DIR/dgemm_seq_mesure.c"

# --- 1. correction ---------------------------------------------------------
echo; echo "[B] correction"
{
  for b in $BUILDS; do
    echo "== check_dgemm -v seq ($b)"
    ./build/$b/testings/check_dgemm -v seq | tail -2
  done
  echo "== check_dgetrf -v seq (o3-haswell)"
  ./build/o3-haswell/testings/check_dgetrf -v seq | tail -2
  echo "== tailles difficiles (o3-haswell)"
  for dims in "1 1 1" "3 3 3" "4 4 4" "5 5 5" "7 7 7" "8 8 8" "9 9 9" "17 17 17" \
              "99 99 99" "333 333 333" "997 997 997" "101 53 211" "13 401 29" \
              "30 13 25"; do
    set -- $dims
    r=$(./build/o3-haswell/testings/perf_dgemm -v seq -M $1 -N $2 -K $3 --check | grep -o "SUCCESS\|FAIL" | head -1)
    echo "M=$1 N=$2 K=$3 : ${r:-?}"
  done
  echo "== transpositions (o3-haswell)"
  for opt in "-A" "-B" "-A -B"; do
    r=$(./build/o3-haswell/testings/perf_dgemm -v seq $opt --check | grep -o "SUCCESS\|FAIL\|not supported" | head -1)
    echo "option $opt : ${r:-?}"
  done
} 2>&1 | tee "$DIR/correction.txt"
echo "--> nombre de FAIL : $(grep -c FAIL "$DIR/correction.txt")"

# --- 2. performance : tableau principal ------------------------------------
echo; echo "[C1] tableau principal"
echo "build,n,gflops" > "$DIR/dgemm_flags.csv"
for b in $BUILDS; do
  for n in 500 1000 1500; do
    med=$(./build/$b/testings/perf_dgemm -M $n -N $n -K $n -v seq -i 5 | mediane)
    echo "$b,$n,$med" | tee -a "$DIR/dgemm_flags.csv"
  done
done
echo "vendor (OpenBLAS, reference) :"
for n in 500 1000 1500; do
  med=$(./build/o3-haswell/testings/perf_dgemm -M $n -N $n -K $n -v vendor -i 5 | mediane)
  echo "vendor,$n,$med" | tee -a "$DIR/dgemm_flags.csv"
done

# --- 3. effet des puissances de 2 ------------------------------------------
echo; echo "[C3] puissances de 2"
echo "build,n,gflops" > "$DIR/dgemm_puissance2.csv"
for n in 511 512 513 1023 1024 1025; do
  for b in o2 o3-haswell; do
    med=$(./build/$b/testings/perf_dgemm -M $n -N $n -K $n -v seq -i 5 | mediane)
    echo "$b,$n,$med" | tee -a "$DIR/dgemm_puissance2.csv"
  done
done

# --- 4. tableau final avec % du pic (40 GFlop/s sur Xeon E5-2680 v3) --------
echo; echo "[D] resume (pic = 40 GFlop/s par coeur)"
awk -F, 'NR>1 && $3!="" {printf "%-12s n=%-5s %8.2f GFlop/s  = %5.1f %% du pic\n", $1, $2, $3, $3/40*100}' \
  "$DIR/dgemm_flags.csv" | tee "$DIR/resume.txt"

echo; echo "Fini. Resultats dans $DIR/ :"; ls -1 "$DIR"
