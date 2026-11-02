# usage: bash scripts/bench_dgemm.sh "<versions>" "<sizes>" [build_dir]
# one thread, median of 5 runs, CSV on stdout
VERSIONS=${1:-"seq vendor"}
SIZES=${2:-"100 200 300 500 700 999 1000"}
BUILD=${3:-build/opt}
echo "version,size,median_gflops"
for v in $VERSIONS; do
  for n in $SIZES; do
    med=$(OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0 \
          $BUILD/testings/perf_dgemm -v $v -M $n -N $n -K $n -i 5 \
          | grep -oE '[0-9.]+e[+-][0-9]+ GFlop/s' | awk '{print $1}' \
          | sort -g | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
    echo "$v,$n,$med"
  done
done
