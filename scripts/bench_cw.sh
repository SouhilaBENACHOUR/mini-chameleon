# usage: bash scripts/bench_cw.sh <dgemm|dgetrf|ddot> "<versions>" "<sizes>" [build_dir]
# cold = 5 fresh processes of perf_X -i 1 ; warm = 5 runs of warm_X ; median of 5, 1 core
R=$1; VERSIONS=$2; SIZES=$3; BUILD=${4:-build/opt}
case $R in
  dgemm)  dims() { echo "-M $1 -N $1 -K $1"; } ;;
  dgetrf) dims() { echo "-M $1 -N $1"; } ;;
  ddot)   dims() { echo "-N $1"; } ;;
esac
median() { grep -oE '[0-9.]+e[+-][0-9]+ GFlop/s' | awk '{print $1}' | sort -g | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}'; }
echo "routine,mode,version,size,median_gflops"
for v in $VERSIONS; do
  for n in $SIZES; do
    for mode in cold warm; do
      prog=perf_$R; [ $mode = warm ] && prog=warm_$R
      med=$(for k in 1 2 3 4 5; do
        OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 taskset -c 0 $BUILD/testings/$prog -v $v $(dims $n) -i 1 | grep GFlop | tail -1
      done | median)
      echo "$R,$mode,$v,$n,$med"
    done
  done
done
