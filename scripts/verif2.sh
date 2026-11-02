B=./build/opt/testings
echo "=== options perf_ddot"; $B/perf_ddot -h
echo "=== check ddot seq"; $B/check_ddot -v seq | tail -2
for v in kij ijk jik; do
  echo "=== check dgemm $v"; $B/check_dgemm -v $v | tail -2
done
echo "=== ddot, courbe 1.3 (1 coeur)"
n=50
while [ $n -le 1000000 ]; do
  echo -n "$n : "
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0 $B/perf_ddot -N $n -v seq | tail -1
  n=$(( n * 125 / 100 ))
done
