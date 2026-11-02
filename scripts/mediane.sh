B=./build/opt/testings
for n in 511 512 513; do
  for v in seq bloc vendor; do
    med=$(OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0 \
          $B/perf_dgetrf -M $n -N $n -v $v -i 5 \
          | grep -oE '[0-9.]+e[+-][0-9]+ GFlop/s' | awk '{print $1}' \
          | sort -g | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
    echo "$v n=$n : $med GFlop/s"
  done
done
