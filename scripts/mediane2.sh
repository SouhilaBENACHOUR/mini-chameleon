B=./build/opt/testings
for n in 504 508 510 512 514 520; do
  med=$(OPENBLAS_NUM_THREADS=1 taskset -c 0 $B/perf_dgetrf -M $n -N $n -v bloc -i 5 \
        | grep -oE '[0-9.]+e[+-][0-9]+ GFlop/s' | awk '{print $1}' \
        | sort -g | awk '{a[NR]=$1} END {print a[int((NR+1)/2)]}')
  echo "bloc n=$n : $med GFlop/s"
done
