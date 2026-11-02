#!/bin/bash
B=./build/opt/testings
echo "=== check dgemm seq";   $B/check_dgemm -v seq  | tail -3
echo "=== check dgetrf seq";  $B/check_dgetrf -v seq  | tail -3
echo "=== check dgetrf bloc"; $B/check_dgetrf -v bloc | tail -3
for s in 200 600 1000; do
  for v in seq bloc vendor; do
    echo "=== perf dgetrf $v N=$s"
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 taskset -c 0 \
      $B/perf_dgetrf -v $v -M $s -N $s --check | tail -3
  done
done
