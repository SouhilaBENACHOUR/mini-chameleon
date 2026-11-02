B=./build/opt/testings
echo "##### dgemm, tailles impaires (carrées et rectangulaires)"
for v in seq kij ijk jik; do
  for dims in "1 1 1" "7 7 7" "99 99 99" "333 333 333" "997 997 997" "101 53 211" "13 401 29"; do
    set -- $dims
    echo -n "$v M=$1 N=$2 K=$3 : "
    $B/perf_dgemm -v $v -M $1 -N $2 -K $3 --check | tail -2 | grep -o "SUCCESS\|FAIL" | head -1
  done
done
echo "##### dgetrf, tailles impaires, blocs non puissances de 2"
for bs in 7 32 33 100; do
  for s in 1 7 99 333 997; do
    for v in unblocked bloc; do
      echo -n "$v b=$bs N=$s : "
      LU_BLOCKSIZE=$bs $B/perf_dgetrf -v $v -M $s -N $s --check | tail -2 | grep -o "SUCCESS\|FAIL" | head -1
    done
  done
done
echo "##### dgetrf rectangulaire"
for dims in "100 60" "60 100" "333 127"; do
  set -- $dims
  echo -n "bloc M=$1 N=$2 : "
  $B/perf_dgetrf -v bloc -M $1 -N $2 --check | tail -2 | grep -o "SUCCESS\|FAIL" | head -1
done
