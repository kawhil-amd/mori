#!/bin/bash
# base vs LL, strictly alternated, old notify only (= the PR's change). 3 reps, min.
echo "tok   base      ll        gain"
for m in 1 32 128 4096; do
  bb=999999; bl=999999
  for rep in 1 2 3; do
    for v in 0 1; do
      /tmp/prof.sh "LL_DISPATCH=$v" $m /sgl-workspace/mori /tmp/ll_$v >/dev/null 2>&1
      x=$(python3 /tmp/kt.py /tmp/ll_$v/r0_kernel_trace.csv 2>/dev/null | awk '{print $1}')
      [ "$x" = "NA" ] && x=""
      if [ -n "$x" ]; then
        if [ $v = 0 ]; then bb=$(python3 -c "print(min($bb,$x))"); else bl=$(python3 -c "print(min($bl,$x))"); fi
      fi
    done
  done
  python3 -c "
b,l=$bb,$bl
print(f'{$m:<5d} {b:9.2f} {l:9.2f}  {100*(l-b)/b:+6.1f}%' if b<999999 and l<999999 else f'{$m} FAILED b={b} l={l}')"
done
