#!/bin/bash
# notify tickets A/B on the clean node. old = /sgl-workspace/mori, new = /sgl-workspace/mori-nt
echo "tok  ll  old      new      gain"
for m in 1 32 4096; do
  for ll in 0 1; do
    bo=999999; bn=999999
    for rep in 1 2 3; do
      for v in old new; do
        root=/sgl-workspace/mori; [ $v = new ] && root=/sgl-workspace/mori-nt
        /tmp/prof.sh "LL_DISPATCH=$ll" $m "$root" /tmp/ab_$v >/dev/null 2>&1
        x=$(python3 /tmp/kt.py /tmp/ab_$v/r0_kernel_trace.csv 2>/dev/null | awk '{print $1}')
        [ "$x" = "NA" ] && x=""
        if [ -n "$x" ]; then
          if [ $v = old ]; then bo=$(python3 -c "print(min($bo,$x))"); else bn=$(python3 -c "print(min($bn,$x))"); fi
        fi
      done
    done
    python3 -c "
o,n=$bo,$bn
print(f'{$m:<4d} {$ll}   {o:8.2f} {n:8.2f}  {100*(n-o)/o:+6.1f}%' if o<999999 and n<999999 else f'{$m} {$ll} FAILED o={o} n={n}')"
  done
done
