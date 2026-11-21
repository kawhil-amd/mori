#!/bin/bash
# in-process base-vs-LL under rocprof: one process, both kernels, separated by symbol name.
set -e
O=/tmp/pp; rm -rf $O; mkdir -p $O
cat > $O/w.sh <<'W'
#!/bin/bash
if [ "$LOCAL_RANK" = "0" ]; then exec rocprofv3 --kernel-trace --output-format csv -d /tmp/pp -o r0 -- "$@"; else exec "$@"; fi
W
chmod +x $O/w.sh
cd /sgl-workspace/mori/tests/python/ops/dispatch_combine_v2
env PYTHONPATH=/sgl-workspace/mori/python MORI_SOURCE_ROOT=/sgl-workspace/mori \
    MORI_V2_LIB_DIR=/sgl-workspace/mori/python/mori \
    BACKENDS=${BK:-hip,hip_ll} COMB_MODE=pull SWEEP=${SW:-1,32,128,4096} ITERS=${IT:-200} WARMUP=20 MODES=graph CHECK=0 \
    python3 -m torch.distributed.run --standalone --nproc_per_node=8 --no-python $O/w.sh python3 bench_ep.py > $O/log 2>&1 || true
python3 - <<'K'
import csv,statistics as st,collections
rows=list(csv.DictReader(open('/tmp/pp/r0_kernel_trace.csv')))
d=collections.defaultdict(list)
for r in rows:
    n=r.get("Kernel_Name") or ""
    if "mori_ep_dispatch" not in n: continue
    try: d[n].append((int(r["End_Timestamp"])-int(r["Start_Timestamp"]))/1000.0)
    except: pass
for n,v in sorted(d.items()):
    v=sorted(v); print(f"{n[:46]:48s} n={len(v):5d} min={v[0]:8.2f} p10={v[len(v)//10]:8.2f} med={st.median(v):8.2f}")
K
