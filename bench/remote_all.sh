#!/bin/bash
# runs INSIDE the container on the Crusoe node.
set -o pipefail
log(){ echo "[$(date +%H:%M:%S)] $*"; }

# ---------- 1. trees ----------
cd /sgl-workspace/mori || exit 1
git remote get-url kc >/dev/null 2>&1 || git remote add kc https://github.com/kawhil-amd/mori.git
log "fetching"; git fetch -q kc feat/epv2-ll-dispatch exp/epv2-notify-tickets || { log FETCH_FAIL; exit 1; }
git checkout -q -f -B llbase kc/feat/epv2-ll-dispatch || { log CHECKOUT_FAIL; exit 1; }
rm -rf /sgl-workspace/mori-nt; git worktree prune
git worktree add -q -f -B ntbr /sgl-workspace/mori-nt kc/exp/epv2-notify-tickets || { log WORKTREE_FAIL; exit 1; }
log "trees: $(git log --oneline -1) | $(git -C /sgl-workspace/mori-nt log --oneline -1)"

# ---------- 2. build ----------
log "ninja"; ninja -C build 2>&1 | tail -2
log "build_ext"; python3 setup.py build_ext --inplace > /tmp/bext.log 2>&1 || { log BEXT_FAIL; tail -15 /tmp/bext.log; exit 1; }
cmp -s build/src/cco/libmori_cco.so python/mori/libmori_cco.so && log LIBS_SYNCED || log LIBS_DIFFER
python3 -c "import mori, torch; print('mori ok, gpus', torch.cuda.device_count())" || { log IMPORT_FAIL; exit 1; }

# ---------- 3. harness ----------
cat > /tmp/prof.sh <<'P'
#!/bin/bash
# prof.sh "<env>" <tok> <srcroot> <outdir>
env_s="$1"; m="$2"; root="$3"; out="$4"
rm -rf "$out"; mkdir -p "$out"
cat > "$out/wrap.sh" <<'W'
#!/bin/bash
if [ "$LOCAL_RANK" = "0" ]; then exec rocprofv3 --kernel-trace --output-format csv -d "$PROF_OUT" -o r0 -- "$@"; else exec "$@"; fi
W
chmod +x "$out/wrap.sh"
cd /sgl-workspace/mori/tests/python/ops/dispatch_combine_v2
env PYTHONPATH=/sgl-workspace/mori/python MORI_SOURCE_ROOT="$root" MORI_V2_LIB_DIR=/sgl-workspace/mori/python/mori PROF_OUT="$out" $env_s \
  COMB_MODE=pull SWEEP=$m ITERS=100 WARMUP=20 MODES=graph CHECK=0 \
  python3 -m torch.distributed.run --standalone --nproc_per_node=8 --no-python "$out/wrap.sh" python3 bench_ep.py > "$out/log" 2>&1
P
chmod +x /tmp/prof.sh
cat > /tmp/kt.py <<'K'
import csv,sys,statistics as st
rows=list(csv.DictReader(open(sys.argv[1])))
v=[]
for r in rows:
    n=r.get("Kernel_Name") or ""
    if "dispatch" not in n: continue
    try: v.append((int(r["End_Timestamp"])-int(r["Start_Timestamp"]))/1000.0)
    except: pass
print(f"{min(v):.2f} {st.median(v):.2f} {len(v)}" if v else "NA NA 0")
K

# ---------- 4. smoke ----------
log "smoke (tok=32, LL, old notify)"
/tmp/prof.sh "LL_DISPATCH=1" 32 /sgl-workspace/mori /tmp/sm
if [ -f /tmp/sm/r0_kernel_trace.csv ]; then log "SMOKE_OK $(python3 /tmp/kt.py /tmp/sm/r0_kernel_trace.csv)"; else log SMOKE_FAIL; tail -20 /tmp/sm/log; exit 1; fi
log READY
