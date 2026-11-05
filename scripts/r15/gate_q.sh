# r15 F: the gates one at a time WITHOUT gpu.lock (coordinator OK: SyncNet only, <= 3 GB; wait while < 4 GB VRAM is free), fix (cf) vs control (c0)
cd /workspace/dub/dubai-f; . scripts/env.sh >/dev/null; B=/workspace/dub/artifacts/r4m; G=/workspace/dub/r15f/gate
free_ok() { f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1); [ "${f:-0}" -ge 4096 ]; }
for r in q2_m03 q2_old03 q2_m18; do
  for t in c0 cf; do
    until [ -s $B/r9f_$r$t/${r}_r9f_$r$t.mp4 ] && ! pgrep -f "comp_f.sh $r $t" > /dev/null; do sleep 20; done
    until free_ok; do echo "waiting for VRAM $(date +%T)"; sleep 30; done
    echo "== gate $r$t start $(date +%T)"; bash scripts/r15/gate_f.sh $r $t > /workspace/dub/r15f/logs/gate_$r$t.log 2>&1; tail -n 3 /workspace/dub/r15f/logs/gate_$r$t.log | cut -c1-300
  done
  python scripts/r10/gate_cmp.py --base $G/${r}c0 --cand $G/${r}cf --md /workspace/dub/r15f/gate.md | head -n 3
  for t in c0 cf; do python - $G/$r$t/art.json <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); f = d.get("final_events", [])
print("   ", sys.argv[1].split("/")[-2], "final events:", [(e["class"], e["t"]) for e in f])
PY
  done
done
echo GATE_Q_DONE
