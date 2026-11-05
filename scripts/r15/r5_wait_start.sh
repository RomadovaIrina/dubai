#!/bin/bash
# r15-w: wait until all 1817 r5 windows have arrived (cap_ext/ + own/cap/) and load, then start r5_chain.sh within the budget
# (coordinator: <= 4.5 GPU-h -> 1500 train steps, E2E old03 + m03)
X=/workspace/dub/r15x; L=/workspace/dub/dubai-d/scripts/r15/r5_sel_list.txt
for i in $(seq 1 360); do
  miss=0; while read rel; do [ -s $X/$rel ] || miss=$((miss + 1)); done < $L
  echo "$(date +%T) missing $miss / 1817 (cap_ext $(ls $X/cap_ext 2>/dev/null | wc -l), own/cap $(ls $X/own/cap 2>/dev/null | wc -l))"
  if [ $miss = 0 ]; then
    sleep 60   # a last file may still be writing
    if /venv/dabai/bin/python - "$X" "$L" <<'EOF'
import sys, torch
X, L = sys.argv[1], sys.argv[2]; bad = []
for rel in open(L).read().split():
    try: d = torch.load(f"{X}/{rel}", map_location="cpu", weights_only=False); assert "cond" in d and "audio" in d
    except Exception as e: bad.append(rel)
print(f"load check: {len(bad)} bad", bad[:3]); sys.exit(1 if bad else 0)
EOF
    then break; fi
  fi
  sleep 60
done
[ $miss = 0 ] || { echo "data not complete after 6 h"; exit 1; }
echo "$(date +%T) data ok -> r5_chain"
cd /workspace/dub/dubai-d && R5_STEPS=1500 R5_IDS=m03 exec bash scripts/r15/r5_chain.sh
