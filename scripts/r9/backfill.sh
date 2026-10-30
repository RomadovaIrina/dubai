#!/bin/bash
# r9: gate rows that the metrics stage did not have yet when a run was made (faith judge, pressed lips, face_gate layout), then manifest.json
# again with the run's own status / flags / timing. Measurement only: no production stage is re-run.  bash scripts/r9/backfill.sh <run dir> ...
set -u
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; R=$DUB_ROOT; . $R/scripts/env.sh >/dev/null; DUB_ROOT=$R; export DUB_ROOT; cd $DUB_ROOT
A=/workspace/dub/artifacts/r4a; B=/workspace/dub/artifacts/r4m; export DUB_ALLOW_NC=1
for D in "$@"; do
  M=$D/manifest.json; ID=$(python3 -c "import json; print(json.load(open('$M'))['id'])"); AT=r9_$ID; HT=r9_${ID}_htd; FT=r9f_$ID; W=$D/work
  export DUB_INPUTS=$W/in
  python3 -c "import json, sys; m = json.load(open('$A/$AT/$ID.metrics.json')); sys.exit(0 if m.get('faith') else 1)" 2>/dev/null \
    || PYTHONPATH=third_party/llama_cuda:${PYTHONPATH:-} python scripts/r5/r5_faith.py $AT $ID > $A/$AT/runs/faith.log 2>&1 || echo "WARN faith $ID"
  if [ -f $B/$FT/${ID}_$FT.mp4 ]; then
    [ -f $W/closed_mouth.json ] || python scripts/r7/r7_closed_mouth.py --ids $ID --variants $FT:$A/$AT --out $W/closed_mouth.json > /dev/null 2>&1
    python scripts/r7/r7_baseline.py freeze --tag $FT --audio $A/$AT --ids $ID --out $W/face_gate.json > /dev/null 2>&1
  fi
  eval "$(python3 -c "
import json, shlex; m = json.load(open('$M'))
print('ST=' + shlex.quote(m['status']), 'FLG=' + shlex.quote(' '.join(f for f in m['flags'] if f != 'output_format_differs_from_source')), 'ERR=' + shlex.quote('\n'.join(m.get('errors') or [])), 'IN=' + shlex.quote(m['input']))")"
  python scripts/r9/collect.py --id $ID --out $D --status "$ST" --flags "$FLG" --errors "$ERR" --stages $W/stages.tsv --nvsmi $W/nvsmi.csv --precheck $W/precheck.json \
    --audio $A/$AT --htd $A/$HT --face $B/$FT --input "$IN" | tail -1
done
