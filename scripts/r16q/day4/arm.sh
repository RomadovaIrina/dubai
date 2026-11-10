#!/bin/bash
# arm.sh SET ID SRC FINAL : defects + all-seconds scan for one arm render; prints one summary line
SET=$1; ID=$2; SRC=$3; FIN=$4; Q=/workspace/dub/q; cd $Q
./run1.sh $ID "$SRC" "$FIN" $SET
/workspace/dub/qenv/bin/python scripts/r16q/scan_breaks.py --series out/$SET/$ID.series.npz --final "$FIN" --src "$SRC" --id $ID --out out/$SET/scan < /dev/null > /dev/null 2>&1
python3 - $SET $ID <<'PY'
import json,sys; S,I=sys.argv[1:]; r=json.load(open(f"out/{S}/{I}.json")); b=json.load(open(f"out/{S}/scan/{I}.breaks.json"))
bk=json.load(open(f"out/{S}/{I}.det.npz.det.json"))["det"]["black"]; g=r["ghost"]; gs=sum(e-s for s,e,_ in g["runs"]); top=[(m["t"],m["rank"],"/".join(m["classes"])) for m in b["moments"] if m["rank"]>=1.2][:5]
print(f"{S}/{I}: black {bk['flag_frames']}fr share {bk.get('black_share_ours')}/src {bk.get('black_share_src')} | seam {r['seam']['excess_median']} p95 {r['seam']['excess_p95']} | ghost {gs:.1f}s p95 {g['excess_p95']} | tail {r['tail']['local_max']} | soft {len(r['mouth']['soft_runs'])} thin {len(r['mouth']['lip_thin_runs'])} | moments>=1.2 {sum(m['rank']>=1.2 for m in b['moments'])} top {top}")
PY
