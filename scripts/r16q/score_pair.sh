#!/bin/bash
# r16 Q: score a paste variant against its current-default twin on ONE clip (K1 --align-scale etc.):
#   speech@0 (eval_final.sh, no noise pass: use the golden noise +-0.01-0.05), spring ratio (r16q_spring.py, the SAME region for both),
#   chin slide_ch + slide (r16q_detect.py --only slide). Prints one PAIR line and writes OUT/pair.json.
#   bash scripts/r16q/score_pair.sh ID VARIANT.mp4 TWIN.mp4 MASTER25.mp4 DUB16K.wav OUT [--region x0,y0,x1,y1] [--route route.json --route-wh W,H]
set -u
ID=$1; V=$(realpath "$2"); T=$(realpath "$3"); M=$(realpath "$4"); D=$(realpath "$5"); O=$(realpath -m "$6"); shift 6
REG=(); RT=()
while [ $# -gt 0 ]; do case "$1" in --region) REG=(--region "$2"); shift 2;; --route|--route-wh) RT+=("$1" "$2"); shift 2;; *) echo "unknown $1"; exit 2;; esac; done
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$R"; PY=/venv/dabai/bin/python; MP=/venv/mediapipe/bin/python; mkdir -p "$O"
for w in var twin; do F=$V; [ $w = twin ] && F=$T
  bash scripts/r16q/eval_final.sh "$F" "$D" "$O/eval_$w" --noise 0 "${RT[@]}" --label ${ID}_$w > "$O/eval_$w.log" 2>&1 &
  nice $PY scripts/quality/r16q_spring.py --final "$F" --master "$M" "${REG[@]}" --out "$O/spring_$w.json" > "$O/spring_$w.log" 2>&1 &
  nice $MP scripts/quality/r16q_detect.py --src "$M" --ours "$F" --out "$O/det_$w.json" --npz "$O/det_$w.npz" --only slide --label ${ID}_$w > "$O/det_$w.log" 2>&1 &
done; wait
python3 - "$O" "$ID" <<'PY'
import json, sys
import numpy as np
O, ID = sys.argv[1], sys.argv[2]; r = {"id": ID}
for w in ("var", "twin"):
    e = json.load(open(f"{O}/eval_{w}/eval.json")); s = json.load(open(f"{O}/spring_{w}.json")); z = np.load(f"{O}/det_{w}.series.npz")
    r[w] = {"speech_at0": e["speech_at0"], "status": e["status"], "spring_ratio": s.get("spring_ratio"), "spring": s["final_metric"]["spring_scale_pct"],
            "floor": s.get("floor_metric", {}).get("spring_scale_pct"), "upper_ratio": s.get("upper_ratio"),
            "chin_median": round(float(np.nanmedian(z["slide_ch"])), 3), "chin_p95": round(float(np.nanpercentile(z["slide_ch"], 95)), 3),
            "slide_flags": int(np.nansum(z["slide"] > 4.0))}
v, t = r["var"], r["twin"]
r["d_speech"] = round(v["speech_at0"] - t["speech_at0"], 3) if None not in (v["speech_at0"], t["speech_at0"]) else None
r["d_spring_ratio"] = round(v["spring_ratio"] - t["spring_ratio"], 3) if None not in (v["spring_ratio"], t["spring_ratio"]) else None
r["d_chin_median"] = round(v["chin_median"] - t["chin_median"], 3)
r["verdict"] = ("speech_drop" if r["d_speech"] is not None and r["d_speech"] < -0.1 else "ok_speech") + "/" + \
               ("spring_down" if r["d_spring_ratio"] is not None and r["d_spring_ratio"] < -0.1 else "spring_same_or_up")
json.dump(r, open(f"{O}/pair.json", "w"), indent=1)
print("PAIR " + json.dumps({k: r[k] for k in ("id", "d_speech", "d_spring_ratio", "d_chin_median", "verdict")} | {"var": {k: v[k] for k in ("speech_at0", "spring_ratio", "chin_median")}, "twin": {k: t[k] for k in ("speech_at0", "spring_ratio", "chin_median")}}))
PY
