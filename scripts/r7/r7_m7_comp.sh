#!/bin/bash
# r7 F: m7 candidate = the m6 run's raw LatentSync layers (no generator rerun) + r7_layer_sr.py (Real-ESRGAN x2 on the layer's lips, temporal +-1)
# + the m6 composite with the source skin texture at full confidence (--tex-tau 1000 --tex-sigma 2.5). Output in the r4m layout so that
# r7_watch.sh / r4_gate.py compare / r6_twitch.sh work unchanged: /workspace/dub/artifacts/r4m/<TAG>/<id>_<TAG>.mp4 + .manifest.json + score.
#   TAG=m7 SRC_TAG=m6 bash scripts/r7/r7_m7_comp.sh [ids]     env: SR=srt1 (layer suffix made by r7_layer_sr.py --temporal 1; "" = the raw layer), EXTRA (composite flags)
set -u
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DUB_ROOT; . $DUB_ROOT/scripts/env.sh >/dev/null; export DUB_ROOT; cd $DUB_ROOT
TAG=${TAG:-m7}; ST=${SRC_TAG:-m6}; SR=${SR:-srt1}; EXTRA=${EXTRA:---tex-tau 1000 --tex-sigma 2.5}; B=/workspace/dub/artifacts/r4m; mkdir -p $B/$TAG; ln -sfn $B/$ST/timing $B/$TAG/timing
V1="--register sim_ring --ring-region upper --edge-px 12 --colour-match upper --sharpen-auto 0.95 --skin-texture source --lip-guard on --retime flow_safe --face-clip on --sanitize-layer 0.83 --sanitize-mode texture --sanitize-sigma 8 --sharpen-lips-max 0.8 --sanitize-guard on --mouth-region 0.38 --paste-guard 0.5"
echo "r7_m7_comp $TAG from $ST layers (SR=$SR) EXTRA=$EXTRA commit $(git rev-parse --short HEAD)" >> $B/$TAG/README.txt
for id in ${*:-03 01 05 02 04}; do
  t0=$(date +%s); W=$B/$ST/work_${id}_$ST; M=$W/${id}_${ST}_25fps.manifest.json; [ -f $M ] || { echo "no $M"; continue; }
  nls=$(python -c "import json;print(json.load(open('$M'))['lipsync']['by_action'].get('LATENT_SYNC',{}).get('frames',0))")
  if [ "$nls" -gt 0 ]; then
    LS=$W/${id}_${ST}_25fps_ls_raw.mp4
    if [ -n "$SR" ]; then
      LS=$W/${id}_${ST}_25fps_ls_$SR.mp4
      [ -f $LS ] || { python scripts/r7/r7_layer_sr.py --ls-raw $W/${id}_${ST}_25fps_ls_raw.mp4 --lm $W/${id}_${ST}_25fps_ls_raw.mediapipe.npz --out $LS --temporal 1 ${SR_EXTRA:-} | grep "^==" ; cp $W/${id}_${ST}_25fps_ls_raw.mediapipe.npz ${LS%.mp4}.mediapipe.npz; }
    fi
    FIX30_LS=$LS FIX30_MASTER_FPS=25 FIX30_TD=$B/$ST/timing FIX30_MODE=r3fix FIX30_THREADS=${FIX30_THREADS:-16} FIX30_ARGS="$V1 $EXTRA" bash scripts/quality/r3_fix30.sh $M $TAG > $B/$TAG/${id}.fix30.log 2>&1
    F=$W/${id}_${ST}_25fps_$TAG.mp4; [ -f $F ] || { echo "FAIL composite $id (see $B/$TAG/${id}.fix30.log)"; continue; }
    mv $F $B/$TAG/${id}_$TAG.mp4; mv $W/${id}_${ST}_25fps_$TAG.manifest.json $B/$TAG/${id}_$TAG.manifest.json; mv $W/${id}_${ST}_25fps_$TAG.fps_composite.json $B/$TAG/${id}_$TAG.fps_composite.json 2>/dev/null; mv $W/${id}_${ST}_25fps_$TAG.log $B/$TAG/${id}_$TAG.log 2>/dev/null
  else   # no LatentSync frames (02): the m6 output is the source + audio already
    cp $B/$ST/${id}_$ST.mp4 $B/$TAG/${id}_$TAG.mp4; cp $B/$ST/${id}_$ST.manifest.json $B/$TAG/${id}_$TAG.manifest.json
  fi
  python - $B/$TAG/${id}_$TAG.manifest.json $B/$ST/${id}_$ST.manifest.json $t0 "$SR" "$EXTRA" <<'PY'
import json, pathlib, sys, time
mp, m6p, t0 = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), float(sys.argv[3]); m = json.loads(mp.read_text()); m6 = json.loads(m6p.read_text())
r = dict(m6.get("r4v") or {}); st = dict(r.get("stage_wall_s") or {}); st["layer_sr_and_composite"] = round(time.time() - t0, 1); r["stage_wall_s"] = st; r["note"] = f"m7 = m6 layers + layer SR '{sys.argv[4]}' + composite {sys.argv[5]}"
vmin = m["source"]["duration_s"] / 60; r["s_per_video_min"] = {k: round(v / vmin, 1) for k, v in st.items() if isinstance(v, (int, float))}; m["r4v"] = r; mp.write_text(json.dumps(m, indent=1))
print("==", mp.stem, "layer SR + composite", st["layer_sr_and_composite"], "s =", round(st["layer_sr_and_composite"] / vmin), "s/video-min")
PY
  python scripts/r4/r4_gate.py score --label $TAG --dir $B/$TAG --ids $id > $B/$TAG/${id}.score.log 2>&1 || echo "WARN score $id"
  MAN=$B/$TAG/%s_$TAG.manifest.json bash scripts/r6/r6_twitch.sh $TAG $B/$TAG _$TAG $id | grep '=='
  bash scripts/r7/r7_frame_gate.sh $B/$TAG:$TAG $id | grep "^=="
  echo "== $TAG $id $(( $(date +%s) - t0 )) s $(date +%T)"
done
echo R7_M7_DONE $TAG
