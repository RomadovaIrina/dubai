#!/bin/bash
# r7 F: m7 candidate = the m6 run's raw LatentSync layers (no generator rerun) + r7_layer_sr.py (Real-ESRGAN x2 on the layer's lips, temporal +-1)
# + the m6 composite with the source skin texture at full confidence (--tex-tau 1000 --tex-sigma 2.5). Output in the r4m layout so that
# r7_watch.sh / r4_gate.py compare / r6_twitch.sh work unchanged: /workspace/dub/artifacts/r4m/<TAG>/<id>_<TAG>.mp4 + .manifest.json + score.
#   TAG=m7 SRC_TAG=m6 bash scripts/r7/r7_m7_comp.sh [ids]     env: DFK=<label> (r8: r8_layer_deflicker.py after the SR, DFK_ARGS), SR=srt1 (layer suffix made by r7_layer_sr.py --temporal 1; "" = the raw layer), EXTRA (composite flags)
set -u
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DUB_ROOT; . $DUB_ROOT/scripts/env.sh >/dev/null; export DUB_ROOT; cd $DUB_ROOT
TAG=${TAG:-m7}; ST=${SRC_TAG:-m6}; SR=${SR-srt1}; EXTRA=${EXTRA:---tex-tau 1000 --tex-sigma 2.5}; B=/workspace/dub/artifacts/r4m; mkdir -p $B/$TAG; ln -sfn $B/$ST/timing $B/$TAG/timing
V1="--register sim_ring --ring-region upper --edge-px 12 --colour-match upper --sharpen-auto 0.95 --skin-texture source --lip-guard on --retime flow_safe --face-clip on --sanitize-layer 0.83 --sanitize-mode texture --sanitize-sigma 8 --sharpen-lips-max 0.8 --sanitize-guard on --mouth-region 0.38 --paste-guard 0.5"
echo "r7_m7_comp $TAG from $ST layers (SR=$SR) EXTRA=$EXTRA commit $(git rev-parse --short HEAD)" >> $B/$TAG/README.txt
for id in ${*:-03 01 05 02 04}; do
  t0=$(date +%s); W=$B/$ST/work_${id}_$ST; M=$W/${id}_${ST}_25fps.manifest.json; [ -f $M ] || { echo "no $M"; continue; }
  nls=$(python -c "import json;print(json.load(open('$M'))['lipsync']['by_action'].get('LATENT_SYNC',{}).get('frames',0))")
  if [ "$nls" -gt 0 ]; then
    RAW=$W/${id}_${ST}_25fps_ls_raw.mp4; LS=$RAW; SRN=$W/${id}_${ST}_25fps_ls_$SR.mp4
    if [ -n "${CFD:-}" ]; then   # r8 F: CodeFormer DETAIL only: CFD=<clean base tag of the same seeded run> CFD_LABEL CFD_ARGS (r8_cf_detail.py)
      if [ "$CFD" = self ]; then BASE=$W/${id}_${ST}_25fps_ls_pre.mp4; BLM=$W/${id}_${ST}_25fps_ls_raw.mediapipe.npz   # this run's pre-CodeFormer dump
      else BW=$B/$CFD/work_${id}_$CFD; BASE=$BW/${id}_${CFD}_25fps_ls_raw.mp4; BLM=$BW/${id}_${CFD}_25fps_ls_raw.mediapipe.npz; fi
      RAW=$W/${id}_${ST}_25fps_ls_cfd_${CFD_LABEL:-cfd}.mp4; SRN=${RAW%.mp4}_$SR.mp4
      for try in 1 2; do [ -s $RAW ] && break; [ $try = 2 ] && echo "== RETRY cf_detail $id"; python scripts/r8/r8_cf_detail.py --base $BASE --cf $W/${id}_${ST}_25fps_ls_raw.mp4 --lm $BLM --out $RAW ${CFD_ARGS:-} | grep "^=="; done; [ -f $RAW ] && cp $BLM ${RAW%.mp4}.mediapipe.npz   # r9: one retry (sporadic libx264 open EINVAL, not reproducible)
      LS=$RAW
    fi
    if [ -n "$SR" ]; then
      LS=$SRN
      for try in 1 2; do [ -s $LS ] && break; [ $try = 2 ] && echo "== RETRY layer_sr $id"; python scripts/r7/r7_layer_sr.py --ls-raw $RAW --lm ${RAW%.mp4}.mediapipe.npz --out $LS --temporal 1 ${SR_EXTRA:-} | grep "^==" ; done; [ -f $LS ] && cp ${RAW%.mp4}.mediapipe.npz ${LS%.mp4}.mediapipe.npz
    fi
    if [ -n "${DFK:-}" ]; then   # r8 F: layer tone deflicker (inner mouth + lower face), cached per DFK label: DFK=dfk1 DFK_ARGS="--sigma 1.5 ..."
      LSD=${LS%.mp4}_$DFK.mp4; [ -f $LSD ] || { python scripts/r8/r8_layer_deflicker.py --ls $LS --lm ${LS%.mp4}.mediapipe.npz --out $LSD ${DFK_ARGS:-} | grep "^=="; cp ${LS%.mp4}.mediapipe.npz ${LSD%.mp4}.mediapipe.npz; }; LS=$LSD
    fi
    FBX=""; if [ -n "${FB:-}" ]; then   # r8 F: confidence fallback to the source mouth, FB=<label> FB_ARGS="--thr-wash ..." (score on the raw layer, cached)
      FBN=$W/${id}_${ST}_25fps_fb_$FB.npz; [ -f $FBN ] || python scripts/r8/r8_fallback.py --work $W --ls $RAW --lm ${RAW%.mp4}.mediapipe.npz --out $FBN ${FB_ARGS:-} | grep "^=="; FBX="--fallback $FBN"
    fi
    if [ -n "${REP:-}" ] && [ -n "$FBX" ]; then   # r9: short glitches re-drawn IN the layer (flow interpolation of the good neighbours) instead of a 1-frame swap to the source mouth
      RL=${REP_LABEL:-rep2}; LSR=${LS%.mp4}_$RL.mp4; FBR=${FBN%.npz}_${TAG}_$RL.npz
      [ -s $LSR ] || python scripts/r9/r9_layer_repair.py --ls $LS --lm ${LS%.mp4}.mediapipe.npz --fb $FBN --src-lm $W/video/master_mediapipe.npz --src-video $W/video/master_25fps.mp4 --out $LSR --fb-out $FBR ${REP_ARGS:-} | grep "^=="
      [ -s $LSR ] && { LS=$LSR; FBX="--fallback $FBR"; }
    fi
    FIX30_LS=$LS FIX30_MASTER_FPS=${LS_FPS:-25} FIX30_TD=$B/$ST/timing FIX30_MODE=r3fix FIX30_THREADS=${FIX30_THREADS:-16} FIX30_ARGS="$V1 $EXTRA $FBX" bash scripts/quality/r3_fix30.sh $M $TAG > $B/$TAG/${id}.fix30.log 2>&1
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
  if [ -z "${SKIP_SCORE:-}" ]; then   # r9: SKIP_SCORE=1 = production only (e2e.sh scores in its metrics stage)
  python scripts/r4/r4_gate.py score --label $TAG --dir $B/$TAG --ids $id > $B/$TAG/${id}.score.log 2>&1 || echo "WARN score $id"
  MAN=$B/$TAG/%s_$TAG.manifest.json bash scripts/r6/r6_twitch.sh $TAG $B/$TAG _$TAG $id | grep '=='
  bash scripts/r7/r7_frame_gate.sh $B/$TAG:$TAG $id | grep "^=="
  fi
  echo "== $TAG $id $(( $(date +%s) - t0 )) s $(date +%T)"
done
echo R7_M7_DONE $TAG
