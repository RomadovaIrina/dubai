#!/bin/bash
# r13 final speed default = f_s8 + KS_LMK_CACHE (k8i7 failed old03 flashes): the 3 management finals rebuilt from the f_s8lmk KeySync renders
# (8 steps, the same dub audio) with cfhf (KS_CMS, end-fade 12, skin-cm 12 robust, guard-fade 10, KS_HF 3, KS_SHARP 0.3);
# m18 / m03 alpha 0.8, old03 both alpha 0 (cfhf) and 0.8 (cfts). Gate numbers per final.
cd /workspace/dub/keysync; K=/workspace/dub/keysync; OUT=/workspace/dub/outbox/r13_ks/final_s8; mkdir -p $OUT
run() {  # <id> <src> <name> <ks render> <alpha>
  id=$1; SRC=$2; NAME=$3; R=$4; TS=$5; O=$K/final/${id}_s8_$NAME; mkdir -p $O/ks/$id qc/$id; ln -sfn $R $O/ks/$id/${id}_$id.mp4; cp final/$id/src.mediapipe.npz $O/
  KS_CMS=1 KS_HF=3 KS_SHARP=0.3 KS_REDO=1 KS_TSMOOTH=$TS KS_FINALDIR=$O KS_OUTDIR=$OUT KS_LAYERARGS="--end-fade 12" KS_FIXARGS="--skin-cm 12 --cm-robust 1 --guard-fade 10" \
    bash ks_final.sh $id $SRC $NAME > $O/final.log 2>&1 || { echo "FAIL $NAME"; tail -5 $O/final.log; return; }
  F=$OUT/${NAME}_KeySync.mp4; echo "READY $F $(stat -c %s $F)"; A=/workspace/dub/artifacts/r4a/r9_${id}_htd/$id
  FPS=$(ffprobe -v error -select_streams v:0 -show_entries stream=r_frame_rate -of csv=p=0 $SRC | python3 -c "import sys; n, d = sys.stdin.read().strip().split(\"/\"); print(float(n) / float(d))")
  ( cd /workspace/dub/dubai && /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video $F --out $O/out.mediapipe.npz > /dev/null 2>&1 )
  /venv/dabai/bin/python ks_colorcheck.py --a $F --b $SRC --box data/box_$id.json --lm $O/out.mediapipe.npz --out qc/$id/${NAME}.json | head -1 | sed "s/^/CC $NAME /" | cut -c1-400
  SH=$(python3 -c "import json; print(-int(json.load(open(\"$O/raw_sync.json\"))[\"global\"][\"av_offset\"]))")
  /venv/dabai/bin/python ks_lag.py --lm $O/out.mediapipe.npz --src-lm $O/src.mediapipe.npz --fps $FPS --shift $SH --out $O/lag | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print(\"LAG $NAME\", d[\"spike_rate\"], \"src\", d[\"source_rate\"])"
  ( cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
    python scripts/quality/r3_flash.py --video $F --src $SRC --lm /workspace/dub/artifacts/r4m/r9g_$id/timing/${id}_source_lm_F.npz --out $O/flash.json > /dev/null 2>&1
    python3 -c "import json; f=json.load(open(\"$O/flash.json\")); print(\"FLASH $NAME\", sum(1 for e in f[\"flash_list\"] if (e.get(\"mc_ratio\") or 0) >= 2.0 and (e.get(\"mc_vs_src\") or 0) >= 2.0))"
    ffmpeg -v error -y -i $F -vf fps=25 -an -c:v libx264 -crf 12 /tmp/s25_$NAME.mp4
    flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video /tmp/s25_$NAME.mp4 --audio $A/dubbed_16k.wav --out /tmp/s_$NAME.json > /dev/null 2>&1
    python3 -c "import json,statistics as st; d=json.load(open(\"/tmp/s_$NAME.json\")); sp=[p[\"conf_at0\"] for p in d[\"per_second\"] if p.get(\"conf_at0\") is not None and (p.get(\"dub_act\") or 0)>=0.5]; print(\"SYNC $NAME\", d[\"global\"], round(st.mean(sp),3))" )
}
run q18 /workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4 m18_s8cfhf $K/fast/f_s8lmk/out/q18/q18_q18.mp4 0.8
run m03 /workspace/dub/inputs/mamakris/story-1774249919482-367803614.MOV m03_s8cfhf $K/fast/f_s8lmk/out/m03/m03_m03.mp4 0.8
run qo3 /workspace/dub/inputs/test_videos_raw/03.mp4 old03_s8cfhf $K/fast/f_s8lmk_o3/out/qo3/qo3_qo3.mp4 0
run qo3 /workspace/dub/inputs/test_videos_raw/03.mp4 old03_s8cfts $K/fast/f_s8lmk_o3/out/qo3/qo3_qo3.mp4 0.8
ls -la $OUT; echo S8FINAL_DONE
