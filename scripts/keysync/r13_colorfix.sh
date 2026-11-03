#!/bin/bash
# r13 viewer feedback "the picture COLOUR changes a couple of times": old03 v1 (no smoothing) rebuilt with the colour fix, all behind flags:
#   KS_CMS=1 (BT.709 colour-managed I/O: no ~1.7 L loss per re-encode), ks_layer --end-fade 12, ks_fix --skin-cm 12 --cm-robust 1 --guard-fade 10
# then the gate numbers (SyncNet speech at0, strong flashes, jerk) + the colour check vs the source, for the fixed and the v1 file.
#   r13_colorfix.sh [tag] (env CF_LAYER / CF_FIX override the flags)
cd /workspace/dub/keysync; T=${1:-cf}; id=qo3; SRC=/workspace/dub/inputs/test_videos_raw/03.mp4; OUT=/workspace/dub/outbox/r13_ks/colorfix; A=/workspace/dub/artifacts/r4a/r9_${id}_htd/$id
O=/workspace/dub/keysync/final/${id}_$T; mkdir -p $O $OUT qc/$id; ln -sfn /workspace/dub/keysync/final/$id/ks $O/ks; cp final/$id/raw_sync.json final/$id/src.mediapipe.npz $O/
KS_CMS=1 KS_TSMOOTH=${CF_TS:-0} KS_REDO=${CF_REDO-1} KS_FINALDIR=$O KS_OUTDIR=$OUT KS_LAYERARGS="${CF_LAYER:---end-fade 12}" KS_FIXARGS="${CF_FIX:---skin-cm 12 --cm-robust 1 --guard-fade 10}" \
  bash ks_final.sh $id $SRC old03_$T > $O/final.log 2>&1 || { echo FINAL_FAIL; tail -5 $O/final.log; exit 1; }
grep -a "==" $O/final.log | cut -c1-200; F=$OUT/old03_${T}_KeySync.mp4; echo "FINAL_READY $F $(stat -c %s $F)"
meas() {  # <video> <tag>
  ( cd /workspace/dub/dubai && /venv/mediapipe/bin/python scripts/quality/mp_landmarks.py --video $1 --out $O/$2.mediapipe.npz > /dev/null 2>&1 )
  /venv/dabai/bin/python ks_colorcheck.py --a $1 --b $SRC --box data/box_$id.json --lm $O/$2.mediapipe.npz --out qc/$id/$2.json | sed "s/^/CC $2 /" | cut -c1-600
  SH=$(python3 -c "import json; print(-int(json.load(open('$O/raw_sync.json'))['global']['av_offset']))")
  /venv/dabai/bin/python ks_lag.py --lm $O/$2.mediapipe.npz --src-lm $O/src.mediapipe.npz --fps 30 --shift $SH --out $OUT/$2_lag | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print('LAG $2', d['spike_rate'], 'src', d['source_rate'])"
  ( cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1
    python scripts/quality/r3_flash.py --video $1 --src $SRC --lm /workspace/dub/artifacts/r4m/r9g_$id/timing/${id}_source_lm_F.npz --out $O/$2_flash.json > /dev/null 2>&1
    python3 -c "import json; f=json.load(open('$O/$2_flash.json')); print('FLASH $2', sum(1 for e in f['flash_list'] if (e.get('mc_ratio') or 0) >= 2.0 and (e.get('mc_vs_src') or 0) >= 2.0))"
    ffmpeg -v error -y -i $1 -vf fps=25 -an -c:v libx264 -crf 12 /tmp/s25_$2.mp4
    flock /workspace/dub/gpu.lock python scripts/quality/sync_perframe.py --video /tmp/s25_$2.mp4 --audio $A/dubbed_16k.wav --out /tmp/s_$2.json > /tmp/s_$2.log 2>&1
    python3 -c "import json,statistics as st; d=json.load(open('/tmp/s_$2.json')); sp=[p['conf_at0'] for p in d['per_second'] if p.get('conf_at0') is not None and (p.get('dub_act') or 0)>=0.5]; print('SYNC $2', d['global'], round(st.mean(sp),3))" )
}
meas $F old03_$T
[ -s qc/$id/v1_flash_done ] || { meas /workspace/dub/outbox/r13_ks/v1check/old03_v1_KeySync.mp4 old03_v1; touch qc/$id/v1_flash_done; }
echo COLORFIX_DONE $T
