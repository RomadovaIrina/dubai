#!/bin/bash
# r9 speed (FACE): the face stages of scripts/r9/e2e.sh (face_gen + face_comp, E2E_FX=1 = "r9 + fix") on a FINISHED audio handoff
# (artifacts/r4a/r9_<id> + r9_<id>_htd), timed per stage, both beds muxed, then the face gate rows. One video at a time on the GPU.
#     V=<variant suffix> bash scripts/r9/face_only.sh <id> [<id> ...]      env: the lever switches (GEN_COMP=0, ...), METRICS=off (timing only)
#   -> artifacts/r4m/r9f_<id><V>/<id>_r9f_<id><V>.mp4, /workspace/dub/r9/face/<id><V>/{timing.json, <id>_{demucs,spleeter}.mp4, logs/}
set -u
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; R=$DUB_ROOT; . $R/scripts/env.sh >/dev/null; DUB_ROOT=$R; export DUB_ROOT; cd $DUB_ROOT
export DUB_ALLOW_NC=1 DUB_ALLOW_NC_BED=1
A=/workspace/dub/artifacts/r4a; B=/workspace/dub/artifacts/r4m; V=${V:-}
for ID in "$@"; do
  AT=r9_$ID; HT=r9_${ID}_htd; GT=r9g_$ID$V; FT=r9f_$ID$V; OUT=/workspace/dub/r9/face/$ID$V; L=$OUT/logs; mkdir -p $L
  export DUB_INPUTS=/workspace/dub/r9/runs/$ID/work/in; SRC=$(ls $DUB_INPUTS/$ID.* | head -1)
  env | grep -E "^(GEN_COMP|FAST_|LS_|CFD_|SR_|REP|E2E_|STAB|MP_)" | sort > $L/env.txt
  echo "face_only $ID V=$V commit $(git rev-parse --short HEAD) $(date -Is) $(tr '\n' ' ' < $L/env.txt)" | tee $L/face.log
  nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 500 > $L/nvsmi.csv 2>/dev/null & NV=$!
  t0=$(date +%s.%N); rm -rf $B/$GT $B/$FT; SP=""; unset MP_SOURCE_NPZ
  SMP=$A/sep/$ID/source_mediapipe.npz   # the audio stage's single-face VIDEO-mode MediaPipe pass of the source (r8_audio.sh)
  if [ "${FAST_MPREUSE:-0}" = 1 ] && [ -f $SMP ]; then   # r9 speed: the generator's master pass from it (frame map; gated) + the composite's source pass IS it (identical)
    export MP_SOURCE_NPZ=$SMP; mkdir -p $B/$GT/timing; cp $SMP $B/$GT/timing/${ID}_source_lm_F_mp.npz; fi
  if [ "${FAST_SRCPREP:-0}" = 1 ]; then   # r9 speed L4: the source-only composite inputs (master timing, source landmarks, rigid motion) on the CPU while LatentSync runs
    WV=$B/$GT/work_${ID}_$GT/video; TD=$B/$GT/timing
    ( until [ -s $WV/master_mediapipe.npz ]; do sleep 2; done; mkdir -p $TD; T=$TD/${ID}_master_timing_F.json; SL=$TD/${ID}_source_lm_F.npz
      python scripts/quality/r3_master_timing.py --src $SRC --master $WV/master_25fps.mp4 --vf fps=25 --out $T && python scripts/quality/r3_source_lm.py --src $SRC --timing $T --out $SL \
        && python scripts/r9/r9_source_rigid.py --src $SRC --source-lm $SL ) > $L/srcprep.log 2>&1 & SP=$!
  fi
  AUDIO=$A/$AT TAG=$GT SKIP_SCORE=1 VIDEO_EXTRA="--codeformer optimized --codeformer-w 1.0 --steps 10 --steps-silent 20 --codeformer-parse off --codeformer-batch 16 ${GEN_EXTRA:-}" \
    bash scripts/r7/r7_e2e.sh $ID > $L/face_gen.log 2>&1
  [ -n "$SP" ] && wait $SP
  t1=$(date +%s.%N)
  REP=1 CFD_LABEL=ia04 CFD_ARGS="--inner-alpha 0.4 ${CFD_EXTRA:-}" SKIP_SCORE=1 TAG=$FT SRC_TAG=$GT CFD=self SR=srt1 SR_EXTRA="--batch 8 --mask down2 ${SR_MORE:-}" FB=fb7 FB_ARGS="$(cat reports/experiments/r8-face/fb7.args)" \
    bash scripts/r7/r7_m7_comp.sh $ID > $L/face_comp.log 2>&1
  t2=$(date +%s.%N); kill $NV 2>/dev/null; wait $NV 2>/dev/null
  VID=$B/$FT/${ID}_$FT.mp4
  [ -f $VID ] || { echo "== $ID$V FAIL: no $VID (see $L)" | tee -a $L/face.log; continue; }
  for v in demucs:$A/$HT/$ID/final_mix_aac.m4a spleeter:$A/$AT/$ID/final_mix_aac.m4a; do
    n=${v%%:*}; m=${v#*:}; ffmpeg -loglevel error -y -i $VID -i $m -map 0:v:0 -map 1:a:0 -c copy -movflags +faststart $OUT/${ID}_$n.mp4; done
  python - $OUT/timing.json $A/$AT/$ID $t0 $t1 $t2 $L/nvsmi.csv "$V" <<'PY'
import json, glob, sys, pathlib
out, ad, t0, t1, t2, nv, v = sys.argv[1], sys.argv[2], *map(float, sys.argv[3:6]), sys.argv[6], sys.argv[7]
m = json.load(open(glob.glob(ad + "/*_r4b.manifest.json")[0])); vmin = m["source"]["duration_s"] / 60
peak = max((int(l.split(",")[1]) for l in open(nv) if l.count(",") >= 2 and l.split(",")[1].strip().isdigit()), default=None)
st = {"face_gen": round(t1 - t0, 1), "face_comp": round(t2 - t1, 1), "face": round(t2 - t0, 1)}
r = {"variant": v, "duration_s": m["source"]["duration_s"], "seconds": st, "s_per_video_min": {k: round(x / vmin, 1) for k, x in st.items()}, "peak_vram_mib": peak}
pathlib.Path(out).write_text(json.dumps(r, indent=1)); print("== timing", json.dumps(r))
PY
  [ "${METRICS:-on}" = off ] && continue
  mkdir -p /workspace/dub/artifacts/r3d/timing; cp -n $B/$GT/timing/${ID}_* /workspace/dub/artifacts/r3d/timing/ 2>/dev/null   # r4_gate's flash / face rows read the source landmarks from this shared cache
  { MAN=$B/$FT/%s_$FT.manifest.json
    python scripts/r4/r4_gate.py score --label $FT --dir $B/$FT --ids $ID > $L/face_score.log 2>&1 || echo "WARN face score"
    MAN=$MAN bash scripts/r6/r6_twitch.sh $FT $B/$FT _$FT $ID | grep '=='
    python scripts/r9/occl_scan.py --src $SRC --out $VID --lm $B/$GT/timing/${ID}_source_lm_F.npz --sheet $OUT/occlusion_scan.jpg --json $OUT/occlusion.json | tail -1
    python scripts/r7/r7_baseline.py freeze --tag $FT --audio $A/$AT --ids $ID --out $OUT/face_gate.json | tail -2
    mkdir -p /workspace/dub/r9/diag $B/r9g_$ID/work_${ID}_r9g_$ID   # artefacts.py reads the source master from r9g_<id> (source-only data)
    [ -e $B/r9g_$ID/work_${ID}_r9g_$ID/video ] || ln -sfn $B/$GT/work_${ID}_$GT/video $B/r9g_$ID/work_${ID}_r9g_$ID/video
    L3=$(ls $B/$GT/work_${ID}_$GT/${ID}_${GT}_25fps_ls_cfd_ia04_srt1_rep*.mp4 2>/dev/null | grep -v "\.mediapipe" | tail -1)
    CUDA_VISIBLE_DEVICES= python scripts/r9/artefacts.py $ID --tag-suffix "$V" --gen $GT ${L3:+--l3 $L3} --out $OUT/art.json | tail -1   # stick / unstick, amplitude, teeth, slide
  } > $L/metrics.log 2>&1
  echo "== $ID$V metrics done $(date +%T)" | tee -a $L/face.log
  # the whole-frame gate is one CPU core for ~1 s per frame: in the background (as e2e.sh), then the gate rows again
  ( bash scripts/r7/r7_frame_gate.sh $B/$FT:$FT $ID | grep "^=="; python scripts/r7/r7_baseline.py freeze --tag $FT --audio $A/$AT --ids $ID --out $OUT/face_gate.json | tail -2
    echo "FRAME_GATE_DONE $ID$V" ) > $L/frame_gate.log 2>&1 < /dev/null &
  disown
done
echo FACE_ONLY_DONE "$@"
