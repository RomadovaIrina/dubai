#!/bin/bash
# Round 9 E2E: ONE command from a raw phone video to the final_r8b pair (management set), no manual steps.
#     bash scripts/r9/e2e.sh <input.mp4> <out_dir>        env: ID=<short id> (default: the input's file stem, sanitised)
#   -> <out_dir>/<id>_DabAI_r9_demucs.mp4 (htdemucs bed, main) + <id>_DabAI_r9_spleeter.mp4 (Spleeter bed) + manifest.json + timing.json
# Stages (= final_r8b, reports/experiments/r8-face/final_r8b_README.md):
#   precheck  ffprobe + 3-face MediaPipe probe -> flags (scripts/r9/precheck.py); a rotated source is made upright first (display orientation kept)
#   audio     audio-r8-v4 from the raw input: scripts/r8/r8_audio.sh (ASR, Qwen 14B Q5_K_M duration-aware, batched Chatterbox, R8_ONEPROC,
#             retake / clause takes / burst align / lip placement / hush / soften, the Spleeter bed remix) = the SPLEETER final mix
#   sep       htdemucs stems on the GPU (scripts/r8/sep_stems.py) + the same dub remixed on the htdemucs bed with the sepH8p flags
#             (scripts/r8/sep_mix.sh: --bed-pauses source --bed-noise-back on --src-duck-db -3) = the DEMUCS final mix (NC, DUB_ALLOW_NC_BED=1)
#   face_gen  LatentSync 1.6 + CodeFormer (NC, DUB_ALLOW_NC=1, no ParseNet, batch 16), 10 steps where the dub speaks / 20 where silent (scripts/r7/r7_e2e.sh)
#   face_comp CodeFormer detail-only (CFD=self) + fast layer SR + tight fallback fb7 + the r7 composite at the source fps (scripts/r7/r7_m7_comp.sh)
#   mux       the face video stream (rendered ONCE) + each mix, stream copy
# Hard cases never crash the command: no face -> the source picture with the dub (flag no_face); a failed face stage -> the same, PARTIAL_SUCCESS;
# a failed htdemucs bed -> only the Spleeter file, PARTIAL_SUCCESS; no audio / failed dub -> ERROR. manifest.json has status + flags + metrics.
set -u
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; R=$DUB_ROOT; . $R/scripts/env.sh >/dev/null; DUB_ROOT=$R; export DUB_ROOT; cd $DUB_ROOT
IN=$(readlink -f "$1"); OUT=$(mkdir -p "$2" && readlink -f "$2")
ID=${ID:-$(basename "${IN%.*}" | tr -c 'A-Za-z0-9_\n-' '_')}
export METRICS=off   # production stages only; every gate / scoring pass runs in the metrics stage below (honest per-stage time)
case "${R9_FAST_AUDIO:-2}" in   # r9 audio speed (reports/experiments/r9-speed-audio/REPORT.md); default 2 = r9fast-audio (user-approved by ear, 30.10); 0 = the r9 E2E audio path
  1) export R9_MP_BG=on DUB_S3_BATCH=1 R9_ONEPROC=on;;   # r9fast-audio-safe: background MediaPipe, batched S3Gen, the stage in the pipeline process
  2) export R9_MP_BG=on DUB_S3_BATCH=1 R9_ONEPROC=on DUB_UNIT_CONC=1 DUB_QWEN_BATCH=1;;   # r9fast-audio: + units concurrently, batched resident Qwen 14B
esac
export DUB_ALLOW_NC=1 DUB_ALLOW_NC_BED=1   # CodeFormer + htdemucs: deliberate NC deviation, user-accepted (brief_r9_e2e.md); InsightFace is not used
A=/workspace/dub/artifacts/r4a; B=/workspace/dub/artifacts/r4m; AT=r9_$ID; HT=r9_${ID}_htd; GT=r9g_$ID; FT=r9f_$ID
W=$OUT/work; L=$OUT/logs; mkdir -p $W $L; T=$W/stages.tsv; : > $T; STATUS=SUCCESS; FL=(); ER=()
export DUB_INPUTS=$W/in; rm -rf $DUB_INPUTS; mkdir -p $DUB_INPUTS
echo "r9 e2e $ID commit $(git rev-parse --short HEAD) input $IN out $OUT $(date -Is)" | tee $L/e2e.log
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 500 > $W/nvsmi.csv 2>/dev/null & NV=$!
trap 'kill $NV 2>/dev/null' EXIT
st() { echo -e "$1\t$(date +%s.%N)" >> $T; echo "== $ID stage $1 $(date +%T)" | tee -a $L/e2e.log; }   # stage start; the next st (or 'end') closes it
fail() { FL+=("$1"); ER+=("$1: ${3:-}"); [ "$2" = ERROR ] && STATUS=ERROR || { [ $STATUS = SUCCESS ] && STATUS=$2; }; echo "== $ID $2: $1 ${3:-}" | tee -a $L/e2e.log; }   # fail <flag> <status> [detail]
collect() { python scripts/r9/collect.py --id $ID --out $OUT --status $STATUS --flags "${FL[*]:-}" --errors "$(printf '%s\n' "${ER[@]:-}")" --stages $T --nvsmi $W/nvsmi.csv --precheck $W/precheck.json \
    --audio $A/$AT --htd $A/$HT --face $B/$FT --input "$IN"; }
finish() {
  st end; kill $NV 2>/dev/null; wait $NV 2>/dev/null
  collect 2>&1 | tee -a $L/e2e.log
  echo "R9_E2E_DONE $ID $STATUS" | tee -a $L/e2e.log; exit 0
}

st precheck
python scripts/r9/precheck.py --input "$IN" --out $W/precheck.json --work $W > $L/precheck.log 2>&1 || fail precheck_crashed PARTIAL_SUCCESS "see logs/precheck.log"
tail -1 $L/precheck.log | tee -a $L/e2e.log
PF=$(python3 -c "import json; print(' '.join(json.load(open('$W/precheck.json')).get('flags', [])))" 2>/dev/null)
for f in $PF; do FL+=("$f"); done
case " $PF " in *" unreadable_input "*|*" no_audio "*) fail no_usable_input ERROR "${PF}"; finish;; esac
EXT=${IN##*.}; ln "$IN" $DUB_INPUTS/$ID.$EXT 2>/dev/null || cp "$IN" $DUB_INPUTS/$ID.$EXT   # a real file named <id>: the stages name their work dirs after the resolved input
if [[ " $PF " == *" rotated_source "* ]]; then   # the pipeline reads coded frames: bake the display rotation in (near-lossless), same display orientation
  st upright
  ffmpeg -loglevel error -y -i "$IN" -map 0:v:0 -map 0:a:0 -c:v libx264 -preset medium -crf 12 -pix_fmt yuv420p -c:a copy -metadata:s:v:0 rotate=0 $W/upright.mp4 \
    && { rm -f $DUB_INPUTS/$ID.*; mv $W/upright.mp4 $DUB_INPUTS/$ID.mp4; } || fail upright_failed PARTIAL_SUCCESS "rotation bake failed"
fi
SRC=$(ls $DUB_INPUTS/$ID.* | head -1)

st audio   # one id, from the raw input; its own tag so nothing of another run is reused (the per-id caches $A/sep/<id> are cleared)
rm -rf $A/sep/$ID $A/$AT $A/${AT}src $A/${AT}pre
bash scripts/r8/r8_audio.sh $AT $ID > $L/audio.log 2>&1
AD=$A/$AT/$ID
[ -f $AD/final_mix_aac.m4a ] && [ -f $AD/dubbed_16k.wav ] || { fail audio_failed ERROR "see logs/audio.log"; finish; }

st sep   # htdemucs: stems on the GPU (models loaded once), then the sepH8p remix of the SAME dub on the CPU
export DUB_SEP_STEMS=/workspace/dub/r8a; rm -rf $DUB_SEP_STEMS/stems_htdemucs/$ID $A/$HT
( export PYTHONPATH=/workspace/dub/sep_pylib:${PYTHONPATH:-}; flock /workspace/dub/gpu.lock python scripts/r8/sep_stems.py --sep htdemucs --out $DUB_SEP_STEMS/stems_htdemucs $ID ) > $L/sep_stems.log 2>&1 \
  || fail htdemucs_stems_failed PARTIAL_SUCCESS "see logs/sep_stems.log"
SEP_SRC=$AT SEP_PREV=$AT bash scripts/r8/sep_mix.sh $HT htdemucs stems "--bed-pauses source --bed-noise-back on --src-duck-db -3" $ID > $L/sep_mix.log 2>&1
[ -f $A/$HT/$ID/final_mix_aac.m4a ] || fail htdemucs_mix_failed PARTIAL_SUCCESS "see logs/sep_mix.log: no demucs file"

VID=""
if [[ " $PF " == *" no_face "* ]]; then FL+=("face_skipped_no_face")
else
  st face_gen
  rm -rf $B/$GT $B/$FT
  AUDIO=$A/$AT TAG=$GT SKIP_SCORE=1 VIDEO_EXTRA="--codeformer optimized --codeformer-w 1.0 --steps 10 --steps-silent 20 --codeformer-parse off --codeformer-batch 16" \
    bash scripts/r7/r7_e2e.sh $ID > $L/face_gen.log 2>&1
  if [ -f $B/$GT/work_${ID}_$GT/${ID}_${GT}_25fps.manifest.json ]; then
    st face_comp
    SKIP_SCORE=1 TAG=$FT SRC_TAG=$GT CFD=self SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7 FB_ARGS="$(cat reports/experiments/r8-face/fb7.args)" \
      bash scripts/r7/r7_m7_comp.sh $ID > $L/face_comp.log 2>&1
    [ -f $B/$FT/${ID}_$FT.mp4 ] && VID=$B/$FT/${ID}_$FT.mp4 || fail face_comp_failed PARTIAL_SUCCESS "see logs/face_comp.log: source picture used"
  else fail face_gen_failed PARTIAL_SUCCESS "see logs/face_gen.log: source picture used"; fi
fi
[ -n "$VID" ] || VID=$SRC

st mux   # the picture rendered once, muxed with each bed (stream copy: within the pair the video stream is bit-identical)
for v in demucs:$A/$HT/$ID/final_mix_aac.m4a spleeter:$AD/final_mix_aac.m4a; do
  n=${v%%:*}; m=${v#*:}; o=$OUT/${ID}_DabAI_r9_$n.mp4; rm -f $o
  [ -f $m ] || continue
  ffmpeg -loglevel error -y -i $VID -i $m -map 0:v:0 -map 1:a:0 -c copy -movflags +faststart $o || fail mux_${n}_failed PARTIAL_SUCCESS
done
[ -f $OUT/${ID}_DabAI_r9_spleeter.mp4 ] || [ -f $OUT/${ID}_DabAI_r9_demucs.mp4 ] || fail no_output ERROR

st metrics   # NOT production: the r8 gate rows on the outputs (logs/metrics.log)
export METRICS=on
{ src=$SRC; O=$A/$AT
  python scripts/quality/r4_audio_metrics.py --mix $O/${ID}_over-original.mp4 --src $src --sep $A/sep/$ID --run $O/$ID --manifest $O/$ID/${ID}_r4b.manifest.json --wer \
    --series $O/${ID}.series.npz --json $O/${ID}.metrics.json || echo "WARN audio metrics"
  PYTHONPATH=third_party/llama_cuda:${PYTHONPATH:-} python scripts/r5/r5_faith.py $AT $ID > $O/runs/faith.log 2>&1 || echo "WARN faith"   # the 14B judge + back-translation -> metrics.json faith
  python scripts/r8/coverage.py --run $O/$ID --id $ID --label $AT --face $A/sep/$ID/source_mediapipe.npz --out $O/cov/cov_$ID | tail -1
  python scripts/r8/pacing.py --run $O/$ID --id $ID --label $AT --face $A/sep/$ID/source_mediapipe.npz --out $O/cov/pace_$ID | tail -1
  python scripts/r8/cuts.py --run $O/$ID --id $ID --label $AT --out $O/cov/cuts_$ID | tail -1
  python scripts/r8/mixdrop.py --run $O/$ID --id $ID --label $AT --out $O/cov/mix_$ID | tail -1
  flock /workspace/dub/gpu.lock python scripts/r7/gap_metrics.py --tag $AT --clips $O/gaps $ID || echo "WARN gaps spleeter"
  if [ -f $A/$HT/$ID/final_mix_aac.m4a ]; then mkdir -p $A/$HT/cov
    python scripts/r8/mixdrop.py --run $A/$HT/$ID --id $ID --label $HT --out $A/$HT/cov/mix_$ID | tail -1
    python scripts/r7/clicks.py --tag $HT --prev $AT --json $A/$HT/cov/clicks.json $ID
    flock /workspace/dub/gpu.lock python scripts/r7/gap_metrics.py --tag $HT --prev $AT --clips $A/$HT/gaps $ID || echo "WARN gaps demucs"; fi
  if [ -f $B/$FT/${ID}_$FT.mp4 ]; then
    python scripts/r4/r4_gate.py score --label $FT --dir $B/$FT --ids $ID > $L/face_score.log 2>&1 || echo "WARN face score"
    MAN=$B/$FT/%s_$FT.manifest.json bash scripts/r6/r6_twitch.sh $FT $B/$FT _$FT $ID | grep '=='
    python scripts/r7/r7_baseline.py freeze --tag $FT --audio $A/$AT --ids $ID --out $W/face_gate.json | tail -2
    python scripts/r7/r7_closed_mouth.py --ids $ID --variants $FT:$A/$AT --out $W/closed_mouth.json | tail -1   # pressed lips
  fi
} > $L/metrics.log 2>&1
if [ -f $B/$FT/${ID}_$FT.mp4 ]; then   # the whole-frame gate is one CPU core for ~1 s per frame at 1080p: in the background (next video starts), then the rows again
  ( bash scripts/r7/r7_frame_gate.sh $B/$FT:$FT $ID | grep "^=="; python scripts/r7/r7_baseline.py freeze --tag $FT --audio $A/$AT --ids $ID --out $W/face_gate.json | tail -2
    collect | tail -1; echo "R9_FRAME_GATE_DONE $ID" ) > $L/frame_gate.log 2>&1 < /dev/null &
  disown; fi
finish
