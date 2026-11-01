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
  2) export R9_MP_BG=on DUB_S3_BATCH=1 R9_ONEPROC=on DUB_UNIT_CONC=1 DUB_QWEN_BATCH=1 R9_CHAIN_INPROC=${R9_CHAIN_INPROC:-on} R9_TTS_PRELOAD=${R9_TTS_PRELOAD:-on};;   # r9fast-audio: + units concurrently, batched resident Qwen 14B, the CPU chain in the audio process (round 2; R9_SEP_OVERLAP below)
esac
case "${R10_FAST_FACE:-1}" in   # r9 speed FACE (reports/experiments/r9-speed-face/REPORT.md): default 1 = stack S2 (S without the mouth-only decode, stick +2) + VAE compile, gate PASS on m03; 0 = the r9 E2E face path
  1) export GEN_COMP=${GEN_COMP:-0} FAST_SRCPREP=${FAST_SRCPREP:-1} FAST_FB_BG=${FAST_FB_BG:-1} FAST_ENC=${FAST_ENC:-ultrafast} FAST_MPREUSE=${FAST_MPREUSE:-1} \
       FC_EXTRA=${FC_EXTRA:---pre-threads=12} LS_VAE_COMPILE=${LS_VAE_COMPILE:-max-autotune-no-cudagraphs} CFD_EXTRA=${CFD_EXTRA:---threads=12};;
esac
[ "${E2E_HALF_RATE:-0}" = 1 ] && export LS_HALF_RATE=1   # r9 speed: LatentSync on every other frame (GPU −40 %), odd frames flow-filled; NOT default (sync margin thin, r10 gate pending)
[ -n "${E2E_LS_RATE:-}" ] && export LS_RATE=$E2E_LS_RATE   # r10 speed: LatentSync on k of every m frames (E2E_LS_RATE=2/3: face GPU −25 %, gated m03 / m19 / old03); half-rate (1/2) fails the flash gate
[ -n "${R10_GEN_SPOOL:-}" ] && export R10_GEN_PY=scripts/r10/gen_client.py   # r10 speed: the generator job goes to the resident generator worker (scripts/r10/gen_worker.sh)
export E2E_FX=${E2E_FX:-1}   # r10: the r9 + fix look (the management set, and the look the face speed stack was gated on) is the default; E2E_FX=0 = the plain r9 composite
export DUB_ALLOW_NC=1 DUB_ALLOW_NC_BED=1  # CodeFormer + htdemucs: deliberate NC deviation, user-accepted (brief_r9_e2e.md); InsightFace is not used
A=/workspace/dub/artifacts/r4a; B=/workspace/dub/artifacts/r4m; AT=r9_$ID; HT=r9_${ID}_htd; GT=r9g_$ID; FT=r9f_$ID
W=$OUT/work; L=$OUT/logs; mkdir -p $W $L; T=$W/stages.tsv; : > $T; STATUS=SUCCESS; FL=(); ER=()
export DUB_INPUTS=$W/in; rm -rf $DUB_INPUTS; mkdir -p $DUB_INPUTS
echo "r9 e2e $ID E2E_REF=${E2E_REF:-same} E2E_FX=${E2E_FX:-0} commit $(git rev-parse --short HEAD) input $IN out $OUT $(date -Is)" | tee $L/e2e.log
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 100 > $W/nvsmi.csv 2>/dev/null & NV=$!
export R10_JOB=$ID; NVML=${R10_NVML:-$W/nvml.jsonl}; NM=""   # r10: NVML sampler, 100 ms, per-process SM util tagged by R10_JOB (a worker passes its one shared R10_NVML file)
[ -z "${R10_NVML:-}" ] && [ -x /venv/main/bin/python ] && { rm -f $NVML; /venv/main/bin/python scripts/r10/gpu_sampler.py --out $NVML > $L/gpu_sampler.log 2>&1 & NM=$!; }
trap 'kill $NV $NM 2>/dev/null' EXIT
st() { echo -e "$1\t$(date +%s.%N)" >> $T; echo "== $ID stage $1 $(date +%T)" | tee -a $L/e2e.log; }   # stage start; the next st (or 'end') closes it
fail() { FL+=("$1"); ER+=("$1: ${3:-}"); [ "$2" = ERROR ] && STATUS=ERROR || { [ $STATUS = SUCCESS ] && STATUS=$2; }; echo "== $ID $2: $1 ${3:-}" | tee -a $L/e2e.log; }   # fail <flag> <status> [detail]
collect() { python scripts/r9/collect.py --id $ID --out $OUT --status $STATUS --flags "${FL[*]:-}" --errors "$(printf '%s\n' "${ER[@]:-}")" --stages $T --nvsmi $W/nvsmi.csv --precheck $W/precheck.json \
    --audio $A/$AT --htd $A/$HT --face $B/$FT --input "$IN"; }
finish() {
  st end; kill $NV $NM 2>/dev/null; wait $NV $NM 2>/dev/null
  collect 2>&1 | tee -a $L/e2e.log
  [ -s $NVML ] && /venv/main/bin/python scripts/r10/gpu_sampler.py --summary $NVML --stages $T --job $ID \
    --vmin $(python3 -c "import json; print(json.load(open('$OUT/timing.json'))['video_min'] or 1)" 2>/dev/null || echo 1) > $OUT/gpu_time.json 2>> $L/gpu_sampler.log \
    && python3 -c "import json; t = json.load(open('$OUT/gpu_time.json'))['total']; print('== $ID NVML GPU-busy', t.get('gpu_busy_s_per_vmin'), 's/vmin, weighted', t.get('gpu_weighted_s_per_vmin'), ', wall', t.get('wall_s_per_vmin'))" | tee -a $L/e2e.log
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

if [ -n "${R10_HEAVY_LOCK:-}" ]; then st wait_heavy; exec 8>"$R10_HEAVY_LOCK"; flock 8; fi   # r10 worker: audio + sep + face_gen (the heavy GPU stages) hold ONE exclusive lock; the wait is its own stage
st audio   # one id, from the raw input; its own tag so nothing of another run is reused (the per-id caches $A/sep/<id> are cleared)
rm -rf $A/sep/$ID $A/$AT $A/${AT}src $A/${AT}pre
export DUB_SEP_STEMS=/workspace/dub/r8a
OVL=${R9_SEP_OVERLAP:-on}   # r9 speed O: the htdemucs stems while the audio stage runs, the demucs remix in parallel with the Spleeter one (same commands = same files)
if [ "$OVL" = on ]; then
  rm -rf $DUB_SEP_STEMS/stems_htdemucs/$ID $A/$HT
  bash scripts/r9/sep_overlap.sh start $ID $DUB_SEP_STEMS/stems_htdemucs $W $L
  export R9_ON_MIXSRC="bash $DUB_ROOT/scripts/r9/sep_overlap.sh mix $ID $AT $HT $W $L"
fi
bash scripts/r8/r8_audio.sh $AT $ID > $L/audio.log 2>&1
unset R9_ON_MIXSRC
AD=$A/$AT/$ID
[ -f $AD/final_mix_aac.m4a ] && [ -f $AD/dubbed_16k.wav ] || { fail audio_failed ERROR "see logs/audio.log"; finish; }

st sep   # htdemucs: stems on the GPU (models loaded once), then the sepH8p remix of the SAME dub on the CPU
if [ "$OVL" = on ] && timeout 900 bash scripts/r9/sep_overlap.sh wait $W; then :   # the demucs mix was made during the audio stage
elif [ "$OVL" = on ] && [ -f $W/sepmix.done ]; then fail htdemucs_stems_failed PARTIAL_SUCCESS "see logs/sep_stems.log"
else   # sequential (R9_SEP_OVERLAP=off, or the overlap never started)
rm -rf $DUB_SEP_STEMS/stems_htdemucs/$ID $A/$HT
( export PYTHONPATH=/workspace/dub/sep_pylib:${PYTHONPATH:-}; flock /workspace/dub/gpu.lock python scripts/r8/sep_stems.py --sep htdemucs --out $DUB_SEP_STEMS/stems_htdemucs $ID ) > $L/sep_stems.log 2>&1 \
  || fail htdemucs_stems_failed PARTIAL_SUCCESS "see logs/sep_stems.log"
SEP_SRC=$AT SEP_PREV=$AT bash scripts/r8/sep_mix.sh $HT htdemucs stems "--bed-pauses source --bed-noise-back on --src-duck-db -3" $ID > $L/sep_mix.log 2>&1
fi
[ -f $A/$HT/$ID/final_mix_aac.m4a ] || fail htdemucs_mix_failed PARTIAL_SUCCESS "see logs/sep_mix.log: no demucs file"

VID=""
if [[ " $PF " == *" no_face "* ]]; then FL+=("face_skipped_no_face")
else
  st face_gen
  rm -rf $B/$GT $B/$FT
  REFX=""; [ -n "${E2E_REF:-}" ] && REFX="--ref-mode ${E2E_REF%%:*} --audio-lead ${E2E_REF#*:}"   # r9: E2E_REF=fixed:0.5 = LatentSync reference from a calm closed-mouth clip (no copy of the source mouth) + its half-frame lag compensated
  unset MP_SOURCE_NPZ; SMP=$A/sep/$ID/source_mediapipe.npz; SP=""   # (as scripts/r9/face_only.sh)
  if [ "${FAST_MPREUSE:-0}" = 1 ] && [ -f $SMP ]; then   # r9 speed: the generator's master pass from the audio stage's source MediaPipe (frame map; gated) + the composite's source pass IS it
    export MP_SOURCE_NPZ=$SMP; mkdir -p $B/$GT/timing; cp $SMP $B/$GT/timing/${ID}_source_lm_F_mp.npz; fi
  if [ "${FAST_SRCPREP:-0}" = 1 ]; then   # r9 speed L4: the source-only composite inputs (master timing, source landmarks, rigid motion) on the CPU while LatentSync runs
    WV=$B/$GT/work_${ID}_$GT/video; TD=$B/$GT/timing
    ( for i in $(seq 1800); do [ -s $WV/master_mediapipe.npz ] && break; sleep 2; done; [ -s $WV/master_mediapipe.npz ] || exit 0; mkdir -p $TD; TJ=$TD/${ID}_master_timing_F.json; SL=$TD/${ID}_source_lm_F.npz
      python scripts/quality/r3_master_timing.py --src $SRC --master $WV/master_25fps.mp4 --vf fps=25 --out $TJ && python scripts/quality/r3_source_lm.py --src $SRC --timing $TJ --out $SL \
        && python scripts/r9/r9_source_rigid.py --src $SRC --source-lm $SL ) > $L/srcprep.log 2>&1 & SP=$!
  fi
  AUDIO=$A/$AT TAG=$GT SKIP_SCORE=1 VIDEO_EXTRA="--codeformer optimized --codeformer-w 1.0 --steps 10 --steps-silent 20 --codeformer-parse off --codeformer-batch 16 $REFX" \
    bash scripts/r7/r7_e2e.sh $ID > $L/face_gen.log 2>&1
  [ -n "$SP" ] && wait $SP
  if [ -f $B/$GT/work_${ID}_$GT/${ID}_${GT}_25fps.manifest.json ]; then
    [ -n "${R10_HEAVY_LOCK:-}" ] && flock -u 8   # r10 worker: the next job's audio + face_gen may start
    if [ -n "${R10_COMP_GATE:-}" ]; then st wait_comp; touch $W/comp_ready; until [ -f $W/comp_go ]; do sleep 1; done; fi   # r10 worker: the composite (~12 GB) starts only once no other job is in its audio stage (~30 GB)
    st face_comp
    [ "$E2E_FX" = 1 ] && export REP=1 CFD_LABEL=ia04 CFD_ARGS="--inner-alpha 0.4 ${CFD_EXTRA:-}"   # r9: short mouth glitches re-drawn in the layer, long ones -> smooth source fallback; CodeFormer detail at 0.4 inside the mouth
    [ "$E2E_FX" = 1 ] || { [ -n "${CFD_EXTRA:-}" ] && export CFD_ARGS="${CFD_ARGS:-} $CFD_EXTRA"; }
    SKIP_SCORE=1 TAG=$FT SRC_TAG=$GT CFD=self SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7 FB_ARGS="$(cat reports/experiments/r8-face/fb7.args)" \
      bash scripts/r7/r7_m7_comp.sh $ID > $L/face_comp.log 2>&1
    [ -f $B/$FT/${ID}_$FT.mp4 ] && VID=$B/$FT/${ID}_$FT.mp4 || fail face_comp_failed PARTIAL_SUCCESS "see logs/face_comp.log: source picture used"
  else fail face_gen_failed PARTIAL_SUCCESS "see logs/face_gen.log: source picture used"; fi
fi
[ -n "${R10_HEAVY_LOCK:-}" ] && flock -u 8
[ -n "$VID" ] || VID=$SRC

st mux   # the picture rendered once, muxed with each bed (stream copy: within the pair the video stream is bit-identical)
for v in demucs:$A/$HT/$ID/final_mix_aac.m4a spleeter:$AD/final_mix_aac.m4a; do
  n=${v%%:*}; m=${v#*:}; o=$OUT/${ID}_DabAI_r9_$n.mp4; rm -f $o
  [ -f $m ] || continue
  ffmpeg -loglevel error -y -i $VID -i $m -map 0:v:0 -map 1:a:0 -c copy -movflags +faststart $o || fail mux_${n}_failed PARTIAL_SUCCESS
done
[ -f $OUT/${ID}_DabAI_r9_spleeter.mp4 ] || [ -f $OUT/${ID}_DabAI_r9_demucs.mp4 ] || fail no_output ERROR

st metrics   # NOT production: the r8 gate rows on the outputs (logs/metrics.log)
[ "${E2E_METRICS:-on}" = off ] && finish   # r9: production only (the gate rows are scored separately)
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
    python scripts/r9/occl_scan.py --src $SRC --out $B/$FT/${ID}_$FT.mp4 --lm $B/$GT/timing/${ID}_source_lm_F.npz --sheet $W/occlusion_scan.jpg --json $W/occlusion.json | tail -1   # something next to the mouth overwritten
  fi
} > $L/metrics.log 2>&1
if [ "${E2E_FRAME_GATE:-on}" = on ] && [ -f $B/$FT/${ID}_$FT.mp4 ]; then   # the whole-frame gate is one CPU core for ~1 s per frame at 1080p: in the background (next video starts), then the rows again
  ( bash scripts/r7/r7_frame_gate.sh $B/$FT:$FT $ID | grep "^=="; python scripts/r7/r7_baseline.py freeze --tag $FT --audio $A/$AT --ids $ID --out $W/face_gate.json | tail -2
    collect | tail -1; echo "R9_FRAME_GATE_DONE $ID" ) > $L/frame_gate.log 2>&1 < /dev/null &
  disown; fi
finish
