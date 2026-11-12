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
export DUB_ALLOW_NC=${DUB_ALLOW_NC:-1} DUB_ALLOW_NC_BED=${DUB_ALLOW_NC_BED:-1}  # CodeFormer + htdemucs: deliberate NC deviation, user-accepted (brief_r9_e2e.md); InsightFace is not used
# r15 T: DUB_ALLOW_NC=0 = the commercially clean face (no CodeFormer: generator --codeformer off, composite without the CF detail layer);
#        DUB_ALLOW_NC_BED=0 = no htdemucs (the Spleeter bed is the only output). Default unchanged (1 / 1); the clean Docker image sets 0 / 0.
A=/workspace/dub/artifacts/r4a; B=/workspace/dub/artifacts/r4m; AT=r9_$ID; HT=r9_${ID}_htd; GT=r9g_$ID; FT=r9f_$ID
W=$OUT/work; L=$OUT/logs; mkdir -p $W $L; T=$W/stages.tsv; : > $T; STATUS=SUCCESS; FL=(); ER=()
export DUB_INPUTS=$W/in; rm -rf $DUB_INPUTS; mkdir -p $DUB_INPUTS
echo "r9 e2e $ID E2E_REF=${E2E_REF:-same} E2E_FX=${E2E_FX:-0} E2E_SYNC=${E2E_SYNC:-0} commit $(git rev-parse --short HEAD) input $IN out $OUT $(date -Is)" | tee $L/e2e.log
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 100 > $W/nvsmi.csv 2>/dev/null & NV=$!
export R10_JOB=$ID; NVML=${R10_NVML:-$W/nvml.jsonl}; NM=""   # r10: NVML sampler, 100 ms, per-process SM util tagged by R10_JOB (a worker passes its one shared R10_NVML file)
[ -z "${R10_NVML:-}" ] && [ -x /venv/main/bin/python ] && { rm -f $NVML; /venv/main/bin/python scripts/r10/gpu_sampler.py --out $NVML > $L/gpu_sampler.log 2>&1 & NM=$!; }
trap 'kill $NV $NM 2>/dev/null' EXIT
st() { echo -e "$1\t$(date +%s.%N)" >> $T; echo "== $ID stage $1 $(date +%T)" | tee -a $L/e2e.log; }   # stage start; the next st (or 'end') closes it
fail() { FL+=("$1"); ER+=("$1: ${3:-}"); [ "$2" = ERROR ] && STATUS=ERROR || { [ $STATUS = SUCCESS ] && STATUS=$2; }; echo "== $ID $2: $1 ${3:-}" | tee -a $L/e2e.log; }   # fail <flag> <status> [detail]
collect() { python scripts/r9/collect.py --id $ID --out $OUT --status $STATUS --flags "${FL[*]:-}" --errors "$(printf '%s\n' "${ER[@]:-}")" --stages $T --nvsmi $W/nvsmi.csv --precheck $W/precheck.json \
    --audio $A/$AT --htd $A/$HT --face $B/$FT --gen $B/$GT --input "$IN" --ks-subst $W/ks/subst.json; }   # r16 TZ 11.2 / 11.3
finish() {
  st end; kill $NV $NM 2>/dev/null; wait $NV $NM 2>/dev/null
  collect 2>&1 | tee -a $L/e2e.log
  [ -s $NVML ] && /venv/main/bin/python scripts/r10/gpu_sampler.py --summary $NVML --stages $T --job $ID \
    --vmin $(python3 -c "import json; print(json.load(open('$OUT/timing.json'))['video_min'] or 1)" 2>/dev/null || echo 1) > $OUT/gpu_time.json 2>> $L/gpu_sampler.log \
    && python3 -c "import json; t = json.load(open('$OUT/gpu_time.json'))['total']; print('== $ID NVML GPU-busy', t.get('gpu_busy_s_per_vmin'), 's/vmin, weighted', t.get('gpu_weighted_s_per_vmin'), ', wall', t.get('wall_s_per_vmin'))" | tee -a $L/e2e.log
  STATUS=$(python3 -c "import json; print(json.load(open('$OUT/manifest.json'))['status'])" 2>/dev/null || echo $STATUS)   # r13: collect folds in the stage verdicts (11.3 2 % rule)
  echo "R9_E2E_DONE $ID $STATUS" | tee -a $L/e2e.log; [ "$STATUS" = ERROR ] && exit 1; exit 0   # (Q audit O3: a failed job exits non-zero)
}

st precheck
python scripts/r9/precheck.py --input "$IN" --out $W/precheck.json --work $W > $L/precheck.log 2>&1 || fail precheck_crashed PARTIAL_SUCCESS "see logs/precheck.log"
tail -1 $L/precheck.log | tee -a $L/e2e.log
PF=$(python3 -c "import json; print(' '.join(json.load(open('$W/precheck.json')).get('flags', [])))" 2>/dev/null)
for f in $PF; do FL+=("$f"); done
case " $PF " in *" unreadable_input "*|*" no_audio "*) fail no_usable_input ERROR "${PF}"; finish;; esac
if [ "${E2E_LIPSYNC:-on}" != off ]; then   # r16 T: every weight the face path needs, before any GPU work (a missing file = a loud failure, not a quiet source picture)
  WF=$(python3 scripts/keysync/r16/weights_check.py --face ${DUB_FACE:-auto} --nc ${DUB_ALLOW_NC:-1} 2>> $L/e2e.log); WRC=$?
  for f in $WF; do FL+=("$f"); done
  [ $WRC != 0 ] && { fail weights_missing ERROR "model weights missing / the weights check failed (rc $WRC) for DUB_FACE=${DUB_FACE:-auto}: see logs/e2e.log"; finish; }
  [[ " $WF " == *" face_latentsync_unavailable "* ]] && export DUB_FACE_ENGINE=${DUB_FACE_ENGINE:-keysync}   # beards then go to KeySync (flagged)
fi
EXT=${IN##*.}; ln "$IN" $DUB_INPUTS/$ID.$EXT 2>/dev/null || cp "$IN" $DUB_INPUTS/$ID.$EXT   # a real file named <id>: the stages name their work dirs after the resolved input
if [[ " $PF " == *" rotated_source "* ]]; then   # the pipeline reads coded frames: bake the display rotation in (near-lossless), same display orientation
  st upright
  ffmpeg -loglevel error -y -i "$IN" -map 0:v:0 -map 0:a:0 -c:v libx264 -preset medium -crf 12 -pix_fmt yuv420p -c:a copy -metadata:s:v:0 rotate=0 $W/upright.mp4 \
    && { rm -f $DUB_INPUTS/$ID.*; mv $W/upright.mp4 $DUB_INPUTS/$ID.mp4; } || fail upright_failed PARTIAL_SUCCESS "rotation bake failed"
fi
SRC=$(ls $DUB_INPUTS/$ID.* | head -1)
if [[ " $PF " == *" variable_fps "* ]]; then   # r13: a VFR phone clip -> constant rate at its AVERAGE rate (the same frame count, even timing), near-lossless;
  st cfr                                     # the composite assumes a constant rate (edge suite: VFR 22.5 avg / 30 nominal -> face_comp_failed)
  AFR=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of csv=p=0 "$SRC")
  ffmpeg -loglevel error -y -i "$SRC" -map 0:v:0 -map 0:a:0 -fps_mode cfr -r "$AFR" -c:v libx264 -preset medium -crf 12 -pix_fmt yuv420p -c:a copy $W/cfr.mp4 \
    && { rm -f $DUB_INPUTS/$ID.*; mv $W/cfr.mp4 $DUB_INPUTS/$ID.mp4; SRC=$DUB_INPUTS/$ID.mp4; } || fail cfr_failed PARTIAL_SUCCESS "VFR -> constant frame rate failed"
fi
FMS_DEF=1920; [ "${DUB_FACE:-auto}" != latentsync ] && FMS_DEF=1280   # r16: KeySync draws 512 px; its paste cost grows with the crop side (coordinator 07.11: 1280 default)
HIRES=""; FMS=${E2E_FACE_MAXSIDE:-$FMS_DEF}   # r15 T (edge res4k_vertical: face_gen_failed on 2160x3840): above FMS px on the long side the face stages
if [ "$FMS" != 0 ] && [ "${E2E_LIPSYNC:-on}" != off ] && [[ " $PF " != *" no_face "* ]]; then   # (generator + composite) run on a copy scaled to FMS
  IFS=, read -r SW SH _ < <(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 "$SRC")   # (r15: csv s=' ' fails to parse in ffprobe 6.1)
  [ -n "${SW:-}" ] && [ -n "${SH:-}" ] || echo "WARN $ID face_res: could not read the source size, face work at full resolution" | tee -a $L/e2e.log   # (the audio stage reads it too:
  if [ "${SW:-0}" -gt "$FMS" ] || [ "${SH:-0}" -gt "$FMS" ]; then                                                       # audio stream-copied, identical);
    st face_res                                                                                                            # scripts/r15/hires_merge.py then adds
    HIRES=$W/hires_src.${SRC##*.}; mv "$SRC" $HIRES                                                                        # the composite's change onto the
    if [ "$SW" -ge "$SH" ]; then SCL="scale=$FMS:-2:flags=lanczos"; else SCL="scale=-2:$FMS:flags=lanczos"; fi                    # full-res source frames
    if ffmpeg -loglevel error -y -i $HIRES -map 0:v:0 -map 0:a:0 -vf $SCL -c:v libx264 -preset medium -crf 12 -pix_fmt yuv420p -c:a copy $DUB_INPUTS/$ID.mp4; then
      SRC=$DUB_INPUTS/$ID.mp4; FL+=("face_work_downscaled")
    else rm -f $DUB_INPUTS/$ID.*; mv $HIRES "$SRC"; HIRES=""; fail face_res_failed PARTIAL_SUCCESS "downscale for the face stages failed"; fi
  fi
fi

. scripts/keysync/r16/gpulock.sh   # r16 T (audit 08.11): the GPU lock is held by a holder process, never by an fd this shell's children inherit
if [ -n "${R10_HEAVY_LOCK:-}" ]; then st wait_heavy   # audio + route + the KeySync GPU section / face_gen hold ONE GPU lock (= GPU_LOCK, gpu.lock)
  [ "$R10_HEAVY_LOCK" = "$GPU_LOCK" ] || echo "== $ID WARN R10_HEAVY_LOCK=$R10_HEAVY_LOCK is not the GPU lock $GPU_LOCK: the GPU lock is used" | tee -a $L/e2e.log
  gpu_acquire $$ || { fail gpu_lock_failed ERROR "could not take the GPU lock $GPU_LOCK"; finish; }; fi   # r10 worker: audio + sep + face_gen (the heavy GPU stages) hold ONE exclusive lock; the wait is its own stage
LONG_S=${E2E_LONG_AUDIO_S:-120}   # r15 T (long5 / long10: the in-process retake ran out of VRAM next to the resident pipeline models, 27-31 GB): above this
DUR_S=$(python3 -c "import json; print(json.load(open('$W/precheck.json'))['probe']['duration_s'])" 2>/dev/null || echo 0)   # many seconds the TTS stage (retake ->
if [ "$LONG_S" != 0 ] && python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" "$DUR_S" "$LONG_S"; then   # clause -> burst_align)
  export R9_ONEPROC=off R9_CHAIN_INPROC=off R8_STAGE_WAIT_VRAM=${R8_STAGE_WAIT_VRAM:-4096} DUB_RETAKE_BATCH_UNITS=${DUB_RETAKE_BATCH_UNITS:-4} R8_STAGE_TTS_BATCH=${R8_STAGE_TTS_BATCH:-0}; FL+=("audio_stage_own_process")   # (+ the stage takes one by one: a batched round of long face-aware takes = +8 GB -> OOM)   # runs in its own process after the pipeline exits (r8_tts_stage.py: the pre-r9-speed-L path,
fi                                                                               # the same stage code, full GPU); shorter clips unchanged (in-process, user-accepted 30.10)
st audio   # one id, from the raw input; its own tag so nothing of another run is reused (the per-id caches $A/sep/<id> are cleared)
rm -rf $A/sep/$ID $A/$AT $A/${AT}src $A/${AT}pre
export DUB_SEP_STEMS=/workspace/dub/r8a
[ "$DUB_ALLOW_NC_BED" = 1 ] || { R9_SEP_OVERLAP=off; FL+=("bed_spleeter_only"); }   # r15 T: clean bed
OVL=${R9_SEP_OVERLAP:-on}   # r9 speed O: the htdemucs stems while the audio stage runs, the demucs remix in parallel with the Spleeter one (same commands = same files)
if [ "$OVL" = on ]; then
  rm -rf $DUB_SEP_STEMS/stems_htdemucs/$ID $A/$HT
  bash scripts/r9/sep_overlap.sh start $ID $DUB_SEP_STEMS/stems_htdemucs $W $L
  export R9_ON_MIXSRC="bash $DUB_ROOT/scripts/r9/sep_overlap.sh mix $ID $AT $HT $W $L"
fi
if [ "${E2E_KS_PRE:-on}" = on ] && [ "${DUB_FACE:-auto}" != latentsync ] && [ "${E2E_LIPSYNC:-on}" != off ]; then   # r17 speed (output-identical): the KeySync
  export KS_PRE_DIR=$W/ks_pre; ( bash scripts/keysync/r16/ks_pre.sh "$SRC" $KS_PRE_DIR ) > $L/ks_pre.log 2>&1 &   # master + landmarks during the audio stage
fi
RPRE=""   # r17 speed (E2E_ROUTE_PRE=1, off by default until the VM guard): the route's dub-independent parts (its 25 fps master + the beard check)
if [ "${E2E_ROUTE_PRE:-0}" = 1 ] && [ "${E2E_LIPSYNC:-on}" != off ] && [ "${DUB_FACE:-auto}" != latentsync ] && [ "${DUB_ROUTE:-on}" = on ] && [ -z "${DUB_ROUTE_JSON:-}" ]; then
  RPRE=$W/route_pre; mkdir -p $RPRE   # the SAME commands as the face_route stage below, run during the audio stage (CPU)
  ( ffmpeg -loglevel error -y -i $SRC -map 0:v:0 -map 0:a:0 -vf fps=25 -c:v libx264 -preset fast -crf 12 -pix_fmt yuv420p -c:a aac $RPRE/master_25fps.mp4 && touch $RPRE/master.done
    ${DUB_MEDIAPIPE_PY:-/venv/mediapipe/bin/python} scripts/quality/r16q_beard.py $SRC --out $RPRE/beard.json > $RPRE/beard.log 2>&1 && touch $RPRE/beard.done; touch $RPRE/finished ) &
fi
if [ -z "${DUB_AUDIO_FROM:-}" ] && ! python3 scripts/voice_env_check.py --python ${DUB_ENV:-/venv/dabai}/bin/python 2>> $L/e2e.log; then   # r16 T: never a wrong voice silently
  fail voice_env_mismatch ERROR "the voice environment is not the accepted one (Chatterbox 5de7a54 + noisereduce): see logs/e2e.log"; finish; fi
if [ -n "${DUB_AUDIO_FROM:-}" ]; then   # r16 T golden test: an ACCEPTED dub (dubbed_16k.wav + final_mix_aac.m4a [+ *.manifest.json]) instead of a new one
  mkdir -p $A/$AT/$ID; cp $DUB_AUDIO_FROM/dubbed_16k.wav $DUB_AUDIO_FROM/final_mix_aac.m4a $A/$AT/$ID/; for m_ in $DUB_AUDIO_FROM/*.manifest.json; do [ -f "$m_" ] && cp "$m_" $A/$AT/$ID/${ID}_${m_##*_}; done   # r16 TZ: renamed to THIS id (<old id>_r4b -> <id>_r4b): the route / subs look up $AD/${ID}_*.manifest.json
  FL+=("audio_reused"); echo "== $ID audio reused from $DUB_AUDIO_FROM" | tee -a $L/e2e.log
else bash scripts/r8/r8_audio.sh $AT $ID > $L/audio.log 2>&1; fi
unset R9_ON_MIXSRC
AD=$A/$AT/$ID
if ! { [ -f $AD/final_mix_aac.m4a ] && [ -f $AD/dubbed_16k.wav ]; }; then   # r13: a clip with no speech (silence / music only) is its own, clear ERROR
  if grep -qs "ASR produced no speech units" $A/${AT}pre/runs/$ID.log $L/audio.log; then fail no_speech ERROR "no speech found in the audio: nothing to dub"
  else fail audio_failed ERROR "see logs/audio.log"; fi; finish; fi

if [ "$DUB_ALLOW_NC_BED" = 1 ]; then   # r15 T: the whole htdemucs bed only with the NC bed allowed
st sep   # htdemucs: stems on the GPU (models loaded once), then the sepH8p remix of the SAME dub on the CPU
if [ "$OVL" = on ] && timeout 900 bash scripts/r9/sep_overlap.sh wait $W; then :   # the demucs mix was made during the audio stage
elif [ "$OVL" = on ] && [ -f $W/sepmix.done ]; then fail htdemucs_stems_failed PARTIAL_SUCCESS "see logs/sep_stems.log"
else   # sequential (R9_SEP_OVERLAP=off, or the overlap never started)
rm -rf $DUB_SEP_STEMS/stems_htdemucs/$ID $A/$HT
( export PYTHONPATH=/workspace/dub/sep_pylib:${PYTHONPATH:-}; gpu_run python scripts/r8/sep_stems.py --sep htdemucs --out $DUB_SEP_STEMS/stems_htdemucs $ID ) > $L/sep_stems.log 2>&1 \
  || fail htdemucs_stems_failed PARTIAL_SUCCESS "see logs/sep_stems.log"
SEP_SRC=$AT SEP_PREV=$AT bash scripts/r8/sep_mix.sh $HT htdemucs stems "--bed-pauses source --bed-noise-back on --src-duck-db -3" $ID > $L/sep_mix.log 2>&1
fi
[ -f $A/$HT/$ID/final_mix_aac.m4a ] || fail htdemucs_mix_failed PARTIAL_SUCCESS "see logs/sep_mix.log: no demucs file"
fi

if [ -n "${DUB_POLYGLOT:-}" ]; then   # r13 8d / non-English target: the units the map assigns another language re-voiced in place on the accepted
  st polyglot                        # English timing (scripts/r13/polyglot.py), BEFORE the face (the lips follow the new speech); default off
  if ! python scripts/r13/polyglot.py --run $AD $([ -d $A/$HT/$ID ] && echo --htd $A/$HT/$ID) > $L/polyglot.log 2>&1; then
    for d in $AD $A/$HT/$ID; do for f in $d/en_*; do [ -f "$f" ] && { b=$(basename $f); rm -f $d/${b#en_}; cp $f $d/${b#en_}; }; done; done
    fail polyglot_failed PARTIAL_SUCCESS "see logs/polyglot.log: the English dub kept"; fi
fi

if [ "${DUB_PACE_FIX:-0}" = 1 ] && [ -z "${DUB_POLYGLOT:-}" ]; then   # r13 7.3 (default off): a phrase heard as rushed (speech rate > 1.25 x the video median,
  st pace_fix                  # or its tempo outside x0.75-1.25) is re-said shorter at natural pace in the same slot; the rest of the mix stays bit-identical
  if ! python scripts/r13/pace_fix.py --run $AD $([ -d $A/$HT/$ID ] && echo --htd $A/$HT/$ID) > $L/pace_fix.log 2>&1; then
    for d in $AD $A/$HT/$ID; do for f in $d/pre_pace_*; do [ -f "$f" ] && { b=$(basename $f); rm -f $d/${b#pre_pace_}; cp $f $d/${b#pre_pace_}; }; done; done
    fail pace_fix_failed PARTIAL_SUCCESS "see logs/pace_fix.log: the accepted dub kept"; fi
fi

VID=""; FACE=${DUB_FACE:-auto}; ROUTE=""
pick_engine() {   # face=auto: KeySync for every face. Irina 08.11 on the beard panels: "кейсинку явно получше стало" (KeySync beats LatentSync on beards),
  [ "$FACE" = auto ] || return 0   # so the route's beard -> latentsync choice is only an OPT-IN (DUB_BEARD_LATENTSYNC=1); DUB_FACE_ENGINE still overrides
  if [ "${DUB_BEARD_LATENTSYNC:-0}" = 1 ]; then FACE=${DUB_FACE_ENGINE:-$1}
  else FACE=${DUB_FACE_ENGINE:-keysync}; [ "$1" = latentsync ] && FL+=("beard_keysync"); fi; }
gpu_release   # r16 T (Q audit O4): the CPU work from here on (route encode / beard, ks_face master / landmarks / shots / crop) runs OUTSIDE the GPU lock;
              # the route active speaker (gpu_run) and ks_face.sh GPU section take it again
if [ -n "${DUB_ROUTE_JSON:-}" ] && [ "${DUB_ROUTE:-on}" != off ]; then   # r16 T: a given route.json (golden test / the backend) instead of computing one
  mkdir -p $W/route; cp $DUB_ROUTE_JSON $W/route/route.json; DUB_ROUTE=given; FL+=("face_route_given")
  ROUTE=$W/route/route.json; read -r KSA ENG < <(python3 -c "import json; d=json.load(open('$ROUTE')); print(int(bool(d.get('keep_source_all'))), d.get('face_engine') or 'keysync')")
  [ "$KSA" = 1 ] && FL+=("face_keep_source_no_visible_speaker"); pick_engine "$ENG"
fi
if [ "${E2E_LIPSYNC:-on}" != off ] && [[ " $PF " != *" no_face "* ]] && [ "$FACE" != latentsync ] && [ "${DUB_ROUTE:-on}" = on ]; then
  st face_route   # r16 Q (reports/experiments/r16-qa/ROUTING.md): which visible face speaks which turn (SyncNet on the SOURCE) + beard -> the engine
  RD=$W/route; mkdir -p $RD; MAN=$(ls $AD/${ID}_*.manifest.json 2>/dev/null | head -1)
  if [ -n "$RPRE" ]; then until [ -f $RPRE/finished ]; do sleep 0.5; done; fi   # r17 E2E_ROUTE_PRE: its master / beard, else computed here as before
  rmaster() { if [ -n "$RPRE" ] && [ -f $RPRE/master.done ]; then cp $RPRE/master_25fps.mp4 $RD/master_25fps.mp4; else
    ffmpeg -loglevel error -y -i $SRC -map 0:v:0 -map 0:a:0 -vf fps=25 -c:v libx264 -preset fast -crf 12 -pix_fmt yuv420p -c:a aac $RD/master_25fps.mp4; fi; }
  rbeard() { if [ -n "$RPRE" ] && [ -f $RPRE/beard.done ]; then cp $RPRE/beard.json $RD/beard.json && cat $RPRE/beard.log >> $L/face_route.log; else
    ${DUB_MEDIAPIPE_PY:-/venv/mediapipe/bin/python} scripts/quality/r16q_beard.py $SRC --out $RD/beard.json >> $L/face_route.log 2>&1; fi; }
  if rmaster \
     && gpu_run python scripts/quality/r16q_active_speaker.py --video $RD/master_25fps.mp4 --manifest $MAN --out $RD/as.json --device ${DUB_ROUTE_DEVICE:-cuda} > $L/face_route.log 2>&1 \
     && rbeard \
     && python3 scripts/r16q/face_route.py --active $RD/as.json --beard $RD/beard.json --out $RD/route.json >> $L/face_route.log 2>&1; then
    ROUTE=$RD/route.json; read -r KSA ENG < <(python3 -c "import json; d=json.load(open('$ROUTE')); print(int(bool(d.get('keep_source_all'))), d.get('face_engine') or 'keysync')")
    echo "== $ID route: keep_source_all $KSA, engine $ENG, faces $(python3 -c "import json; print([(f['mode'], round(f.get('silence_s', 0), 1)) for f in json.load(open('$ROUTE'))['faces']])")" | tee -a $L/e2e.log
    [ "$KSA" = 1 ] && FL+=("face_keep_source_no_visible_speaker")   # (made a visible PARTIAL_SUCCESS below)
    pick_engine "$ENG"
  else FL+=("face_route_failed"); echo "== $ID WARN face_route failed (see logs/face_route.log): the dubbed-units mapping is used" | tee -a $L/e2e.log; ROUTE=""; fi
fi
[ "$FACE" = auto ] && FACE=keysync
if [ "${E2E_LIPSYNC:-on}" = off ]; then FL+=("lipsync_off_by_request")   # r13 interface: the user asked for a voice-only dub
elif [[ " $PF " == *" no_face "* ]]; then FL+=("face_skipped_no_face")
elif [[ " ${FL[*]} " == *" face_keep_source_no_visible_speaker "* ]]; then   # r16: the dub voice belongs to no visible face: the source picture
  echo "== $ID off-screen voice: nobody on screen speaks the dubbed lines -> the source picture with the dub (SUCCESS + W_OFFSCREEN_VOICE, Irina 09.11)" | tee -a $L/e2e.log
elif [ "$FACE" = keysync ]; then   # r16 T: the KeySync chain (scripts/keysync/r16/ks_face.sh: K1 paste + K2 mouth + Q route, accepted by eye 07.11)
  st face_keysync                 # DUB_FACE=latentsync (or auto + a beard) = the r15 LatentSync path below
  FL+=("face_keysync")   # a GPU lock held since the audio stage (GPU_RDY) goes on to ks_face.sh's GPU section, which releases it before its CPU post
  bash scripts/keysync/r16/ks_face.sh $ID $SRC $AD $W $W/ks_face.mp4 $ROUTE > $L/face_keysync.log 2>&1; KRC=$?   # which releases it before its CPU post
  if [ $KRC = 0 ] && [ -s $W/ks_face.mp4 ]; then
    VID=$W/ks_face.mp4; grep -E "^KS_(FACES|PROBE|CHECK|PASTE|FACE_DONE)" $L/face_keysync.log | cut -c1-400 | tee -a $L/e2e.log
    python3 -c "import json,sys; d=json.load(open('$W/ks/faces.json')); sys.exit(0 if len(d['faces']) > 1 or d.get('source_tracks') or d['mode'] == 'multi_face' else 1)" 2>/dev/null && FL+=("face_keysync_multi_face")
    KS25=$W/ks/master_25fps.mp4   # the KeySync result is on the 25 fps master grid (no retime)
    for f in $(cat $W/ks/flags.txt 2>/dev/null); do FL+=("$f"); done   # ks_probe_failed / ks_probe_low_conf / ks_probe_clamped / ks_check_offset
  elif [ $KRC = 3 ]; then grep -E "^KS_FACE_KEEP_SOURCE" $L/face_keysync.log | tee -a $L/e2e.log   # no-face rule (too small / inset / absent) or off-screen voice: the source picture
    if grep -q -x ks_offscreen_voice $W/ks/flags.txt 2>/dev/null; then FL+=("ks_offscreen_voice")   # r16c T (Irina 09.11 rule 2): SUCCESS + W_OFFSCREEN_VOICE, no refund
    else fail face_keysync_no_face PARTIAL_SUCCESS "no face fit for lip-sync (too small, inset or absent): the source picture with the dub"; fi
  else fail face_gen_failed PARTIAL_SUCCESS "see logs/face_keysync.log: source picture used"; fi
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
  export R13_GPU_DONE_FILE=$W/gen_gpu_done; rm -f $R13_GPU_DONE_FILE   # read by the resident generator only (R10_GEN_EARLY_RELEASE)
  CFX="--codeformer optimized --codeformer-w 1.0 --codeformer-parse off --codeformer-batch 16"; CFDV=self   # r15 T: DUB_ALLOW_NC=0 -> the clean face
  if [ "$DUB_ALLOW_NC" = 1 ] && [ ! -s third_party/CodeFormer/weights/CodeFormer/codeformer.pth ]; then   # r16 T: the NC weights are not on this host / volume
    echo "== $ID WARN LatentSync face: DUB_ALLOW_NC=1 but third_party/CodeFormer/weights/CodeFormer/codeformer.pth is missing -> the CLEAN face (no CodeFormer)" | tee -a $L/e2e.log
    export DUB_ALLOW_NC=0; FL+=("codeformer_missing_clean_face"); fi
  [ "$DUB_ALLOW_NC" = 1 ] || { CFX="--codeformer off"; CFDV=""; FL+=("face_clean_no_codeformer"); }
  AUDIO=$A/$AT TAG=$GT SKIP_SCORE=1 VIDEO_EXTRA="$CFX --steps 10 --steps-silent 20 $REFX" \
    bash scripts/r7/r7_e2e.sh $ID > $L/face_gen.log 2>&1 & GENP=$!
  if [ "${R10_GEN_EARLY_RELEASE:-0}" = 1 ] && [ -n "${R10_HEAVY_LOCK:-}" ]; then   # r13: the resident generator signals the end of its GPU work (scripts/r10/gen_worker.py)
    while kill -0 $GENP 2>/dev/null; do [ -f $W/gen_gpu_done ] && { gpu_release; echo "== $ID heavy lock released early: generator GPU done $(date +%T)" | tee -a $L/e2e.log; break; }; sleep 0.5; done
  fi
  wait $GENP
  if [ -n "$SP" ]; then [ -f $B/$GT/work_${ID}_$GT/${ID}_${GT}_25fps.manifest.json ] || kill $SP 2>/dev/null; wait $SP 2>/dev/null; fi   # r13: no generator output -> stop the source prep (it would wait 1 h for the master)
  if [ -f $B/$GT/work_${ID}_$GT/${ID}_${GT}_25fps.manifest.json ]; then
    gpu_release   # r10 worker: the next job's audio + face_gen may start
    if [ -n "${R10_COMP_GATE:-}" ]; then st wait_comp; touch $W/comp_ready; until [ -f $W/comp_go ]; do sleep 1; done; fi   # r10 worker: the composite (~12 GB) starts only once no other job is in its audio stage (~30 GB)
    st face_comp
    [ "$E2E_FX" = 1 ] && export REP=1 CFD_LABEL=ia04 CFD_ARGS="--inner-alpha 0.4 ${CFD_EXTRA:-}"   # r9: short mouth glitches re-drawn in the layer, long ones -> smooth source fallback; CodeFormer detail at 0.4 inside the mouth
    [ "$E2E_FX" = 1 ] || { [ -n "${CFD_EXTRA:-}" ] && export CFD_ARGS="${CFD_ARGS:-} $CFD_EXTRA"; }
    FBL=fb7; FBA="$(cat reports/experiments/r8-face/fb7.args)"; SYX=""
    case "${E2E_SYNC:-0}" in   # r12: the r11 composite sync fix (reports/experiments/r11-composite-sync): k5 (user: "реально лучше") / k7 (k5 without the colour-match exclusion, gate-safer); needs E2E_FX=1 (the layer repair)
      1|k5|k7) FBL=fb7s; FBA="$FBA --open-gate 0.04 --dark-transient on --thr-wash-lips 0.2"; export REP_ARGS="--spikes off ${REP_ARGS:-}" REP_LABEL=rep2s
               SYX="--mouth-region 0.55"; [ "$E2E_SYNC" = k7 ] || SYX="$SYX --cm-mouth-excl 0.5";;
    esac
    if [ "$DUB_ALLOW_NC" != 1 ] && [ "${DUB_CLEAN_LIPCON:-0}" = 1 ]; then   # r15 F2 (agent F, experiment/r15-repair 6168df3; eye 05.11: "no CF + adaptive lipcon ~ prod CF"):
      LCX="--thr-lipcon 0.45 --lipcon-mode adaptive --lipcon-rel 0.6 --lipcon-win 25 --lipcon-max-frac 0.03"   # the clean face: a lip-vs-skin contrast dip
      FBL=${FBL}la; FBA="$FBA $LCX"; export REP_LABEL=${REP_LABEL:-rep2}la REP2_FB_ARGS="$(cat reports/experiments/r8-face/fb7.args) $LCX"   # (< 0.45 of the source AND < 0.6 x
      FL+=("face_clean_lipcon")   # the video's +-1 s median, <= 3 % of the frames) is re-drawn by the layer repair, re-scored after it (2nd pass); CPU only. Default OFF until F's gates pass
    fi
    SKIP_SCORE=1 TAG=$FT SRC_TAG=$GT CFD=$CFDV SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=$FBL FB_ARGS="$FBA" EXTRA="--tex-tau 1000 --tex-sigma 2.5 $SYX" \
      bash scripts/r7/r7_m7_comp.sh $ID > $L/face_comp.log 2>&1
    [ -f $B/$FT/${ID}_$FT.mp4 ] && VID=$B/$FT/${ID}_$FT.mp4 || fail face_comp_failed PARTIAL_SUCCESS "see logs/face_comp.log: source picture used"
  else fail face_gen_failed PARTIAL_SUCCESS "see logs/face_gen.log: source picture used"; fi
fi
gpu_release
[ -n "$VID" ] || VID=$SRC
for f in ks_probe_failed ks_probe_low_conf ks_check_offset ks_dissolve_hard_switch face_route_failed ks_shot_failed; do   # r16 T (Q audit O1): degraded sync is not SUCCESS
  if [[ " ${FL[*]} " == *" $f "* ]]; then ER+=("$f: lip-sync degraded (see logs)"); [ $STATUS = SUCCESS ] && STATUS=PARTIAL_SUCCESS; echo "== $ID PARTIAL_SUCCESS: $f" | tee -a $L/e2e.log; fi
done
if [ -n "$HIRES" ]; then   # r15 T: the face work was downscaled: back to the source resolution
  if [ "$VID" = "$SRC" ]; then VID=$HIRES   # no face result: the full-res source picture
  else
    st face_hires; HI=$HIRES; LOS=$SRC
    if [ -n "${KS25:-}" ]; then LOS=$KS25; HI=$W/hires_25fps.mp4   # KeySync: merge on its 25 fps grid (the full-res source at 25 fps, near-lossless)
      ffmpeg -loglevel error -y -i $HIRES -map 0:v:0 -an -vf fps=25 -c:v libx264 -preset veryfast -crf 12 -pix_fmt yuv420p $HI || HI=$HIRES; fi
    if python scripts/r15/hires_merge.py --hi $HI --lo-src $LOS --lo-comp $VID --out $W/face_hires.mp4 --json $W/face_hires.json --threads ${HIRES_THREADS:-16} > $L/face_hires.log 2>&1; then
      VID=$W/face_hires.mp4; tail -1 $L/face_hires.log | tee -a $L/e2e.log
    else VID=$HIRES; fail face_hires_failed PARTIAL_SUCCESS "see logs/face_hires.log: source picture used"; fi
  fi
fi

SUBS=${E2E_SUBS:-off}; WM=${E2E_WATERMARK:-}   # r13 spec block 9: burned subtitles (ASS, word timing on the stretched dub, 4 styles + karaoke) and / or a watermark; default off
if [ "$SUBS" != off ] || [ -n "$WM" ]; then   # E2E_SUBS=on E2E_SUBS_LANG=dub|src E2E_SUBS_STYLE=white|yellow|trendy|outline E2E_SUBS_KARAOKE=on|off; E2E_WATERMARK=<png>|text:DabAI E2E_WM_CORNER E2E_WM_OPACITY
  st subs   # CPU (libass + libx264 at the source size / fps); the picture is re-encoded ONCE, then muxed with each bed below
  SA=(--no-subs)
  if [ "$SUBS" != off ]; then MAN=$(ls $AD/${ID}_*.manifest.json 2>/dev/null | head -1)
    SA=(--manifest "$MAN" --lang ${E2E_SUBS_LANG:-dub} --style ${E2E_SUBS_STYLE:-yellow} --karaoke ${E2E_SUBS_KARAOKE:-on}); fi
  [ -n "$WM" ] && SA+=(--watermark "$WM" --wm-corner ${E2E_WM_CORNER:-br} --wm-opacity ${E2E_WM_OPACITY:-0.6})
  if SUBS_THREADS=${SUBS_THREADS:-8} python scripts/r7/r7_subs.py burn --video $VID "${SA[@]}" --out $W/final_subs.mp4 > $L/subs.log 2>&1; then
    VID=$W/final_subs.mp4; [ -f $W/final_subs.mp4.ass ] && cp $W/final_subs.mp4.ass $OUT/${ID}.ass   # the ASS for a player / the backend (soft subs)
  else fail subs_failed PARTIAL_SUCCESS "see logs/subs.log: the output has no subtitles / watermark"; fi
fi

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
  gpu_run python scripts/r7/gap_metrics.py --tag $AT --clips $O/gaps $ID || echo "WARN gaps spleeter"
  if [ -f $A/$HT/$ID/final_mix_aac.m4a ]; then mkdir -p $A/$HT/cov
    python scripts/r8/mixdrop.py --run $A/$HT/$ID --id $ID --label $HT --out $A/$HT/cov/mix_$ID | tail -1
    python scripts/r7/clicks.py --tag $HT --prev $AT --json $A/$HT/cov/clicks.json $ID
    gpu_run python scripts/r7/gap_metrics.py --tag $HT --prev $AT --clips $A/$HT/gaps $ID || echo "WARN gaps demucs"; fi
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
