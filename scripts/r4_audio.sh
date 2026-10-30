#!/bin/bash
# Round 4 AUDIO agent: ONE command, all five videos from the raw input, the same flags for every video (no per-video settings).
#   1. audio stage = v4's audio-r4 (r3_audio_r4.sh flags: r2 policy, speakable text, lexicon, cut snap, equal-power fades, Qwen on the GPU)
#   2. bed remix + static loudness (r3_remix.py; v4 = audio-r4b defaults)
#   3. new audio over the UNTOUCHED source video (<id>_over-original.mp4) + r4_audio_metrics.py
#     bash scripts/r4_audio.sh <tag> [ids]     -> /workspace/dub/artifacts/r4a/<tag>/{<id>/ (mix dir), runs/, <id>_over-original.mp4, <id>.metrics.json}
# env: AUDIO_EXTRA = extra run_clean_pipeline_05.py flags; REMIX_ARGS = r3_remix.py flags; REUSE=<tag> = take the TTS run of another tag (bed-only A/B);
#      PAR = videos in parallel on the GPU (default 2)
set -u
DUB_ROOT=${DUB_ROOT:-/workspace/dub/dubai}; export DUB_ROOT; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
TAG=$1; shift; IDS=${*:-01 02 03 04 05}; A=/workspace/dub/artifacts/r4a; O=$A/$TAG; mkdir -p $O/runs $O/mix_src $A/sep
export PYTHONPATH=third_party/llama_cuda${PYTHONPATH:+:$PYTHONPATH} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True DUB_CUT_SNAP=1 DUB_EQPOWER_FADE=1
echo "r4_audio $TAG commit $(git rev-parse --short HEAD) AUDIO_EXTRA='${AUDIO_EXTRA:-}' REMIX_ARGS='${REMIX_ARGS:-}' REUSE='${REUSE:-}'" >> $O/README.txt
TH=$(( 24 / ${PAR:-2} )); [ $TH -lt 2 ] && TH=2
one() {
  id=$1; src=$(ls ${DUB_INPUTS:-/workspace/dub/inputs/test_videos_raw}/$id.* | head -1); s=$(date +%s); t_start=$(date +%s.%N)
  if [ -n "${REUSE:-}" ]; then W=$(readlink -f $A/$REUSE/mix_src/$id); else
    python ${R9_PIPE_PY:-scripts/pilot/run_clean_pipeline_05.py} --input $src --output $O/runs/${id}_r4a.mp4 --target-lang en --alignment burst \
      --stop-after audio --codeformer off --face-stack mediapipe --audio-policy r2 --qwen-gpu-layers -1 --tts-text-norm on --qa-whisper-compute int8_float16 \
      ${AUDIO_EXTRA:-} --work-dir $O/runs/$id > $O/runs/${id}.log 2>&1 || { echo "FAIL audio $id"; return; }
    W=$O/runs/$id/work_$(basename ${src%.*}); cp $O/runs/${id}_r4a.manifest.json $W/${id}_r4a.manifest.json
  fi
  t_audio=$(date +%s.%N)
  if [ "${R4_PIPELINE_ONLY:-0}" = 1 ]; then ln -sfn $W $O/mix_src/$id; echo "== $id $TAG $(( $(date +%s) - s )) s (pipeline only)"; \
    python3 -c "import json; json.dump({'audio_or_reuse': $t_audio - $t_start}, open('$O/runs/${id}.steps.json','w'))"; return; fi   # speed track R7_LEAN: the retake base needs no mix
  export OMP_NUM_THREADS=$TH MKL_NUM_THREADS=$TH DUB_THREADS=$TH   # CPU quota ~30: remix / metrics of PAR videos must not oversubscribe (bed A/B hit load 475)
  [ -d $A/sep/$id ] || { mkdir -p $A/sep/$id; cp $W/bg_44k.wav $W/vocals_44k.wav $A/sep/$id/; }
  if [ "${LIP_PLACE:-off}" = on ]; then   # r7: dub blocks moved into the source speech spans where the face is visible (scripts/r7/lip_place.py)
    python -c "import sys; sys.path.insert(0,'scripts/pilot'); import clean_face_stack as c, pathlib; c.run_mediapipe(pathlib.Path('$src'), pathlib.Path('$A/sep/$id/source_mediapipe.npz'))" > /dev/null 2>&1
    python scripts/r7/lip_place.py --work $W --out $O/placed/$id --manifest $(ls $W/${id}_*.manifest.json | head -1) --face $A/sep/$id/source_mediapipe.npz ${LIP_ARGS:-} > $O/runs/${id}.place.log 2>&1 || { echo "FAIL place $id"; return; }
    W=$O/placed/$id; fi
  if [ "${PHRASE_FIT:-off}" = on ]; then   # r8 v4: every placed piece re-timed as a whole (tempo 0.9-1.1, start +-0.5 s) to cover the source speech (scripts/r8/phrase_fit.py)
    python scripts/r8/phrase_fit.py --placed $W --out $O/pfit/$id > $O/runs/${id}.pfit.log 2>&1 && W=$O/pfit/$id || echo "WARN phrase_fit $id"; fi
  t_place=$(date +%s.%N)
  if [ "${HUSH_PLACED:-off}" = on ]; then   # r7_audio_v3a.1: non-word TTS material at re-taken piece edges -> -30 dB, timing kept (scripts/r7/hush_placed.py)
    python scripts/r7/hush_placed.py --placed $W --out $O/hushed/$id > $O/runs/${id}.hush.log 2>&1 || { echo "FAIL hush $id"; return; }
    W=$O/hushed/$id; fi
  if [ "${FINAL_SOFTEN:-off}" = on ]; then   # r8 v4: the last abrupt word ends (hush / gate / placement edges) softened on the final dub (scripts/r8/soften_edges.py)
    python scripts/r8/soften_edges.py --work $W --out $O/soft/$id > $O/runs/${id}.soft.log 2>&1 && W=$O/soft/$id || echo "WARN final soften $id"; fi
  t_hush=$(date +%s.%N)
  ln -sfn $W $O/mix_src/$id
  [ -n "${R9_ON_MIXSRC:-}" ] && { ( eval "$R9_ON_MIXSRC" ) < /dev/null > /dev/null 2>&1 & }   # r9 speed O: the final dub exists -> the demucs remix starts (scripts/r9/sep_overlap.sh)
  python scripts/quality/r3_remix.py --src $O/mix_src --out $O ${REMIX_ARGS:-} $id > $O/runs/${id}.remix.log 2>&1 || { echo "FAIL remix $id"; return; }
  t_remix=$(date +%s.%N)
  ffmpeg -loglevel error -y -i $src -i $O/$id/final_mix_aac.m4a -map 0:v:0 -map 1:a:0 -c copy -shortest $O/${id}_over-original.mp4
  [ "${METRICS:-on}" = on ] && python scripts/quality/r4_audio_metrics.py --mix $O/${id}_over-original.mp4 --src $src --sep $A/sep/$id --run $O/$id --manifest $O/$id/${id}_r4b.manifest.json --wer --series $O/${id}.series.npz \
    --json $O/${id}.metrics.json > $O/runs/${id}.metrics.log 2>&1 || echo "WARN metrics $id"
  t_end=$(date +%s.%N)
  python3 -c "import json,sys; t=[float(x) for x in sys.argv[1:]]; json.dump({'audio_or_reuse':t[1]-t[0],'lip_place':t[2]-t[1],'hush':t[3]-t[2],'remix':t[4]-t[3],'mux+metrics(not production)':t[5]-t[4]},open('$O/runs/${id}.steps.json','w'))" $t_start $t_audio $t_place $t_hush $t_remix $t_end
  echo "== $id $TAG $(( $(date +%s) - s )) s"
}
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv,noheader,nounits -lms 1000 > $O/nvsmi.csv 2>/dev/null & NV=$!
P=(); for id in $IDS; do one $id & P+=($!); [ ${#P[@]} -ge ${PAR:-2} ] && { wait ${P[0]}; P=("${P[@]:1}"); }; done; for p in "${P[@]}"; do wait $p; done
kill $NV 2>/dev/null; echo "peak VRAM MiB $(awk -F, '{if($2>m)m=$2}END{print m}' $O/nvsmi.csv)" | tee -a $O/README.txt
echo R4_AUDIO_DONE $TAG
