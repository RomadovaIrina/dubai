#!/bin/bash
# r8-sep: the audio-r8-v4 mix (tag r8v4c) rebuilt with ONLY the bed changed: same dub takes / placement / hush / soften (r8v4c/mix_src), the same
# remix flags as scripts/r8/r8_audio.sh, plus --sep / --sep-bg and optional relaxations; then the gate rows vs r8v4c (gap words + voice, mixdrop, clicks).
#     bash scripts/r8/sep_mix.sh <tag> <sep> <stems|residual> ["extra r3_remix flags"] [ids]
set -u
cd /workspace/dub/sep; R=/workspace/dub/sep; . scripts/env.sh >/dev/null; export DUB_ROOT=$R PYTHONPATH=/workspace/dub/sep_pylib:${PYTHONPATH:-} DUB_ALLOW_NC_BED=1
T=$1; SEP=$2; BG=$3; EXTRA=${4:-}; shift 4; IDS=${*:-01 02 03 04 05}
A=/workspace/dub/artifacts/r4a; P=${SEP_PREV:-r8v4c}; O=$A/$T; mkdir -p $O/runs $O/cov; ln -sfn $A/${SEP_SRC:-$P}/mix_src $O/mix_src   # SEP_SRC = the tag whose dub (mix_src) is used (default SEP_PREV); SEP_PREV = the tag whose dub (mix_src) and gate rows are used
BASE="--src-duck-db -6 --unit-level off --gap-bed-cap-db 10 --gap-bed-cap-scope edges --gap-clean on --gap-clean-nr-ns 0.7 --dub-gate on --gap-clean-floor 3"
echo "sep_mix $T commit $(git rev-parse --short HEAD) sep=$SEP bg=$BG BASE='$BASE' EXTRA='$EXTRA' (dub = $P/mix_src)" >> $O/README.txt
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 DUB_THREADS=6 CUDA_VISIBLE_DEVICES=""   # remix + detectors on the CPU; only the gap ASR takes the GPU (gpu.lock)
for id in $IDS; do
  python scripts/quality/r3_remix.py --src $A/${SEP_SRC:-$P}/mix_src --out $O $BASE $EXTRA --sep $SEP --sep-bg $BG $id > $O/runs/$id.remix.log 2>&1 || { echo "FAIL remix $id"; continue; }
  ffmpeg -loglevel error -y -i $(ls /workspace/dub/inputs/test_videos_raw/$id.* | head -1) -i $O/$id/final_mix_aac.m4a -map 0:v:0 -map 1:a:0 -c copy -shortest $O/${id}_over-original.mp4
  python scripts/r8/mixdrop.py --run $O/$id --id $id --label $T --out $O/cov/mix_$id 2>/dev/null | tail -1
done
python scripts/r7/clicks.py --tag $T --prev $P --json $O/cov/clicks.json $IDS 2>/dev/null
unset CUDA_VISIBLE_DEVICES
if [ "${SEP_ASR:-on}" = on ]; then flock /workspace/dub/gpu.lock python scripts/r7/gap_metrics.py --tag $T --prev $P --clips $O/gaps $IDS > $O/runs/gaps.log 2>&1 || echo "WARN gaps"
else CUDA_VISIBLE_DEVICES="" python scripts/r7/gap_metrics.py --tag $T --asr off $IDS > $O/runs/gaps.log 2>&1; fi   # SEP_ASR=off: no Whisper (the GPU is busy), the rest of the gate
grep -h "^0[1-5]" $O/runs/gaps.log | cut -c1-400
echo "SEP_MIX_DONE $T"
