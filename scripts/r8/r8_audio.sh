#!/bin/bash
# Round 8 AUDIO (audio-r8-v1): the dub speaks whenever the source speaks, on and off screen (management on final_r7: 02 / 05 "keeps
# disappearing", the silent "about to vomit" mouth), and sentence starts follow the speaker (user on 04: "strange starts and pauses").
# On a base of r6/r7 takes (REUSE tag with mix_src/<id> = work dirs on the pre-placement timeline; v3a1 rebuilt by scripts/r8/rebase.py):
#   1. retake.py --cover-all on       units whose source speech outlasts the take (on OR off screen) re-translated longer (Qwen 14B duration-aware,
#                                     same judge) and re-taken (normal / deliberate pace); every other unit bit-identical
#   2. burst_align.py --cover-all on  the take aligned to the source bursts, tempo R8_TMIN..R8_TMAX (0.9-1.1), where it beats the placement
#   3. the r7 chain with lip_place --pace on --cover-all on (onset anchor, sentence gap floor, gate-aware blocks, dictation splits), hush, remix
#   then scripts/r8/coverage.py + pacing.py rows per video (<tag>/cov/).
#     REUSE=v3a1r R7_PREV=r8repro2 bash scripts/r8/r8_audio.sh <tag> [ids]      R8_ALIGN=off skips step 2
set -u
DUB_ROOT=$(cd "$(dirname "$0")/../.." && pwd); cd $DUB_ROOT; R=$DUB_ROOT; . scripts/env.sh >/dev/null; DUB_ROOT=$R; export DUB_ROOT
TAG=$1; shift; IDS=${*:-01 02 03 04 05}; A=/workspace/dub/artifacts/r4a; BASE=${REUSE:?REUSE=<base tag>}; S=$A/${TAG}src; mkdir -p $S/mix_src $S/src16
s=$(date +%s)
for id in $IDS; do
  w=$(readlink -f $A/$BASE/mix_src/$id); s16=$S/src16/${id}_16k.wav; fc=$A/sep/$id/source_mediapipe.npz
  ffmpeg -loglevel error -y -i "$(ls /workspace/dub/inputs/test_videos_raw/$id.* | head -1)" -vn -ac 1 -ar 16000 -c:a pcm_s16le $s16
  f=""; for t in ${R7_PREV:-} $BASE; do [ -f $A/$t/$id.metrics.json ] && { f="--faith $A/$t/$id.metrics.json"; break; }; done
  t0=$(date +%s.%N)
  PYTHONPATH=$DUB_ROOT/third_party/llama_cuda:${PYTHONPATH:-} python scripts/r7/retake.py --work $w --out $S/rt/$id --face $fc $f --src16 $s16 --cover-all on ${R8_RETAKE_ARGS:-} > $S/rt_$id.log 2>&1 \
    || { echo "FAIL retake $id (see $S/rt_$id.log)"; continue; }
  t1=$(date +%s.%N); W=$S/rt/$id
  if [ "${R8_ALIGN:-on}" = on ]; then
    python scripts/r7/burst_align.py --work $W --out $S/ba/$id --face $fc --cover-all on --tmin ${R8_TMIN:-0.9} --tmax ${R8_TMAX:-1.1} ${R8_ALIGN_ARGS:-} > $S/ba_$id.log 2>&1 && W=$S/ba/$id || echo "WARN burst_align $id (see $S/ba_$id.log)"
  fi
  t2=$(date +%s.%N)
  if [ "${R8_CLOSE:-off}" = on ]; then   # v2: the take's inner pauses where the source talks closed, the piece slowed to keep its span (scripts/r8/close_inner.py)
    python scripts/r8/close_inner.py --work $W --out $S/ci/$id --tmin ${R8_TMIN:-0.9} > $S/ci_$id.log 2>&1 && W=$S/ci/$id || echo "WARN close_inner $id (see $S/ci_$id.log)"
  fi
  t3=$(date +%s.%N); ln -sfn $W $S/mix_src/$id
  python3 -c "import json,sys; t=[float(x) for x in sys.argv[1:]]; json.dump({'retake':t[1]-t[0],'burst_align':t[2]-t[1],'close_inner':t[3]-t[2]},open('$S/$id.steps.json','w'))" $t0 $t1 $t2 $t3
  echo "== $id src $(( $(date +%s) - s )) s"
done
RETAKE=off REUSE=${TAG}src LIP_ARGS="--max-shift 2.0 --grow-unit 1.5 --grow 0.5 --pace on --cover-all on ${R8_LIP_ARGS:-}" \
  REMIX_ARGS="--src-duck-db -6 --unit-level off --gap-bed-cap-db 10 --gap-bed-cap-scope edges --gap-clean on --gap-clean-nr-ns 0.7 --dub-gate on" \
  R7_PREV=${R7_PREV:-} bash scripts/r7/r7_audio.sh $TAG $IDS
mkdir -p $A/$TAG/cov
for id in $IDS; do
  python scripts/r8/coverage.py --run $A/$TAG/$id --id $id --label $TAG --face $A/sep/$id/source_mediapipe.npz --out $A/$TAG/cov/cov_$id 2>/dev/null | tail -1
  python scripts/r8/pacing.py --run $A/$TAG/$id --id $id --label $TAG --face $A/sep/$id/source_mediapipe.npz --out $A/$TAG/cov/pace_$id 2>/dev/null | tail -1
done
echo "R8_AUDIO_DONE $TAG $(( $(date +%s) - s )) s"
