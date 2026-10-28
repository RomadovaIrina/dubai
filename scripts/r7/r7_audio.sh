#!/bin/bash
# Round 7 AUDIO: ONE command, all five videos, the same flags for every video = r6_audio_v2 (scripts/r6/r6_audio.sh: takes, tempo, pauses, bed) +
#   LIP_PLACE=on   dub blocks moved (no stretch, no new pause) into the source speech spans where the face is visible (scripts/r7/lip_place.py)
#   --gap-clean on bed in dub gaps: residual-voice / burst detector -> room tone, denoise where the source speaks (stationary over a room tone,
#                  else non-stationary 0.7: 05 residue -22.5 -> -29 dB) (scripts/r7/gap_clean.py)
#   On the published r6_audio_v1 takes (already unit-levelled): REUSE=r6v1src REMIX_ARGS="... --unit-level off ..." (README_r7.md)
#   v2: --dub-gate on (the take's noisy pauses >= 0.6 s -> -30 dB, sin^2 20-ms ramps: the 04 crackle), lip limits 2.0 / 1.5 / 0.5 s, 20-ms equal-power fades
#   then the gap metric + listen clips (scripts/r7/gap_metrics.py, vs R7_PREV when set).
#   v3a.1 (default, HUSH_PLACED=on): non-word TTS material at re-taken piece edges ducked on the placed dub, timing kept (scripts/r7/hush_placed.py)
#   v3a (default, RETAKE=on; off = v2): units where the on-screen speaker outlasts the take, or the judge scored < 4, re-translated (Qwen 14B,
#     duration-aware) and re-taken (Chatterbox) before lip placement; faith rows from <REUSE or the fresh <tag>pre run, else R7_PREV> metrics.
#     bash scripts/r7/r7_audio.sh <tag> [ids]       REUSE=<tag> = take the TTS of an earlier r6/r7 run (no Qwen / Chatterbox); R6_QWEN=14b|7b
set -u
DUB_ROOT=${DUB_ROOT:-/workspace/dub/dubai}; export DUB_ROOT; cd $DUB_ROOT; TAG=$1; shift; IDS=${*:-01 02 03 04 05}
export HUSH_PLACED=${HUSH_PLACED:-on} LIP_PLACE=${LIP_PLACE:-on} LIP_ARGS="${LIP_ARGS:---max-shift 2.0 --grow-unit 1.5 --grow 0.5}" REMIX_ARGS="${REMIX_ARGS:---src-duck-db -6 --unit-level on --gap-bed-cap-db 10 --gap-bed-cap-scope edges --gap-clean on --gap-clean-nr-ns 0.7 --dub-gate on}"
s=$(date +%s); A=/workspace/dub/artifacts/r4a
if [ "${RETAKE:-on}" = on ]; then   # v3a (default): longer takes where the speaker is on screen (scripts/r7/retake.py), then the r7 chain on them
  R=$DUB_ROOT; . scripts/env.sh >/dev/null; DUB_ROOT=$R; export DUB_ROOT   # env.sh resets DUB_ROOT
  BASE=${REUSE:-}
  if [ -z "$BASE" ]; then BASE=${TAG}pre
    if [ "${R7_LEAN:-off}" = on ]; then R4_PIPELINE_ONLY=1 R5_FAITH=off bash scripts/r6/r6_audio.sh $BASE $IDS || exit 1   # speed track: the takes only; retake judges
    else RETAKE=off bash scripts/r7/r7_audio.sh $BASE $IDS || exit 1; fi; fi   # fresh takes + their faith judge (+ a mix nobody uses)
  mkdir -p $A/${TAG}src/mix_src $A/${TAG}src/src16
  for id in $IDS; do
    w=$(readlink -f $A/$BASE/mix_src/$id); o=$A/${TAG}src/runs/$id/work; s16=$A/${TAG}src/src16/${id}_16k.wav
    ffmpeg -loglevel error -y -i "$(ls /workspace/dub/inputs/test_videos_raw/$id.* | head -1)" -vn -ac 1 -ar 16000 -c:a pcm_s16le $s16
    f=""; for t in $BASE ${R7_PREV:-}; do [ -f $A/$t/$id.metrics.json ] && { f="--faith $A/$t/$id.metrics.json"; break; }; done   # same text => same judge rows
    [ "${R7_LEAN:-off}" = on ] && [ -z "${REUSE:-}" ] && f="--faith judge"
    fc=$A/sep/$id/source_mediapipe.npz; [ -f $fc ] && fc="--face $fc" || fc=""
    t0=$(date +%s.%N); PYTHONPATH=$DUB_ROOT/third_party/llama_cuda:${PYTHONPATH:-} python scripts/r7/retake.py --work $w --out $o $fc $f --src16 $s16 > $A/${TAG}src_$id.log 2>&1 \
      && ln -sfn $o $A/${TAG}src/mix_src/$id || { echo "FAIL retake $id (see $A/${TAG}src_$id.log)"; exit 1; }
    echo "RETAKE_TIME $id $(python3 -c "import time;print(round(time.time()-$t0,1))")" >> $A/${TAG}src/times.txt
  done
  export REUSE=${TAG}src
fi
bash scripts/r6/r6_audio.sh $TAG $IDS
. scripts/env.sh >/dev/null
python scripts/r7/gap_metrics.py --tag $TAG ${R7_PREV:+--prev $R7_PREV} --clips $A/$TAG/gaps $IDS > $A/$TAG/runs/gaps.log 2>&1 || echo "WARN gaps"
echo "R7_AUDIO_DONE $TAG $(( $(date +%s) - s )) s"
