#!/bin/bash
# Round 7 AUDIO publish (= r6_publish.sh + the r7 gap listen clips and the placement rows): gate NEW vs PREV (r6_audio_v1 reproduced on this server)
# on all five videos; WATCH=1 / FORCE_WATCH=1 publish a WATCH candidate (failing rows listed in review.md, no READY):
#   outbox/<id>_r7_audio_<ver>[_WATCH]_over-original.mp4, _AB_prev-then-new.mp4, _review.md (r4_review 10-s rows + known moments + gate rows +
#   lip placement + gap rows), _gaps_AB.m4a (every dub gap: one tick = PREV, two ticks = NEW), r7_audio_<ver>_regression.md
#   handoff/out/F/audio-r7-<ver>/ (dubbed_16k.wav, dubbed_24k.wav, final_mix_aac.m4a, manifest per video, README.txt, READY unless WATCH)
#     bash scripts/r7/r7_publish.sh <ver> <new_tag> <prev_tag>
set -u
DUB_ROOT=${DUB_ROOT:-/workspace/dub/dubai}; export DUB_ROOT; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
VER=$1; NEW=$2; PREV=$3; A=/workspace/dub/artifacts/r4a; OB=${OB:-/workspace/dub/outbox}; HB=${HB:-/workspace/dub/handoff/out/F}; C=$(git rev-parse --short HEAD)
REG=$A/$NEW/r7_audio_${VER}_regression.md
python scripts/quality/r4_audio_gate.py --prev $PREV --prev-run $PREV --new $NEW --title "r7_audio_$VER (commit $C)" --out $REG > /dev/null; rc=$?
KIND=""; if [ $rc -ne 0 ]; then
  [ "${FORCE_WATCH:-0}" = 1 ] || { echo "GATE FAILED -> not published (see $REG)"; grep -A40 "## Verdict" $REG; exit 1; }; KIND="_WATCH"; fi
[ "${WATCH:-0}" = 1 ] && KIND="_WATCH"
python scripts/r7/gap_metrics.py --tag $PREV --asr on 01 02 03 04 05 > /dev/null 2>&1   # PREV gap rows for the AB tables
python scripts/r7/gap_metrics.py --tag $NEW --prev $PREV --clips $A/$NEW/gaps 01 02 03 04 05 > /dev/null 2>&1
H=$HB/audio-r7-$VER; mkdir -p $OB $H
for id in 01 02 03 04 05; do
  src=$(ls /workspace/dub/inputs/test_videos_raw/$id.* | head -1); N=$OB/${id}_r7_audio_${VER}${KIND}
  cp $A/$NEW/${id}_over-original.mp4 ${N}_over-original.mp4
  printf "file '%s'\nfile '%s'\n" $A/$PREV/${id}_over-original.mp4 $A/$NEW/${id}_over-original.mp4 > $A/$NEW/_ab_$id.txt
  ffmpeg -loglevel error -y -f concat -safe 0 -i $A/$NEW/_ab_$id.txt -c copy ${N}_AB_prev-then-new.mp4; rm -f $A/$NEW/_ab_$id.txt
  cp $A/$NEW/gaps/${id}_gaps_AB.m4a ${N}_gaps_AB.m4a
  python scripts/quality/r4_review.py --id $id --prev-series $A/$PREV/$id.series.npz --new-series $A/$NEW/$id.series.npz --new-manifest $A/$NEW/$id/${id}_r4b.manifest.json \
    --prev-manifest $A/$PREV/$id/${id}_r4b.manifest.json --prev-metrics $A/$PREV/$id.metrics.json --new-metrics $A/$NEW/$id.metrics.json --out ${N}_review.md \
    --title "(r7_audio_$VER$KIND, commit $C; PREVIOUS = $PREV, NEW = $(basename ${N}_over-original.mp4))" > /dev/null
  python scripts/r5/r5_known_check.py --prev $PREV --new $NEW --id $id >> ${N}_review.md
  python scripts/r7/r7_rows.py --prev $PREV --new $NEW --id $id >> ${N}_review.md
  cat $A/$NEW/gaps/${id}_gaps.md >> ${N}_review.md
  [ -n "$KIND" ] && { echo; echo "## Gate rows that fail on $id"; grep "^- $id:" $REG; } >> ${N}_review.md
  mkdir -p $H/$id; cp $A/$NEW/$id/dubbed_16k.wav $A/$NEW/$id/dubbed_24k.wav $A/$NEW/$id/final_mix_aac.m4a $A/$NEW/$id/${id}_r4b.manifest.json $H/$id/
done
cp $REG $OB/; cp $REG $H/
cat > $H/README.txt <<EOF
r7_audio_$VER$KIND commit $C tag $NEW; gate vs $PREV: $([ $rc -eq 0 ] && echo PASS || echo "FAIL, see r7_audio_${VER}_regression.md")
= r6_audio_v1/v2 (takes, tempo, pause positions, bed mix) + r7:
  lip placement: where the face is visible, dub blocks are moved (never stretched) into the source speech spans; splits only at pb1 boundaries.
  gap clean: in dub gaps the bed's leaked voice / separator bursts -> room tone, denoise where the source speaks.
Per video: dubbed_16k.wav / dubbed_24k.wav (dub voice, NEW timing), final_mix_aac.m4a (final mix, -16 LUFS), <id>_r4b.manifest.json
  (units with moved placements; lip_place.dub_speech = the dub's speech spans on the new timeline, use them (or VAD on dubbed_16k.wav) for
  --ls-split-dub-silence; lip_place.shifts = every block's move).
EOF
du -sh $H; [ -z "$KIND" ] && touch $H/READY; echo "PUBLISHED r7_audio_$VER$KIND -> $OB, handoff $H"
