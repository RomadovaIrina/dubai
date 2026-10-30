#!/bin/bash
# Round 5 AUDIO publish: gate NEW vs pb1 (reproduced on this server = prev and prev-run) on all five videos -> only on PASS (or FORCE_WATCH=1, which
# publishes a WATCH candidate with the failing rows in review.md):
#   outbox/<id>_r5_audio_<ver>_over-original.mp4, <id>_r5_audio_<ver>_AB_prev-then-new.mp4, <id>_r5_audio_<ver>_review.md
#   (r4_review.py 10-s rows + every known moment of r4_audio_v1_known_issues.md / pb1_listen.md re-measured), r5_audio_<ver>_regression.md,
#   handoff/out/F/audio-r5-<ver>/ (dubbed_16k.wav, dubbed_24k.wav, final_mix_aac.m4a, manifest per video, READY)
#     bash scripts/r5/r5_publish.sh <ver> <new_tag> [prev_tag=pb1]
set -u
DUB_ROOT=${DUB_ROOT:-/workspace/dub/dubai}; export DUB_ROOT; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
VER=$1; NEW=$2; PREV=${3:-pb1}; A=/workspace/dub/artifacts/r4a; OB=/workspace/dub/outbox; C=$(git rev-parse --short HEAD)
REG=$A/$NEW/r5_audio_${VER}_regression.md
python scripts/quality/r4_audio_gate.py --prev $PREV --prev-run $PREV --new $NEW --title "r5_audio_$VER (commit $C)" --out $REG > /dev/null; rc=$?
KIND=""; if [ $rc -ne 0 ]; then
  [ "${FORCE_WATCH:-0}" = 1 ] || { echo "GATE FAILED -> not published (see $REG)"; grep -A40 "## Verdict" $REG; exit 1; }; KIND="_WATCH"; fi
H=/workspace/dub/handoff/out/F/audio-r5-$VER; mkdir -p $OB $H
for id in 01 02 03 04 05; do
  src=$(ls ${DUB_INPUTS:-/workspace/dub/inputs/test_videos_raw}/$id.* | head -1); N=$OB/${id}_r5_audio_${VER}${KIND}
  cp $A/$NEW/${id}_over-original.mp4 ${N}_over-original.mp4
  printf "file '%s'\nfile '%s'\n" $A/$PREV/${id}_over-original.mp4 $A/$NEW/${id}_over-original.mp4 > $A/$NEW/_ab_$id.txt
  ffmpeg -loglevel error -y -f concat -safe 0 -i $A/$NEW/_ab_$id.txt -c copy ${N}_AB_prev-then-new.mp4; rm -f $A/$NEW/_ab_$id.txt
  python scripts/quality/r4_review.py --id $id --prev-series $A/$PREV/$id.series.npz --new-series $A/$NEW/$id.series.npz --new-manifest $A/$NEW/$id/${id}_r4b.manifest.json \
    --prev-manifest $A/$PREV/$id/${id}_r4b.manifest.json --prev-metrics $A/$PREV/$id.metrics.json --new-metrics $A/$NEW/$id.metrics.json --out ${N}_review.md \
    --title "(r5_audio_$VER$KIND, commit $C; PREVIOUS = pb1, NEW = $(basename ${N}_over-original.mp4))" > /dev/null
  python scripts/r5/r5_known_check.py --prev $PREV --new $NEW --id $id >> ${N}_review.md
  [ -n "$KIND" ] && { echo "## Gate rows that fail on $id"; grep "^- $id:" $REG; } >> ${N}_review.md
  mkdir -p $H/$id; cp $A/$NEW/$id/dubbed_16k.wav $A/$NEW/$id/dubbed_24k.wav $A/$NEW/$id/final_mix_aac.m4a $A/$NEW/$id/${id}_r4b.manifest.json $H/$id/
done
cp $REG $OB/; cp $REG $H/
echo "r5_audio_$VER$KIND commit $C tag $NEW; gate vs pb1: $([ $rc -eq 0 ] && echo PASS || echo FAIL, see regression.md). Per video: dubbed_16k.wav / dubbed_24k.wav (dub voice), final_mix_aac.m4a (final mix, -16 LUFS), <id>_r4b.manifest.json (units, dub timing: dub-silence spans for --ls-split-dub-silence)" > $H/README.txt
du -sh $H; [ -z "$KIND" ] && touch $H/READY; echo "PUBLISHED r5_audio_$VER$KIND -> $OB, handoff $H"
