#!/bin/bash
# Round 4 AUDIO publish: regression gate NEW vs PREVIOUS on all five videos -> only on PASS: outbox/<id>_r4_audio_<ver>_over-original.mp4,
# <id>_review.md, <id>_r4_audio_<ver>_AB_prev-then-new.mp4 (the same original video with the PREVIOUS audio, then the NEW one), r4_audio_<ver>_regression.md,
# and the video-agent handoff handoff/out/V/audio-r4a-<ver>/ (dubbed_16k.wav, final_mix_aac.m4a, manifests, READY).
#   bash scripts/r4_audio_publish.sh <ver> <new_tag> [prev_tag (mix-only metrics, default ref_v4)] [prev_run (stem metrics, default v4repro)] [prev_media_dir]
set -u
DUB_ROOT=/workspace/dub/dubai; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
VER=$1; NEW=$2; PREV=${3:-ref_v4}; PRUN=${4:-v4repro}; PMED=${5:-/workspace/dub/reference/v4}; A=/workspace/dub/artifacts/r4a; OB=/workspace/dub/outbox
REG=$A/$NEW/r4_audio_${VER}_regression.md; C=$(git rev-parse --short HEAD)
python scripts/quality/r4_audio_gate.py --prev $PREV --prev-run $PRUN --new $NEW --title "r4_audio_$VER (commit $C)" --out $REG > /dev/null; rc=$?
[ $rc -eq 0 ] || { echo "GATE FAILED -> not published (see $REG)"; grep -A30 "## Verdict" $REG; exit 1; }
H=/workspace/dub/handoff/out/V/audio-r4a-$VER; mkdir -p $OB $H
for id in 01 02 03 04 05; do
  src=$(ls ${DUB_INPUTS:-/workspace/dub/inputs/test_videos_raw}/$id.* | head -1); pm=$(ls $PMED/${id}_*.mp4 | head -1)
  cp $A/$NEW/${id}_over-original.mp4 $OB/${id}_r4_audio_${VER}_over-original.mp4
  ffmpeg -loglevel error -y -i $src -i $pm -map 0:v:0 -map 1:a:0 -c copy -shortest $A/$NEW/_prev_$id.mp4
  printf "file '%s'\nfile '%s'\n" $A/$NEW/_prev_$id.mp4 $A/$NEW/${id}_over-original.mp4 > $A/$NEW/_ab_$id.txt
  ffmpeg -loglevel error -y -f concat -safe 0 -i $A/$NEW/_ab_$id.txt -c copy $OB/${id}_r4_audio_${VER}_AB_prev-then-new.mp4; rm -f $A/$NEW/_prev_$id.mp4 $A/$NEW/_ab_$id.txt
  python scripts/quality/r4_review.py --id $id --prev-series $A/$PREV/$id.series.npz --new-series $A/$NEW/$id.series.npz --new-manifest $A/$NEW/$id/${id}_r4b.manifest.json \
    --prev-manifest $A/$PRUN/$id/${id}_r4b.manifest.json --prev-metrics $A/$PRUN/$id.metrics.json --new-metrics $A/$NEW/$id.metrics.json --out $OB/${id}_review.md --title "(r4_audio_$VER, commit $C; PREVIOUS = $(basename $pm), NEW = ${id}_r4_audio_${VER}_over-original.mp4)" > /dev/null
  mkdir -p $H/$id; cp $A/$NEW/$id/dubbed_16k.wav $A/$NEW/$id/final_mix_aac.m4a $A/$NEW/$id/${id}_r4b.manifest.json $H/$id/
done
cp $REG $OB/; cp $REG $H/; echo "r4_audio_$VER commit $C tag $NEW; per video: dubbed_16k.wav (dub voice), final_mix_aac.m4a (final mix), <id>_r4b.manifest.json" > $H/README.txt
du -sh $H; touch $H/READY; echo "PUBLISHED r4_audio_$VER -> $OB, handoff $H"
