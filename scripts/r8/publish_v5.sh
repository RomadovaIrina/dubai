#!/bin/bash
# r8-sep: audio-r8-v5 (htdemucs bed, user-approved 29.10, NC deviation) + audio-r8-v5-spleeter (the same dub on the r8v4 Spleeter bed) from a
# published dub handoff. Same remix flags as audio-r8-v4; the handoff dub is already gated -> --dub-gate off. READY only when every video
# has 0 gap words (Whisper on the mix gaps) on the htdemucs mix.
#     bash scripts/r8/publish_v5.sh /workspace/dub/handoff/out/F/audio-r8-v4s v4s
set -u
cd /workspace/dub/sep; H=$1; L=$2; IDS="01 02 03 04 05"; A=/workspace/dub/artifacts/r4a; OUT=/workspace/dub/handoff/out/F; LOG=/workspace/dub/r8a/sep_logs
SRC=h_$L; SPL=v5spl_$L; HTD=v5_$L; HFLAGS="--bed-pauses source --bed-noise-back on --src-duck-db -3 --dub-gate off"
bash scripts/r8/from_handoff.sh $H $SRC $IDS
SEP_SRC=$SRC SEP_ASR=off bash scripts/r8/sep_mix.sh $SPL spleeter stems "--dub-gate off" $IDS > $LOG/$SPL.out 2>&1
SEP_SRC=$SRC SEP_PREV=$SPL bash scripts/r8/sep_mix.sh $HTD htdemucs stems "$HFLAGS" $IDS > $LOG/$HTD.out 2>&1
pub() {   # $1 tag, $2 handoff name, $3 bed text
  D=$OUT/$2; rm -rf $D.tmp; mkdir -p $D.tmp
  for id in $IDS; do mkdir -p $D.tmp/$id; cp $A/$1/$id/${id}_r4b.manifest.json $D.tmp/$id/; for f in dubbed_16k.wav dubbed_24k.wav final_mix_aac.m4a; do cp -L $A/$1/$id/$f $D.tmp/$id/; done; done
  git rev-parse --short HEAD > $D.tmp/COMMIT
  { echo "$2 (r8-sep, tag $1): dub = $H (its dubbed_16k / 24k, manifest timing, unchanged), bed = $3."
    echo "Remix = the audio-r8-v4 flags (--src-duck-db -6 --unit-level off --gap-bed-cap-db 10 --gap-bed-cap-scope edges --gap-clean on --gap-clean-nr-ns 0.7 --gap-clean-floor 3)"
    echo "  with --dub-gate off (the handoff dub is already gated) and: $4. Same layout as audio-r8-v4: <id>/{dubbed_16k.wav, dubbed_24k.wav, final_mix_aac.m4a (-16 LUFS), <id>_r4b.manifest.json}."
    echo "Code: /workspace/dub/sep (experiment/r8-sep) scripts/r8/publish_v5.sh; report reports/experiments/r8-sep/README.md."; } > $D.tmp/README.txt
  rm -rf $D; mv $D.tmp $D
}
pub $SPL audio-r8-v5-spleeter "Spleeter 2stems (MIT; = the audio-r8-v4 bed)" "nothing else"
pub $HTD audio-r8-v5 "Demucs v4 htdemucs (NC weights trained on MUSDB18-HQ: a deliberate, user-accepted deviation 29.10)" "--sep htdemucs $HFLAGS"
cp reports/experiments/r8-sep/README.md $OUT/audio-r8-v5/README_r8-sep.md
for id in 02 04 05; do cp /workspace/dub/outbox/sep/${id}_audio_sep_listen.md $OUT/audio-r8-v5/$id/ 2>/dev/null; done
ok=1; for id in $IDS; do w=$(python3 -c "import json;print(json.load(open(//.gaps.json)).get(gap_words,?))" 2>/dev/null); echo "$id gap_words $w"; [ "$w" = 0 ] || ok=0; done | tee $LOG/${HTD}_gate.txt
grep -q -v "gap_words 0" $LOG/${HTD}_gate.txt && ok=0 || ok=1
for d in audio-r8-v5 audio-r8-v5-spleeter; do [ $ok = 1 ] && date +%FT%T > $OUT/$d/READY || echo "gap words: see $LOG/${HTD}_gate.txt" > $OUT/$d/NOT_READY; done
WATCH_PREV=$SPL WATCH_PREV_LABEL="v5-spleeter" WATCH_NAME="v5spleeter-then-v5" bash scripts/r8/sep_watch.sh $HTD 02 04 05 > $LOG/${HTD}_watch.log 2>&1
for id in 02 04 05; do cp /workspace/dub/outbox/sep/${id}_audio_sep_listen.md $OUT/audio-r8-v5/$id/; done
echo "PUBLISH_V5_DONE $L ok=$ok"
