#!/bin/bash
# r8-sep: a remix source (mix_src) from a published audio handoff: <tag>/mix_src/<id>/ = the handoff dub (already dub-gated + softened: remix it with
# --dub-gate off) + manifest + the Spleeter stems of the r8 runs (/workspace/dub/r8a/stems/<id>) as bg_44k / vocals_44k.
#     bash scripts/r8/from_handoff.sh /workspace/dub/handoff/out/F/audio-r8-v4s <tag> [ids]
H=$1; T=$2; shift 2; IDS=${*:-01 02 03 04 05}; A=/workspace/dub/artifacts/r4a; M=$A/$T/srcdirs
for id in $IDS; do
  d=$M/$id; mkdir -p $d $A/$T/mix_src
  for f in dubbed_16k.wav dubbed_24k.wav; do ln -sfn $H/$id/$f $d/$f; done
  cp $H/$id/${id}_r4b.manifest.json $d/${id}_h.manifest.json
  ln -sfn /workspace/dub/r8a/stems/$id/bg_44k.wav $d/bg_44k.wav; ln -sfn /workspace/dub/r8a/stems/$id/vocals_44k.wav $d/vocals_44k.wav
  ln -sfn $d $A/$T/mix_src/$id
done
echo "FROM_HANDOFF $H -> $A/$T/mix_src ($IDS)"
