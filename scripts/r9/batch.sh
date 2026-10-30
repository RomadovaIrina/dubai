#!/bin/bash
# r9: the mamakris E2E set, one video at a time on this server's GPU (honest timing). bash scripts/r9/batch.sh <short id> ...
set -u
cd "$(dirname "$0")/../.."; M=/workspace/dub/inputs/mamakris; R=/workspace/dub/r9/runs; mkdir -p $R
declare -A F=([m19]=story-1774250832175-746471562.mp4 [m01]=story-1773579769284-320619146.MP4 [m04]=story-1774249930391-60765207.MOV
  [m03]=story-1774249919482-367803614.MOV [m15]=story-1774250573471-666527943.MP4 [m17]=story-1774250795589-441186215.mp4
  [m18]=story-1774250817106-616905094.mp4 [m21]=story-1774250904597-834383232.mp4 [m20]=story-1774250884789-692894647.mp4)
for id in "$@"; do
  rm -rf $R/$id; ID=$id bash scripts/r9/e2e.sh $M/${F[$id]} $R/$id > $R/$id.out 2>&1; tail -2 $R/$id.out
done
echo R9_BATCH_DONE "$@"
