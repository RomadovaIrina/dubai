#!/bin/bash
# r10 GPU accounting: warm production E2E (defaults, metrics off), one at a time
cd /workspace/dub/dubai; M=/workspace/dub/inputs/mamakris; O=/workspace/dub/inputs/test_videos_raw
for j in m19:$M/story-1774250832175-746471562.mp4 m03:$M/story-1774249919482-367803614.MOV 03:$O/03.mp4; do
  id=${j%%:*}; in=${j#*:}; rm -rf /workspace/dub/r10/acct/$id
  ID=$id E2E_METRICS=off bash scripts/r9/e2e.sh $in /workspace/dub/r10/acct/$id > /workspace/dub/r10/acct/$id.out 2>&1
  tail -3 /workspace/dub/r10/acct/$id.out
done
echo ACCT_DONE
