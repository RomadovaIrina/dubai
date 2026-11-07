#!/bin/bash
# r16 L: training jobs for ks_dump.py from an explicit CLEAN list (audit #5). The list = "<tag> <512 crop 25 fps> <own audio 16 kHz>" per line;
# any tag / path of the eval or QA set (q0*, m03, m18, old03 = test_videos_raw/03, o03*) is refused. ks_lcm_jit.py refuses such dumps too.
#   l_train_jobs.sh <clean list> <out jobs file>
set -e; : > $2
while read tag vid aud; do
  [ -z "$tag" ] && continue
  if echo "$tag $vid $aud" | grep -qE '(^|[/_ ])(q0[0-9]|m03|m18|old03|o03)|test_videos_raw/03|story-1774249919482|story-1774250817106'; then echo "REFUSED (test set): $tag"; continue; fi
  echo "$tag $vid $aud" >> $2
done < $1
wc -l $2
