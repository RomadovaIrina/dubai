#!/bin/bash
# r11 SYNC: arms in parallel on several ids.  bash scripts/r11s/sweep.sh "<ids>" "tag=flags" ["tag=flags" ...]  (env LS / FB / BASEARGS pass through)
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); ids=$1; shift
for id in $ids; do for arm in "$@"; do t=${arm%%=*}; x=${arm#*=}; bash $R/scripts/r11s/arm.sh $id $t "$x" & sleep 2; done; done; wait
