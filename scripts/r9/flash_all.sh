#!/bin/bash
# mouth "flash" frames (scripts/quality/r3_flash.py, the gate detector: 1-frame spikes of texture / brightness / colour in the lip box vs the source) on r9 vs r9fx
R=${R9ROOT:-/workspace/dub/dubai}; cd $R; . scripts/env.sh >/dev/null; export OMP_NUM_THREADS=6; B=/workspace/dub/artifacts/r4m; D=/workspace/dub/r9/diag
for id in "$@"; do src=$(ls /workspace/dub/r9/runs/$id/work/in/$id.* | head -1); lm=$B/r9g_$id/timing/${id}_source_lm_F.npz
  FX=r9f_${id}fx; [ $id = m03 ] && FX=r9f_m03fx3; while [ ! -f $B/$FX/${id}_$FX.mp4 ]; do sleep 30; done
  for T in r9f_$id $FX; do python scripts/quality/r3_flash.py --video $B/$T/${id}_$T.mp4 --src $src --lm $lm --out $D/flash_$T.json > /dev/null 2>&1; echo "== $T done"; done
done; echo FLASH_DONE
