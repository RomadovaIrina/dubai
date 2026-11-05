#!/bin/bash
# r15 F3 fix 3: CodeFormer weight sweep (w 0.5 / 0.7 / 1.0 with MediaPipe 5 points) + CF outside the inner mouth only (inner-alpha 0 on the
# production CF layer); each -> CF detail layer (CPU) -> layer SR (GPU) -> production composite (CPU). GPU steps: no E2E_RUNNING, < 2 GB, gpu.lock.
set -u; R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd $R; . scripts/env.sh >/dev/null; export DUB_ROOT=$R DUB_ALLOW_NC=1
B=/workspace/dub/artifacts/r4m; L=/workspace/dub/r15f3/logs; mkdir -p $L
gpu_wait(){ while [ -e /workspace/dub/r15d/E2E_RUNNING ] || [ -e /workspace/dub/r15t/E2E_RUNNING ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -ge 2048 ]; do sleep 30; done; }
SPECS=${SPECS:-"q2_m03:r9g_q2_m03fu:fu q2_old03:r9g_q2_old03:"}
for s in $SPECS; do IFS=: read id ST A <<< "$s"; W=$B/$ST/work_${id}_$ST; P=$W/${id}_${ST}_25fps; LMN=${P}_ls_raw.mediapipe.npz
  for w in 05 07 10; do CF=${P}_ls_cfw$w.mp4
    [ -s $CF ] || { gpu_wait; echo "== CF $id w$w $(date +%T)"; flock /workspace/dub/gpu.lock python scripts/r15/cf_layer.py --work $W --pre ${P}_ls_pre.mp4 --lm $LMN --w 0.${w#0} --out $CF | grep "^=="; }
    [ "$w" = 10 ] && [ ! -s ${P}_ls_cfw10.mp4 ] && continue
    [ -s ${P}_ls_cfd_w$w.mp4 ] || python scripts/r8/r8_cf_detail.py --base ${P}_ls_pre.mp4 --cf $CF --lm $LMN --out ${P}_ls_cfd_w$w.mp4 --inner-alpha 0.4 | grep "^=="
    cp $LMN ${P}_ls_cfd_w$w.mediapipe.npz
  done
  [ -s ${P}_ls_cfd_ia00.mp4 ] || python scripts/r8/r8_cf_detail.py --base ${P}_ls_pre.mp4 --cf ${P}_ls_raw.mp4 --lm $LMN --out ${P}_ls_cfd_ia00.mp4 --inner-alpha 0.0 | grep "^=="
  cp $LMN ${P}_ls_cfd_ia00.mediapipe.npz
  for lab in w05 w07 w10 ia00; do RW=${P}_ls_cfd_$lab.mp4; S=${RW%.mp4}_srt1.mp4; [ -s $S ] && continue
    gpu_wait; echo "== SR $id $lab $(date +%T)"
    flock /workspace/dub/gpu.lock python scripts/r7/r7_layer_sr.py --ls-raw $RW --lm ${RW%.mp4}.mediapipe.npz --out $S --temporal 1 --batch 8 --mask down2 | grep "^==" | cut -c1-200
    cp ${RW%.mp4}.mediapipe.npz ${S%.mp4}.mediapipe.npz
  done
done
echo "== CF GPU part done $(date +%T)"
for s in $SPECS; do IFS=: read id ST A <<< "$s"
  for lab in w05 w07 w10 ia00; do ( GT=$ST bash scripts/r15/comp_f2.sh $id ${A}$lab $lab ) & done; wait
done
echo CF_DONE $(date +%T)
