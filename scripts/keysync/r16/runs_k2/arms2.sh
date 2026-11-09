#!/bin/bash
export KS_LORA=/workspace/dub/k2/lora/lora_pick.pt; K=/workspace/dub/k2/m18; F=/workspace/dub/intwt/scripts/keysync/r16/ks_face.sh
while pgrep -f "[d]ay3.sh" > /dev/null; do sleep 20; done
for arm in D_e2eaud_gate E_e2eaud_off; do G="thr=0.12,cap=0.25,hi=8,lo=2"; [ $arm = E_e2eaud_off ] && G=""
  env KS_GATE="$G" bash $F m18_$arm /workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4 $K/ad_e2e $K/w_$arm $K/$arm.mp4 > $K/$arm.log 2>&1
  echo "$arm rc $? $(grep -h "^KS_PROBE" $K/$arm.log | cut -c1-80)"
  GLOG_minloglevel=2 /venv/mediapipe/bin/python /workspace/dub/dubai/scripts/quality/mp_landmarks.py --video $K/$arm.mp4 --out $K/$arm.npz > /dev/null 2>&1
done
cd $K && /venv/dabai/bin/python lips.py e2e.npz D_e2eaud_gate.npz E_e2eaud_off.npz | grep "9.5"; /venv/dabai/bin/python open.py e2e.npz D_e2eaud_gate.npz E_e2eaud_off.npz
echo ARMS2_DONE
