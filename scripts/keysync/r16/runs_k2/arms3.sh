#!/bin/bash
export KS_LORA=/workspace/dub/k2/lora/lora_pick.pt; K=/workspace/dub/k2/m18; F=/workspace/dub/intwt/scripts/keysync/r16/ks_face.sh; R=$(cat $K/neutral_ref.txt)
for arm in F_e2eaud_ref G_acc_ref; do AD=$K/ad_e2e; [ $arm = G_acc_ref ] && AD=$K/ad
  env KS_REF=$R bash $F m18_$arm /workspace/dub/inputs/mamakris/story-1774250817106-616905094.mp4 $AD $K/w_$arm $K/$arm.mp4 > $K/$arm.log 2>&1
  echo "$arm ref $R rc $? $(grep -h "^KS_PROBE" $K/$arm.log | cut -c1-80)"
  GLOG_minloglevel=2 /venv/mediapipe/bin/python /workspace/dub/dubai/scripts/quality/mp_landmarks.py --video $K/$arm.mp4 --out $K/$arm.npz > /dev/null 2>&1
done
cd $K && /venv/dabai/bin/python lips.py golden.npz A_e2e.npz G_acc_ref.npz e2e.npz D_e2eaud_gate.npz F_e2eaud_ref.npz; /venv/dabai/bin/python open.py golden.npz A_e2e.npz G_acc_ref.npz D_e2eaud_gate.npz F_e2eaud_ref.npz
cd /workspace/dub/intwt; for a in A_e2e G_acc_ref; do /venv/dabai/bin/python scripts/quality/sync_perframe.py --video $K/$a.mp4 --audio $K/ad/dubbed_16k.wav --out /tmp/$a.s.json > /dev/null 2>&1; python3 -c "
import json; s=json.load(open(/tmp/.s.json)); sp=[p[conf_at0] for p in s[per_second] if p.get(conf_at0) is not None and (p.get(dub_act) or 0)>=0.5]
print( speech@0 %.2f off %s % (sum(sp)/max(len(sp),1), s[global][av_offset]))"; done
echo ARMS3_DONE
