#!/bin/bash
# r16 L: render eval segments with a student LoRA (GPU, locked) + metrics: cheek / jaw motion vs source, SyncNet, black cavity.
#   l_eval.sh <lora.pt (lora_meta.json next to it)> <name> [jobs file] [ks_fast extra args]
# fast2 = 2 LCM steps, keyframes = the prod TRT engine (DeepCache 2:0), interp student in torch (LoRA merged; TRT gives the same within 56 dB).
R=/workspace/dub/r16l; K=/workspace/dub/keysync; Q=/workspace/dub/dubai/scripts/quality; L=$1; N=$2; J=${3:-$R/eval/jobs_a.txt}; shift $(( $# < 3 ? $# : 3 )); X="$*"
O=$R/eval/$N; mkdir -p $O $R/lmk; cd $K/code; . $K/venv/bin/activate
T0=$(date +%s)
KS_DC=2:0 KS_LMK_CACHE=$R/lmk flock /workspace/dub/gpu.lock python $R/repo/scripts/keysync/r16/l/ks_fast_l.py --jobs $J --out $O --cfg _s8 --vae-half --trt ${TRTDIR:-$R/trt_kf} \
  --interp-lora $L --interp-steps 2 --vae-batch 8 --vae-compile $X > $O/render.log 2>&1
echo "render $(( $(date +%s) - T0 )) s"; grep KS_JOB $O/render.log | cut -c1-250
deactivate
while read tag vid aud; do
  G=$O/$tag/${tag}_$tag.mp4; [ -s $G ] || { echo "MISSING $G"; continue; }
  [ -s $R/data/videos/$tag.mp.npz ] || /venv/mediapipe/bin/python $Q/mp_landmarks.py --video $vid --out $R/data/videos/$tag.mp.npz > /dev/null 2>&1
  /venv/mediapipe/bin/python $Q/mp_landmarks.py --video $G --out $O/$tag.mp.npz > /dev/null 2>&1
  python3 $R/repo/scripts/keysync/r16/l/l_cheek.py --gen $O/$tag.mp.npz --src $R/data/videos/$tag.mp.npz --out $O/$tag.cheek.json > /dev/null
  (cd /workspace/dub/dubai && SYNC_DEVICE=cpu /venv/dabai/bin/python $R/repo/scripts/quality/sync_perframe.py --video $G --audio $aud --out $O/$tag.sync.json > $O/$tag.sync.log 2>&1; echo cpu > $O/$tag.sync.dev)
  /venv/dabai/bin/python $K/k_cavity.py --pairs gen=$G:$O/$tag.mp.npz src=$vid:$R/data/videos/$tag.mp.npz > $O/$tag.cav.json 2>/dev/null
  /venv/dabai/bin/python $R/repo/scripts/keysync/r16/l/l_flow.py --t0 ${FLOW_T0:-1} --t1 ${FLOW_T1:-5} --gen $G --gen-lm $O/$tag.mp.npz --src $vid --src-lm $R/data/videos/$tag.mp.npz --out $O/$tag.flow.json > /dev/null 2>&1
  /venv/dabai/bin/python $R/repo/scripts/keysync/r16/l/l_boundary.py --gen $G --gen-lm $O/$tag.mp.npz --src $vid --src-lm $R/data/videos/$tag.mp.npz --out $O/$tag.bnd.json > /dev/null 2>&1
done < $J
python3 $R/repo/scripts/keysync/r16/l/l_agg.py $O
