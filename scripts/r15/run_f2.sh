#!/bin/bash
# r15 F2 driver: CodeFormer off (b) and CodeFormer off + source detail transfer (c) vs production (a) on the cached q2 layers.
#   bash scripts/r15/run_f2.sh            (old03, m18, m03 = the fixed m03fu run)
# GPU only for the layer SR (Real-ESRGAN, BSD-3) under gpu.lock, and only when no E2E_RUNNING marker exists and < 2 GB VRAM is used.
set -u
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd $R; . scripts/env.sh >/dev/null; export DUB_ROOT=$R DUB_ALLOW_NC=1
B=/workspace/dub/artifacts/r4m; L=/workspace/dub/r15f2/logs; O=/workspace/dub/outbox/r15_f2; mkdir -p $L $O
SPECS=${SPECS:-"q2_old03:r9g_q2_old03:c0 q2_m18:r9g_q2_m18:c0 q2_m03:r9g_q2_m03fu:fu"}   # id : generator run : production composite suffix (arm a)
SDT_ARGS=${SDT_ARGS:-}
gpu_wait(){ while [ -e /workspace/dub/r15d/E2E_RUNNING ] || [ -e /workspace/dub/r15t/E2E_RUNNING ] || [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -ge 2048 ]; do sleep 30; done; }
pre(){ local id=$1 ST=$2; local W=$B/$ST/work_${id}_$ST; local P=$W/${id}_${ST}_25fps; echo "$W $P"; }
echo "== F2 start $(date +%T) $(git rev-parse --short HEAD)"
# 1. layers (CPU): b = the pre-CodeFormer dump as is, c = + source detail transfer
for s in $SPECS; do IFS=: read id ST A <<< "$s"; read W P <<< "$(pre $id $ST)"; LMN=${P}_ls_raw.mediapipe.npz
  [ -e ${P}_ls_cfd_off.mp4 ] || ln -s $(basename ${P}_ls_pre.mp4) ${P}_ls_cfd_off.mp4; cp $LMN ${P}_ls_cfd_off.mediapipe.npz
  [ -s ${P}_ls_cfd_sdt1.mp4 ] || python scripts/r15/src_detail.py --base ${P}_ls_pre.mp4 --lm $LMN --src $W/video/master_25fps.mp4 --src-lm $W/video/master_mediapipe.npz --out ${P}_ls_cfd_sdt1.mp4 $SDT_ARGS | grep "^==" | sed "s/^/$id /"
  cp $LMN ${P}_ls_cfd_sdt1.mediapipe.npz
done
# 2. layer SR (GPU, gated, one at a time)
for s in $SPECS; do IFS=: read id ST A <<< "$s"; read W P <<< "$(pre $id $ST)"
  for lab in off sdt1; do RW=${P}_ls_cfd_$lab.mp4; S=${RW%.mp4}_srt1.mp4; [ -s $S ] && continue
    gpu_wait; echo "== SR $id $lab $(date +%T)"
    flock /workspace/dub/gpu.lock python scripts/r7/r7_layer_sr.py --ls-raw $RW --lm ${RW%.mp4}.mediapipe.npz --out $S --temporal 1 --batch 8 --mask down2 | grep "^==" | cut -c1-200
    cp ${RW%.mp4}.mediapipe.npz ${S%.mp4}.mediapipe.npz
  done
done
echo "== F2 GPU part done $(date +%T)"; touch /workspace/dub/r15f2/SR_DONE
# 3. composites (CPU; the SR layers exist, so r7_m7_comp never touches the GPU): two at a time
for s in $SPECS; do IFS=: read id ST A <<< "$s"
  for lab in off sdt1; do sx=${A/c0/}$([ $lab = off ] && echo b || echo c)
    ( GT=$ST bash scripts/r15/comp_f2.sh $id $sx $lab ) &
  done; wait
done
# 4. metrics: layer sharpness, tone_flicker, mouth_pop (CPU)
for s in $SPECS; do IFS=: read id ST A <<< "$s"; read W P <<< "$(pre $id $ST)"; SRC=$(ls /workspace/dub/r14d/eval/$id/e2e/work/in/$id.* | head -1); SL=$B/r9g_$id/timing/${id}_source_lm_F.npz
  for arm in a:cfd_ia04:$A b:cfd_off:${A/c0/}b c:cfd_sdt1:${A/c0/}c; do IFS=: read an lab sx <<< "$arm"
    python scripts/r15/layer_sharp.py --label "$id $an" --layer ${P}_ls_${lab}_srt1.mp4 --lm ${P}_ls_raw.mediapipe.npz --src $W/video/master_25fps.mp4 --src-lm $W/video/master_mediapipe.npz --json /workspace/dub/r15f2/sharp.jsonl &
    V=$B/r9f_$id$sx/${id}_r9f_$id$sx.mp4
    ( python scripts/r12/tone_flicker.py --video $V --src $SRC --src-lm $SL --label "$id $an" > $L/flick_$id$an.txt 2>&1; python scripts/r12/mouth_pop.py --video $V --src $SRC --src-lm $SL --label "$id $an" >> $L/flick_$id$an.txt 2>&1 ) &
  done; wait
done
cat $L/flick_*.txt | grep "^==" > /workspace/dub/r15f2/flicker.txt
# 5. gate incl. SyncNet (no lock, one at a time, same GPU check) for b and c, compared with a (the base gate re-run if it is missing)
for s in $SPECS; do IFS=: read id ST A <<< "$s"
  [ -s /workspace/dub/r15f/gate/$id$A/face_gate.json ] || { gpu_wait; GT=$ST bash scripts/r15/gate_f.sh $id $A > $L/gate_$id$A.log 2>&1; tail -n 1 $L/gate_$id$A.log; }
  for sx in ${A/c0/}b ${A/c0/}c; do gpu_wait; GT=$ST bash scripts/r15/gate_f.sh $id $sx > $L/gate_$id$sx.log 2>&1; tail -n 1 $L/gate_$id$sx.log
    python scripts/r10/gate_cmp.py --base /workspace/dub/r15f/gate/$id$A --cand /workspace/dub/r15f/gate/$id$sx --md /workspace/dub/r15f2/gate.md | head -n 3; done
done
echo F2_DONE $(date +%T)
