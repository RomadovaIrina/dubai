#!/bin/bash
# r16 Q: the e2e LatentSync PROD face stage (quality = q2 + F3, = scripts/r9/e2e.sh face_gen + face_comp with dabai_job mode_env quality)
# on the EXISTING real final mixes (artifacts/r4a/r9_<id>/<id>, the 14:55-15:11 e2e audio K2 uses), no audio regen, no job timeout.
# GPU lock: r4_video.sh locks gpu.lock itself around the generator (do NOT add an outer flock: self-deadlock). Usage: bash ls_face.sh <id> -> /workspace/dub/outbox/r16_q/ls/<id>.mp4
set -u; ID=$1
cd /workspace/dub/int; R=$PWD; . scripts/env.sh >/dev/null; export DUB_ROOT=$R; cd $R
set -a; . /workspace/dub/.env; set +a
export METRICS=off GEN_COMP=0 FAST_SRCPREP=1 FAST_FB_BG=1 FAST_ENC=ultrafast FAST_MPREUSE=1 LS_VAE_COMPILE=max-autotune-no-cudagraphs CFD_EXTRA=--threads=12
export E2E_FX=1 E2E_SYNC=1 DUB_ALLOW_NC=${DUB_ALLOW_NC:-0} LS_CMS_READ=1 FC_CMS=1 LS_STRIDE=8 LS_OVERLAP=continue
export FC_EXTRA="--pre-threads=12 --sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.2 --reg-smooth 4 --mouth-region 0.45"
A=/workspace/dub/artifacts/r4a; B=/workspace/dub/artifacts/r4m; AT=r9_$ID; AD=$A/$AT/$ID; GT=r16qg_$ID; FT=r16qf_$ID
O=/workspace/dub/r16q/lsface/$ID; L=$O/logs; mkdir -p $L /workspace/dub/outbox/r16_q/ls
export DUB_INPUTS=/workspace/dub/r16q/lsface/in; mkdir -p $DUB_INPUTS; cp -f /workspace/dub/r16q/clips/$ID.mp4 $DUB_INPUTS/$ID.mp4; SRC=$DUB_INPUTS/$ID.mp4
echo "== $ID start $(date +%T) audio $AD"; SP=""; CFDV=""; [ "$DUB_ALLOW_NC" = 1 ] && CFDV=self
if [ "${SKIP_GEN:-0}" = 1 ] && [ -f $B/$GT/work_${ID}_$GT/${ID}_${GT}_25fps.manifest.json ]; then rm -rf $B/$FT; else rm -rf $B/$GT $B/$FT
unset MP_SOURCE_NPZ; SMP=$A/sep/$ID/source_mediapipe.npz; SP=""; CFDV=""; [ "$DUB_ALLOW_NC" = 1 ] && CFDV=self
if [ -f $SMP ]; then export MP_SOURCE_NPZ=$SMP; mkdir -p $B/$GT/timing; cp $SMP $B/$GT/timing/${ID}_source_lm_F_mp.npz; fi
WV=$B/$GT/work_${ID}_$GT/video; TD=$B/$GT/timing
( for i in $(seq 1800); do [ -s $WV/master_mediapipe.npz ] && break; sleep 2; done; [ -s $WV/master_mediapipe.npz ] || exit 0; mkdir -p $TD; TJ=$TD/${ID}_master_timing_F.json; SL=$TD/${ID}_source_lm_F.npz
  python scripts/quality/r3_master_timing.py --src $SRC --master $WV/master_25fps.mp4 --vf fps=25 --out $TJ && python scripts/quality/r3_source_lm.py --src $SRC --timing $TJ --out $SL \
    && python scripts/r9/r9_source_rigid.py --src $SRC --source-lm $SL ) > $L/srcprep.log 2>&1 & SP=$!
CFX="--codeformer optimized --codeformer-w 1.0 --codeformer-parse off --codeformer-batch 16"; CFDV=self; [ "$DUB_ALLOW_NC" = 1 ] || { CFX="--codeformer off"; CFDV=""; }   # no codeformer.pth on r16-qt: default = the clean face (= e2e DUB_ALLOW_NC=0)
env AUDIO=$A/$AT TAG=$GT SKIP_SCORE=1 VIDEO_EXTRA="$CFX --steps 10 --steps-silent 20" bash scripts/r7/r7_e2e.sh $ID > $L/face_gen.log 2>&1
fi
echo "== $ID gen done $(date +%T)"; wait $SP 2>/dev/null
[ -f $B/$GT/work_${ID}_$GT/${ID}_${GT}_25fps.manifest.json ] || { echo "FAIL gen $ID"; exit 1; }
export REP=1 CFD_LABEL=ia04 CFD_ARGS="--inner-alpha 0.4 ${CFD_EXTRA:-}"
FBA="$(cat reports/experiments/r8-face/fb7.args) --open-gate 0.04 --dark-transient on --thr-wash-lips 0.2"; export REP_ARGS="--spikes off" REP_LABEL=rep2s
SKIP_SCORE=1 TAG=$FT SRC_TAG=$GT CFD=$CFDV SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7s FB_ARGS="$FBA" EXTRA="--tex-tau 1000 --tex-sigma 2.5 --mouth-region 0.55 --cm-mouth-excl 0.5" \
  bash scripts/r7/r7_m7_comp.sh $ID > $L/face_comp.log 2>&1
V=$B/$FT/${ID}_$FT.mp4; [ -f $V ] || { echo "FAIL comp $ID"; exit 1; }
ffmpeg -loglevel error -y -i $V -i $AD/final_mix_aac.m4a -map 0:v:0 -map 1:a:0 -c copy -movflags +faststart /workspace/dub/outbox/r16_q/ls/$ID.mp4 && echo "== $ID DONE $(date +%T)"
