#!/bin/bash
# r15 F2: production composite (E2E_FX=1 E2E_SYNC=1 + q2 + F3, fb7s fallback, rep2s repair), as scripts/r15/comp_f.sh, but on the layer
# <run>_25fps_ls_cfd_<lab>.mp4 (lab = off: the pre-CodeFormer dump; sdt1: + source detail transfer) instead of the CodeFormer detail layer.
# CPU only: refuses to run unless the layer SR (srt1) already exists (run_f2.sh makes it under gpu.lock).
#   GT=<generator run> bash scripts/r15/comp_f2.sh <id> <sfx> <lab>
set -u; id=$1; sfx=$2; lab=$3
R=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd); cd $R; . scripts/env.sh >/dev/null; export DUB_ROOT=$R DUB_ALLOW_NC=1
ST=${GT:-r9g_$id}; B=/workspace/dub/artifacts/r4m; P=$B/$ST/work_${id}_$ST/${id}_${ST}_25fps
[ -s ${P}_ls_cfd_${lab}_srt1.mp4 ] || { echo "== comp_f2 $id$sfx: no SR layer ${P}_ls_cfd_${lab}_srt1.mp4 (refusing: it would run the GPU without the lock)"; exit 1; }
Q2="--sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.2"; F3=${F3:-"--reg-smooth 4 --mouth-region 0.45"}
FBA="$(cat reports/experiments/r8-face/fb7.args) --open-gate 0.04 --dark-transient on --thr-wash-lips 0.2"; RL=rep2s; FL=fb7s$lab
[ -n "${LC:-}" ] && { FBA="$FBA --thr-lipcon $LC"; RL=rep2slc; FL=fb7s${lab}lc; }   # r15 F2: + the lip-contrast flag, own FB / repair labels
# LCA=<thr>: the ADAPTIVE lip-contrast flag (below thr of the source AND < 0.6 x the video's own 2 s median, <= 3 % of the frames), scored on
# the pre-SR layer AND again on the repaired post-SR layer (second repair pass, r7_m7_comp REP2_FB_ARGS); labels fb7s<lab>la / rep2sla
LCX=""; [ -n "${LCA:-}" ] && { LCX="--thr-lipcon $LCA --lipcon-mode adaptive --lipcon-rel 0.6 --lipcon-win 25 --lipcon-max-frac ${LCA_CAP:-0.03}"; FBA="$FBA $LCX"; RL=rep2sla; FL=fb7s${lab}la; }
export REP2_FB_ARGS="${LCX:+$(cat reports/experiments/r8-face/fb7.args) $LCX}"
# LT=color|median|both: r15 F3 lip temporal stabilisation of the final layer (LT_EXTRA = more lip_temporal.py args); own repair label
# (<RL>x) so the production / earlier repaired layers are never rewritten; arm a (lab ia04) reuses the production fallback score fb7s
export LT_ARGS="" LT_LABEL=""; [ -n "${LT:-}" ] && { LT_ARGS="--mode $LT ${LT_EXTRA:-}"; LT_LABEL=$LT${LT_TAG:-}; RL=${RL}x; [ "$lab" = ia04 ] && [ -z "${LC:-}${LCA:-}" ] && FL=fb7s; }
mkdir -p /workspace/dub/r15f2/logs
FC_CMS=1 LS_CMS_READ=1 SKIP_SCORE=1 TAG=r9f_$id$sfx SRC_TAG=$ST CFD=self CFD_LABEL=$lab SR=srt1 SR_EXTRA="--batch 8 --mask down2" \
  FB=$FL FB_ARGS="$FBA" REP=1 REP_ARGS="--spikes off" REP_LABEL=$RL FC_EXTRA="--pre-threads=8 $Q2 $F3" \
  EXTRA="--tex-tau 1000 --tex-sigma 2.5 --mouth-region 0.55 --cm-mouth-excl 0.5" bash scripts/r7/r7_m7_comp.sh $id > /workspace/dub/r15f2/logs/comp_$id$sfx.log 2>&1
ls -la $B/r9f_$id$sfx/${id}_r9f_$id$sfx.mp4 2>&1 | tail -1 | cut -c1-200; grep "^== .*layer_repair\|^== .*fallback {" /workspace/dub/r15f2/logs/comp_$id$sfx.log | cut -c1-300
