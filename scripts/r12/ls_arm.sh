#!/bin/bash
# r12 LS: one LatentSync generator lever arm on the baseline's audio (artifacts/r4a/r9_<id>, from ls_base.sh) + the k5 composite + the NVML
# GPU-busy seconds of the generator / composite + the gate vs the k5 base (r10/gate/<id>base).
#   bash scripts/r12/ls_arm.sh <id> <arm> "<VIDEO_EXTRA override, '' = production>" [ENV=VAL ...]
#   e.g. bash scripts/r12/ls_arm.sh m18 t3 "" LS_CFG_TRUNC=3      -> artifacts/r4m/r9g_<id><arm>, r9f_<id><arm>; /workspace/dub/r12/arms/<id><arm>.json
# No flock around the generator (r4_video.sh takes gpu.lock itself); the composite (layer SR on the GPU) runs under gpu.lock.
set -u
id=$1; arm=$2; VX=${3:-}; shift 3
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null; export DUB_ROOT=/workspace/dub/dubai DUB_ALLOW_NC=1 METRICS=off DUB_INPUTS=/workspace/dub/r10/runs/$id/work/in R10_JOB=$id$arm
A=/workspace/dub/artifacts/r4a; B=/workspace/dub/artifacts/r4m; GT=r9g_$id$arm; FT=r9f_$id$arm; O=/workspace/dub/r12/arms; mkdir -p $O
[ -n "$VX" ] || VX="--codeformer optimized --codeformer-w 1.0 --steps 10 --steps-silent 20 --codeformer-parse off --codeformer-batch 16"
export GEN_COMP=0 FAST_SRCPREP=0 FAST_FB_BG=1 FAST_ENC=ultrafast FC_EXTRA=--pre-threads=12 LS_VAE_COMPILE=${LS_VAE_COMPILE:-max-autotune-no-cudagraphs} CFD_EXTRA=--threads=12   # the e2e R10_FAST_FACE=1 preset
NV=$O/$id$arm.nv.jsonl; ST=$O/$id$arm.stages.tsv; : > $ST; rm -f $NV
/venv/main/bin/python scripts/r10/gpu_sampler.py --out $NV > /dev/null 2>&1 & SP=$!
echo -e "face_gen\t$(date +%s.%N)" >> $ST
[ -f $B/$GT/work_${id}_$GT/${id}_${GT}_25fps.manifest.json ] || env "$@" AUDIO=$A/r9_$id TAG=$GT SKIP_SCORE=1 VIDEO_EXTRA="$VX" bash scripts/r7/r7_e2e.sh $id > $O/$id$arm.gen.log 2>&1
echo -e "face_comp\t$(date +%s.%N)" >> $ST
FBA="$(cat reports/experiments/r8-face/fb7.args) --open-gate 0.04 --dark-transient on --thr-wash-lips 0.2"
[ -f $B/$FT/${id}_$FT.mp4 ] || flock /workspace/dub/gpu.lock env SKIP_SCORE=1 TAG=$FT SRC_TAG=$GT CFD=self CFD_LABEL=ia04 CFD_ARGS="--inner-alpha 0.4 --threads=12" SR=srt1 SR_EXTRA="--batch 8 --mask down2" \
  FB=fb7s FB_ARGS="$FBA" REP=1 REP_ARGS="--spikes off" REP_LABEL=rep2s EXTRA="--tex-tau 1000 --tex-sigma 2.5 --mouth-region 0.55 --cm-mouth-excl 0.5" bash scripts/r7/r7_m7_comp.sh $id > $O/$id$arm.comp.log 2>&1
echo -e "end\t$(date +%s.%N)" >> $ST; kill $SP 2>/dev/null; wait $SP 2>/dev/null
vmin=$(python3 -c "import json; print(json.load(open('/workspace/dub/r10/runs/$id/timing.json'))['video_min'])")
/venv/main/bin/python scripts/r10/gpu_sampler.py --summary $NV --stages $ST --vmin $vmin --job $id$arm > $O/$id$arm.gpu.json 2>/dev/null
python3 - $O/$id$arm.gpu.json $B/$GT/work_${id}_$GT/${id}_${GT}_25fps.manifest.json <<'PY'
import json, sys
g = json.load(open(sys.argv[1])); s = g["stages"]
try:
    m = json.load(open(sys.argv[2])); ls = [c for c in (m.get("latentsync_calls") or [])]
except Exception: ls = []
print("== gpu", {k: {"busy_s_per_vmin": v.get("gpu_busy_s_per_vmin"), "wall_s": round(v["wall_s"], 1)} for k, v in s.items()}, "total busy s/vmin", g["total"].get("gpu_busy_s_per_vmin"))
PY
[ -f $B/$FT/${id}_$FT.mp4 ] || { echo "== arm $id$arm FAILED (see $O/$id$arm.gen.log / comp.log)"; exit 1; }
bash scripts/r10/gate.sh $id Y$arm $B/$FT/${id}_$FT.mp4 > $O/$id$arm.gate.log 2>&1
python scripts/r10/gate_cmp.py --base /workspace/dub/r10/gate/${id}base --cand /workspace/dub/r10/gate/${id}Y$arm --md /workspace/dub/r12/gate.md | head -n 3
