#!/bin/bash
# r15 W: one face ARM on the FINISHED q2 + F3 baseline's dub (scripts/r15/base_q2.sh): the face_gen + face_comp stages exactly as
# scripts/r9/e2e.sh runs them under dabai_job (E2E_FX=1, E2E_SYNC=1 = k5, R10_FAST_FACE=1 stack, q2 + F3 flags), the generator swapped by env
# (R10_GEN_PY / LS_* / ...), timed with the NVML per-job sampler (GPU-busy s per video-minute of the WHOLE face stage incl. model loads).
#   V=<arm> [env ...] bash scripts/r15/arm.sh <old03|m03|m18> ...
#   -> /workspace/dub/r15w/eval/<V>_<id>/<V>_<id>.mp4 (demucs bed, as the baseline's main output) + timing.json + logs/
set -u
cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null
export DUB_ALLOW_NC=1 DUB_ALLOW_NC_BED=1 METRICS=off
V=${V:?V=<arm name>}; X=/workspace/dub/r15w/eval; A=/workspace/dub/artifacts/r4a; B=/workspace/dub/artifacts/r4m
# the production face defaults (e2e.sh R10_FAST_FACE=1, E2E_FX=1, E2E_SYNC=1) + q2 + F3 (eval_r14.sh COMMON)
export GEN_COMP=${GEN_COMP:-0} FAST_SRCPREP=${FAST_SRCPREP:-1} FAST_FB_BG=${FAST_FB_BG:-1} FAST_ENC=${FAST_ENC:-ultrafast} FAST_MPREUSE=${FAST_MPREUSE:-1} \
       LS_VAE_COMPILE=${LS_VAE_COMPILE:-max-autotune-no-cudagraphs} CFD_EXTRA=${CFD_EXTRA:---threads=12}
export LS_CMS_READ=${LS_CMS_READ:-1} FC_CMS=${FC_CMS:-1} FC_EXTRA=${FC_EXTRA:---pre-threads=12 --sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.2 --reg-smooth 4 --mouth-region 0.45}
for id in "$@"; do
  ID=q2_$id; AT=r9_$ID; HT=r9_${ID}_htd; GT=r9g_${ID}_$V; FT=r9f_${ID}_$V; O=$X/${V}_$id; L=$O/logs; W=$O/work; mkdir -p $L $W
  [ -s $O/${V}_$id.mp4 ] && { echo "have ${V}_$id"; continue; }
  [ -f /workspace/dub/r15w/skip/${V}_$id ] && { echo "== SKIP ${V}_$id (r15w/skip)"; continue; }   # r15 Z: prune queued arms of a running chain
  export DUB_INPUTS=$X/$ID/e2e/work/in; SRC=$(ls $DUB_INPUTS/$ID.* 2>/dev/null | head -1)
  [ -n "$SRC" ] && [ -f $A/$AT/$ID/dubbed_16k.wav ] || { echo "FAIL ${V}_$id: no baseline audio / input for $ID"; continue; }
  env | grep -E "^(R10_GEN_PY|R13_|LS_|TBDUB|BON_|KF_|GEN_|FAST_|FC_|CFD_|E2E_)" | sort > $L/env.txt
  echo "arm ${V}_$id commit $(git rev-parse --short HEAD) $(date -Is) $(tr '\n' ' ' < $L/env.txt)" | tee $L/face.log
  export R10_JOB=${V}_$id; NV=$W/nvml.jsonl; rm -f $NV; /venv/main/bin/python scripts/r10/gpu_sampler.py --out $NV > $L/gpu_sampler.log 2>&1 & NM=$!
  T=$W/stages.tsv; : > $T; st() { echo -e "$1\t$(date +%s.%N)" >> $T; }
  rm -rf $B/$GT $B/$FT; unset MP_SOURCE_NPZ; SP=""
  st face_gen
  SMP=$A/sep/$ID/source_mediapipe.npz
  if [ "$FAST_MPREUSE" = 1 ] && [ -f $SMP ]; then export MP_SOURCE_NPZ=$SMP; mkdir -p $B/$GT/timing; cp $SMP $B/$GT/timing/${ID}_source_lm_F_mp.npz; fi
  if [ "$FAST_SRCPREP" = 1 ]; then
    WV=$B/$GT/work_${ID}_$GT/video; TD=$B/$GT/timing
    ( for i in $(seq 1800); do [ -s $WV/master_mediapipe.npz ] && break; sleep 2; done; [ -s $WV/master_mediapipe.npz ] || exit 0; mkdir -p $TD; TJ=$TD/${ID}_master_timing_F.json; SL=$TD/${ID}_source_lm_F.npz
      python scripts/quality/r3_master_timing.py --src $SRC --master $WV/master_25fps.mp4 --vf fps=25 --out $TJ && python scripts/quality/r3_source_lm.py --src $SRC --timing $TJ --out $SL \
        && python scripts/r9/r9_source_rigid.py --src $SRC --source-lm $SL ) > $L/srcprep.log 2>&1 & SP=$!
  fi
  AUDIO=$A/$AT TAG=$GT SKIP_SCORE=1 VIDEO_EXTRA="--codeformer optimized --codeformer-w 1.0 --steps 10 --steps-silent 20 --codeformer-parse off --codeformer-batch 16 ${GEN_EXTRA:-}" \
    bash scripts/r7/r7_e2e.sh $ID > $L/face_gen.log 2>&1
  if [ -n "$SP" ]; then [ -f $B/$GT/work_${ID}_$GT/${ID}_${GT}_25fps.manifest.json ] || kill $SP 2>/dev/null; wait $SP 2>/dev/null; fi
  nf=$(grep -c "LatentSync FAILED" $B/$GT/${ID}_$GT.log 2>/dev/null); nf=${nf:-0}   # a generator error = a pass-through segment = no lip-sync: not a valid arm
  if [ "$nf" -gt 0 ]; then echo "== ${V}_$id FAIL: $nf generator segments failed -> pass-through: $(grep -m1 'LatentSync FAILED' $B/$GT/${ID}_$GT.log | cut -c1-220)" | tee -a $L/face.log
  elif [ -f $B/$GT/work_${ID}_$GT/${ID}_${GT}_25fps.manifest.json ]; then
    st face_comp
    FBA="$(cat reports/experiments/r8-face/fb7.args) --open-gate 0.04 --dark-transient on --thr-wash-lips 0.2"
    CFDV=self; [ "${ARM_CFD:-self}" = off ] && CFDV=""   # ARM_CFD=off: no CodeFormer-detail step (needed when the generator ran with --codeformer off: no _ls_pre dump)
    REP=1 CFD_LABEL=ia04 CFD_ARGS="--inner-alpha 0.4 $CFD_EXTRA" REP_ARGS="--spikes off ${REP_ARGS:-}" REP_LABEL=rep2s \
    SKIP_SCORE=1 TAG=$FT SRC_TAG=$GT CFD=$CFDV SR=srt1 SR_EXTRA="--batch 8 --mask down2" FB=fb7s FB_ARGS="$FBA" EXTRA="--tex-tau 1000 --tex-sigma 2.5 --mouth-region 0.55 --cm-mouth-excl 0.5" \
      bash scripts/r7/r7_m7_comp.sh $ID > $L/face_comp.log 2>&1
  else echo "== ${V}_$id face_gen FAILED (see $L/face_gen.log)" | tee -a $L/face.log; fi
  st end; kill $NM 2>/dev/null; wait $NM 2>/dev/null
  VID=$B/$FT/${ID}_$FT.mp4
  if [ -f $VID ]; then
    ffmpeg -loglevel error -y -i $VID -i $A/$HT/$ID/final_mix_aac.m4a -map 0:v:0 -map 1:a:0 -c copy -movflags +faststart $O/${V}_$id.mp4
    vmin=$(python3 -c "import json,glob; print(json.load(open(glob.glob('$A/$AT/$ID/*_r4b.manifest.json')[0]))['source']['duration_s'] / 60)")
    /venv/main/bin/python scripts/r10/gpu_sampler.py --summary $NV --stages $T --job ${V}_$id --vmin $vmin > $O/timing.json 2>> $L/gpu_sampler.log
    python3 -c "import json; d = json.load(open('$O/timing.json')); t = d['total']; print('== ${V}_$id face GPU-busy', t.get('gpu_busy_s_per_vmin'), 's/vmin (job-attributed), weighted', t.get('gpu_weighted_s_per_vmin'), ', wall', t.get('wall_s_per_vmin'), '|', {k: v.get('gpu_busy_s_per_vmin') for k, v in d['stages'].items()})" 2>&1 | tee -a $L/face.log
  else echo "== ${V}_$id FAIL: no $VID (see $L)" | tee -a $L/face.log; fi
done
echo "ARM_DONE $V"
