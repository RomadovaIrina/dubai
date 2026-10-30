#!/bin/bash
# Round 4 V queue 7 (new server, all renders lost): rebuild v4-repro (gate PREV of v1) + r4_video_v1 (gate PREV of v2), then the v2 arms:
#   task 5: chin3 / chin5 (v1 + --chin-clip 0.3 / 0.5, CPU on the v4-repro raw layers)
#   task 4: v5colour (v1 + v5's --sanitize-colour), v5 E2E on 04 (the run the user saw the dark half face in) + r4_face_metrics
#   task 6: g25 / g25nb (guidance 2.5 with / without the bridge, NEW generator run on the v4-repro audio, v1 composite)
# GPU jobs run one at a time under /workspace/dub/gpu.lock; CPU composites + scores overlap them. Resumable: finished outputs are skipped.
# Scores in dependency order: v4repro WITH its own ASR first, the arms (same muxed audio, md5-checked) reuse its WER.
#   bash scripts/r4/r4_queue_q7.sh [ids]
set -u
DUB_ROOT=/workspace/dub/dubai; . $DUB_ROOT/scripts/env.sh >/dev/null; cd $DUB_ROOT
B=/workspace/dub/artifacts/r4v; IDS=${*:-04 03 05 01 02}; mkdir -p $B /workspace/dub/artifacts/r3d
until grep -q SETUP_DONE /workspace/dub/setup.log && grep -q "BUILD_EXIT 0" $B/llama_build.log; do sleep 30; done
[ -L /workspace/dub/artifacts/r3d/timing ] || ln -s $B/v4repro/timing /workspace/dub/artifacts/r3d/timing   # r4_gate / r3_scorecard source landmarks
V1="--register sim_ring --ring-region upper --edge-px 12 --colour-match upper --sharpen-auto 0.95 --skin-texture source --lip-guard on --retime flow_safe --face-clip on --sanitize-layer 0.83 --sanitize-mode texture --sanitize-sigma 8 --sharpen-lips-max 0.8"
ASR="--asr-from $B/v4repro:v4repro"
score() { local l=$1 d=$2 id=$3; shift 3
  python scripts/r4/r4_gate.py score --label $l --dir $d --ids $id "$@" > $d/${id}_$l.score.out 2>&1 && echo "== scored $id $l" || echo "FAIL score $id $l"; }
arm() { local id=$1 d=$2 a=$3; [ -f $d/${id}_$a.mp4 ] || OUT_DIR=$d FIX30_THREADS=8 bash scripts/r4/r4_ablate.sh $id $a; score $a $d $id $ASR; }
cpu_arms() {   # after v4repro of one id: its own score, then v1 + task 4/5 arms (CPU) + scores
  local id=$1; score v4repro $B/v4repro $id
  arm $id $B/v1 v1 & for a in chin3 chin5 v5colour; do arm $id $B/arms $a & done; wait
}
for id in $IDS; do
  [ -f $B/v4repro/${id}_v4repro.manifest.json ] || bash scripts/r4/r4_v4repro.sh $id
  cpu_arms $id > $B/q7_cpu_$id.out 2>&1 &
done
mkdir -p $B/gen; [ -e $B/gen/timing ] || ln -s $B/v4repro/timing $B/gen/timing
for id in $IDS; do
  for a in g25 g25nb; do
    G="--anchor-filter flow:4 --anchor-round off --affine-precision fp32 --mask-track head --master-timing-mode fps --guidance 2.5 --max-ls-frames 400"
    [ $a = g25 ] && G="$G --bridge-len 8 --bridge-steps 4"
    [ -f $B/gen/${id}_$a.manifest.json ] || GEN_ARGS="$G" FIX30_ARGS="$V1" bash scripts/r4/r4_video.sh $id $B/audio-r4b/$id $B/audio-r4b/$id/${id}_r4b.manifest.json $B/gen $a
    (until grep -q "== scored $id v4repro" $B/q7_cpu_$id.out 2>/dev/null; do sleep 20; done; score $a $B/gen $id $ASR) &   # overlaps the next generator run
  done
done
mkdir -p $B/v5; [ -f $B/v5/04_e2e_v5.mp4 ] || flock /workspace/dub/gpu.lock bash scripts/r3_e2e_v5.sh 04 $B/v5 > $B/v5/run.out 2>&1
python scripts/r4/r4_face_metrics.py --video $B/v5/04_e2e_v5.mp4 --src ${DUB_INPUTS:-/workspace/dub/inputs/test_videos_raw}/04.mp4 --src-lm $B/v4repro/timing/04_source_lm_F.npz --out $B/v5/04_e2e_v5.r4face.json > $B/v5/face.out 2>&1
wait
cat $B/q7_cpu_*.out | grep -E "^==|FAIL"
echo Q7_DONE
