#!/bin/bash
# r14 F: fresh box -> setup -> r9 models -> pynvml -> the LS q2 E2E baselines (E2E_FX=1 E2E_SYNC=1 + q2 flags; every intermediate layer kept)
# for old03 then m03 (when its upload is complete), the k5-q2 gate bases, then the per-stage flicker diagnosis at the user's timestamps.
#   setsid nohup bash scripts/r12/r14_chain.sh > /workspace/dub/r14/logs/chain.out 2>&1 &
cd /workspace/dub/dubai; mkdir -p /workspace/dub/r14/logs /workspace/dub/r14/diag
export E2E_METRICS=off DUB_ALLOW_NC=1 LS_CMS_READ=1 FC_CMS=1 FC_EXTRA="--pre-threads=12 --sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.2"
until grep -q SETUP_DONE /workspace/dub/setup.log; do sleep 30; done; echo "== setup done $(date +%T)"
[ -f /workspace/dub/r9_models.done ] || { bash scripts/r9/r9_models.sh > /workspace/dub/r9_models.log 2>&1 && touch /workspace/dub/r9_models.done; }; tail -1 /workspace/dub/r9_models.log
/venv/main/bin/python -c "import pynvml" 2>/dev/null || uv pip install --python /venv/main/bin/python nvidia-ml-py 2>&1 | tail -1
M=/workspace/dub/inputs/mamakris; T=/workspace/dub/inputs/test_videos_raw; R=/workspace/dub/r10/runs; B=/workspace/dub/artifacts/r4m; mkdir -p $R
declare -A F=([m03]=$M/story-1774249919482-367803614.MOV [m18]=$M/story-1774250817106-616905094.mp4 [old03]=$T/03.mp4)
declare -A WIN=([old03]="3-4,5-6,20.8-21,24,27-28" [m03]="")
ready() { local f=$1 s0 s1; [ -f $f ] || return 1; s0=$(stat -c %s $f); sleep 60; s1=$(stat -c %s $f); [ "$s0" = "$s1" ] && ffprobe -v error -show_entries format=duration -of csv=p=0 $f > /dev/null 2>&1; }
for id in old03 m03; do
  until ready ${F[$id]}; do sleep 30; done; echo "== input $id ready $(date +%T)"
  [ -f $R/$id/${id}_DabAI_r9_demucs.mp4 ] || E2E_FX=1 E2E_SYNC=1 ID=$id bash scripts/r9/e2e.sh ${F[$id]} $R/$id > $R/$id.out 2>&1
  grep -h "NVML GPU-busy\|R9_E2E_DONE" $R/$id/logs/e2e.log | tail -2
  [ -n "${WIN[$id]}" ] && { . scripts/env.sh >/dev/null; python scripts/r12/stage_diag.py $id r9g_$id $B/r9f_$id/${id}_r9f_$id.mp4 "${WIN[$id]}" --out /workspace/dub/r14/diag/${id}_stages.md; }
  DUB_INPUTS=$R/$id/work/in bash scripts/r10/gate.sh $id "" > /workspace/dub/r14/logs/gate_${id}base.log 2>&1; echo "== $id base + gate $(date +%T)"
done
echo R14_CHAIN_DONE
