#!/bin/bash
# r15 W: metrics of several arms on one held-out video, all on the q2 + F3 baseline's dub: SyncNet (scripts/r11/sync_speech.py, the gate's,
# cached per arm), the mouth flicker (scripts/r14/flicker_metric.py, at the user's eye spots) and the mouth opening on voiced frames
# (scripts/r14/mouth_open.py, ratios vs the first arm = q2). Arm "q2" = the baseline E2E output; any other = scripts/r15/arm.sh's.
#   bash scripts/r15/metrics.sh <old03|m03|m18> q2 <arm> [<arm> ...]   -> eval/metrics_<id>.txt (+ .json files)
set -u
cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai; . scripts/env.sh >/dev/null
X=/workspace/dub/r15w/eval; A=/workspace/dub/artifacts/r4a; id=$1; shift
declare -A AT=([old03]="3.5 5.5 20.9 24 27.5" [m03]="17.5 18.5 19.5 20.5" [m18]="2.5 3.5 4.5 5.5")
vid(){ [ "$1" = q2 ] && echo $X/q2_$id/q2_$id.mp4 || echo $X/${1}_$id/${1}_$id.mp4; }
WAV=$A/r9_q2_$id/q2_$id/dubbed_16k.wav; SRC=$(ls $X/q2_$id/e2e/work/in/q2_$id.* | head -1)
LM=$(find /workspace/dub/artifacts/r4m -path "*r9g_q2_${id}/*" -name "q2_${id}_source_lm_F.npz" | head -1)
VS=(); for a in "$@"; do v=$(vid $a); [ -s $v ] || { echo "missing $a ($v)"; continue; }; VS+=("$a=$v")
  [ -s $X/${a}_$id.sync.txt ] || flock /workspace/dub/gpu.lock python scripts/r11/sync_speech.py --video $v --audio $WAV --label ${a}_$id 2>&1 | grep "^== sync" > $X/${a}_$id.sync.txt; done
{ for a in "$@"; do cat $X/${a}_$id.sync.txt 2>/dev/null; done
  python scripts/r14/flicker_metric.py --lm $LM --video src=$SRC "${VS[@]}" --at ${AT[$id]} --json $X/flicker_$id.json 2>&1 | tail -n $(( ${#VS[@]} + 3 ))
  python scripts/r14/mouth_open.py --wav $WAV --video "${VS[@]}" --json $X/mouth_$id.json 2>&1 | tail -n $(( ${#VS[@]} + 3 ))
  for a in "$@"; do [ -s $X/${a}_$id/timing.json ] && python3 -c "import json; t = json.load(open('$X/${a}_$id/timing.json'))['total']; print('gpu ${a}_$id face GPU-busy', t.get('gpu_busy_s_per_vmin'), 's/vmin')"; done
} | tee $X/metrics_$id.txt
