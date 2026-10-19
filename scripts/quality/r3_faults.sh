#!/bin/bash
# Round 3 (track A), spec 11 fault injection on one video (audio-r2, r2best generator):  r3_faults.sh <id> <arm>...
#   lserr     LatentSync raises RuntimeError on its 2nd call -> that segment PASS_THROUGH_LS_FAILED, worker survives
#   lmdrop2   MediaPipe landmarks removed on 2 % of the frames (single frames) + three 8-frame gaps (> the 5-frame hold -> split)
#   lmdrop10  10 % of the frames removed in 3-frame runs (all held) -> degraded > 2 % -> PARTIAL_SUCCESS expected
set -u
cd /workspace/dub/dubai; export DUB_ROOT=/workspace/dub/dubai
id=$1; shift; O=/workspace/dub/artifacts/r3/faults; M=/workspace/dub/artifacts/r3/masters/$id; mkdir -p $O
for arm in "$@"; do
  W=$O/work_${id}_$arm; mkdir -p $W; ln -sfn $M/master_25fps.mp4 $W/master_25fps.mp4; ln -sfn $M/master_rigid.npz $W/master_rigid.npz; ln -sfn $M/master_mediapipe_probe2.npz $W/master_mediapipe_probe2.npz
  X=""
  case $arm in
    lserr) ln -sfn $M/master_mediapipe.npz $W/master_mediapipe.npz; X="--inject-ls-error 2";;
    lmdrop2|lmdrop10) /venv/dabai/bin/python - $M/master_mediapipe.npz $W/master_mediapipe.npz $arm <<'PY'
import sys, numpy as np
z = dict(np.load(sys.argv[1])); lm = z["lm"].copy(); n = len(lm); rng = np.random.default_rng(11); face = np.flatnonzero(np.isfinite(lm[:, 0, 0])); drop = []
if sys.argv[3] == "lmdrop2":
    drop = list(rng.choice(face, int(0.02 * len(face)), replace=False))
    for c in rng.choice(face[20:-20], 3, replace=False): drop += list(range(c, c + 8))
else:
    for c in rng.choice(face[5:-5], int(0.10 * len(face) / 3), replace=False): drop += [c, c + 1, c + 2]
drop = np.array(sorted(set(d for d in drop if d < n))); lm[drop] = np.nan
if "matrix" in z: z["matrix"] = z["matrix"].copy(); z["matrix"][drop] = np.nan
z["lm"] = lm; z["presence"] = np.isfinite(lm[:, 0, 0]); np.savez_compressed(sys.argv[2], **z); print("dropped", len(drop), "of", len(face), "face frames")
PY
    ;;
    *) echo "unknown $arm"; continue;;
  esac
  [ -f $O/${id}_$arm.manifest.json ] || DUB_CPUS=${DUB_CPUS:-0-7} scripts/nothp /venv/dabai/bin/python scripts/quality/candidate_video_stage.py --id $id --tag $arm --manifest /workspace/dub/artifacts/r3/audio-r2/$id.manifest.json \
    --dubbed-dir /workspace/dub/handoff/in/C/audio-r2/$id --out $O --work-out $W --mux final_mix --gate-merge-gap-s 1.2 --mouth-only-pasteback on --mask-smooth 3 --ls-landmark-smooth 2 \
    --codeformer-scope mouth --codeformer off --face-stack mediapipe --anchor-filter flow:4 --anchor-round off --affine-precision fp32 --mask-track head --window-batch-size 1 $X > $O/${id}_$arm.log 2>&1 || echo "FAIL $id $arm"
  grep "^== " $O/${id}_$arm.log | cut -c1-200
  /venv/dabai/bin/python -c "import json; m=json.load(open('$O/${id}_$arm.manifest.json')); print('$arm', m.get('video_status'), m['lipsync']['by_action'])"
done
