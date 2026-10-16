#!/bin/bash
# Track B: whole-video candidates on finished baseline runs (video stage only; the run's audio is copied into every candidate).
#   bash scripts/quality/run_candidates_full.sh 03 04 05 01
# ROOT/<id>/: lossless stage videos (removed at the end), ROOT/final/<id>_<tag>.mp4 (+ manifest, eval, mouth_jitter_hr, face_temporal)
set -u
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DUB_ROOT; . "${DUB_ROOT}/scripts/env.sh"; cd "${DUB_ROOT}"
BASE="${BASELINE_ROOT:-/workspace/dub/artifacts/baseline}"; ROOT="${CAND_ROOT:-/workspace/dub/artifacts/restoration-temporal/full}"; F="$ROOT/final"; mkdir -p "$F"
for id in "$@"; do
  M="$BASE/$id/final/${id}_final.manifest.json"; D="$ROOT/$id"; mkdir -p "$D"; t0=$(date +%s)
  echo "===== $id $(date -Iseconds)"
  python scripts/quality/decompose_video_stage.py --manifest "$M" --frames all --out "$D" --stages-out S0_ls_raw S1_ls_cf S3_final 2>&1 | grep -E "^window|piece|times|Traceback|Error" ; t1=$(date +%s)
  python scripts/quality/restore_stage.py --dir "$D" --kind gfpgan14_fixed 2>&1 | grep -E "seconds|Traceback|Error"; t2=$(date +%s)
  python scripts/quality/temporal_mouth_stabilize.py --dir "$D" --tag C2T_nocf --boundary on --skin-texture source 2>&1 | grep -E '"seconds"|Traceback|Error' | cut -c1-120
  python scripts/quality/temporal_mouth_stabilize.py --dir "$D" --tag C1C2T_gfpgan --detail S1_gfpgan14_fixed --boundary on --alpha 0.5 --gain 1.3 --skin-texture source 2>&1 | grep -E '"seconds"|Traceback|Error' | cut -c1-120
  t3=$(date +%s)
  echo "  timing: decompose(LS+CF) $((t1-t0)) s, gfpgan $((t2-t1)) s, stabilise x2 $((t3-t2)) s"
  for tag in C2T_nocf C1C2T_gfpgan; do python scripts/quality/candidate_eval.py --manifest "$M" --frames "$D/$tag.mp4" --tag "$tag" --out "$F" 2>&1 | tail -1 | cut -c1-400; done
  rm -f "$D"/*.mp4; echo "===== $id done $(( $(date +%s) - t0 )) s"
done
echo ALL_CANDIDATES_DONE
