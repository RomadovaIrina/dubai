#!/bin/bash
# Week 2 CodeFormer A/B: re-run ONLY the video stage on a baseline work dir (audio stages reused) with the optimized
# CodeFormer, then measure: quality_manifest (+SyncNet), face_metrics, codeformer_quality (locality/seams/geometry).
# Usage: scripts/quality/run_codeformer_ab.sh <id> <tag> [candidate_video_stage.py args...]
#   scripts/quality/run_codeformer_ab.sh 04 cf_w05 --codeformer optimized --codeformer-w 0.5 --codeformer-batch 8
# Outputs: /tmp/dabai_quality/candidates/<id>_<tag>.mp4 + manifest + .nvsmi.log + .log, manifests/ + frames/ analyses.
set -uo pipefail
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; . "${DUB_ROOT}/scripts/env.sh"; cd "${DUB_ROOT}"
id="$1"; tag="$2"; shift 2
Q=/tmp/dabai_quality; C="${Q}/candidates"; mkdir -p "$C"
echo "===== $id $tag start $(date -Iseconds) args: $*"
s=$(date +%s)
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 2 > "${C}/${id}_${tag}.nvsmi.log" 2>/dev/null & NV=$!
python scripts/quality/candidate_video_stage.py --id "$id" --tag "$tag" "$@" > "${C}/${id}_${tag}.log" 2>&1; rc=$?
kill $NV 2>/dev/null; wait $NV 2>/dev/null
echo "===== $id $tag exit $rc wall $(( $(date +%s) - s ))s peak_nvsmi_MiB=$(sort -n "${C}/${id}_${tag}.nvsmi.log" | tail -1)"
[ $rc -eq 0 ] || exit $rc
python scripts/quality/quality_manifest.py --run-dir "$C" --tag "$tag" --ids "$id" --syncnet > "${C}/${id}_${tag}.quality.log" 2>&1 || echo "quality_manifest failed"
python scripts/quality/face_metrics.py --run-dir "$C" --tag "$tag" --ids "$id" > "${C}/${id}_${tag}.face_metrics.log" 2>&1 || echo "face_metrics failed"
python scripts/quality/codeformer_quality.py --id "$id" --tag "$tag" --run-dir "$C" > "${C}/${id}_${tag}.cfq.log" 2>&1 || echo "codeformer_quality failed"
python - "$id" "$tag" <<'PY'
import json, pathlib, sys
i, t = sys.argv[1:]; Q = pathlib.Path("/tmp/dabai_quality")
m = json.loads((Q / "candidates" / f"{i}_{t}.manifest.json").read_text()); L = m["lipsync"]; cf = L.get("codeformer")
q = json.loads((Q / "manifests" / f"{i}_{t}.quality.json").read_text()); fm = json.loads((Q / "manifests" / f"{i}_{t}.face_metrics.json").read_text())
cq = json.loads((Q / "manifests" / f"{i}_{t}.codeformer_quality.json").read_text())
sn = q.get("syncnet", {}); segs = fm["segments"]
def mean(k): return round(sum(s[k] for s in segs) / max(len(segs), 1), 3)
print(f"== {i} {t}: validation {m['validation']['pass']} | stage s {m['stage_seconds'].get('codeformer')} cf / {m['stage_seconds'].get('latentsync')} ls / total {m['stage_seconds']['total']} | "
      f"cf {cf if isinstance(cf, str) else {k: cf[k] for k in ('eligible_frames', 'master_frames', 'seconds')} } totals {None if isinstance(cf, str) else cf['totals']}")
print(f"   syncnet out {sn.get('output', {}).get('confidence')} / offset {sn.get('output', {}).get('av_offset_frames')} (orig {sn.get('original', {}).get('confidence')}) | "
      f"mouth sharp {mean('mouth_sharp_ratio')} upper sharp {mean('upper_sharp_ratio')} flicker {mean('flicker_ratio')} | geometry identical {cq['geometry']['identical']} untouched bit-identical {cq['untouched_frames']['bit_identical']} "
      f"outside-face px max {cq['locality']['outside_dilated_face_px_max']} seam ratio {cq['seam']['ratio_mean']} (p95 {cq['seam']['ratio_p95']}) eyes sharp {cq['sharpness']['upper_face_ratio_mean']} mask jitter {cq['mask_motion']['centroid_rel_jitter_px_mean']}")
PY
