#!/bin/bash
# Round 2 track B, stage 1: composite variants + every measurement on one decompose window dir (25 fps).
#   bash scripts/quality/r2_stage1.sh <id> <win dir> [variants...]      (default: all below)
set -u
id=$1; win=$2; shift 2
DUB_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; export DUB_ROOT; . "${DUB_ROOT}/scripts/env.sh" >/dev/null; cd "${DUB_ROOT}"
R=/workspace/dub/artifacts/restoration-temporal/r2; NPZ=$R/work/work_$id/master_mediapipe.npz; FA=/workspace/dub/common/from_A/artifacts/lipsync-temporal/r2
declare -A V=( [V0]="--variant clean0" [V1t]="--reg trans" [V1]="--reg sim" [V2]="--reg sim --ring on" [M1]="--reg none --mask ellipse_raw" [M2]="--reg none --mask nosecut"
               [V3]="--reg sim --ring on --mask nosecut" [V3e]="--reg sim --ring on --mask ellipse_raw" )
names=${*:-"V0 V1t V1 V2 M1 M2 V3e V3"}
A=$(python -c "import json;print(json.load(open('$win/window_manifest.json'))['window'][0])"); B=$(python -c "import json;print(json.load(open('$win/window_manifest.json'))['window'][1])")
for n in $names; do python scripts/quality/composite_register.py --win $win --npz $NPZ --name $n ${V[$n]} | python -c "import json,sys;d=json.loads(sys.stdin.read());print('  comp',d['name'],d.get('est_misreg_px'),d.get('kinds'),d['composite_wall_s_per_ls_video_min'],'s/LS-min')"; done
bash scripts/quality/r2_slide_win.sh $id $win $names S0_ls_raw | cut -c1-260
for n in $names; do
  python scripts/quality/r2_position_check.py --video $win/$n.mp4 --master $R/work/work_$id/master_25fps.mp4 --npz $FA/masters/$id/master_mediapipe.npz --offset $A --frames $A:$B --manifest $win/full_index.manifest.json --out $win/$n.pos.json | cut -c1-260
  python - $win $n $FA/anchor/${id}_rigid.npz <<'PY'
import json, sys, numpy as np, pathlib
sys.path.insert(0, "scripts/pilot"); import anchor_filters as af
win, n, rig = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
H = np.load(rig)["H"]; z = np.load(win / f"{n}.maskboxes.npz") if (win / f"{n}.maskboxes.npz").exists() else np.load(win / f"{n}.boxes.npz")
bx = np.full((len(H), 4), np.nan); bx[z["frames"]] = z["boxes"]; cxy = np.stack([(bx[:, 0] + bx[:, 2]) / 2, bx[:, 1] + 0.76 * (bx[:, 3] - bx[:, 1])], 1)
r = [np.linalg.norm((cxy[t] - cxy[t - 1]) - (af._apply(H[t], cxy[t - 1][None])[0] - cxy[t - 1])) for t in range(1, len(H)) if np.isfinite(cxy[t, 0]) and np.isfinite(cxy[t - 1, 0])]
o = {"mask_centre_resid_vs_head_px": {"mean": round(float(np.mean(r)), 4), "p95": round(float(np.percentile(r, 95)), 4), "n": len(r)}}
(win / f"{n}.masklag.json").write_text(json.dumps(o)); print("  masklag", n, o["mask_centre_resid_vs_head_px"])
PY
done
for n in $names; do python scripts/quality/mouth_jitter_hr.py --video $win/$n.mp4 --reference $win/master.mp4 --manifest $win/window_manifest.json --out $win/$n.mjhr.json --tag $n > $win/$n.mjhr.log 2>&1 & done; wait
flock /workspace/dub/gpu.lock python scripts/quality/window_syncnet.py --dir $win --stages master S0_ls_raw $names --out $win/syncnet.json 2>/dev/null | grep -v "^$"
echo STAGE1_WIN_DONE $win
