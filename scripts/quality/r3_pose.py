#!/usr/bin/env python3
"""Round 3 (track A), spec 5.4 / 5.5: per-frame head pose (MediaPipe facial transformation matrix, clean_face_stack.pose_from_matrix) on the
25 fps masters -> share of face frames with |yaw| > 30 deg (and |pitch| > 30), whether the voice-only pre-check fires (> 30 % of face
frames), and how many lip-sync-eligible frames a per-frame gate at 30 / 20 deg would pass through.   r3_pose.py ID... [--md OUT]"""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot"))
import clean_face_stack as cfs
M = pathlib.Path("/workspace/dub/artifacts/r3/masters")
rows = []; out = {}
args = [a for a in sys.argv[1:] if not a.startswith("--")]; md = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
args = [a for a in args if a != md]
for v in args:
    z = np.load(cfs.run_mediapipe(M / v / "master_25fps.mp4", M / v / "master_mediapipe.npz"))
    pose = cfs.pose_from_matrix(z["matrix"]); face = np.isfinite(pose[:, 0]); n = len(pose); nf = int(face.sum())
    yaw, pitch = np.abs(pose[:, 0]), np.abs(pose[:, 1])
    r = {"frames": n, "face_frames": nf, "face_share": round(nf / n, 3)}
    if nf:
        r.update({"yaw_p50": round(float(np.nanmedian(yaw)), 1), "yaw_p99": round(float(np.nanpercentile(yaw[face], 99)), 1), "yaw_max": round(float(np.nanmax(yaw)), 1),
                  "share_yaw_gt30": round(float((yaw[face] > 30).mean()), 4), "share_pitch_gt30": round(float((pitch[face] > 30).mean()), 4),
                  "share_yaw_gt20": round(float((yaw[face] > 20).mean()), 4)})
        r["precheck_fires"] = bool(r["share_yaw_gt30"] > cfs.POSE_PRECHECK["max_fraction"])
        r["gate30_frames"] = int((face & (yaw > 30)).sum()); r["gate20_frames"] = int((face & (yaw > 20)).sum())
        # longest run over 30 deg (s)
        b = face & (yaw > 30); best = cur = 0
        for x in b: cur = cur + 1 if x else 0; best = max(best, cur)
        r["longest_gt30_s"] = round(best / 25, 2)
    else:
        r["precheck_fires"] = None; r["note"] = "no face -> spec 5.4 'no face' branch (voice-only / abort per product rule)"
    out[v] = r
    rows.append(f"| {v} | {nf}/{n} ({r['face_share']:.0%}) | {r.get('yaw_p50','–')} / {r.get('yaw_p99','–')} / {r.get('yaw_max','–')} | "
                f"{100 * r.get('share_yaw_gt30', 0):.2f} % | {100 * r.get('share_pitch_gt30', 0):.2f} % | {r.get('longest_gt30_s','–')} | {r['precheck_fires']} | {r.get('gate30_frames','–')} / {r.get('gate20_frames','–')} |")
hdr = ["| video | face frames | |yaw| p50 / p99 / max (deg) | face frames |yaw| > 30 | |pitch| > 30 | longest > 30 run s | pre-check fires (> 30 %) | per-frame gate frames 30 / 20 deg |",
       "|---|---|---|---|---|---|---|---|"]
txt = "\n".join(hdr + rows); print(txt)
if md: pathlib.Path(md).write_text(txt + "\n"); pathlib.Path(md).with_suffix(".json").write_text(json.dumps(out, indent=1))
