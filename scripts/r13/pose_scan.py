#!/usr/bin/env python3
"""r13 spec 5.4 / 5.5: per-frame head pose (MediaPipe Face Landmarker transformation matrix -> yaw / pitch / roll, clean_face_stack.pose_from_matrix,
the same call the production pre-check uses) on a video; the 5.5 decision with the production rule (POSE_PRECHECK: > 30 % of face frames
|yaw| > 30 deg -> voice-only) and, for the report, the same share for |pitch| and for max(|yaw|, |pitch|).
    python scripts/r13/pose_scan.py --out DIR video [video ...]      -> DIR/<stem>.pose.json (+ the yaw / pitch series, every frame)"""
import argparse, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot"))
import clean_face_stack as cfs  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--every", type=int, default=1); ap.add_argument("videos", nargs="+")
    a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for v in a.videos:
        v = pathlib.Path(v); npz = cfs.run_mediapipe(v, out / f"{v.stem}_mp.npz", mode="video", num_faces=1, every=a.every); z = np.load(npz)
        pose = cfs.pose_from_matrix(z["matrix"]); idx = np.arange(0, len(pose), a.every); pose = pose[idx]; face = np.isfinite(pose[:, 0])
        thr, mf = cfs.POSE_PRECHECK["yaw_deg"], cfs.POSE_PRECHECK["max_fraction"]; nf = max(int(face.sum()), 1)
        fy = float((np.abs(pose[face, 0]) > thr).sum() / nf); fp = float((np.abs(pose[face, 1]) > thr).sum() / nf)
        fa = float((np.maximum(np.abs(pose[face, 0]), np.abs(pose[face, 1])) > thr).sum() / nf)
        r = {"video": str(v), "frames": int(len(pose)), "face_frames": int(face.sum()), "face_frac": round(float(face.mean()), 3),
             "yaw_p1_p50_p99": np.round(np.nanpercentile(pose[:, 0], [1, 50, 99]), 1).tolist() if face.any() else None,
             "pitch_p1_p50_p99": np.round(np.nanpercentile(pose[:, 1], [1, 50, 99]), 1).tolist() if face.any() else None,
             "frac_yaw_gt30": round(fy, 4), "frac_pitch_gt30": round(fp, 4), "frac_any_gt30": round(fa, 4),
             "decision": "no_face" if not face.any() else ("voice_only" if fy > mf else "lipsync"), "rule": cfs.POSE_PRECHECK}
        (out / f"{v.stem}.pose.json").write_text(json.dumps({**r, "yaw": np.round(pose[:, 0], 1).tolist(), "pitch": np.round(pose[:, 1], 1).tolist()}))
        print(f"== {v.name}: faces {r['face_frac']:.0%}, |yaw|>30 {fy:.1%}, |pitch|>30 {fp:.1%}, any {fa:.1%}, yaw p1/50/99 {r['yaw_p1_p50_p99']} -> {r['decision']}", flush=True)


if __name__ == "__main__":
    main()
