#!/usr/bin/env python3
"""r15 F: scan the held-out videos for fast face motion (r9_layer_repair.face_motion on the 25 fps master) and list which repaired runs of the
q2 layer repair sit in it. -> per video: motion p50 / p95 / p99, the fast frames (> thr), and every repaired run with the max motion over its
span +-2 frames.   python scripts/r15/motion_scan.py old03 m03 m18 [--thr 0.06] [--json out.json]"""
import sys, json, pathlib, argparse
import numpy as np, cv2
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "r9")); import r9_layer_repair as lr
ap = argparse.ArgumentParser(); ap.add_argument("ids", nargs="+"); ap.add_argument("--thr", type=float, default=0.06); ap.add_argument("--tag", default="q2"); ap.add_argument("--json", default="")
a = ap.parse_args(); out = {}
for idn in a.ids:
    W = pathlib.Path(f"/workspace/dub/artifacts/r4m/r9g_{a.tag}_{idn}/work_{a.tag}_{idn}_r9g_{a.tag}_{idn}"); P = f"{a.tag}_{idn}_r9g_{a.tag}_{idn}_25fps"
    lm = np.load(W / "video/master_mediapipe.npz")["lm"].astype(np.float64); c = cv2.VideoCapture(str(W / "video/master_25fps.mp4")); F = []
    while len(F) < len(lm):
        ok, f = c.read()
        if not ok: break
        F.append(f)
    mot = lr.face_motion(F, lm); del F
    rj = json.loads((W / f"{P}_ls_cfd_ia04_srt1_rep2s.repair.json").read_text())
    runs = [[b, e, round(float(mot[max(0, b - 3):min(len(mot), e + 4)].max()), 4)] for b, e in rj["repaired"]]
    fast = np.flatnonzero(mot > a.thr)
    out[idn] = {"frames": len(mot), "p50": round(float(np.percentile(mot, 50)), 4), "p95": round(float(np.percentile(mot, 95)), 4), "p99": round(float(np.percentile(mot, 99)), 4),
                "fast_frames": int(len(fast)), "fast_s": [round(k / 25, 2) for k in fast[:60]], "repaired_runs": runs,
                "repaired_in_motion": [r + [round(r[0] / 25, 2)] for r in runs if r[2] > a.thr]}
    print(f"== {idn}: motion p50 {out[idn]['p50']} p95 {out[idn]['p95']} p99 {out[idn]['p99']}; fast (> {a.thr}) {len(fast)} frames; repaired runs {len(runs)}, in motion {out[idn]['repaired_in_motion']}", flush=True)
    print("   all repaired runs [b, e, motion]:", runs, flush=True)
    np.save(f"/workspace/dub/r15f/motion_{idn}.npy", mot)
if a.json: pathlib.Path(a.json).write_text(json.dumps(out, indent=1))
