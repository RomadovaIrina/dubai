#!/usr/bin/env python3
"""r7 F: WHOLE-FRAME gate. Everything outside a small mouth region must be the source: |out - src| ~ 0 there, on EVERY frame.
(User, r6 m5 05 25-26 s: 'a rectangle with the face plus a background from another frame is pasted' while the camera shows trees.
The mouth-crop sheets could not see it; this gate looks at the whole frame.)

Per source frame k: grey |out_k - src_k| > --thr (default 24, well above codec noise: crf 13 re-encode of a phone h264 is < 8), outside the
mouth region = the source lips hull (MediaPipe on the SOURCE frames) dilated by --mouth-iod inter-ocular distances (default 0.6: the composite
pastes <= --mouth-region 0.38 IOD + the 12 px feather) -> the defect area in px and in % of the frame; a frame WITHOUT a source face has no
mouth region, so any change there is a defect. A video PASSES when no frame has more than --max-pct (0.2 %) of the frame changed outside
the mouth, and the mean outside-mouth |diff| (changed px only) stays < --max-mean.
Output: <out>.json (per-frame area, the worst frames), a line '== <label> <id> PASS|FAIL ...' and --sheet: top-N whole-frame sheet SRC | OUT | DIFF.
    python scripts/r7/r7_frame_gate.py --src SRC.mp4 --out-video OUT.mp4 --lm <src mediapipe npz> --out report.json [--sheet sheet.jpg --top 12]
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "quality"))
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]


def frames(path):
    import av
    c = av.open(path)
    for fr in c.decode(c.streams.video[0]):
        yield fr.to_ndarray(format="gray")
    c.close()


def frames_bgr(path, keep: set):
    import av
    c = av.open(path); out = {}
    for k, fr in enumerate(c.decode(c.streams.video[0])):
        if k in keep: out[k] = fr.to_ndarray(format="bgr24")
        if k >= max(keep): break
    c.close(); return out


def mouth_mask(lm_k, H, W, iod_mult):
    import cv2
    if lm_k is None or not np.isfinite(lm_k[LIPS_OUTER]).all() or not np.isfinite(lm_k[[33, 263]]).all(): return None
    iod = float(np.linalg.norm(lm_k[33, :2] - lm_k[263, :2])); hull = cv2.convexHull(lm_k[LIPS_OUTER, :2].astype(np.float32))[:, 0]
    m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [np.round(hull).astype(np.int32)], 1)
    r = int(max(2, round(iod_mult * iod)))
    return cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0


def main() -> int:
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True); ap.add_argument("--out-video", required=True); ap.add_argument("--lm", required=True, help="MediaPipe npz of the SOURCE frames (clean_face_stack.run_mediapipe)")
    ap.add_argument("--out", required=True); ap.add_argument("--label", default=""); ap.add_argument("--id", default="")
    ap.add_argument("--thr", type=float, default=24.0); ap.add_argument("--mouth-iod", type=float, default=0.6)
    ap.add_argument("--max-pct", type=float, default=0.2, help="FAIL if any frame changes more than this %% of the frame outside the mouth region")
    ap.add_argument("--max-mean", type=float, default=60.0, help="FAIL if the mean |diff| of the changed outside-mouth pixels of the worst frame exceeds this (a pasted crop is ~ 80+, a feather halo ~ 30)")
    ap.add_argument("--sheet", default=None); ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    lm = np.load(a.lm)["lm"]; area = []; mean = []; nface = 0; H = W = None
    for k, (s, o) in enumerate(zip(frames(a.src), frames(a.out_video))):
        if H is None: H, W = s.shape
        d = cv2.absdiff(s, o); mm = mouth_mask(lm[k] if k < len(lm) else None, H, W, a.mouth_iod)
        if mm is not None: nface += 1; d[mm] = 0
        ch = d > a.thr; n = int(ch.sum()); area.append(n); mean.append(float(d[ch].mean()) if n else 0.0)
    area = np.array(area); pct = 100.0 * area / (H * W); worst = np.argsort(-pct)[: a.top]
    fail_frames = [int(k) for k in np.flatnonzero(pct > a.max_pct)]
    verdict = "PASS" if not fail_frames or max(mean[k] for k in fail_frames) < a.max_mean else "FAIL"
    if fail_frames and verdict == "PASS": verdict = "WARN"   # area over the limit but faint (a feather halo, a colour-match halo)
    rep = {"id": a.id, "label": a.label, "src": a.src, "out": a.out_video, "frames": int(len(area)), "frames_with_face": nface, "thr": a.thr, "mouth_iod": a.mouth_iod,
           "max_pct": a.max_pct, "max_mean": a.max_mean, "verdict": verdict, "worst_pct": round(float(pct.max()), 3) if len(pct) else None,
           "frames_over": len(fail_frames), "frames_over_list": fail_frames[:200], "p99_pct": round(float(np.percentile(pct, 99)), 3) if len(pct) else None,
           "worst": [{"frame": int(k), "t_s": round(k / 30.0, 2), "pct": round(float(pct[k]), 3), "mean_diff": round(mean[k], 1)} for k in worst]}
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(a.out).write_text(json.dumps(rep, indent=1))
    print(f"== frame_gate {a.label} {a.id} {verdict} worst {rep['worst_pct']} % at frame {int(worst[0]) if len(worst) else -1} (mean |d| {mean[int(worst[0])] if len(worst) else 0:.0f}); frames over {a.max_pct} %: {len(fail_frames)}; p99 {rep['p99_pct']} %; face in {nface}/{len(area)}")
    if a.sheet and len(worst):
        ks = sorted(int(k) for k in worst); S = frames_bgr(a.src, set(ks)); O = frames_bgr(a.out_video, set(ks)); tiles = []
        for k in ks:
            d = cv2.absdiff(cv2.cvtColor(S[k], cv2.COLOR_BGR2GRAY), cv2.cvtColor(O[k], cv2.COLOR_BGR2GRAY)); mm = mouth_mask(lm[k] if k < len(lm) else None, H, W, a.mouth_iod)
            dm = cv2.applyColorMap(np.clip(d.astype(np.float32) * 3, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
            if mm is not None: cv2.drawContours(dm, cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (255, 255, 255), 1)
            row = np.hstack([S[k], O[k], dm]); sc = 360.0 / H
            row = cv2.resize(row, None, fx=sc, fy=sc); cv2.putText(row, f"{k} {k / 30:.2f}s {pct[k]:.2f}% |d| {mean[k]:.0f}", (6, 24), 0, 0.7, (0, 0, 255), 2); tiles.append(row)
        cols = 2 if len(tiles) > 4 else 1; rows_ = [np.hstack(tiles[i:i + cols]) if len(tiles[i:i + cols]) == cols else np.hstack(tiles[i:i + cols] + [np.zeros_like(tiles[0])] * (cols - len(tiles[i:i + cols]))) for i in range(0, len(tiles), cols)]
        cv2.imwrite(a.sheet, np.vstack(rows_), [cv2.IMWRITE_JPEG_QUALITY, 80])
    return 0 if verdict != "FAIL" else 2


if __name__ == "__main__":
    sys.exit(main())
