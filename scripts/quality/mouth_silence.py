#!/usr/bin/env python3
"""Does the dubbed speaker keep his mouth shut when the dub is silent? (2026-10-14 human feedback: "открывание рта когда звука нет никакого",
"губы не должны дёргаться или шевелиться, когда на англе речь закончилась, а на русском нет — пусть закроет рот").

Per output frame inside LatentSync segments: insightface 2d106 inner-lip gap / inter-brow distance (face_aware_latentsync_accel.mouth_open) on the
OUTPUT and on the 25 fps MASTER. Frames are classed by the dub track (Silero VAD of dubbed_16k, +-pad): dub_silent / dub_speech, and by the
source (manifest speech_intervals): src_speech / src_silent. Reported per class: mean / p90 mouth opening, share of "open" frames (> closed
level + 0.08; closed level = 10th percentile of the master opening in source-silent frames) and motion (mean |frame-to-frame change|).
The key row is dub_silent & src_speech: the master mouth moves there (the speaker talks), the output mouth must not.
Contact sheet of the worst dub-silent frames (output | master) with --sheet.

    python scripts/quality/mouth_silence.py --manifest RUN/final/03_final.manifest.json [--video OTHER.mp4] [--pad 0.12] [--sheet out.jpg] [--json out.json]
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parents[1] / "third_party" / "latentsync"))
FPS = 25


def frames_at(path: str, idx: list[int]) -> dict:
    import cv2
    want = set(idx); out = {}; cap = cv2.VideoCapture(str(path)); i = 0
    while want:
        ok, f = cap.read()
        if not ok: break
        if i in want: out[i] = f; want.discard(i)
        i += 1
    cap.release(); return out


def vad(wav16: str) -> list:
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
    ts = get_speech_timestamps(read_audio(wav16, sampling_rate=16000), load_silero_vad(), sampling_rate=16000, return_seconds=True)
    return [(t["start"], t["end"]) for t in ts]


def mask(iv, n, pad):
    m = np.zeros(n, bool)
    for a, b in iv:
        m[max(0, int(np.floor((a - pad) * FPS))):min(n, int(np.ceil((b + pad) * FPS)))] = True
    return m


def main() -> int:
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--video", default=None); ap.add_argument("--pad", type=float, default=0.12)
    ap.add_argument("--sheet", default=None); ap.add_argument("--json", default=None)
    ap.add_argument("--scope", choices=["ls", "all"], default="ls", help="ls = LatentSync frames only; all = every frame (frames without a face are skipped): "
                    "needed for --silent-mouth freeze, whose frozen frames are outside LatentSync")
    a = ap.parse_args()
    import face_aware_latentsync as fa
    from face_aware_latentsync_accel import mouth_open
    man = json.loads(pathlib.Path(a.manifest).read_text()); vid = a.video or man["output"]; master = str(pathlib.Path(man["work_dir"]) / "master_25fps.mp4")
    n = man["lipsync"]["master_frames"]
    ls = np.zeros(n, bool)
    for s in man["lipsync"]["segments"]:
        if s["action"] == "LATENT_SYNC": ls[s["start_frame"]:min(n, s["end_frame"])] = True
    dub = mask(vad(man["dubbed_track"]["wav16k"]), n, a.pad); src = mask([(v["start"], v["end"]) for v in man["speech_intervals"]], n, 0.0)
    if a.scope == "all":
        ls = np.ones(n, bool)
    idx = [int(i) for i in np.flatnonzero(ls)]
    det = fa.face_detector()
    O = frames_at(vid, idx); Mst = frames_at(master, idx)
    oo = np.full(n, np.nan); om = np.full(n, np.nan); so = np.full((n, 20, 2), np.nan); sm = np.full((n, 20, 2), np.nan)
    for i in idx:
        for src_fr, arr, shp in ((O.get(i), oo, so), (Mst.get(i), om, sm)):
            if src_fr is None: continue
            bb, lm = det(cv2.cvtColor(src_fr, cv2.COLOR_BGR2RGB))
            if bb is not None:
                arr[i] = mouth_open(lm); lm = np.asarray(lm, np.float64)
                org = 0.5 * (lm[[43, 48, 49, 51, 50]].mean(0) + lm[101:106].mean(0)); unit = np.linalg.norm(lm[101:106].mean(0) - lm[[43, 48, 49, 51, 50]].mean(0))
                shp[i] = (lm[52:72] - org) / max(unit, 1e-6)    # the 20 mouth points in a face frame: lip SHAPE, not only the gap
    # pixel motion of the mouth in a head-stabilised crop: similarity transform from the MASTER brows + nose (same geometry for output and master,
    # so only the mouth content differs), 64x40 grey mouth crop, mean |frame - previous frame|
    BR = [43, 48, 49, 51, 50] + list(range(101, 106)) + [74, 77, 83, 86]
    ref_pts = None; po = np.full(n, np.nan); pm = np.full(n, np.nan); prev = {}
    for i in idx:
        if Mst.get(i) is None or O.get(i) is None: continue
        bb, lm = det(cv2.cvtColor(Mst[i], cv2.COLOR_BGR2RGB))
        if bb is None: prev = {}; continue
        lm = np.asarray(lm, np.float32)
        if ref_pts is None:
            ref_pts = lm[BR].copy(); mx = lm[52:72]; box = (mx[:, 0].min(), mx[:, 1].min(), mx[:, 0].max(), mx[:, 1].max())
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2; w = (box[2] - box[0]) * 1.5; h = w * 0.62
            crop = (int(cx - w / 2), int(cy - h / 2), int(w), int(h))
        M, _ = cv2.estimateAffinePartial2D(lm[BR], ref_pts)
        if M is None: prev = {}; continue
        cur = {}
        for key, fr in (("o", O[i]), ("m", Mst[i])):
            wimg = cv2.warpAffine(fr, M, (fr.shape[1], fr.shape[0]))
            x, y, w_, h_ = crop; c = cv2.cvtColor(wimg[max(0, y):y + h_, max(0, x):x + w_], cv2.COLOR_BGR2GRAY)
            cur[key] = cv2.resize(c, (64, 40)).astype(np.float32)
        if prev and i - prev["i"] == 1:
            po[i] = float(np.abs(cur["o"] - prev["o"]).mean()); pm[i] = float(np.abs(cur["m"] - prev["m"]).mean())
        prev = {**cur, "i": i}
    closed = float(np.nanpercentile(om[ls & ~src], 10)) if np.isfinite(om[ls & ~src]).any() else float(np.nanpercentile(om[ls], 5))
    thr = closed + 0.08
    def stats(sel, arr, shp, pix):
        v = arr[sel]; v = v[np.isfinite(v)]
        d = np.abs(np.diff(arr)); d = np.concatenate([[np.nan], d]); dv = d[sel]; dv = dv[np.isfinite(dv)]
        c = shp - shp.mean(1, keepdims=True)                                   # shape relative to the mouth centre (head / jaw translation removed)
        ds = np.concatenate([[np.nan], np.linalg.norm(np.diff(c, axis=0), axis=2).mean(1)]); dsv = ds[sel]; dsv = dsv[np.isfinite(dsv)]
        return {"frames": int(sel.sum()), "open_mean": round(float(v.mean()), 3) if len(v) else None, "open_p90": round(float(np.percentile(v, 90)), 3) if len(v) else None,
                "open_share": round(float((v > thr).mean()), 3) if len(v) else None, "motion": round(float(dv.mean()), 4) if len(dv) else None,
                "shape_motion": round(float(dsv.mean()), 4) if len(dsv) else None,
                "pixel_motion": round(float(np.nanmean(pix[sel])), 3) if np.isfinite(pix[sel]).any() else None}
    classes = {"dub_silent&src_speech": ls & ~dub & src, "dub_silent&src_silent": ls & ~dub & ~src, "dub_speech": ls & dub}
    res = {"closed_level": round(closed, 3), "open_threshold": round(thr, 3), "ls_frames": int(ls.sum())}
    for k, sel in classes.items():
        res[k] = {"output": stats(sel, oo, so, po), "master": stats(sel, om, sm, pm)}
    print(json.dumps(res, indent=1))
    if a.sheet:
        sel = np.flatnonzero(ls & ~dub & np.isfinite(oo)); worst = sel[np.argsort(-oo[sel])][:8] if len(sel) else []
        tiles = []
        for i in sorted(int(x) for x in worst):
            bb, _ = det(cv2.cvtColor(Mst[i], cv2.COLOR_BGR2RGB))
            if bb is None: continue
            x1, y1, x2, y2 = [int(v) for v in bb]; h = y2 - y1; y1 = max(0, y1 + h // 3); y2 = min(Mst[i].shape[0], y2 + h // 8)
            pair = [cv2.resize(fr[y1:y2, max(0, x1):x2], (220, 220)) for fr in (O[i], Mst[i])]
            t = np.concatenate(pair, 1); cv2.putText(t, f"{i / FPS:.2f}s out|src open {oo[i]:.2f}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1); tiles.append(t)
        if tiles:
            rows = [np.concatenate(tiles[k:k + 2] + [np.zeros_like(tiles[0])] * (2 - len(tiles[k:k + 2])), 1) for k in range(0, len(tiles), 2)]
            cv2.imwrite(a.sheet, np.concatenate(rows, 0))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
