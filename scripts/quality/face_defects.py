#!/usr/bin/env python3
"""Frame-by-frame face defect map of a dubbed output vs its 25 fps master (2026-10-14, human review of R10 04: "22-я секунда, что это?").

For every frame: the pipeline MODE that produced it (from the manifest + the dub VAD + the freeze runs) and three defect indicators, all
"output minus source" so that the source's own camera shake / blur cancels:
  contour   mean |out - src| in a band (+-6 % of the face width) around the SOURCE jaw / cheek contour (insightface 2d106 points 0-32 below
            the nose): a second / shifted contour = the "smeared jaw" and the freeze ghost
  outside   mean |out - src| in a ring OUTSIDE the source face contour (8-20 % of the face width away): pixels of another head or another
            background pasted next to the face (ghost)
  shake     |frame-to-frame motion of the output lower face - same for the source| (phase correlation, px): the lower face moving on its
            own (freeze misalignment, LatentSync crop jitter)
Modes: LS_SPEECH (LatentSync, dub speaks), LS_SILENT (LatentSync frame where the dub is silent: closed-mouth reference / gate margins),
FREEZE (mouth_freeze frames), ORIGINAL (source frame), NO_FACE. Detector on the CPU (CUDA_VISIBLE_DEVICES='' is set here).

    python scripts/quality/face_defects.py --manifest RUN/final/04_final.manifest.json --out DIR [--top 12]
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parents[1] / "third_party" / "latentsync"))
FPS = 25
MODES = ["ORIGINAL", "LS_SPEECH", "LS_SILENT", "FREEZE", "NO_FACE"]


def modes_of(man: dict, n: int) -> np.ndarray:
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
    dub = np.zeros(n, bool)
    if pathlib.Path(man["dubbed_track"]["wav16k"]).exists():
        iv = [(t["start"], t["end"]) for t in get_speech_timestamps(read_audio(man["dubbed_track"]["wav16k"], sampling_rate=16000), load_silero_vad(), sampling_rate=16000, return_seconds=True)]
    else:   # work dir pruned: the placed dub parts of the manifest
        iv = [tuple(p["placed"]) for u in man["units"] for p in u["alignment"].get("placements", []) if p.get("placed")]
    for s_, e_ in iv:
        dub[max(0, int((s_ - 0.12) * FPS)):min(n, int((e_ + 0.12) * FPS))] = True
    mode = np.zeros(n, np.int32)
    for s in man["lipsync"]["segments"]:
        if s["action"] == "LATENT_SYNC":
            a, b = s["start_frame"], min(n, s["end_frame"]); mode[a:b] = np.where(dub[a:b], 1, 2)
    fz = (man["lipsync"].get("mouth_freeze") or {}).get("keyframes") or []
    for run in fz:
        for c in run:
            if c.get("key") is not None:
                a, b = int(round(c["chunk"][0] * FPS)), int(round(c["chunk"][1] * FPS)); mode[a:b] = np.where(mode[a:b] == 0, 3, mode[a:b])
    return mode


def main() -> int:
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--video", default=None); ap.add_argument("--out", required=True); ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    import face_aware_latentsync as fa
    man = json.loads(pathlib.Path(a.manifest).read_text()); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    mp = pathlib.Path(man["work_dir"]) / "master_25fps.mp4"
    if not mp.exists():   # work dir pruned: rebuild the 25 fps master the same way the runner does
        tmp = out / f"master_{pathlib.Path(man['input']).stem}"; tmp.mkdir(exist_ok=True); mp, _ = fa.make_master(pathlib.Path(man["input"]), tmp)
    O = fa.read_frames(a.video or man["output"]); M = fa.read_frames(str(mp)); n = min(len(O), len(M))
    mode = modes_of(man, n); det = fa.face_detector()
    contour = np.full(n, np.nan); outside = np.full(n, np.nan); shake = np.full(n, np.nan); boxes = {}
    prev = None
    for i in range(n):
        bb, lm = det(cv2.cvtColor(M[i], cv2.COLOR_BGR2RGB))
        if bb is None:
            mode[i] = 4 if mode[i] == 0 else mode[i]; prev = None; continue
        lm = np.asarray(lm, np.float32); c = lm[:33]; nose = lm[[76, 82, 80], 1].mean(); fw = float(c[:, 0].max() - c[:, 0].min())
        hull = cv2.convexHull(c.astype(np.int32)); face = np.zeros(M[i].shape[:2], np.uint8); cv2.fillConvexPoly(face, hull, 1)
        low = np.zeros_like(face); low[int(nose):, :] = 1
        k = max(3, int(0.06 * fw)); band = (cv2.dilate(face, np.ones((k, k), np.uint8)) - cv2.erode(face, np.ones((k, k), np.uint8))) * low
        r1, r2 = max(3, int(0.08 * fw)), max(5, int(0.20 * fw)); ring = (cv2.dilate(face, np.ones((r2, r2), np.uint8)) - cv2.dilate(face, np.ones((r1, r1), np.uint8))) * low
        d = cv2.absdiff(O[i], M[i]).astype(np.float32).mean(2)
        contour[i] = float(d[band > 0].mean()) if band.any() else np.nan; outside[i] = float(d[ring > 0].mean()) if ring.any() else np.nan
        x1, x2 = int(c[:, 0].min()), int(c[:, 0].max()); y1, y2 = int(nose) + 4, int(c[:, 1].max()); u1 = int(lm[43:52, 1].min())
        boxes[i] = (x1, x2, u1, int(nose), y1, y2)
        if prev is not None and x2 - x1 > 16 and y2 - y1 > 16:
            px1, px2, pu1, pn, py1, py2 = prev
            def sh(F, ya, yb):
                A = cv2.cvtColor(F[i - 1][ya:yb, px1:px2], cv2.COLOR_BGR2GRAY).astype(np.float32); B = cv2.cvtColor(F[i][ya:yb, px1:px2], cv2.COLOR_BGR2GRAY).astype(np.float32)
                (dx, dy), _ = cv2.phaseCorrelate(A, B); return np.array([dx, dy])
            rel_o = sh(O, py1, py2) - sh(O, pu1, pn); rel_m = sh(M, py1, py2) - sh(M, pu1, pn)
            shake[i] = float(np.linalg.norm(rel_o - rel_m))
        prev = boxes[i]
    # per-mode summary
    summ = {}
    for k, name in enumerate(MODES):
        sel = mode == k
        summ[name] = {"frames": int(sel.sum()), "seconds": round(sel.sum() / FPS, 1),
                      "contour_mean": round(float(np.nanmean(contour[sel])), 2) if np.isfinite(contour[sel]).any() else None,
                      "contour_p90": round(float(np.nanpercentile(contour[sel], 90)), 2) if np.isfinite(contour[sel]).any() else None,
                      "outside_mean": round(float(np.nanmean(outside[sel])), 2) if np.isfinite(outside[sel]).any() else None,
                      "shake_mean": round(float(np.nanmean(shake[sel])), 2) if np.isfinite(shake[sel]).any() else None}
    # switches between modes (face frames only)
    sw = [(i, MODES[mode[i - 1]], MODES[mode[i]]) for i in range(1, n) if mode[i] != mode[i - 1] and mode[i] != 4 and mode[i - 1] != 4]
    res = {"frames": n, "modes": summ, "switches": len(sw), "switches_per_min": round(len(sw) / (n / FPS / 60), 1),
           "switch_list": [{"t": round(i / FPS, 2), "from": f, "to": t} for i, f, t in sw]}
    # worst moments: contour + outside, z-scored against the ORIGINAL frames (= codec noise level)
    base_c = np.nanmedian(contour[mode == 0]) if np.isfinite(contour[mode == 0]).any() else np.nanmedian(contour)
    score = np.nan_to_num(contour - base_c) + np.nan_to_num(outside - np.nanmedian(outside[mode == 0]) if np.isfinite(outside[mode == 0]).any() else 0)
    order = []
    for i in np.argsort(-score):
        if all(abs(i - j) > 12 for j in order): order.append(int(i))
        if len(order) >= a.top: break
    res["worst"] = [{"t": round(i / FPS, 2), "mode": MODES[mode[i]], "contour": round(float(contour[i]), 1), "outside": round(float(outside[i]), 1),
                     "shake": None if not np.isfinite(shake[i]) else round(float(shake[i]), 1)} for i in sorted(order)]
    stem = pathlib.Path(man["input"]).stem
    (out / f"{stem}_face_defects.json").write_text(json.dumps(res, indent=1))
    np.savez(out / f"{stem}_face_defects.npz", mode=mode, contour=contour, outside=outside, shake=shake)
    # timeline plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    tt = np.arange(n) / FPS; cols = {0: "#9aa0a6", 1: "#1a73e8", 2: "#e37400", 3: "#9334e6", 4: "#ffffff"}
    fig, ax = plt.subplots(3, 1, figsize=(14, 7), sharex=True)
    for axx, (y, name) in zip(ax, ((contour, "contour band |out-src|"), (outside, "outside ring |out-src|"), (shake, "lower-face shake px (out - src)"))):
        axx.plot(tt, y, color="#202124", lw=0.8); axx.set_ylabel(name, fontsize=8)
        for k in range(4):
            sel = mode == k; axx.fill_between(tt, 0, np.nanmax(y[np.isfinite(y)]) if np.isfinite(y).any() else 1, where=sel, color=cols[k], alpha=0.18, step="mid", label=MODES[k])
    ax[0].legend(loc="upper right", fontsize=7, ncol=4); ax[-1].set_xlabel("s"); fig.suptitle(f"{stem}: face defects by pipeline mode"); fig.tight_layout()
    fig.savefig(out / f"{stem}_face_defects.png", dpi=110)
    # contact sheet of the worst moments: source | output | |diff|
    tiles = []
    for i in sorted(order):
        if i not in boxes: continue
        x1, x2, u1, nb, y1, y2 = boxes[i]; w = x2 - x1; xa, xb = max(0, x1 - w // 3), min(M[i].shape[1], x2 + w // 3); ya, yb = max(0, u1 - w // 6), min(M[i].shape[0], y2 + w // 4)
        s_, o_ = M[i][ya:yb, xa:xb], O[i][ya:yb, xa:xb]; d_ = cv2.applyColorMap(np.clip(cv2.absdiff(o_, s_).max(2) * 4, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
        t3 = [cv2.resize(x, (200, int(200 * (yb - ya) / max(1, xb - xa)))) for x in (s_, o_, d_)]; h = min(x.shape[0] for x in t3); t3 = [x[:h] for x in t3]
        tile = np.concatenate(t3, 1); cv2.putText(tile, f"{i / FPS:.2f}s {MODES[mode[i]]}", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1); tiles.append(tile)
    if tiles:
        h = min(t.shape[0] for t in tiles); tiles = [t[:h] for t in tiles]
        rows = [np.concatenate(tiles[k:k + 2] + [np.zeros_like(tiles[0])] * (2 - len(tiles[k:k + 2])), 1) for k in range(0, len(tiles), 2)]
        cv2.imwrite(str(out / f"{stem}_worst.jpg"), np.concatenate(rows, 0))
    print(json.dumps({k: v for k, v in res.items() if k not in ("switch_list",)}, indent=1)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
