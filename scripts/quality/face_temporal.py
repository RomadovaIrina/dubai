#!/usr/bin/env python3
"""Temporal stability of the face in a dubbed output vs its 25 fps master ("the mouth / facial expression twitches", 2026-10-13 feedback).

SyncNet says nothing about how calm the face looks, so this measures the candidate causes separately:
  1. LS <-> original switches        from the manifest segments: count, per minute, segment lengths, short LS islands / short original gaps
  2. boundary discontinuity          image jump (mouth / upper face) at every switch frame and in a +-5 frame window, output vs master
  3. whole-face texture jump         upper-face Laplacian sharpness 5 frames before vs after each switch (CodeFormer / VAE re-render vs original)
  4. mouth jitter INSIDE LS segments landmark acceleration (2nd difference) of the lip points, face-normalised, output vs master
  5/6. temporal flicker              2nd temporal difference of the pixels (|f+1 - 2f + f-1|) in fixed mouth / upper crops, output vs master
  7. crop / landmark / mask jitter   acceleration of LatentSync's own insightface bbox (the mouth-only paste-back mask follows it) and of its
                                     3 affine points (eyebrow centres + nose, rounded to integers like LatentSync) on the MASTER frames
Landmarks = LatentSync's FaceDetector (insightface 2d106) on master and output frames; all positions are expressed in a face frame
(origin = master eye-brow midpoint, unit = master inter-brow distance) so head motion cancels and output/master are comparable.
Regions: ls_core (LS frames > 5 from a switch), switch (+-5 frames of a switch), orig (pass-through frames > 5 from a switch).

    python scripts/quality/face_temporal.py --manifest RUN/final/03_final.manifest.json --out RUN/face_temporal --tag reviewed [--sheets 6] [--clips 4]
"""
from __future__ import annotations
import argparse, json, math, pathlib, subprocess, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parents[1] / "third_party" / "latentsync"))
FPS = 25
BROW_L, BROW_R, NOSE = [43, 48, 49, 51, 50], list(range(101, 106)), [74, 77, 83, 86]   # LatentSync image_processor.affine_transform


def read_all(p) -> list:
    import cv2
    cap = cv2.VideoCapture(str(p)); out = []
    while True:
        ok, f = cap.read()
        if not ok: break
        out.append(f)
    cap.release(); return out


def detect(frames_bgr) -> list:
    import cv2
    import face_aware_latentsync as fa
    det = fa.face_detector(); out = []
    for f in frames_bgr:
        bb, lm = det(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        out.append((None, None) if bb is None else (np.asarray(bb, np.float32), np.asarray(lm, np.float32)))
    return out


def interp_nan(a: np.ndarray) -> np.ndarray:
    a = a.copy(); idx = np.arange(len(a))
    for c in range(a.shape[1]):
        m = np.isfinite(a[:, c])
        if m.sum() >= 2: a[~m, c] = np.interp(idx[~m], idx[m], a[m, c])
    return a


def accel(x: np.ndarray) -> np.ndarray:
    """|x[t+1] - 2 x[t] + x[t-1]| per frame (0 at the ends), x: (T, D) -> (T,) mean over D."""
    out = np.zeros(len(x))
    if len(x) >= 3: out[1:-1] = np.nanmean(np.abs(x[2:] - 2 * x[1:-1] + x[:-2]), axis=1)
    return out


def segments_of(man: dict) -> list:
    segs = sorted([s for s in man["lipsync"]["segments"] if s["action"] != "SPLIT"], key=lambda s: s["start_frame"])
    return [{"start": s["start_frame"], "end": s["end_frame"], "ls": s["action"] == "LATENT_SYNC", "action": s["action"]} for s in segs]


def switch_frames(segs: list) -> list:
    sw = []
    for a, b in zip(segs, segs[1:]):
        if a["ls"] != b["ls"]:
            sw.append({"frame": b["start"], "t": round(b["start"] / FPS, 3), "dir": "LS->orig" if a["ls"] else "orig->LS",
                       "prev_len_s": round((a["end"] - a["start"]) / FPS, 2), "next_len_s": round((b["end"] - b["start"]) / FPS, 2)})
    return sw


def crop(fr, box):
    x1, y1, x2, y2 = [int(round(v)) for v in box]; H, W = fr.shape[:2]
    return fr[max(0, y1):min(H, y2), max(0, x1):min(W, x2)]


def main() -> int:
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--video", default=None, help="default: manifest output")
    ap.add_argument("--master", default=None, help="default: <work_dir>/master_25fps.mp4"); ap.add_argument("--original", default=None, help="default: manifest input (for review clips)")
    ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--sheets", type=int, default=6); ap.add_argument("--clips", type=int, default=4); ap.add_argument("--win", type=int, default=5)
    a = ap.parse_args()
    man = json.loads(pathlib.Path(a.manifest).read_text()); vid = a.video or man["output"]
    master_p = a.master or str(pathlib.Path(man["work_dir"]) / "master_25fps.mp4")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); stem = f"{pathlib.Path(man['input']).stem}_{a.tag}"
    M = read_all(master_p); O = read_all(vid); n = min(len(M), len(O)); M, O = M[:n], O[:n]
    segs = segments_of(man); sw = switch_frames(segs); dur_min = n / FPS / 60
    ls = np.zeros(n, bool)
    for s in segs:
        if s["ls"]: ls[s["start"]:min(n, s["end"])] = True
    near = np.zeros(n, bool)
    for s in sw: near[max(0, s["frame"] - a.win):min(n, s["frame"] + a.win)] = True
    region = {"ls_core": ls & ~near, "switch": near, "orig": ~ls & ~near}

    # ---- landmarks (master + output), face frame from the MASTER
    dm, do = detect(M), detect(O)
    def lm_arr(d):
        return np.stack([lm if lm is not None else np.full((106, 2), np.nan, np.float32) for _, lm in d])
    def bb_arr(d):
        return np.stack([bb if bb is not None else np.full(4, np.nan, np.float32) for bb, _ in d])
    LM, LO, BM = lm_arr(dm), lm_arr(do), bb_arr(dm)
    bl, br, no = LM[:, BROW_L].mean(1), LM[:, BROW_R].mean(1), LM[:, NOSE].mean(1)
    origin = interp_nan(0.5 * (bl + br)); unit = interp_nan(np.linalg.norm(br - bl, axis=1)[:, None])[:, 0]
    unit_s = np.convolve(np.pad(unit, 12, mode="edge"), np.ones(25) / 25, mode="valid")              # 1 s smoothed face scale
    first = next(i for i in range(n) if np.isfinite(LM[i]).all())
    nose_y = LM[first, NOSE, 1].max(); mouth_idx = [k for k in range(33, 106) if LM[first, k, 1] > nose_y + 0.15 * unit[first] and k not in NOSE]
    def norm(L): return (L - origin[:, None, :]) / unit_s[:, None, None]
    NM, NO = norm(LM), norm(LO)
    mouth_m, mouth_o = NM[:, mouth_idx].reshape(n, -1), NO[:, mouth_idx].reshape(n, -1)
    upper_idx = BROW_L + BROW_R + list(range(33, 43)) + list(range(87, 101))                           # brows + eyes
    upper_m, upper_o = NM[:, upper_idx].reshape(n, -1), NO[:, upper_idx].reshape(n, -1)
    open_m = NM[:, mouth_idx, 1].max(1) - NM[:, mouth_idx, 1].min(1); open_o = NO[:, mouth_idx, 1].max(1) - NO[:, mouth_idx, 1].min(1)
    acc = {"mouth_lm_out": accel(mouth_o), "mouth_lm_master": accel(mouth_m), "upper_lm_out": accel(upper_o), "upper_lm_master": accel(upper_m),
           "open_out": accel(open_o[:, None]), "open_master": accel(open_m[:, None])}
    # LatentSync crop inputs on the master: bbox (mouth-only mask follows it) and the 3 rounded affine points, in pixels
    pts3 = np.concatenate([np.round(bl), np.round(br), np.round(no)], axis=1)
    acc["bbox_px"] = accel(BM); acc["affine_pts_px"] = accel(pts3)

    # ---- fixed crops (from the 1 s median-smoothed master bbox) -> pixel motion / flicker / sharpness
    bbs = interp_nan(BM); k = 12; bbm = np.stack([np.median(bbs[max(0, i - k):i + k + 1], axis=0) for i in range(n)])
    def regions_of(i):
        x1, y1, x2, y2 = bbm[i]; w, h = x2 - x1, y2 - y1
        return (x1 + 0.15 * w, y1 + 0.55 * h, x2 - 0.15 * w, y2 + 0.08 * h), (x1, y1, x2, y1 + 0.45 * h)
    gm = {"mouth": [], "upper": []}; go = {"mouth": [], "upper": []}
    for i in range(n):
        mb, ub = regions_of(i)
        for key, box in (("mouth", mb), ("upper", ub)):
            ref = crop(M[i], box); sz = (max(8, ref.shape[1]), max(8, ref.shape[0]))
            gm[key].append(cv2.resize(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32))
            go[key].append(cv2.resize(cv2.cvtColor(crop(O[i], box), cv2.COLOR_BGR2GRAY), (64, 64)).astype(np.float32))
    pix = {}
    for key in ("mouth", "upper"):
        A, B = np.stack(gm[key]), np.stack(go[key])
        pix[f"{key}_flicker_out"] = accel(B.reshape(n, -1)); pix[f"{key}_flicker_master"] = accel(A.reshape(n, -1))
        d1o = np.zeros(n); d1o[1:] = np.abs(B[1:] - B[:-1]).mean((1, 2)); d1m = np.zeros(n); d1m[1:] = np.abs(A[1:] - A[:-1]).mean((1, 2))
        pix[f"{key}_motion_out"] = d1o; pix[f"{key}_motion_master"] = d1m
    sharp_o = np.array([float(cv2.Laplacian(cv2.cvtColor(crop(O[i], regions_of(i)[1]), cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) for i in range(n)])
    sharp_m = np.array([float(cv2.Laplacian(cv2.cvtColor(crop(M[i], regions_of(i)[1]), cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()) for i in range(n)])

    def agg(sig_o, sig_m):
        r = {}
        for name, msk in region.items():
            mm = msk & np.isfinite(sig_o) & np.isfinite(sig_m) & (sig_m > 0)
            if mm.sum() < 3: r[name] = None; continue
            r[name] = {"out": round(float(np.mean(sig_o[mm])), 5), "master": round(float(np.mean(sig_m[mm])), 5), "ratio": round(float(np.mean(sig_o[mm]) / max(np.mean(sig_m[mm]), 1e-9)), 3),
                       "p95_ratio": round(float(np.percentile(sig_o[mm], 95) / max(np.percentile(sig_m[mm], 95), 1e-9)), 3)}
        return r
    summary = {"mouth_landmark_jitter": agg(acc["mouth_lm_out"], acc["mouth_lm_master"]), "mouth_open_jitter": agg(acc["open_out"], acc["open_master"]),
               "upper_landmark_jitter": agg(acc["upper_lm_out"], acc["upper_lm_master"]),
               "mouth_pixel_flicker": agg(pix["mouth_flicker_out"], pix["mouth_flicker_master"]), "upper_pixel_flicker": agg(pix["upper_flicker_out"], pix["upper_flicker_master"]),
               "mouth_pixel_motion": agg(pix["mouth_motion_out"], pix["mouth_motion_master"]), "upper_pixel_motion": agg(pix["upper_motion_out"], pix["upper_motion_master"]),
               "upper_sharpness": agg(sharp_o, sharp_m)}
    # master-side crop inputs (identical for every candidate with the same master): px acceleration inside LS frames
    summary["latentsync_crop_inputs_master"] = {"bbox_accel_px_ls_mean": round(float(np.nanmean(acc["bbox_px"][ls])), 3) if ls.any() else None,
                                                "affine_pts_accel_px_ls_mean": round(float(np.nanmean(acc["affine_pts_px"][ls])), 3) if ls.any() else None,
                                                "bbox_accel_px_ls_p95": round(float(np.nanpercentile(acc["bbox_px"][ls], 95)), 3) if ls.any() else None}
    # ---- switches
    W = a.win
    for s in sw:
        f = s["frame"]; lo, hi = max(1, f - W), min(n - 1, f + W)
        def rng(x): return float(np.mean(x[lo:hi]))
        s["mouth_jump_out"] = round(float(pix["mouth_motion_out"][f]), 2); s["mouth_jump_master"] = round(float(pix["mouth_motion_master"][f]), 2)
        s["upper_jump_out"] = round(float(pix["upper_motion_out"][f]), 2); s["upper_jump_master"] = round(float(pix["upper_motion_master"][f]), 2)
        s["mouth_flicker_win_ratio"] = round(rng(pix["mouth_flicker_out"]) / max(rng(pix["mouth_flicker_master"]), 1e-6), 2)
        s["upper_flicker_win_ratio"] = round(rng(pix["upper_flicker_out"]) / max(rng(pix["upper_flicker_master"]), 1e-6), 2)
        b4, af = sharp_o[max(0, f - W):f], sharp_o[f:min(n, f + W)]
        s["upper_sharp_before_after"] = [round(float(np.mean(b4)), 1), round(float(np.mean(af)), 1)] if len(b4) and len(af) else None
        s["upper_sharp_step_pct"] = round(100 * abs(np.mean(af) - np.mean(b4)) / max(np.mean(b4), 1e-6), 1) if len(b4) and len(af) else None
        mb4, maf = sharp_m[max(0, f - W):f], sharp_m[f:min(n, f + W)]   # same step on the master: natural motion / focus changes at that moment
        s["upper_sharp_step_master_pct"] = round(100 * abs(np.mean(maf) - np.mean(mb4)) / max(np.mean(mb4), 1e-6), 1) if len(mb4) and len(maf) else None
        s["mouth_lm_accel_win_ratio"] = round(float(np.nanmean(acc["mouth_lm_out"][lo:hi]) / max(np.nanmean(acc["mouth_lm_master"][lo:hi]), 1e-9)), 2)
        s["severity"] = round((s["mouth_jump_out"] / max(s["mouth_jump_master"], 1.0)) + (s["upper_jump_out"] / max(s["upper_jump_master"], 1.0)) + s["upper_sharp_step_pct"] / 10, 2)
    ls_lens = [round((x["end"] - x["start"]) / FPS, 2) for x in segs if x["ls"]]
    gaps = [round((x["end"] - x["start"]) / FPS, 2) for j, x in enumerate(segs) if not x["ls"] and 0 < j < len(segs) - 1 and segs[j - 1]["ls"] and segs[j + 1]["ls"]]
    summary["switches"] = {"count": len(sw), "per_min": round(len(sw) / dur_min, 2), "ls_segments": len(ls_lens), "ls_seconds": round(float(ls.sum()) / FPS, 2),
                           "ls_len_s": {"min": min(ls_lens) if ls_lens else None, "median": float(np.median(ls_lens)) if ls_lens else None},
                           "short_ls_lt_1s": sum(x < 1.0 for x in ls_lens), "short_ls_lt_2s": sum(x < 2.0 for x in ls_lens),
                           "orig_gaps_between_ls": len(gaps), "short_orig_gaps_lt_1_2s": sum(x < 1.2 for x in gaps), "short_orig_gaps_lt_2s": sum(x < 2.0 for x in gaps),
                           "mean_upper_sharp_step_pct": round(float(np.mean([s["upper_sharp_step_pct"] for s in sw if s["upper_sharp_step_pct"] is not None])), 1) if sw else None,
                           "mean_upper_sharp_step_master_pct": round(float(np.mean([s["upper_sharp_step_master_pct"] for s in sw if s.get("upper_sharp_step_master_pct") is not None])), 1) if sw else None,
                           "mean_mouth_jump_ratio": round(float(np.mean([s["mouth_jump_out"] / max(s["mouth_jump_master"], 1.0) for s in sw])), 2) if sw else None}
    # ---- worst 1 s windows of mouth jitter INSIDE LS (switch frames excluded)
    j_ratio = np.where(region["ls_core"], acc["mouth_lm_out"] / np.maximum(np.nanmedian(acc["mouth_lm_master"][ls]) if ls.any() else 1e-3, 1e-6), np.nan)
    win = []
    for i0 in range(0, n - 25, 12):
        v = j_ratio[i0:i0 + 25]
        if np.isfinite(v).sum() >= 15: win.append((float(np.nanmean(v)), i0))
    win.sort(reverse=True); worst_win = []
    for v, i0 in win:
        if all(abs(i0 - w["frame"]) > 25 for w in worst_win): worst_win.append({"frame": i0, "t": round(i0 / FPS, 2), "mouth_lm_jitter_vs_master_median": round(v, 2)})
        if len(worst_win) >= a.sheets: break
    summary["worst_ls_jitter_windows"] = worst_win
    summary["worst_switches"] = sorted(sw, key=lambda s: -s["severity"])[:a.sheets]
    # ---- per-second timeline (for plots)
    secs = int(n // FPS); tl = []
    for s_ in range(secs):
        sl = slice(s_ * FPS, (s_ + 1) * FPS)
        tl.append({"t": s_, "ls_frac": round(float(ls[sl].mean()), 2), "mouth_lm_acc_out": round(float(np.nanmean(acc["mouth_lm_out"][sl])), 5),
                   "mouth_lm_acc_master": round(float(np.nanmean(acc["mouth_lm_master"][sl])), 5), "upper_flicker_out": round(float(np.mean(pix["upper_flicker_out"][sl])), 3),
                   "upper_flicker_master": round(float(np.mean(pix["upper_flicker_master"][sl])), 3), "mouth_flicker_out": round(float(np.mean(pix["mouth_flicker_out"][sl])), 3),
                   "mouth_flicker_master": round(float(np.mean(pix["mouth_flicker_master"][sl])), 3)})
    # ---- contact sheets: rows = output / master, columns = frames f-W .. f+W, crop = smoothed face box (+ margin)
    def sheet(f0, path, title):
        tiles_o, tiles_m = [], []
        for i in range(max(0, f0 - W), min(n, f0 + W + 1)):
            x1, y1, x2, y2 = bbm[i]; w, h = x2 - x1, y2 - y1; box = (x1 - 0.1 * w, y1 - 0.25 * h, x2 + 0.1 * w, y2 + 0.15 * h)
            for lst, src in ((tiles_o, O), (tiles_m, M)):
                t = cv2.resize(crop(src[i], box), (150, 180)); tag = ("LS" if ls[i] else "or")
                cv2.putText(t, f"{i / FPS:.2f} {tag}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 255) if ls[i] else (255, 255, 255), 1, cv2.LINE_AA); lst.append(t)
        canvas = np.vstack([np.hstack(tiles_o), np.hstack(tiles_m)]); bar = np.zeros((22, canvas.shape[1], 3), np.uint8)
        cv2.putText(bar, title + "   (top: output, bottom: master)", (4, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(path), np.vstack([bar, canvas])); return str(path)
    sheets = []
    for s in summary["worst_switches"]:
        sheets.append(sheet(s["frame"], out / f"{stem}_switch_{s['t']:06.2f}.jpg", f"{stem} switch {s['dir']} t={s['t']}s sev={s['severity']}"))
    for w_ in worst_win:
        sheets.append(sheet(w_["frame"] + 12, out / f"{stem}_lsjitter_{w_['t']:06.2f}.jpg", f"{stem} in-LS jitter t={w_['t']}s x{w_['mouth_lm_jitter_vs_master_median']}"))
    # ---- review clips: original | output side by side, real speed and 0.5x, around the worst switches
    clips = []
    orig = a.original or man["input"]
    for s in summary["worst_switches"][:a.clips]:
        t0 = max(0.0, s["t"] - 1.5); c = out / f"{stem}_clip_{s['t']:06.2f}.mp4"
        flt = "[0:v]scale=-2:640,setsar=1[a];[1:v]scale=-2:640,setsar=1[b];[a][b]hstack=inputs=2[v]"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{t0:.2f}", "-t", "3", "-i", str(orig), "-ss", f"{t0:.2f}", "-t", "3", "-i", str(vid),
                        "-filter_complex", flt, "-map", "[v]", "-map", "1:a?", "-c:v", "libx264", "-crf", "18", "-c:a", "aac", str(c)], check=False)
        slow = c.with_name(c.stem + "_slow.mp4")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(c), "-filter:v", "setpts=2.0*PTS", "-an", "-c:v", "libx264", "-crf", "18", str(slow)], check=False)
        clips += [str(c), str(slow)]
    rep = {"video": vid, "master": master_p, "frames": n, "duration_s": round(n / FPS, 2), "mouth_landmark_indices": mouth_idx, "regions_frames": {k: int(v.sum()) for k, v in region.items()},
           "summary": summary, "switch_list": sw, "timeline": tl, "sheets": sheets, "clips": clips}
    (out / f"{stem}.face_temporal.json").write_text(json.dumps(rep, indent=1))
    s = summary
    def r(x, k="ls_core"): return None if not x or not x.get(k) else x[k]["ratio"]
    print(f"== {stem}: switches {s['switches']['count']} ({s['switches']['per_min']}/min), LS segs {s['switches']['ls_segments']} short<1s {s['switches']['short_ls_lt_1s']} <2s {s['switches']['short_ls_lt_2s']}, "
          f"short orig gaps<1.2s {s['switches']['short_orig_gaps_lt_1_2s']}")
    print(f"   out/master ratios [ls_core | switch]: mouth-lm jitter {r(s['mouth_landmark_jitter'])} | {r(s['mouth_landmark_jitter'], 'switch')}; open jitter {r(s['mouth_open_jitter'])}; "
          f"upper-lm jitter {r(s['upper_landmark_jitter'])} | {r(s['upper_landmark_jitter'], 'switch')}; mouth flicker {r(s['mouth_pixel_flicker'])} | {r(s['mouth_pixel_flicker'], 'switch')}; "
          f"upper flicker {r(s['upper_pixel_flicker'])} | {r(s['upper_pixel_flicker'], 'switch')}; upper sharp {r(s['upper_sharpness'])}")
    print(f"   switch: mean upper-sharp step {s['switches']['mean_upper_sharp_step_pct']} % (master {s['switches']['mean_upper_sharp_step_master_pct']} %), mean mouth jump ratio {s['switches']['mean_mouth_jump_ratio']}; crop inputs {s['latentsync_crop_inputs_master']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
