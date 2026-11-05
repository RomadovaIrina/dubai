#!/usr/bin/env python3
"""r8 F: per-frame ARTEFACT SCORE of the LatentSync layer and the CONFIDENCE FALLBACK to the source mouth (spec 5.5 spirit: "no lip-sync for a
moment" beats a mouth explosion; management on final_r7 01 / 03 starts: "mouth explosions").

All features are measured on the 25 fps master timeline, layer vs the SOURCE master at the same frame, so head / camera motion that the
source also has does not count:
  jerk     |o[t] - (o[t-1] + o[t+1]) / 2| of the layer's inner opening (13-14 / IOD) minus the same of the source (the generated mouth
           snapping open / shut between frames)
  tone     inner-mouth luma HF residual (r8_mouth_flicker model: luma not explained by the opening, 2nd difference) of the layer minus the source
  app      mouth-box appearance jump: mean |crop[t] - (crop[t-1] + crop[t+1]) / 2| (grey, box from frame t's landmarks, same box in the
           neighbours), layer minus 1.5 x source (cavity / teeth blobs, colour pops, lip-shape collapse)
  coll / wash  lip-box contrast loss / brightening vs the source at the same frame (the washed pink blob), transient part
  dark     near-black cavity: the layer's 5th-percentile mouth luma vs the source's typical open-mouth cavity over the video
  geo      rigid-landmark disagreement layer vs source (eyes / nose, IOD): the generated face does not sit where the source face is
  joint    the app / jerk excess at a 16-frame window joint counts x --joint-gain (window-joint jumps)
score = max over the features, each scaled so 1 = a clear artefact (--thr-*). fallback weight fb (0..1): score -> clip, max-filter
+-(--dilate) frames, then a smooth ramp (Gaussian --ramp-sigma) = crossfade to the source; a 16-frame window with more than --win-frac of
its frames flagged falls back as a whole. fps_composite --fallback FILE multiplies its master weights by (1 - fb).
    python scripts/r8/r8_fallback.py --work WORK_DIR --ls LAYER.mp4 --lm LAYER.npz --out FB.npz [--wav dubbed_16k.wav]
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, cv2

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from r8_mouth_flicker import LIPS_INNER, LIPS_OUTER
RIG = [33, 133, 362, 263, 168, 6, 197, 4]


def hf(x):
    o = np.full(len(x), np.nan); o[1:-1] = np.abs(x[1:-1] - 0.5 * (x[:-2] + x[2:])); return o


def read_feats(video, lm, n):
    """per frame: opening, inner luma, area, grey mouth crops (box from that frame, 48 x 48) of frames t-1, t, t+1 in frame t's box."""
    c = cv2.VideoCapture(str(video)); frames = []
    for k in range(n):
        ok, f = c.read()
        if not ok: break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    c.release(); n = len(frames)
    op = np.full(n, np.nan); yin = np.full(n, np.nan); ar = np.full(n, np.nan); app = np.full(n, np.nan); boxes = [None] * n
    for k in range(n):
        p = lm[k]
        if not np.isfinite(p[LIPS_INNER + LIPS_OUTER + [33, 263]]).all(): continue
        iod = float(np.linalg.norm(p[33, :2] - p[263, :2]))
        if iod < 5: continue
        op[k] = np.linalg.norm(p[13, :2] - p[14, :2]) / iod; q = p[LIPS_INNER, :2]; ar[k] = cv2.contourArea(q.astype(np.float32)) / iod ** 2
        m = np.zeros(frames[k].shape, np.uint8); cv2.fillPoly(m, [np.round(q).astype(np.int32)], 1); m = cv2.erode(m, np.ones((3, 3), np.uint8))
        if m.sum() >= 6: yin[k] = float(frames[k][m > 0].mean())
        cx, cy = p[LIPS_OUTER, 0].mean(), p[LIPS_OUTER, 1].mean(); r = 0.55 * iod
        boxes[k] = (int(cy - r), int(cy + r), int(cx - r), int(cx + r))
    def crop(k, b):
        y0, y1, x0, x1 = b; H, W = frames[k].shape
        if y0 < 0 or x0 < 0 or y1 > H or x1 > W or y1 - y0 < 8: return None
        return cv2.resize(frames[k][y0:y1, x0:x1], (48, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
    for k in range(1, n - 1):
        if boxes[k] is None: continue
        cs = [crop(j, boxes[k]) for j in (k - 1, k, k + 1)]
        if any(c_ is None for c_ in cs): continue
        app[k] = float(np.abs(cs[1] - 0.5 * (cs[0] + cs[2])).mean())
    return op, yin, ar, app, n


def tone_hf(yin, op, ar, open_min=0.03):
    v = np.isfinite(yin) & np.isfinite(op) & (op >= open_min); r = np.full(len(yin), np.nan)
    if v.sum() > 30:
        X = np.stack([np.ones(v.sum()), op[v], ar[v]], 1); coef, *_ = np.linalg.lstsq(X, yin[v], rcond=None); r[v] = yin[v] - X @ coef
    return hf(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="work dir with video/master_25fps.mp4 + master_mediapipe.npz and the 25fps manifest")
    ap.add_argument("--ls", required=True); ap.add_argument("--lm", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--thr-jerk", type=float, default=0.05); ap.add_argument("--thr-tone", type=float, default=14.0); ap.add_argument("--thr-app", type=float, default=9.0)
    ap.add_argument("--thr-geo", type=float, default=0.12); ap.add_argument("--thr-coll", type=float, default=0.45)
    ap.add_argument("--rel-stats", choices=["off", "on"], default="off", help="r11 FACE: wash / coll / dark relative to the LAYER's own statistics (its steady offset vs the source, its own typical cavity), not absolute: for generator variants with another tone (a tighter generation mask)")
    ap.add_argument("--thr-wash", type=float, default=0.12); ap.add_argument("--thr-dark", type=float, default=0.6); ap.add_argument("--knee", type=float, default=0.5, help="a feature counts from knee x thr (0) to thr (1)")
    ap.add_argument("--joint-gain", type=float, default=1.3); ap.add_argument("--dilate", type=int, default=2); ap.add_argument("--ramp-sigma", type=float, default=2.0); ap.add_argument("--merge-gap", type=int, default=6)
    ap.add_argument("--win-frac", type=float, default=0.4); ap.add_argument("--window", type=int, default=16)
    ap.add_argument("--wash-region", choices=["mouth", "lips"], default="mouth", help="r11 SYNC (m18 / m03: 20 / 25 'wash' flags, nearly all a CLOSED "
                    "generated mouth over an OPEN source mouth: the source's dark cavity makes the layer look 'brighter'; each flag swapped the source's "
                    "open mouth in = a sync hit): mouth = luma of the source-lip box (r8); lips = the lip RING only (outer minus inner contour, each "
                    "frame's own landmarks: layer lips on the layer, source lips on the source), so cavity / teeth never count")
    ap.add_argument("--dark-transient", choices=["off", "on"], default="off", help="r11 SYNC: the black-hole score counts only its transient part "
                    "(minus the +-6-frame median over the layer): a held open vowel with a dark cavity is a mouth, a near-black pop is an artefact")
    ap.add_argument("--open-gate", type=float, default=0.0, help="r11 SYNC: wash / coll flags need the layer and source inner openings (IOD) to differ "
                    "by less than this (a different opening legitimately changes the mouth box luma / contrast); 0 = off")
    ap.add_argument("--og-short", type=int, default=0, help="r11 SYNC, with --open-gate: layer closures (opening < 0.04 IOD) of <= this many frames "
                    "between open frames (> 0.07) keep the r8 wash / coll test (gate-safe: the flash detector counts them); 0 = off")
    ap.add_argument("--thr-wash-lips", type=float, default=0.0, help="r11 SYNC, with --open-gate: where the openings differ, the lip-ring brightness "
                    "transient (layer ring vs source ring, minus its +-6-frame median) >= this flags a washed mouth (+ its neighbours above half); 0 = off")
    ap.add_argument("--thr-lipcon", type=float, default=0.0, help="r15 F2 (m03 'lips vanish', agent T): the lip-vs-skin a* / lightness contrast "
                    "(scripts/r15/lipcon.py, layer as a fraction of the source at the same frame) below this AND >= --lipcon-drop under its +-6-frame median "
                    "flags the frame (1 at thr - 0.1, 0.5 at thr), with --lipcon-same on only where the layer and source openings match (a different opening "
                    "legitimately changes the lip line, old03): a pale / vanished lip pop, re-drawn by the layer repair or faded to the source; 0 = off")
    ap.add_argument("--joint-seg", choices=["off", "on"], default="on", help="r15 F3 (audit): window joints / whole-window fallback per segment (off = the r8 stepping from each segment start to the END OF THE VIDEO, mixing the phases of later segments)")
    ap.add_argument("--lipcon-drop", type=float, default=0.1); ap.add_argument("--lipcon-same", choices=["off", "on"], default="off", help="r15 F2: lipcon only where the openings match (--open-gate)")
    ap.add_argument("--lipcon-mode", choices=["drop", "adaptive"], default="drop", help="r15 F2: drop = below thr AND --lipcon-drop under the +-6-frame "
                    "median; adaptive = below thr AND below --lipcon-rel x the video's own +-(--lipcon-win)-frame median ratio (a transient dip of this video)")
    ap.add_argument("--lipcon-rel", type=float, default=0.6); ap.add_argument("--lipcon-win", type=int, default=25)
    ap.add_argument("--lipcon-max-frac", type=float, default=0.0, help="r15 F2: at most this fraction of the LatentSync frames flagged by lipcon (the deepest "
                    "dips first; a per-video budget so a steadily pale video cannot be re-drawn wholesale); 0 = no cap")
    ap.add_argument("--lipcon-only", choices=["off", "on"], default="off", help="r15 F2: score lipcon alone (the second pass on the repaired layer)")
    a = ap.parse_args(); work = pathlib.Path(a.work)
    if a.thr_lipcon > 0:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "r15")); import lipcon
    lml = np.load(a.lm)["lm"]; lms = np.load(work / "video/master_mediapipe.npz")["lm"]; n = min(len(lml), len(lms))
    ol, yl, al_, apl, n1 = read_feats(a.ls, lml, n); os_, ys, as_, aps, n2 = read_feats(work / "video/master_25fps.mp4", lms, n); n = min(n1, n2)
    man = json.loads(next(work.glob("*_25fps.manifest.json")).read_text())
    ls_mask = np.zeros(n, bool); starts = []; seg_end = {}
    for s in man["lipsync"]["segments"]:
        if s["action"] == "LATENT_SYNC":
            ls_mask[s["start_frame"]:min(n, s["end_frame"])] = True; starts.append(s["start_frame"]); seg_end[s["start_frame"]] = min(n, s["end_frame"])
    jerk = np.nan_to_num(hf(ol[:n]) - hf(os_[:n]), nan=0.0)
    tone = np.nan_to_num(tone_hf(yl[:n], ol[:n], al_[:n]) - np.nan_to_num(tone_hf(ys[:n], os_[:n], as_[:n]), nan=0.0), nan=0.0)
    appx = np.nan_to_num(apl[:n] - 1.5 * np.nan_to_num(aps[:n], nan=0.0), nan=0.0)
    coll = np.zeros(n); wash = np.zeros(n); washl = np.zeros(n); sp5 = np.full(n, np.nan); lp5 = np.full(n, np.nan); lcl = np.full((n, 2), np.nan); lcs = np.full((n, 2), np.nan)
    cl = cv2.VideoCapture(str(a.ls)); cs = cv2.VideoCapture(str(work / "video/master_25fps.mp4"))
    for k in range(n):
        okl, fl = cl.read(); oks, fs = cs.read()
        if not (okl and oks): break
        q = lms[k]
        if a.thr_lipcon > 0: lcl[k] = lipcon.measure(fl, lml[k]); lcs[k] = lipcon.measure(fs, q)
        if not np.isfinite(q[LIPS_OUTER + LIPS_INNER + [33, 263]]).all(): continue
        iod = float(np.linalg.norm(q[33, :2] - q[263, :2])); m = np.zeros(fs.shape[:2], np.uint8)
        cv2.fillPoly(m, [np.round(cv2.convexHull(q[LIPS_OUTER, :2].astype(np.float32))[:, 0]).astype(np.int32)], 1)
        r_ = max(1, int(round(0.12 * iod))); m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r_ + 1,) * 2)) > 0
        if m.sum() < 20: continue
        gl = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY).astype(np.float32)[m]; gs = cv2.cvtColor(fs, cv2.COLOR_BGR2GRAY).astype(np.float32)[m]
        coll[k] = 1 - gl.std() / max(gs.std(), 1.0)                       # lip / cavity contrast lost vs the source (washed blob)
        wash[k] = (gl.mean() - gs.mean()) / max(gs.mean(), 1.0)           # the mouth region brighter than the source (pale / pink pop)
        if a.wash_region == "lips" or a.thr_wash_lips > 0:   # r11 SYNC: lip ring vs lip ring (each frame's own contours), cavity / teeth excluded
            def ring(p, shape):
                if not np.isfinite(p[LIPS_OUTER + LIPS_INNER]).all(): return None
                r = np.zeros(shape, np.uint8); cv2.fillPoly(r, [np.round(p[LIPS_OUTER, :2]).astype(np.int32)], 1); cv2.fillPoly(r, [np.round(p[LIPS_INNER, :2]).astype(np.int32)], 0)
                r = cv2.erode(r, np.ones((3, 3), np.uint8)); return r > 0 if r.sum() >= 15 else None
            rl, rs = ring(lml[k], fs.shape[:2]), ring(q, fs.shape[:2])
            if rl is not None and rs is not None:
                ll = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY).astype(np.float32)[rl]; ls_ = cv2.cvtColor(fs, cv2.COLOR_BGR2GRAY).astype(np.float32)[rs]
                washl[k] = (ll.mean() - ls_.mean()) / max(ls_.mean(), 1.0)
            if a.wash_region == "lips": wash[k] = washl[k]
        sp5[k], lp5[k] = np.percentile(gs, 5), np.percentile(gl, 5)
    cl.release(); cs.release()
    # black hole: the layer's darkest mouth pixels vs the source's TYPICAL open-mouth cavity over the video (not the same frame: a dub-open mouth over a
    # closed source mouth is a legitimate dark cavity; a near-black slot darker than this person's cavity ever is, is not)
    so_ = np.isfinite(os_[:n]) & (os_[:n] > 0.06) & np.isfinite(sp5)
    ref_cav = float(np.median(sp5[so_])) if so_.sum() > 10 else float(np.nanpercentile(sp5, 10))
    if a.rel_stats == "on":   # r11 FACE: a generator that renders a consistently darker cavity (a tighter mask: LS_MASK=mask3.png) is not a black hole;
        lt_ = np.isfinite(lp5) & (ls_mask[:n] > 0) if len(ls_mask) >= n else np.isfinite(lp5)   # only the darker of the source's cavity and the layer's OWN typical one counts
        if lt_.sum() > 10: ref_cav = min(ref_cav, float(np.nanpercentile(lp5[lt_], 10)))
    dark = np.where(np.isfinite(lp5), (ref_cav - lp5) / (ref_cav + 10.0), 0.0)
    geo = np.zeros(n)
    for k in range(n):
        p, q = lml[k], lms[k]
        if np.isfinite(p[RIG]).all() and np.isfinite(q[RIG]).all():
            iod = float(np.linalg.norm(q[33, :2] - q[263, :2])); geo[k] = float(np.median(np.linalg.norm(p[RIG, :2] - q[RIG, :2], axis=1)) / max(iod, 1))
    # window joints: frames within 1 of a 16-frame boundary of a LatentSync segment
    joint = np.zeros(n, bool)
    E_ = (lambda s0: seg_end[s0]) if a.joint_seg == "on" else (lambda s0: n)   # r15 F3 (audit): each segment's OWN 16-frame phase, up to its end
    for s0 in starts:
        for b in range(s0, E_(s0), a.window):
            joint[max(0, b - 1):min(n, b + 1)] = True
    sc = lambda x, t: np.clip((x - a.knee * t) / ((1 - a.knee) * t), 0, 1)
    tmed = lambda x: np.array([np.median(x[max(0, k - 6):k + 7]) for k in range(len(x))])   # the transient part (a pop), not a steady offset
    F = {"jerk": sc(jerk, a.thr_jerk), "tone": sc(tone, a.thr_tone), "app": sc(appx, a.thr_app), "geo": sc(geo, a.thr_geo),
         "coll": sc(coll - np.minimum(tmed(coll), 0.15 if a.rel_stats == "off" else 1e9), a.thr_coll),
         "wash": sc(wash - np.minimum(tmed(wash), 0.05 if (a.rel_stats == "off" and a.wash_region == "mouth") else 1e9), a.thr_wash), "dark": sc(dark, a.thr_dark)}   # r11 --rel-stats: the whole steady offset removed
    if a.dark_transient == "on":   # r11 SYNC: the pop, not a held dark cavity
        dk = np.where(ls_mask[:n], dark, np.nan); dmed = np.array([np.nanmedian(dk[max(0, k - 6):k + 7]) if np.isfinite(dk[max(0, k - 6):k + 7]).any() else 0.0 for k in range(n)])
        F["dark"] = sc(dark - np.maximum(dmed, 0.0), a.thr_dark)
    if a.open_gate > 0:   # r11 SYNC: wash / contrast compared only between mouths of a similar opening
        same = np.nan_to_num(np.abs(ol[:n] - os_[:n]), nan=1.0) < a.open_gate
        if a.og_short > 0:   # r11 SYNC: a SHORT closure of the layer between open frames over an open source keeps the r8 wash test (the gate's
            # flash detector counts these 2-3-frame closures; m03 92-94 / 361-362 / 517-518 / 528-529)
            o_ = np.nan_to_num(ol[:n], nan=1.0); cl_ = o_ < 0.04; k = 0
            while k < n:
                if not cl_[k]: k += 1; continue
                e = k
                while e < n and cl_[e]: e += 1
                if e - k <= a.og_short and o_[max(0, k - 2):k].max(initial=0) > 0.07 and o_[e:e + 2].max(initial=0) > 0.07: same[k:e] = True
                k = e
        for k_ in ("wash", "coll"): F[k_] = F[k_] * same
        if a.thr_wash_lips > 0:   # r11 SYNC: where the openings differ, a washed mouth is judged on the lip ring alone (transient part), with
            # hysteresis: the neighbours of a flagged frame count from half the threshold (m18 410-412: 0.18 / 0.36 / 0.16 = one pale 3-frame pop)
            wt = washl - tmed(washl); fl_ = sc(wt, a.thr_wash_lips); hi_ = fl_ > 0.5
            nb_ = np.zeros(n, bool); nb_[1:] |= hi_[:-1]; nb_[:-1] |= hi_[1:]
            F["washl"] = np.where(nb_ & (wt > 0.5 * a.thr_wash_lips), 1.0, fl_) * ~same
    if a.thr_lipcon > 0:   # r15 F2: the lips lose their colour / lightness contrast to the skin (transient), whatever the opening
        lc = lipcon.contrast(lcl, lcs); lcm = np.array([np.nanmedian(lc[max(0, k - 6):k + 7]) if np.isfinite(lc[max(0, k - 6):k + 7]).any() else np.nan for k in range(n)])
        if a.lipcon_mode == "adaptive":
            R_ = a.lipcon_win; lcw = np.array([np.nanmedian(lc[max(0, k - R_):k + R_ + 1]) if np.isfinite(lc[max(0, k - R_):k + R_ + 1]).any() else np.nan for k in range(n)])
            F["lipcon"] = np.nan_to_num(np.clip((a.thr_lipcon + 0.1 - lc) / 0.2, 0, 1) * (lc < a.lipcon_rel * lcw), nan=0.0)
        else: F["lipcon"] = np.nan_to_num(np.clip((a.thr_lipcon + 0.1 - lc) / 0.2, 0, 1) * (lc < np.minimum(lcm, 1.0) - a.lipcon_drop), nan=0.0)
        if a.lipcon_same == "on" and a.open_gate > 0: F["lipcon"] = F["lipcon"] * (np.nan_to_num(np.abs(ol[:n] - os_[:n]), nan=1.0) < a.open_gate)
    if a.thr_lipcon > 0 and a.lipcon_max_frac > 0:   # r15 F2: per-video budget, the deepest dips kept
        cap = int(a.lipcon_max_frac * max(1, ls_mask[:n].sum())); on_ = np.flatnonzero((F["lipcon"] > 0.5) & ls_mask[:n])
        if len(on_) > cap: F["lipcon"][on_[np.argsort(np.nan_to_num(lc[on_], nan=9))[cap:]]] = 0.0
    if a.thr_lipcon > 0 and a.lipcon_only == "on": F = {"lipcon": F["lipcon"]}
    for k_ in ("jerk", "app"):
        if k_ not in F: continue
        F[k_] = np.where(joint, np.clip(F[k_] * a.joint_gain, 0, 1), F[k_])
    score = np.max(np.stack(list(F.values()), 1), 1) * ls_mask
    fb = score.copy()
    flag = score > 0.5; k = 0   # gaps of <= --merge-gap frames between flagged frames are filled: one held fallback, no blinking in and out
    idx = np.flatnonzero(flag)
    for i0, i1 in zip(idx[:-1], idx[1:]):
        if 1 < i1 - i0 <= a.merge_gap + 1 and ls_mask[i0:i1].all(): fb[i0:i1] = np.maximum(fb[i0:i1], 1.0)
    if a.dilate > 0: fb = np.array([fb[max(0, k - a.dilate):k + a.dilate + 1].max() for k in range(n)])
    # whole-window fallback where a window is mostly bad
    for s0 in starts:
        for b in range(s0, E_(s0), a.window):
            seg = slice(b, min(E_(s0), b + a.window))
            if ls_mask[seg].any() and (score[seg] > 0.5).mean() > a.win_frac: fb[seg] = 1.0
    if a.ramp_sigma > 0:
        R_ = int(3 * a.ramp_sigma); k_ = np.exp(-0.5 * (np.arange(-R_, R_ + 1) / a.ramp_sigma) ** 2); k_ /= k_.sum(); fb = np.convolve(np.pad(fb, R_, mode="edge"), k_, "same")[R_:-R_]
        fb = np.maximum(fb, np.clip(score, 0, 1))   # the flagged frame itself is never softened
    fb = np.clip(fb, 0, 1) * ls_mask
    nls = max(1, ls_mask.sum()); rep = {"frames": int(n), "ls_frames": int(ls_mask.sum()), "fallback_frames_gt05": int((fb > 0.5).sum()), "fallback_pct_gt05": round(100 * float((fb > 0.5).sum()) / nls, 2),
           "fallback_weight_pct": round(100 * float(fb.sum()) / nls, 2), "by_feature_gt05": {k_: int((v_ > 0.5).sum()) for k_, v_ in F.items()},
           "top_s": [[round(int(k) / 25, 2), round(float(score[k]), 2), max(F, key=lambda f_: F[f_][k])] for k in sorted(np.argsort(-score)[:15])], "args": vars(a)}
    np.savez(a.out, fb=fb.astype(np.float32), score=score.astype(np.float32), ls_mask=ls_mask, **{f"f_{k_}": v_.astype(np.float32) for k_, v_ in F.items()},
             raw_jerk=jerk.astype(np.float32), raw_tone=tone.astype(np.float32), raw_app=appx.astype(np.float32), raw_geo=geo.astype(np.float32), raw_coll=coll.astype(np.float32), raw_wash=wash.astype(np.float32), raw_dark=dark.astype(np.float32))
    pathlib.Path(a.out).with_suffix(".json").write_text(json.dumps(rep, indent=1))
    print("== fallback", json.dumps({k_: v_ for k_, v_ in rep.items() if k_ not in ("args", "top_s")}), flush=True)


if __name__ == "__main__":
    main()
