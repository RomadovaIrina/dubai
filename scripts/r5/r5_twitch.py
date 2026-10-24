#!/usr/bin/env python3
"""Round 5 blocker #0: whole-video FACE TWITCH detector (user + Max: "the face still twitches, and that moment simply kills it").

A twitch = the pasted lower face moves / changes in one or two frames in a way the SOURCE face does not. Measured on the OUTPUT at the
source frame rate against the source video, every frame, no manifest needed (the manifest only annotates the events):

  region      the paste area: face oval below the nose base (source landmarks of the shown frame, r3_source_lm.py npz), dilated by 0.12 iod,
              split into  skin = region minus the lips hull dilated 0.22 mouth width (jaw / chin / cheeks / mask ring)  and  lips.
  motion      G_k = F_out(k-1 -> k) - F_src(k-1 -> k)  (DIS flow, same ROI in both videos): the output's motion relative to the source's own
              motion. Where the output IS the source (w = 0, outside the mask) G = 0 up to coding noise. skin_motion_k = mean |G_k| on skin.
  appearance  D_k = lowpass(out_k - src_k) (sigma 0.06 iod), the paste's low-frequency residual; skin_app_k = mean |D_k - D_{k-1}| on skin
              (a crossfade / closed-mouth <-> lip-sync switch / window joint that changes the jaw's shading or blends two jaw positions).
  lips snap   lips_k = mean |G_k| on the lips hull (mouth jumps beyond what the source mouth does; articulation itself is in it too, so it
              only counts through its spike over the local level).
  units       motion px are scaled to iod = 100 px (so a small face is not under-rated); appearance in grey levels.
  per frame   each channel minus its rolling median (+-8 frames) = the SPIKE over the local level, divided by the channel's twitch scale
              (SCALE below: a spike of 1.0 = clearly visible on a 2-s half-speed review); score_k = max over channels.
  events      local maxima of score_k >= --thr (default 1.0), merged within 6 frames; each with time, channel values, face iod, head speed
              (source rigid points, iod / s) and, with --manifest, the nearest LatentSync segment start / end / crossfade / 16-frame window
              joint in source frames -> "cause".
  summary     twitch_sum (sum of event scores), n_events, top20_mean, p99 of score_k: the gate rows (lower = better).

    python scripts/r5/r5_twitch.py score --video OUT.mp4 --src SRC.mp4 --lm <id>_source_lm_F.npz [--manifest MAN] --out x.twitch.json
    python scripts/r5/r5_twitch.py sheet --twitch x.twitch.json [--other y.twitch.json --other-label v4] --top 20 --outdir DIR
        -> DIR/<stem>_top20.jpg (contact sheet: per event out k-2..k+2, src below) + DIR/<stem>_top20_halfspeed.mp4 (2 s clips, src | out, 0.5x)"""
from __future__ import annotations
import argparse, json, math, pathlib, subprocess, sys
import numpy as np
import cv2

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "quality")); sys.path.insert(0, str(HERE.parent / "pilot"))
LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
             132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
NOSE_BASE = [2, 98, 327]; RIGID = [33, 133, 362, 263, 168, 6, 197, 4]
# twitch scales (spike over the local level that reads as a visible jump at half speed): skin motion px @ iod 100, appearance grey levels,
# lips motion px @ iod 100. Calibrated on m3 / v4 events reviewed on contact sheets (r5 F, see reports/experiments/r5-face/README.md).
SCALE = {"skin_motion": 0.6, "skin_app": 2.0, "lips": 1.6}
P = 16


def decode(path):
    import av
    c = av.open(str(path)); s = c.streams.video[0]
    for fr in c.decode(s): yield fr.to_ndarray(format="bgr24")
    c.close()


def rolling_median(x, r):
    y = np.full_like(x, np.nan)
    for i in range(len(x)):
        w = x[max(0, i - r):i + r + 1]; w = w[np.isfinite(w)]
        if len(w): y[i] = np.median(w)
    return y


def masks(lm, H, W):
    """(roi, skin mask, lips mask, iod) for source landmarks lm of the shown frame, or None."""
    if not np.isfinite(lm[FACE_OVAL + LIPS_OUTER + NOSE_BASE + [33, 263]]).all(): return None
    p = lm[:, :2]; iod = float(np.linalg.norm(p[33] - p[263]))
    if iod < 8: return None
    ny = float(p[NOSE_BASE, 1].mean()); ov = p[FACE_OVAL]; low = ov[ov[:, 1] >= ny - 1]
    hull = cv2.convexHull(np.concatenate([low, [[ov[:, 0].min(), ny], [ov[:, 0].max(), ny]]]).astype(np.float32))[:, 0]
    pad = int(0.12 * iod) + 4; x0, y0 = np.floor(hull.min(0)).astype(int) - pad; x1, y1 = np.ceil(hull.max(0)).astype(int) + pad
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    if x1 - x0 < 16 or y1 - y0 < 16: return None
    reg = np.zeros((y1 - y0, x1 - x0), np.uint8); cv2.fillPoly(reg, [np.round(hull - [x0, y0]).astype(np.int32)], 1)
    reg = cv2.dilate(reg, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.12 * iod) + 1,) * 2))
    mw = float(np.linalg.norm(p[61] - p[291])); lips = np.zeros_like(reg)
    cv2.fillPoly(lips, [np.round(cv2.convexHull(p[LIPS_OUTER].astype(np.float32))[:, 0] - [x0, y0]).astype(np.int32)], 1)
    lips_d = cv2.dilate(lips, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(0.22 * mw) + 1,) * 2))
    return (y0, y1, x0, x1), (reg > 0) & (lips_d == 0), lips > 0, iod


def cmd_score(a):
    z = np.load(a.lm); lm = z["lm"].astype(np.float64); t = z["t"].astype(np.float64); n = len(lm)
    cv2.setNumThreads(4); dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    ch = {k: np.full(n, np.nan) for k in ("skin_motion", "skin_app", "lips")}; iods = np.full(n, np.nan); touched = np.zeros(n, bool)
    po = ps = None; Dprev = None; k = -1
    for k, (fo, fs) in enumerate(zip(decode(a.video), decode(a.src))):
        if k >= n: break
        H, W = fs.shape[:2]; mk = masks(lm[k], H, W)
        if mk is None or po is None:
            po, ps, Dprev = fo, fs, None; continue
        (y0, y1, x0, x1), skin, lips, iod = mk; iods[k] = iod; sc = 100.0 / iod
        go = lambda f: cv2.cvtColor(f[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        diff = np.abs(fo[y0:y1, x0:x1].astype(np.int16) - fs[y0:y1, x0:x1].astype(np.int16)).max(2)
        touched[k] = bool((diff[skin | lips] > 6).mean() > 0.02) if (skin | lips).any() else False
        if not touched[k] and not (np.abs(po[y0:y1, x0:x1].astype(np.int16) - ps[y0:y1, x0:x1].astype(np.int16)).max(2) > 6).mean() > 0.02:
            for c_ in ch: ch[c_][k] = 0.0   # output = source here and one frame before: nothing pasted, no twitch possible
            po, ps, Dprev = fo, fs, None; continue
        Fo = dis.calc(go(po), go(fo), None); Fs = dis.calc(go(ps), go(fs), None); G = np.linalg.norm(Fo - Fs, axis=2) * sc
        if skin.sum() > 30: ch["skin_motion"][k] = float(G[skin].mean())
        if lips.sum() > 10: ch["lips"][k] = float(G[lips].mean())
        sg = 0.06 * iod; D = cv2.GaussianBlur(cv2.cvtColor(fo[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY).astype(np.float32) - go(fs).astype(np.float32), (0, 0), sg)
        if Dprev is not None and Dprev[1] == (y0, y1, x0, x1) and skin.sum() > 30:
            yy, xx = np.mgrid[0:y1 - y0, 0:x1 - x0].astype(np.float32)
            Fb = dis.calc(go(fs), go(ps), None)   # k -> k-1: pull D_{k-1} onto frame k's geometry
            Dw = cv2.remap(Dprev[0], xx + Fb[..., 0], yy + Fb[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            ch["skin_app"][k] = float(np.abs(D - Dw)[skin].mean())
        elif skin.sum() > 30:
            ch["skin_app"][k] = 0.0 if Dprev is None and not touched[k] else np.nan
        Dprev = (D, (y0, y1, x0, x1)); po, ps = fo, fs
        if a.max_frames and k >= a.max_frames: break
    nf = k + 1
    spike = {}
    for c_, x in ch.items():
        x = x[:nf]; base = rolling_median(x, 8); spike[c_] = np.clip(x - base, 0, None) / SCALE[c_]
    score = np.nanmax(np.stack([np.nan_to_num(spike[c_], nan=0.0) for c_ in spike]), 0)
    # head speed (source rigid points), iod / s
    rp = lm[:nf][:, RIGID, :2]; ctr = np.nanmean(rp, 1); dt = np.diff(t[:nf], prepend=np.nan)
    head = np.linalg.norm(np.diff(ctr, axis=0, prepend=np.nan), axis=1) / np.maximum(iods[:nf], 1) / np.where(dt > 0, dt, np.nan)
    # events: local maxima >= thr, merged within 6 frames
    ev = []; order = np.argsort(-score)
    used = np.zeros(nf, bool)
    for i in order:
        if score[i] < a.thr: break
        if used[max(0, i - 6):i + 7].any(): continue
        used[i] = True; ev.append(int(i))
    ann = annotate(a.manifest, t[:nf]) if a.manifest else None
    events = []
    for i in ev:
        e = {"frame": i, "t": round(float(t[i]), 3), "score": round(float(score[i]), 3), "channel": max(spike, key=lambda c_: np.nan_to_num(spike[c_][i])),
             **{c_: (round(float(ch[c_][i]), 3) if np.isfinite(ch[c_][i]) else None) for c_ in ch}, "iod": round(float(iods[i]), 1) if np.isfinite(iods[i]) else None,
             "head_speed_iod_s": round(float(np.nanmax(head[max(0, i - 2):i + 3])), 2) if np.isfinite(head[max(0, i - 2):i + 3]).any() else None}
        if ann is not None: e.update(ann(i))
        events.append(e)
    sc_ = np.sort([e["score"] for e in events])[::-1]
    summ = {"twitch_sum": round(float(sc_.sum()), 2), "n_events": len(events), "top20_mean": round(float(sc_[:20].mean()), 3) if len(sc_) else 0.0,
            "p99": round(float(np.percentile(score, 99)), 3), "frames": nf, "touched_frames": int(touched[:nf].sum()),
            "n_events_ge2": int((sc_ >= 2).sum()), "by_cause": {}}
    for e in events: summ["by_cause"][e.get("cause", "?")] = round(summ["by_cause"].get(e.get("cause", "?"), 0) + e["score"], 2)
    out = {"video": str(a.video), "src": str(a.src), "thr": a.thr, "scale": SCALE, "summary": summ, "events": events,
           "series": {"t": [round(float(x), 3) for x in t[:nf]], "score": [round(float(x), 3) for x in score], **{c_: [None if not np.isfinite(v) else round(float(v), 3) for v in ch[c_][:nf]] for c_ in ch}}}
    pathlib.Path(a.out).write_text(json.dumps(out))
    print(json.dumps(summ))
    return 0


def annotate(manifest, t):
    """-> f(i): nearest LS segment start / end / window joint (in source frames at time t) and a cause label."""
    man = json.loads(pathlib.Path(manifest).read_text()); fps_m = float((man.get("lipsync") or {}).get("master_fps") or 25)
    segs = [s for s in man["lipsync"]["segments"] if s["action"] == "LATENT_SYNC"]
    ts = lambda m: m / fps_m
    marks = []
    for s in segs:
        xf = s.get("crossfade") or {}
        marks.append(("seg_start", ts(s["start_frame"]), xf.get("in") or 0)); marks.append(("seg_end", ts(s["end_frame"]), xf.get("out") or 0))
        for b in range(s["start_frame"] + P, s["end_frame"], P): marks.append(("window_joint", ts(b), 0))
    def inside(tt): return any(ts(s["start_frame"]) <= tt < ts(s["end_frame"]) for s in segs)
    def f(i):
        tt = float(t[i]); best = min(marks, key=lambda m: abs(m[1] - tt)) if marks else None
        d = {"in_ls": inside(tt)}
        if best is not None:
            dd = tt - best[1]; d.update({"nearest": best[0], "nearest_dt_s": round(dd, 3)})
            win = (best[2] + 2) / fps_m if best[0] != "window_joint" else 1.5 / fps_m
            d["cause"] = best[0] if abs(dd) <= win + 0.5 / 30 else ("inside_ls" if d["in_ls"] else "outside_ls")
        return d
    return f


def crop_box(lm, H, W):
    p = lm[:, :2]; iod = float(np.linalg.norm(p[33] - p[263])); c = p[[61, 291, 2, 152]].mean(0)
    s = 1.6 * iod; x0, y0 = int(max(0, c[0] - s)), int(max(0, c[1] - 1.1 * s)); x1, y1 = int(min(W, c[0] + s)), int(min(H, c[1] + 0.9 * s))
    return x0, y0, x1, y1


def cmd_sheet(a):
    import av
    d = json.loads(pathlib.Path(a.twitch).read_text()); ev = sorted(d["events"], key=lambda e: -e["score"])[:a.top]
    out = pathlib.Path(a.outdir); out.mkdir(parents=True, exist_ok=True); stem = a.stem or pathlib.Path(a.twitch).name.split(".")[0]
    if not ev: print("no events"); return 0
    z = np.load(a.lm) if a.lm else None
    want = {}
    for e in ev:
        for j in range(e["frame"] - 2, e["frame"] + 3): want[j] = None
    fo, fs = {}, {}
    for path, store in ((d["video"], fo), (d["src"], fs)):
        for k, f in enumerate(decode(path)):
            if k in want: store[k] = f
            if k > max(want): break
    rows = []; T = 112
    for r_, e in enumerate(ev):
        k = e["frame"]; f0 = fs[k]; H, W = f0.shape[:2]
        if z is not None and np.isfinite(z["lm"][k][[33, 263, 61, 291, 2, 152]]).all(): x0, y0, x1, y1 = crop_box(z["lm"][k].astype(np.float64), H, W)
        else: x0, y0, x1, y1 = 0, 0, W, H
        sc = T / max(y1 - y0, 1); tw = int(round((x1 - x0) * sc))
        top = [cv2.resize(fo[j][y0:y1, x0:x1], (tw, T)) if j in fo else np.zeros((T, tw, 3), np.uint8) for j in range(k - 2, k + 3)]
        bot = [cv2.resize(fs[j][y0:y1, x0:x1], (tw, T)) if j in fs else np.zeros((T, tw, 3), np.uint8) for j in range(k - 2, k + 3)]
        row = np.concatenate([np.concatenate(top, 1), np.concatenate(bot, 1)], 0)
        lab = np.full((2 * T, 170, 3), 255, np.uint8)
        for li, s in enumerate([f"#{r_ + 1}  t={e['t']:.2f}s", f"score {e['score']:.2f} {e['channel']}", f"cause {e.get('cause', '-')}",
                                f"dt {e.get('nearest_dt_s', '-')}", f"head {e.get('head_speed_iod_s')}", "top OUT / bottom SRC"]):
            cv2.putText(lab, s, (4, 18 + 30 * li), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)
        rows.append(np.concatenate([lab, row], 1))
    wmax = max(r.shape[1] for r in rows); rows = [np.pad(r, ((0, 4), (0, wmax - r.shape[1]), (0, 0)), constant_values=255) for r in rows]
    sheet = out / f"{stem}_top{a.top}.jpg"; cv2.imwrite(str(sheet), np.concatenate(rows, 0), [cv2.IMWRITE_JPEG_QUALITY, 88])
    if a.clips:   # 2-s clips at half speed, src | out, upscaled lower-face crop of the whole clip, labelled, concatenated
        tmp = out / f".{stem}_clips"; tmp.mkdir(exist_ok=True); lst = []
        for r_, e in enumerate(ev):
            ss = max(0.0, e["t"] - 1.0); c_ = tmp / f"{r_:02d}.mp4"
            k = e["frame"]; H, W = fs[k].shape[:2]
            x0, y0, x1, y1 = crop_box(z["lm"][k].astype(np.float64), H, W) if z is not None and np.isfinite(z["lm"][k][[33, 263, 61, 291, 2, 152]]).all() else (0, 0, W, H)
            cw, chh = (x1 - x0) // 2 * 2, (y1 - y0) // 2 * 2; sc = 2 if chh < 240 else 1
            txt = f"#{r_ + 1} t={e['t']:.2f}s score {e['score']:.1f} {e.get('cause', '')}".replace(":", " ")
            fc = (f"[0:v]trim={ss}:{ss + 2},setpts=2*(PTS-STARTPTS),crop={cw}:{chh}:{x0}:{y0},scale=iw*{sc}:ih*{sc}[s];"
                  f"[1:v]trim={ss}:{ss + 2},setpts=2*(PTS-STARTPTS),crop={cw}:{chh}:{x0}:{y0},scale=iw*{sc}:ih*{sc}[o];"
                  f"[s][o]hstack,pad=iw:ih+28:0:28:white,drawtext=text='{txt}   SRC | OUT  0.5x':x=6:y=6:fontsize=16:fontcolor=black,"
                  f"scale=trunc(iw/2)*2:trunc(ih/2)*2,fps=30[v]")
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", d["src"], "-i", d["video"], "-filter_complex", fc, "-map", "[v]", "-an",
                            "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", str(c_)], check=False)
            if c_.exists(): lst.append(c_)
        if lst:
            # different crop sizes -> pad every clip to the largest
            dims = [tuple(map(int, subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(c)],
                                                  capture_output=True, text=True).stdout.strip().split(","))) for c in lst]
            mw_, mh_ = max(x for x, _ in dims), max(y for _, y in dims); ins = []; fcs = []
            for i, c in enumerate(lst): ins += ["-i", str(c)]; fcs.append(f"[{i}:v]pad={mw_}:{mh_}:0:0:white,setsar=1[v{i}]")
            fcs.append("".join(f"[v{i}]" for i in range(len(lst))) + f"concat=n={len(lst)}:v=1:a=0[v]")
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *ins, "-filter_complex", ";".join(fcs), "-map", "[v]", "-c:v", "libx264", "-crf", "20",
                            "-pix_fmt", "yuv420p", str(out / f"{stem}_top{a.top}_halfspeed.mp4")], check=False)
        for c in tmp.glob("*.mp4"): c.unlink()
        tmp.rmdir()
    print(sheet)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("score"); s.add_argument("--video", required=True); s.add_argument("--src", required=True); s.add_argument("--lm", required=True)
    s.add_argument("--manifest", default=None); s.add_argument("--out", required=True); s.add_argument("--thr", type=float, default=1.0); s.add_argument("--max-frames", type=int, default=0)
    h = sp.add_parser("sheet"); h.add_argument("--twitch", required=True); h.add_argument("--lm", default=None); h.add_argument("--top", type=int, default=20)
    h.add_argument("--outdir", required=True); h.add_argument("--stem", default=None); h.add_argument("--clips", action="store_true")
    a = ap.parse_args(); return cmd_score(a) if a.cmd == "score" else cmd_sheet(a)


if __name__ == "__main__":
    raise SystemExit(main())
