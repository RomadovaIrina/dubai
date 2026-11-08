#!/usr/bin/env python3
"""r16 K2: multi-shot support for the KeySync face stage (audit orch. 5 / paste 12 / prep 4: one job + one reference + the track box / lfm / align
smoothing and the landmark bridge all ran ACROSS scene cuts -> shot B got A's identity / box). ks_shots.sh cuts the video at hard cuts and runs
ks_face.sh once per shot, so every temporal filter is shot-local by construction; this file holds its helpers:
  detect   --video MASTER [--mp master_mp.npz] --out shots.json: shot bounds on the 25 fps timeline. (1) hard cuts: per frame a 64 x 36 HSV
           histogram (8 x 4 x 4) + the 32 x 18 luma; cut before frame k when the histogram distance d_k (half L1, 0..1) > --thr (0.35) AND
           d_k > --ratio (4) x the median d of the +-12 neighbours AND the luma structure changed (mean |diff| > --dy 0.10; q03's title animation
           step 0.067, q05's real cuts 0.23-0.29). (2) with --mp: a run of > --gap-s (2 s) with NO face between two face runs (fades / dissolves
           through a title card, q03) -> a bound in its middle. Bounds closer than --min-s (0.4 s) to the previous one / the end are dropped.
           A dissolve straight from face A to face B (no faceless gap) is NOT split. "segments": each shot cut to its face range (first -> last
           face frame; KeySync crashes on a clip that starts faceless), minus the fade / dissolve frames at those edges (up to --trans-max-s);
           the faceless or fading head / tail (>= --tail-s) / faceless shots = "source".
  slice    --mp NPZ --k0 --k1 --out: the MediaPipe npz cut to one shot (ks_face.sh KS_MP_NPZ: no second landmark pass).
  route    --route R --t0 --t1 --out R_shot: the route.json restricted to [t0, t1) and shifted to the shot's time (faces outside dropped).
  manifest --manifest M --t0 --t1 --out M_shot: the dub manifest's units the same way (the no-route fallback of ks_e2e_prep.py).
  encode   --video V --k0 --k1 --out O: frames [k0, k1) of V re-encoded EXACTLY like ks_plain's output (raw rgb pipe, ks_cms, x264 crf 16 medium),
           so the per-shot outputs concatenate with -c copy (no extra generation over the pasted shots).
  concat   --list a.mp4 b.mp4 ... --audio MIX --frames N --out OUT: concat -c:v copy + the final mix stream-COPIED (never re-encoded, no -shortest)."""
import argparse, json, os, subprocess, sys
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
FPS = 25.0
ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["detect", "slice", "route", "manifest", "encode", "concat"])
ap.add_argument("--mp", default=""); ap.add_argument("--gap-s", type=float, default=2.0, dest="gap_s"); ap.add_argument("--dy", type=float, default=0.10)
ap.add_argument("--tail-s", type=float, default=1.0, dest="tail_s"); ap.add_argument("--trans-thr", type=float, default=0.08, dest="trans_thr")
ap.add_argument("--trans-n", type=int, default=3, dest="trans_n"); ap.add_argument("--trans-max-s", type=float, default=2.0, dest="trans_max_s")
ap.add_argument("--video"); ap.add_argument("--out", required=True); ap.add_argument("--route"); ap.add_argument("--manifest")
ap.add_argument("--t0", type=float); ap.add_argument("--t1", type=float); ap.add_argument("--k0", type=int); ap.add_argument("--k1", type=int)
ap.add_argument("--list", nargs="+"); ap.add_argument("--audio"); ap.add_argument("--frames", type=int, default=0)
ap.add_argument("--dy-cut", type=float, default=0.15, dest="dy_cut"); ap.add_argument("--ratio-y", type=float, default=8.0, dest="ratio_y"); ap.add_argument("--thr2", type=float, default=0.15)
ap.add_argument("--thr", type=float, default=0.35); ap.add_argument("--ratio", type=float, default=4.0); ap.add_argument("--min-s", type=float, default=0.4, dest="min_s")
a = ap.parse_args()


def nframes(v):
    return int(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", v],
                              capture_output=True, text=True).stdout.strip() or 0)


if a.cmd == "detect":
    w, h = 64, 36
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", a.video, "-map", "0:v:0", "-an", "-vf", f"fps=25,scale={w}:{h}:flags=area", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         stdout=subprocess.PIPE)
    H, Y = [], []
    while True:
        r = p.stdout.read(w * h * 3)
        if len(r) < w * h * 3: break
        f = np.frombuffer(r, np.uint8).reshape(h, w, 3); hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).ravel(); H.append(hist / max(hist.sum(), 1))
        Y.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (32, 18), interpolation=cv2.INTER_AREA).astype(np.float32) / 255)
    n = len(H); H = np.array(H); d = np.zeros(n); dy = np.zeros(n)
    if n > 1: d[1:] = 0.5 * np.abs(H[1:] - H[:-1]).sum(1); dy[1:] = np.abs(np.array(Y[1:]) - np.array(Y[:-1])).mean((1, 2))
    cuts = []
    for k in range(1, n):
        nb = np.r_[d[max(1, k - 12):k], d[k + 1:k + 13]]; med = float(np.median(nb)) if len(nb) else 0.0
        nby = np.r_[dy[max(1, k - 12):k], dy[k + 1:k + 13]]; medy = float(np.median(nby)) if len(nby) else 0.0
        if d[k] > a.thr and d[k] > a.ratio * max(med, 0.01) and dy[k] > a.dy: cuts.append(("cut", k))
        # r16 T: a cut between two shots of the SAME room / palette (q02 40.9 s Vera -> the man: d 0.23 < thr, but the luma structure jumps 0.167):
        # a strong structural spike (dy > --dy-cut, > --ratio-y x the neighbours' median) with a clear histogram step (d > --thr2)
        elif dy[k] > a.dy_cut and dy[k] > a.ratio_y * max(medy, 0.005) and d[k] > a.thr2: cuts.append(("cut", k))
    # gradual transitions (fades / dissolves through a title card, q03) are not cut above; the face presence splits them: every run of > --gap-s
    # (2 s = ks_e2e_prep's bridge) without ANY face between two face runs -> a boundary in its middle (no face there: source pixels either way)
    gaps = []
    if a.mp:
        z = np.load(a.mp); LA = z["lm_all"] if "lm_all" in z.files else z["lm"][:, None]
        pres = np.isfinite(LA.reshape(len(LA), LA.shape[1], -1)).all(2).any(1)[:n]; i = 0; G = int(round(a.gap_s * FPS))
        while i < len(pres):
            j = i
            while j < len(pres) and pres[j] == pres[i]: j += 1
            if not pres[i] and i > 0 and j < len(pres) and j - i > G: gaps.append([i, j]); cuts.append(("gap", (i + j) // 2))
            i = j
    cuts.sort(key=lambda c: c[1]); mn = int(round(a.min_s * FPS)); b = [0]; why = []
    for w_, k in cuts:
        if k - b[-1] >= mn and n - k >= mn: b.append(k); why.append(w_)
    b.append(n); shots = [[b[i], b[i + 1]] for i in range(len(b) - 1)]
    # segments: per shot, KeySync runs only from its first to its last face frame (KeySync cannot START on a faceless frame: "No valid landmarks
    # found", q03 shot 2 opens on the title card); a faceless head (>= 1 frame) / tail (>= --tail-s 1 s) / whole shot = the source picture
    # + transition frames at a face range's edges stay source too (q03 29.5 s: the face dissolving in under a title -> KeySync's opaque, globally
    # colour-matched lower face pasted over a half-transparent frame = a bright patch erasing the title). Transition zone = >= --trans-n frames with
    # d > --trans-thr within +-5 frames, dilated 2 frames (static talking heads: <= 2 isolated frames > 0.08 per clip -> never a zone).
    B = (d > a.trans_thr).astype(int); Z = np.convolve(B, np.ones(11, int), "same") >= a.trans_n
    Z = np.convolve(Z.astype(int), np.ones(5, int), "same") > 0; TM = int(round(a.trans_max_s * FPS))
    segs = []; trans = []   # r16 T: the transition zones trimmed off a face range's edges (ks_dissolve check / unmix / assemble in ks_shots.sh)
    for s0, s1 in shots:
        if not a.mp: segs.append([s0, s1, "face"]); continue
        on = np.flatnonzero(pres[s0:s1])
        if len(on) == 0: segs.append([s0, s1, "source"]); continue
        f0, f1 = s0 + int(on[0]), s0 + int(on[-1]) + 1
        g0, g1 = f0, f1
        while f0 < min(f1, g0 + TM) and Z[f0]: f0 += 1
        while f1 > max(f0, g1 - TM) and Z[f1 - 1]: f1 -= 1
        if f1 - f0 <= 0: segs.append([s0, s1, "source"]); continue
        if s1 - f1 < int(round(a.tail_s * FPS)) and not Z[f1:s1].any(): f1 = s1
        if f0 > g0:   # face dissolving IN over [z0, f0): T = the frame before the zone, B = the first clean face frame
            z0 = f0
            while z0 > s0 and Z[z0 - 1]: z0 -= 1
            if z0 > 0: trans.append({"seg": [f0, f1], "edge": "in", "d0": int(z0), "d1": int(f0), "t_frame": int(z0 - 1), "b_frame": int(f0)})
        if f1 < g1:   # face dissolving OUT over [f1, z1): T = the frame after the zone, B = the last clean face frame
            z1 = f1
            while z1 < s1 and Z[z1]: z1 += 1
            if z1 < n: trans.append({"seg": [f0, f1], "edge": "out", "d0": int(f1), "d1": int(z1), "t_frame": int(z1), "b_frame": int(f1 - 1)})
        if f0 > s0: segs.append([s0, f0, "source"])
        segs.append([f0, f1, "face"])
        if f1 < s1: segs.append([f1, s1, "source"])
    out = {"video": a.video, "frames": n, "fps": FPS, "shots": shots, "segments": segs, "transitions": trans, "bounds": [[int(k), w_] for k, w_ in zip(b[1:-1], why)], "face_gaps": gaps,
           "cut_scores": {int(k): [round(float(d[k]), 3), round(float(dy[k]), 3)] for w_, k in cuts if w_ == "cut"},
           "params": {"thr": a.thr, "ratio": a.ratio, "dy": a.dy, "min_s": a.min_s, "gap_s": a.gap_s if a.mp else None}}
    json.dump(out, open(a.out, "w"), indent=1)
    print("KS_SHOTS " + json.dumps({"frames": n, "segments": [[s0, s1, round(s0 / FPS, 2), round(s1 / FPS, 2), kd] for s0, s1, kd in segs], "bounds": out["bounds"],
                                    "scores": out["cut_scores"]}), flush=True)

elif a.cmd == "slice":   # the per-frame arrays of a MediaPipe npz (mp_landmarks.py) cut to [k0, k1)
    z = np.load(a.mp); T = len(z["lm"])
    np.savez_compressed(a.out, **{k: (z[k][a.k0:a.k1] if z[k].ndim and len(z[k]) == T else z[k]) for k in z.files})

elif a.cmd in ("route", "manifest"):
    t0, t1 = a.t0, a.t1
    clip = lambda rs: [[round(max(s0, t0) - t0, 3), round(min(s1, t1) - t0, 3)] for s0, s1 in rs if min(s1, t1) > max(s0, t0)]
    if a.cmd == "route":
        R = json.load(open(a.route)); faces = []
        for f in R.get("faces", []):
            pr = clip([f.get("present_s", [0, 1e9])])
            if not pr: continue
            g = dict(f); g["present_s"] = pr[0]
            for k in ("speak_ranges", "silence_ranges"):
                if k in f: g[k] = clip(f[k])
            g["silence_s"] = round(sum(s1 - s0 for s0, s1 in g.get("silence_ranges", [])), 3); faces.append(g)
        R2 = dict(R); R2["faces"] = faces; R2["offscreen_speech_ranges"] = clip(R.get("offscreen_speech_ranges", []))
        R2["shot"] = {"t0": t0, "t1": t1}; json.dump(R2, open(a.out, "w"), indent=1)
        print(f"route shot {t0:.2f}-{t1:.2f}: {len(faces)} faces ({[(f.get('track'), f.get('mode')) for f in faces]})")
    else:
        M = json.load(open(a.manifest)); U = []
        for u in M.get("units") or []:
            s, e = float(u.get("start", 0)), float(u.get("end", 0))
            if min(e, t1) > max(s, t0): v = dict(u); v["start"] = round(max(s, t0) - t0, 3); v["end"] = round(min(e, t1) - t0, 3); U.append(v)
        M2 = dict(M); M2["units"] = U; json.dump(M2, open(a.out, "w"), indent=1); print(f"manifest shot {t0:.2f}-{t1:.2f}: {len(U)} units")

elif a.cmd == "encode":
    import ks_cms
    pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", a.video], capture_output=True, text=True).stdout.strip().split(",")
    W, Hh = int(pr[0]), int(pr[1])
    r = subprocess.Popen(["ffmpeg", "-v", "error", "-i", a.video] + ks_cms.dec(f"select=between(n\\,{a.k0}\\,{a.k1 - 1})", "rgb24") + ["-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE)
    w = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{Hh}", "-r", "25", "-i", "-",
                          "-c:v", "libx264", "-crf", "16", "-preset", "medium"] + ks_cms.enc() + [a.out], stdin=subprocess.PIPE)
    m = 0
    while True:
        b_ = r.stdout.read(W * Hh * 3)
        if len(b_) < W * Hh * 3: break
        w.stdin.write(b_); m += 1
    w.stdin.close(); w.wait(); print(f"encode {a.video} [{a.k0}, {a.k1}) -> {a.out}: {m} frames"); sys.exit(0 if m == a.k1 - a.k0 else 1)

elif a.cmd == "concat":
    lst = a.out + ".list.txt"; open(lst, "w").write("".join(f"file '{os.path.abspath(v)}'\n" for v in a.list))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-i", a.audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "copy",
                    "-movflags", "+faststart", a.out], check=True)
    got = nframes(a.out); parts = [nframes(v) for v in a.list]
    print(f"KS_SHOTS_CONCAT {a.out}: {got} frames from {parts}" + (f" (master {a.frames})" if a.frames else ""), flush=True)
    if a.frames and got != a.frames: print(f"KS WARN shots concat {got} frames != master {a.frames}", flush=True); sys.exit(1)
