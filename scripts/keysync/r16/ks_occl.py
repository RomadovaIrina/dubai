#!/usr/bin/env python3
"""r16c K2: occlusion guard + pose gate after the paste (Irina 09.11, q05 / q02: "если подносите хуйню к лицу, сильно закрывая — не дорисовываем";
a strong head turn -> the original; a square box corner must never show).
  out_k = S_k + E_k * (1 - w_k) * (P_k - S_k)
  S = the source master (25 fps), P = the pasted KeySync output (ks_face.sh), same frame grid.
  w_k in [0, 1] = "take the original" for frame k, from max of
     hand   MediaPipe Hands: the share of the MOUTH box (lips + 0.3 mouth width) under a hand hull; > --hand-on (0.25) -> gated.
            A hand over the face that does NOT cover the mouth (q02: a foreground hand at the chin) is not gated: its hull (dilated) is cut out
            of E, so the hand stays original and the lips stay lip-synced.
     occl   an object over the mouth (q05 cup): the share of lip-hull pixels whose a*/b* are within --lip-tol of the speaker's lip colour
            (median over clean frames) drops below --lip-on (0.5) x its usual value
     pose   |yaw| (nose tip vs the cheek extremes: (dL - dR) / (dL + dR)) -> clip((|yaw| - y0) / (y1 - y0)); by default ADAPTIVE:
            y0 = max(0.40, this speaker's p90 |yaw| + 0.08), y1 = y0 + 0.12 (a turn well beyond the habitual pose; q05 0.44, q02 0.66)
     absent no source landmarks -> 1
  then hysteresis (on > 0.5, off < 0.3), dilated --pad frames, a --ramp frame linear crossfade at both ends, runs < --min-run frames dropped.
  E_k = the feathered face-oval hull of the source landmarks (dilated --grow x face height, blurred): the paste can never reach the background, so
        no crop-box corner / window can show (q05 turns). With no landmarks E = 0 (the source).
Streaming, colour-managed (ks_cms, BT.709), x264 crf 16 like ks_plain; the audio stream COPIED from --audio (default: P).
    python ks_occl.py --master S.mp4 --paste P.mp4 --lm S_mp.npz --out OUT.mp4 [--dump scores.json] [--y0 0.28 --y1 0.40 ...]"""
import argparse, json, os, subprocess, sys
import numpy as np, cv2

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
ap = argparse.ArgumentParser(); ap.add_argument("--master", required=True); ap.add_argument("--paste", required=True); ap.add_argument("--lm", required=True)
ap.add_argument("--out", default=""); ap.add_argument("--audio", default=""); ap.add_argument("--dump", default="")
ap.add_argument("--y0", type=float, default=-1); ap.add_argument("--y1", type=float, default=-1)   # -1 = adaptive per speaker
ap.add_argument("--zon", type=float, default=4.0); ap.add_argument("--zspan", type=float, default=4.0)
ap.add_argument("--hand-on", type=float, default=0.25, dest="hand_on"); ap.add_argument("--lip-on", type=float, default=0.5, dest="lip_on"); ap.add_argument("--lip-tol", type=float, default=8.0, dest="lip_tol")
ap.add_argument("--pad", type=int, default=3); ap.add_argument("--ramp", type=int, default=4); ap.add_argument("--min-run", type=int, default=4, dest="min_run")
ap.add_argument("--grow", type=float, default=0.06); ap.add_argument("--no-hands", action="store_true", dest="no_hands"); a = ap.parse_args()
pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", a.master], capture_output=True, text=True).stdout.strip().split(",")
W, H = int(pr[0]), int(pr[1]); FB = W * H * 3
z = np.load(a.lm); LM = z["lm"][..., :2].astype(np.float32); N = len(LM)


def reader(path, cms=False):
    import ks_cms
    dec = ks_cms.dec(None, "rgb24") if cms else []
    p = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", path] + dec + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=FB * 2)
    while True:
        b = p.stdout.read(FB)
        if len(b) < FB: return
        yield np.frombuffer(b, np.uint8).reshape(H, W, 3)


# ---- pass 1: per-frame scores from the source
ok = np.isfinite(LM.reshape(N, -1)).all(1)
yaw = np.full(N, np.nan); hand = np.zeros(N); PATCH = {}
HANDS = None
if not a.no_hands:
    import mediapipe as mp
    HANDS = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.4, min_tracking_confidence=0.4)
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
HULLS = {}; LIPAB = {}   # per frame: the hand hulls (pass 2 keeps the hand pixels original); the lip pixels' a*, b*
for k, f in enumerate(reader(a.master)):
    if k >= N: break
    if not ok[k]: continue
    p = LM[k]; dl = np.linalg.norm(p[1] - p[234]); dr = np.linalg.norm(p[454] - p[1]); yaw[k] = (dl - dr) / max(dl + dr, 1)
    mw = float(np.linalg.norm(p[61] - p[291]))
    mx0, my0 = p[LIPS].min(0) - 0.3 * mw; mx1, my1 = p[LIPS].max(0) + 0.3 * mw   # the mouth box
    lm_ = cv2.fillPoly(np.zeros((H, W), np.uint8), [cv2.convexHull(p[LIPS].astype(np.float32))[:, 0].astype(np.int32)], 1)
    lab = cv2.cvtColor(f, cv2.COLOR_RGB2LAB)[lm_ > 0].astype(np.float32); LIPAB[k] = lab[:, 1:] - 128 if len(lab) else np.zeros((0, 2))
    if HANDS is not None:
        r = HANDS.process(np.ascontiguousarray(f))
        if r.multi_hand_landmarks:
            hs = [cv2.convexHull(np.float32([[l.x * W, l.y * H] for l in hl.landmark]))[:, 0].astype(np.int32) for hl in r.multi_hand_landmarks]
            HULLS[k] = hs
            hm = cv2.fillPoly(np.zeros((H, W), np.uint8), hs, 1)[max(0, int(my0)):int(my1), max(0, int(mx0)):int(mx1)]
            hand[k] = float(hm.mean()) if hm.size else 0.0   # the share of the MOUTH box under a hand
if a.y0 < 0:   # adaptive (default): a turn well beyond this speaker's habitual pose (q02 sits at |yaw| 0.5-0.6 and KeySync is fine there;
    ay = np.abs(yaw[np.isfinite(yaw)])   # q05 faces the camera, p90 0.36, and breaks at 0.52); the face-oval limit already stops box corners
    a.y0 = float(max(0.40, np.percentile(ay, 90) + 0.08)) if len(ay) else 0.40; a.y1 = a.y0 + 0.12
    print(f"KS_OCCL pose gate adaptive: |yaw| {a.y0:.2f} -> {a.y1:.2f} (speaker p90 {np.percentile(ay, 90) if len(ay) else 0:.2f})", flush=True)
pose = np.clip((np.abs(yaw) - a.y0) / max(a.y1 - a.y0, 1e-3), 0, 1); pose[~ok] = 0
# mouth covered by an object (cup): the share of lip-hull pixels whose a*/b* match the speaker's lip colour (median over clean frames) drops
ks = sorted(LIPAB); clean = [k for k in ks if hand[k] < 0.05 and pose[k] == 0 and len(LIPAB[k]) > 20]
REFAB = np.median(np.concatenate([LIPAB[k] for k in clean[:: max(1, len(clean) // 200)]]), 0) if clean else np.zeros(2)
match = np.full(N, np.nan)
for k in ks:
    if len(LIPAB[k]) > 20: match[k] = float((np.linalg.norm(LIPAB[k] - REFAB, axis=1) < a.lip_tol).mean())
mref = float(np.nanmedian(match[clean])) if clean else 1.0
occl = np.clip((a.lip_on * mref - np.nan_to_num(match, nan=mref)) / max(a.lip_on * mref * 0.5, 1e-3), 0, 1)   # 0 at lip_on x the usual match, 1 at half of that
handg = np.clip((hand - a.hand_on) / a.hand_on, 0, 1)   # 0 below hand_on of the mouth box covered, 1 at 2 x
s = np.maximum.reduce([handg, occl, pose]); s[~ok] = 1.0
hand = handg; mad = match; med, sp = mref, 1.0
# ---- hysteresis, min run, dilation, crossfade ramp
on = np.zeros(N, bool); st = False
for k in range(N):
    st = s[k] > 0.5 if not st else s[k] > 0.3; on[k] = st
i = 0
while i < N:
    if not on[i]: i += 1; continue
    j = i
    while j < N and on[j]: j += 1
    if j - i < a.min_run and ok[i:j].all(): on[i:j] = False
    i = j
g = on.copy()
for d in range(1, a.pad + 1): g[d:] |= on[:-d]; g[:-d] |= on[d:]
w = g.astype(np.float32)
for d in range(1, a.ramp + 1):   # linear ramps outside each gated run (no hard switch)
    lv = 1.0 - d / (a.ramp + 1)
    w[d:] = np.maximum(w[d:], g[:-d] * lv); w[:-d] = np.maximum(w[:-d], g[d:] * lv)
runs = []; i = 0
while i < N:
    if g[i]:
        j = i
        while j < N and g[j]: j += 1
        why = max((("hand", hand[i:j].max()), ("occl", occl[i:j].max()), ("pose", pose[i:j].max()), ("absent", float((~ok[i:j]).any()))), key=lambda x: x[1])[0]
        runs.append([i, j, round(i / 25, 2), round(j / 25, 2), why]); i = j
    else: i += 1
info = {"frames": N, "gated_frames": int(g.sum()), "runs": runs, "params": {k: getattr(a, k) for k in ("y0", "y1", "zon", "zspan", "pad", "ramp", "min_run", "grow")}}
print("KS_OCCL " + json.dumps({k: info[k] for k in ("frames", "gated_frames", "runs")}), flush=True)
if a.dump:
    json.dump(info | {"yaw": np.round(np.nan_to_num(yaw), 3).tolist(), "hand": hand.tolist(), "occl": np.round(occl, 3).tolist(), "occl_z": np.round(np.nan_to_num((mad - med) / sp), 2).tolist(),
                      "pose": np.round(pose, 3).tolist(), "w": np.round(w, 3).tolist()}, open(a.dump, "w"))
if not a.out: sys.exit()
# ---- pass 2: composite (streaming)
import ks_cms
tmp = a.out + ".v.mp4"
wr = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "25", "-i", "-",
                       "-c:v", "libx264", "-crf", "16", "-preset", "medium"] + ks_cms.enc() + [tmp], stdin=subprocess.PIPE)
P = reader(a.paste, True); n = 0
for k, S in enumerate(reader(a.master, True)):
    Pk = next(P, None)
    if Pk is None: break
    if k < N and ok[k] and w[k] < 0.999:
        p = LM[k]; fh = float(np.linalg.norm(p[10] - p[152]))
        hull = cv2.convexHull(p[OVAL].astype(np.float32))[:, 0].astype(np.int32)
        E = cv2.fillPoly(np.zeros((H, W), np.uint8), [hull], 255)
        gr = max(1, int(a.grow * fh)); E = cv2.dilate(E, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * gr + 1, 2 * gr + 1)))
        if k in HULLS:   # a hand over the face (not covering the mouth, else the frame is gated): its pixels stay original
            hm = cv2.dilate(cv2.fillPoly(np.zeros((H, W), np.uint8), HULLS[k], 255), np.ones((max(3, int(0.04 * fh)),) * 2, np.uint8))
            E = cv2.subtract(E, hm)
        E = cv2.GaussianBlur(E.astype(np.float32) / 255, (0, 0), max(1.0, 0.03 * fh))[..., None]
        o = S.astype(np.float32) + E * (1 - w[k]) * (Pk.astype(np.float32) - S.astype(np.float32))
        out = np.clip(o + 0.5, 0, 255).astype(np.uint8)
    else: out = S
    wr.stdin.write(np.ascontiguousarray(out).tobytes()); n += 1
wr.stdin.close(); wr.wait()
subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", tmp, "-i", a.audio or a.paste, "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "copy", "-movflags", "+faststart", a.out], check=True)
os.remove(tmp); print(f"KS_OCCL wrote {a.out}: {n} frames, {int(g.sum())} gated", flush=True)
