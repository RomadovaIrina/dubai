#!/usr/bin/env python3
"""r19 (Irina 14.11, old03 end: "слишком резкий переход лица": the drawn face with an open mouth cut to the source face with a closed one in 0.25 s).
The tail hand-over (drawn face -> source face after the face's last strong dub speech) as per-master-frame weights for ks_plain --gate-w, with a choice of
WHEN and HOW FAST, decided on the whole video (ks_plain's own --tail-src decides per paste chunk and only knows a linear ramp at a fixed time):
  --mode plain     the ramp starts at t0 = the end of the last strong speech (+ --hold); = ks_plain --tail-src
  --mode similar   the ramp starts at the first frame >= t0, within --search s, where the DRAWN and the SOURCE mouths are alike: |openness_drawn -
                   openness_source| <= --thr for --hold-frames frames in a row (openness = inner-lip gap / mouth width, MediaPipe 13-14 / 61-291);
                   none found -> t0 with --ramp-fallback s
The ramp is a smoothstep of --ramp s. Prints the mouth jump at t0 and at the chosen start. --merge occl.json: the max with another weights json (ks_occl).
    ks_tailw.py --dub dub_f0.wav --lm lm_f0.npz --ks-lm post_f0/raw_lm.npz --frames N --out tailw.json [--mode similar --ramp 0.5]"""
import argparse, json, wave
import numpy as np
ap = argparse.ArgumentParser(); [ap.add_argument("--" + k, required=True) for k in ("dub", "lm", "ks-lm", "out")]
ap.add_argument("--frames", type=int, required=True); ap.add_argument("--fps", type=float, default=25.0); ap.add_argument("--shift", type=int, default=0)
ap.add_argument("--mode", default="plain", choices=["plain", "similar"]); ap.add_argument("--hold", type=float, default=0.0); ap.add_argument("--ramp", type=float, default=0.25)
ap.add_argument("--min-tail", type=float, default=1.0); ap.add_argument("--end-thr", type=float, default=0.25); ap.add_argument("--search", type=float, default=1.5)
ap.add_argument("--thr", type=float, default=0.06); ap.add_argument("--hold-frames", type=int, default=2); ap.add_argument("--ramp-fallback", type=float, default=1.0); ap.add_argument("--merge", default="")
a = ap.parse_args(); N, FPS = a.frames, a.fps
w_ = wave.open(a.dub); sr = w_.getframerate(); x = np.frombuffer(w_.readframes(w_.getnframes()), np.int16).astype(np.float32); hop = sr // 50; nf = len(x) // hop
e = np.sqrt((x[:nf * hop].reshape(nf, hop) ** 2).mean(1)) if nf else np.zeros(0); p99 = max(float(np.percentile(e, 99)) if nf else 1.0, 1.0)
on = e > 0.03 * p99; st = (np.convolve(e, np.ones(5) / 5, mode="same") if nf else e) > a.end_thr * p99
last = (np.flatnonzero(st)[-1] + 1) / 50 if st.any() else ((np.flatnonzero(on)[-1] + 1) / 50 if on.any() else 0.0); t0 = last + a.hold; dur = N / FPS
z = np.load(a.lm); LM = z["lm"][..., :2].astype(np.float32); J = z["j"] if "j" in z.files else np.arange(len(LM)); KL = np.load(getattr(a, "ks_lm"))["lm"][..., :2].astype(np.float32)
def opn(p):
    if not np.isfinite(p[[13, 14, 61, 291]]).all(): return np.nan
    return float(np.linalg.norm(p[13] - p[14]) / max(float(np.linalg.norm(p[61] - p[291])), 1e-3))
os_ = np.array([opn(LM[min(int(J[min(k, len(J) - 1)]), len(LM) - 1)]) for k in range(N)]); og = np.array([opn(KL[int(np.clip(k - a.shift, 0, len(KL) - 1))]) for k in range(N)])
info = {"last_speech_s": round(last, 2), "t0_s": round(t0, 2), "tail_s": round(dur - t0, 2), "mode": a.mode, "applied": False}
w = np.zeros(N, np.float32)
if on.any() and dur - t0 >= a.min_tail:
    k0 = min(N - 1, int(np.floor(t0 * FPS))); ks, ramp = k0, a.ramp; d = np.abs(og - os_)
    info["jump_at_t0"] = None if not np.isfinite(d[k0]) else round(float(d[k0]), 3); info["open_drawn_src_at_t0"] = [None if not np.isfinite(v) else round(float(v), 3) for v in (og[k0], os_[k0])]
    if a.mode == "similar":
        found = None
        for k in range(k0, min(N - a.hold_frames, k0 + int(a.search * FPS))):
            if np.all(np.isfinite(d[k:k + a.hold_frames])) and np.all(d[k:k + a.hold_frames] <= a.thr): found = k; break
        if found is None: ramp = a.ramp_fallback; info["similar"] = "not found within %.1f s -> t0, ramp %.2f s" % (a.search, ramp)
        else: ks = found; info["similar"] = "found %.2f s after t0" % ((found - k0) / FPS)
    u = np.clip((np.arange(N) - ks) / max(ramp * FPS, 1.0), 0, 1); w = (u * u * (3 - 2 * u)).astype(np.float32)
    info.update({"applied": True, "start_s": round(ks / FPS, 2), "ramp_s": ramp, "jump_at_start": None if not np.isfinite(d[ks]) else round(float(d[ks]), 3)})
if a.merge:
    m = np.asarray(json.load(open(a.merge))["w"], np.float32); n_ = min(len(m), N); w[:n_] = np.maximum(w[:n_], m[:n_]); info["merged"] = a.merge
json.dump({"w": [round(float(v), 4) for v in w], "tail": info}, open(a.out, "w")); print("KS_TAILW " + json.dumps(info))
