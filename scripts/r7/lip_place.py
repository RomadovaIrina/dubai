#!/usr/bin/env python3
"""r7 --lip-place: while the face is visible, put the dub speech INSIDE the source speech spans (m5 = m4 + r6 audio lost lip sync because of where
r6 places speech: 05 mouth/dub mismatch 1.74 -> 2.64 s, 04 0.77 -> 2.60 s). r6's takes, tempo, fades and pause POSITIONS stay exactly as they are:
  * the finished dub track (dubbed_24k.wav) is cut into BLOCKS = audible runs separated by real dub silences >= min_gap_s (50 ms). A block is
    never split and never stretched, so no new pause appears and the tempo is r6's;
  * every block may move by up to +-max_shift_s; a DP over the blocks (10 ms grid) minimises
        sum over 10 ms frames of  w(t) * ([dub speaks & source silent] + [source speaks & dub silent])     (= r4_mouth_audio's two error kinds)
        + lam * |shift|                                                                                    (stay at r6's placement unless it helps)
    with w = 1 where MediaPipe sees a face on the source, w_off (0: r6 placement kept off screen) elsewhere;
  * constraints: order kept; a gap never shrinks below min(its r6 length, unit_gap 0.15 s); a gap inside one sentence never grows by more than
    grow_s (0.35 s: a pause stays a short pause); the track never moves past 0 / the video end.
Writes <out>/ = symlinks of the work dir + shifted dubbed_24k.wav / dubbed_16k.wav + the manifest with every placement moved by its block's shift
(placed, track_start, spread pause times; "lip_shift_s"), so the remix, metrics and the face handoff all read the new timing.
    python scripts/r7/lip_place.py --work W --out W2 --manifest W/<id>_r4a.manifest.json --face A/sep/<id>/source_mediapipe.npz --src SRC.mp4"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot"))
FR = 0.01


def blocks_of(y, sr, min_gap_s=0.05, floor_rel_db=-55.0):
    h = int(FR * sr); n = len(y) // h; L = 20 * np.log10(np.sqrt((y[:n * h].reshape(n, h) ** 2).mean(1)) + 1e-9)
    act = L > max(-70.0, np.percentile(L[L > -100], 95) + floor_rel_db) if (L > -100).any() else np.zeros(n, bool)
    d = np.diff(np.concatenate([[0], act.astype(np.int8), [0]])); on, off = list(np.flatnonzero(d == 1)), list(np.flatnonzero(d == -1)); bl = []
    for a, b in zip(on, off):
        if bl and a - bl[-1][1] < int(min_gap_s / FR): bl[-1][1] = b
        else: bl.append([a, b])
    return [(a * FR, b * FR) for a, b in bl], act, L


def face_mask(face_npz, n, dur):
    if not face_npz or not pathlib.Path(face_npz).exists(): return np.ones(n, bool), None
    lm = np.load(face_npz)["lm"]; vis = np.isfinite(lm.reshape(len(lm), -1)).all(1); fps = len(vis) / dur
    return vis[np.minimum((np.arange(n) * FR * fps).astype(int), len(vis) - 1)], round(float(vis.mean()), 3)


def plan(blocks, act, S, w, unit_of, max_shift_s=0.6, lam=0.15, unit_gap=0.15, grow_s=0.35, dur=None):
    n = len(S); K = int(round(max_shift_s / FR)); sh = np.arange(-K, K + 1); c = w * (1 - 2 * S.astype(float))   # frame cost of dub speech here
    cs = np.concatenate([[0], np.cumsum(np.pad(c, (0, 2 * K + 1)))])
    B = len(blocks); st = [int(round(a / FR)) for a, _ in blocks]; en = [int(round(b / FR)) for _, b in blocks]
    cost = np.full((B, len(sh)), np.inf); arg = np.zeros((B, len(sh)), int)
    def bcost(i, k):   # dub frames of block i (its own activity, internal short silences excluded) moved by k frames
        a, b = st[i] + k, en[i] + k
        if a < 0 or (dur is not None and b > n): return np.inf
        idx = np.flatnonzero(act[st[i]:en[i]]) + a; return float(c[idx].sum()) + lam * abs(k)
    for j, k in enumerate(sh): cost[0, j] = bcost(0, k)
    for i in range(1, B):
        g0 = st[i] - en[i - 1]; gmin = min(g0, int(round(unit_gap / FR))); same = unit_of[i] == unit_of[i - 1] and unit_of[i] is not None
        gmax = g0 + int(round(grow_s / FR)) if same else 10 ** 9
        for j, k in enumerate(sh):
            bc = bcost(i, k)
            if not np.isfinite(bc): continue
            # previous block's shift kp: gap = g0 + k - kp in [gmin, gmax]  ->  kp in [k + g0 - gmax, k + g0 - gmin]
            lo, hi = max(-K, k + g0 - gmax), min(K, k + g0 - gmin)
            if lo > hi: continue
            seg = cost[i - 1, lo + K:hi + K + 1]; m = int(np.argmin(seg))
            if np.isfinite(seg[m]): cost[i, j] = seg[m] + bc; arg[i, j] = lo + K + m
    j = int(np.argmin(cost[-1])); out = [0] * B
    for i in range(B - 1, -1, -1): out[i] = int(sh[j]); j = arg[i, j]
    return [k * FR for k in out], float(cost[-1].min())


def apply(y, sr, blocks, shifts, fade_s=0.005):
    z = np.zeros_like(y); f = int(fade_s * sr)
    for (a, b), s in zip(blocks, shifts):
        i0, i1 = int(round(a * sr)), min(len(y), int(round(b * sr))); seg = y[i0:i1].copy()
        if len(seg) > 2 * f and f: seg[:f] *= np.linspace(0, 1, f); seg[-f:] *= np.linspace(1, 0, f)
        j0 = i0 + int(round(s * sr)); j0 = max(0, min(len(z) - len(seg), j0)); z[j0:j0 + len(seg)] += seg
    return z


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); ap.add_argument("--out", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--face"); ap.add_argument("--max-shift", type=float, default=0.6); ap.add_argument("--lam", type=float, default=0.15)
    ap.add_argument("--w-off", type=float, default=0.0); ap.add_argument("--grow", type=float, default=0.35); a = ap.parse_args()
    W, O = pathlib.Path(a.work).resolve(), pathlib.Path(a.out); O.mkdir(parents=True, exist_ok=True); man = json.loads(pathlib.Path(a.manifest).read_text())
    y, sr = sf.read(str(W / "dubbed_24k.wav"), dtype="float32"); y = y if y.ndim == 1 else y.mean(1)
    blocks, act, _ = blocks_of(y, sr); n = len(act); dur = len(y) / sr
    S = np.zeros(n, bool)
    for s in man["speech_intervals"]: S[int(s["start"] / FR):int(np.ceil(s["end"] / FR))] = True
    fv, fvis = face_mask(a.face, n, man["source"]["duration_s"]); w = np.where(fv, 1.0, a.w_off)
    pls = [(u["id"], p) for u in man["units"] for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed")]
    def unit_at(t0, t1):
        best = max(pls, key=lambda q: min(t1, q[1]["placed"][1]) - max(t0, q[1]["placed"][0]), default=None)
        return best[0] if best and min(t1, best[1]["placed"][1]) - max(t0, best[1]["placed"][0]) > 0 else None
    unit_of = [unit_at(a_, b_) for a_, b_ in blocks]
    shifts, _ = plan(blocks, act, S, w, unit_of, a.max_shift, a.lam, grow_s=a.grow, dur=dur)
    z = apply(y, sr, blocks, shifts)
    def err(D):
        return float(((D & ~S) | (S & ~D))[fv].sum() * FR)
    D0 = act.copy(); D1 = np.zeros(n, bool)
    for (b0, b1), s in zip(blocks, shifts):
        i0, i1 = int(round(b0 / FR)), int(round(b1 / FR)); k = int(round(s / FR)); src = act[i0:i1]; D1[max(0, i0 + k):max(0, i0 + k) + len(src)] |= src[:max(0, n - max(0, i0 + k))]
    for f in W.iterdir():
        if f.name in ("dubbed_24k.wav", "dubbed_16k.wav") or f.name.endswith(".manifest.json"): continue
        (O / f.name).unlink(missing_ok=True); (O / f.name).symlink_to(f)
    sf.write(str(O / "dubbed_24k.wav"), z, sr, subtype="PCM_16")
    import librosa
    sf.write(str(O / "dubbed_16k.wav"), librosa.resample(z, orig_sr=sr, target_sr=16000), 16000, subtype="PCM_16")
    mid = [(b0 + b1) / 2 for b0, b1 in blocks]
    def shift_at(t0, t1):
        k = max(range(len(blocks)), key=lambda i: min(t1, blocks[i][1]) - max(t0, blocks[i][0])) if blocks else None
        return shifts[k] if k is not None and min(t1, blocks[k][1]) - max(t0, blocks[k][0]) > 0 else 0.0
    moved = []
    for u in man["units"]:
        al = u.get("alignment") or {}
        for p in al.get("placements", []):
            if not p.get("placed"): continue
            s = round(shift_at(*p["placed"]), 3); p["lip_shift_s"] = s; p["placed"] = [round(p["placed"][0] + s, 3), round(p["placed"][1] + s, 3)]
            if "track_start" in p: p["track_start"] = round(p["track_start"] + s, 3)
            for q in (p.get("spread") or {}).get("pauses", []):
                if "t" in q: q["t"] = round(q["t"] + s, 3)
            if s: moved.append(s)
        if "track_start" in al and al.get("placements"): al["track_start"] = round(al["track_start"] + al["placements"][0].get("lip_shift_s", 0.0), 3)
    rep = {"blocks": len(blocks), "moved_blocks": int(sum(1 for s in shifts if abs(s) > 1e-6)), "shift_abs_mean_s": round(float(np.mean(np.abs(shifts))), 3) if shifts else 0,
           "shift_abs_max_s": round(float(np.max(np.abs(shifts))), 3) if shifts else 0, "face_visible_frac": fvis,
           "mismatch_face_visible_s_before": round(err(D0), 2), "mismatch_face_visible_s_after": round(err(D1), 2),
           "max_shift_s": a.max_shift, "lam": a.lam, "w_off": a.w_off, "grow_s": a.grow, "shifts": [[round(b0, 2), round(b1, 2), round(s, 2)] for (b0, b1), s in zip(blocks, shifts)]}
    man["lip_place"] = rep; man["dubbed_track"] = {**man.get("dubbed_track", {}), "lip_placed": True}
    (O / pathlib.Path(a.manifest).name).write_text(json.dumps(man, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rep.items() if k != "shifts"}))


if __name__ == "__main__":
    main()
