#!/usr/bin/env python3
"""r8 v4: whole-phrase tempo fit after placement (management: "the dub disappears"; user: no unnatural splits). r8v3 stopped splitting phrases at
unnatural points, so a phrase that is shorter than the Russian, or that burst_align sped up to 1.1 to fit a burst, leaves the speaker talking
alone at its end (04 "our app reacts to all this." 10.3-11.4 s, 05 "I'm just walking around," 31.7-32.3 s). On the PLACED dub (after lip_place),
every placed piece is re-timed as a WHOLE (no new cut):
  * tempo t in [--tmin, --tmax] of the raw take (the piece's own atempo already counted), start moved by <= --max-shift s;
  * never into a neighbour (>= --min-gap s to the previous / next piece), never starting > --early s before the speaker's onset under it, and the gap
    to a neighbour of the same sentence never grows beyond max(its current size, the source pause there + 0.35 s) (r8 v3 inserted-gap rule);
  * score = source speech covered by the piece's speech - 0.3 x its speech over source silence - --tempo-w x |1 - t| x seconds; kept when it
    beats the current timing by >= --min-gain s.
The piece's samples are re-rendered with ffmpeg atempo (pitch kept) at the new start; its 0.06-0.12 s pre / post roll is the take's silence,
10-ms fades at its edges. Manifest: placement placed / track_start / atempo updated, placement["r8_phrase_fit"]; lip_place.dub_speech recomputed.
    python scripts/r8/phrase_fit.py --placed <lip_place out dir> --out <dir>"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, tempfile
import numpy as np, soundfile as sf
FR = 0.02


def atempo(seg, sr, t):
    if abs(t - 1) < 1e-3: return seg
    with tempfile.TemporaryDirectory() as d:
        a, b = pathlib.Path(d) / "a.wav", pathlib.Path(d) / "b.wav"; sf.write(str(a), seg, sr, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(a), "-filter:a", f"atempo={t:.5f}", str(b)], check=True)
        return sf.read(str(b), dtype="float32")[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--placed", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tmin", type=float, default=0.9); ap.add_argument("--tmax", type=float, default=1.1); ap.add_argument("--max-shift", type=float, default=0.5)
    ap.add_argument("--min-gap", type=float, default=0.15); ap.add_argument("--early", type=float, default=0.2); ap.add_argument("--tempo-w", type=float, default=1.0)
    ap.add_argument("--min-gain", type=float, default=0.15); a = ap.parse_args()
    W, O = pathlib.Path(a.placed).resolve(), pathlib.Path(a.out); O.mkdir(parents=True, exist_ok=True)
    mp = next(W.glob("*.manifest.json")); man = json.loads(mp.read_text()); D = man["source"]["duration_s"]
    y, sr = sf.read(str(W / "dubbed_24k.wav"), dtype="float32"); y = y if y.ndim == 1 else y.mean(1); z = y.copy(); h = int(FR * sr); n = len(y) // h + 1
    L = 20 * np.log10(np.sqrt((np.pad(y, (0, n * h - len(y)))[:n * h].reshape(n, h) ** 2).mean(1)) + 1e-9); ref = float(np.percentile(L[L > -90], 95))
    S = np.zeros(n, bool)
    for s in man["speech_intervals"]: S[int(s["start"] / FR):min(n, int(np.ceil(s["end"] / FR)))] = True
    d_ = np.diff(np.concatenate([[0], S.astype(np.int8), [0]])); ons = np.flatnonzero(d_ == 1)
    P = sorted([(u["id"], p) for u in man["units"] for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed")], key=lambda x: x[1]["placed"][0])
    srcpause = lambda t0, t1: max([(min(t1, b * FR) - max(t0, a_ * FR)) for a_, b in zip(np.flatnonzero(d_ == -1), np.flatnonzero(d_ == 1)[1:]) if b * FR > t0 and a_ * FR < t1] or [0.0])
    rep = []
    for q, (uid, p) in enumerate(P):
        p0, p1 = p["placed"]; i0, i1 = int(p0 * sr), min(len(y), int(p1 * sr)); seg = y[i0:i1].copy(); Ln = (i1 - i0) / sr
        if Ln < 0.3: continue
        act = L[i0 // h:i1 // h] > ref - 30   # the piece's speech frames, relative to its own start
        if not act.any(): continue
        t_cur = float(p.get("atempo") or 1.0)
        prev_end = P[q - 1][1]["placed"][1] if q else 0.0; next_start = P[q + 1][1]["placed"][0] if q + 1 < len(P) else D
        same_prev = q and P[q - 1][0] == uid; same_next = q + 1 < len(P) and P[q + 1][0] == uid
        cap_prev = max(p0 - prev_end, srcpause(prev_end, p0) + 0.35) if same_prev else 1e9
        cap_next = max(next_start - p1, srcpause(p1, next_start) + 0.35) if same_next else 1e9
        def score(t, s0):
            k = t_cur / t; m = len(act); idx = np.flatnonzero(act); Lp = Ln * k   # length scales by t_cur / t
            fr = (s0 / FR + idx * k).astype(int); fr = fr[(fr >= 0) & (fr < n)]
            if not len(fr): return -1e9, Lp
            M = np.zeros(n, bool); M[fr.min():fr.max() + 1] = True
            M &= np.convolve(np.isin(np.arange(n), fr).astype(float), np.ones(5), "same") > 0   # speech frames, small gaps bridged
            cov = float((M & S).sum() * FR); sil = float((M & ~S).sum() * FR)
            return cov - 0.3 * sil - a.tempo_w * abs(1 - t) * Lp, Lp
        base, _ = score(t_cur, p0); best = (base, t_cur, p0)
        first_speech = p0 + np.flatnonzero(act)[0] * FR
        on_under = ons[(ons * FR >= first_speech - 1.0) & (ons * FR <= first_speech + 1.0)]
        for t in np.arange(a.tmin, a.tmax + 1e-6, 0.02):
            for ds in np.arange(-a.max_shift, a.max_shift + 1e-6, 0.04):
                s0 = p0 + ds; sc, Lp = score(t, s0); e0 = s0 + Lp
                if s0 < prev_end + a.min_gap or e0 > next_start - a.min_gap or s0 < 0 or e0 > D: continue
                if s0 - prev_end > cap_prev + 1e-6 or next_start - e0 > cap_next + 1e-6: continue
                fs = s0 + (first_speech - p0) * (t_cur / t)
                if len(on_under) and fs < on_under.min() * FR - a.early: continue   # never start before the speaker
                if sc > best[0]: best = (sc, round(float(t), 3), round(float(s0), 3))
        if best[0] - base < a.min_gain: continue
        _, t, s0 = best; new = atempo(seg, sr, t / t_cur); f = int(0.01 * sr); new[:f] *= np.linspace(0, 1, f); new[-f:] *= np.linspace(1, 0, f)
        z[i0:i1] = 0.0; j0 = int(s0 * sr); z[j0:j0 + len(new)] += new[:max(0, len(z) - j0)]
        p["r8_phrase_fit"] = {"old": [round(p0, 3), round(p1, 3), t_cur], "tempo": t, "gain_s": round(best[0] - base, 2)}
        p["placed"] = [s0, round(s0 + len(new) / sr, 3)]; p["track_start"] = s0; p["atempo"] = t
        rep.append({"unit": uid, "text": p["text"], **p["r8_phrase_fit"], "new": p["placed"]})
    for f in W.iterdir():
        if f.name in ("dubbed_24k.wav", "dubbed_16k.wav") or f.name.endswith(".manifest.json"): continue
        (O / f.name).unlink(missing_ok=True); (O / f.name).symlink_to(f.resolve())
    sf.write(str(O / "dubbed_24k.wav"), z, sr, subtype="FLOAT")
    import librosa
    sf.write(str(O / "dubbed_16k.wav"), librosa.resample(z, orig_sr=sr, target_sr=16000), 16000, subtype="PCM_16")
    L2 = 20 * np.log10(np.sqrt((np.pad(z, (0, n * h - len(z)))[:n * h].reshape(n, h) ** 2).mean(1)) + 1e-9); act2 = L2 > ref - 30
    d2 = np.diff(np.concatenate([[0], act2.astype(np.int8), [0]]))
    if "lip_place" in man: man["lip_place"]["dub_speech"] = [[round(s * FR, 2), round(e * FR, 2)] for s, e in zip(np.flatnonzero(d2 == 1), np.flatnonzero(d2 == -1))]
    man["r8_phrase_fit"] = {"pieces": rep, "tmin": a.tmin, "tmax": a.tmax}
    (O / mp.name).write_text(json.dumps(man, indent=1, ensure_ascii=False)); print(json.dumps({"refit": len(rep), "gain_s": round(sum(r["gain_s"] for r in rep), 2)}))


if __name__ == "__main__":
    main()
