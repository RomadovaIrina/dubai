#!/usr/bin/env python3
"""r8 v2: Chatterbox's own pauses INSIDE a placed piece, where the SOURCE keeps talking, closed (r8v1 02: 2.0 s of the remaining dropouts, e.g. 8.7 s
"дня буквально", 100.3 s "или минералочки"). On the pre-placement timeline (after retake / burst_align, before lip_place), per placed piece:
  * the take's silent runs (10-ms frames > --below-db under the take's speech p95) >= --min-pause s, >= 0.1 s from the piece edges, during which
    the source speaks over >= --src-frac of the run;
  * each such pause is shortened to --keep s (its middle cut out, 10-ms equal-power crossfade in the silence), and the whole piece is slowed back to
    its old length (ffmpeg atempo, pitch kept), so its start, end and every other unit stay where they are and the speech fills the pause;
  * the piece's tempo (its atempo x this) never goes below --tmin: a pause is only partly closed when the budget runs out (largest pauses first).
Manifest: placement["atempo"] updated, placement["r8_close_inner"] = the cut pauses. Every other sample is bit-identical.
    python scripts/r8/close_inner.py --work W --out W2 [--tmin 0.9]"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys, tempfile
import numpy as np, soundfile as sf
FR = 0.01


def atempo(seg, sr, t):
    if abs(t - 1) < 1e-3: return seg
    with tempfile.TemporaryDirectory() as d:
        a, b = pathlib.Path(d) / "a.wav", pathlib.Path(d) / "b.wav"; sf.write(str(a), seg, sr, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(a), "-filter:a", f"atempo={t:.5f}", str(b)], check=True)
        return sf.read(str(b), dtype="float32")[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tmin", type=float, default=0.9); ap.add_argument("--min-pause", type=float, default=0.2); ap.add_argument("--keep", type=float, default=0.1)
    ap.add_argument("--below-db", type=float, default=35.0); ap.add_argument("--src-frac", type=float, default=0.5); a = ap.parse_args()
    W, O = pathlib.Path(a.work).resolve(), pathlib.Path(a.out); O.mkdir(parents=True, exist_ok=True)
    mp = next(W.glob("*_r4a.manifest.json")); man = json.loads(mp.read_text()); D = man["source"]["duration_s"]
    y, sr = sf.read(str(W / "dubbed_24k.wav"), dtype="float32"); y = y if y.ndim == 1 else y.mean(1); z = y.copy(); h = int(FR * sr)
    Lall = 20 * np.log10(np.sqrt((y[:len(y) // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-9); ref = float(np.percentile(Lall[Lall > -90], 95))
    n = int(D / FR) + 1; S = np.zeros(n, bool)
    for s in man["speech_intervals"]: S[int(s["start"] / FR):int(np.ceil(s["end"] / FR))] = True
    rep = []; xf = int(0.01 * sr); rp = np.sin(np.linspace(0, np.pi / 2, xf))
    for u in man["units"]:
        for p in (u.get("alignment") or {}).get("placements", []):
            if not p.get("placed"): continue
            i0, i1 = int(p["placed"][0] * sr), int(p["placed"][1] * sr); seg = y[i0:i1]; Ln = len(seg)
            if Ln < int(0.6 * sr): continue
            L = Lall[i0 // h:i1 // h]; sil = L < ref - a.below_db; d = np.diff(np.concatenate([[0], sil.astype(np.int8), [0]]))
            cand = []
            for r0, r1 in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                if (r1 - r0) * FR < a.min_pause or r0 * FR < 0.1 or (len(L) - r1) * FR < 0.1: continue
                g0, g1 = (i0 // h) + r0, (i0 // h) + r1; f = float(S[g0:g1].mean()) if g1 <= n else 0.0
                if f >= a.src_frac: cand.append([r0, r1])
            if not cand: continue
            t_cur = float(p.get("atempo") or 1.0); budget = Ln * (1 - a.tmin / t_cur) if t_cur > a.tmin else 0.0
            if budget < 0.05 * sr: continue
            cuts = []
            for r0, r1 in sorted(cand, key=lambda r: r[0] - r[1]):   # largest first
                want = (r1 - r0) * h - int(a.keep * sr); take = int(min(want, budget))
                if take < int(0.05 * sr): continue
                m = (r0 + r1) * h // 2; cuts.append((m - take // 2, m - take // 2 + take)); budget -= take
            if not cuts: continue
            cuts.sort(); parts = []; last = 0
            for c0, c1 in cuts: parts.append(seg[last:c0]); last = c1
            parts.append(seg[last:])
            out = parts[0]
            for q in parts[1:]:   # 10-ms equal-power crossfade at each cut (both sides are the take's silence)
                k = min(xf, len(out), len(q)); out = np.concatenate([out[:len(out) - k], out[len(out) - k:] * rp[::-1][:k] + q[:k] * rp[:k], q[k:]])
            r = len(out) / Ln; st = atempo(out.astype(np.float32), sr, r)   # slower by r: back to the old length
            st = st[:Ln] if len(st) >= Ln else np.concatenate([st, np.zeros(Ln - len(st), np.float32)])
            f = int(0.005 * sr); st[:f] *= np.linspace(0, 1, f); st[-f:] *= np.linspace(1, 0, f)
            z[i0:i1] = st; p["atempo"] = round(t_cur * r, 4)
            p["r8_close_inner"] = [[round(p["placed"][0] + c0 / sr, 2), round((c1 - c0) / sr, 2)] for c0, c1 in cuts]
            rep.append({"unit": u["id"], "text": p["text"], "placed": p["placed"], "cuts": p["r8_close_inner"], "tempo": p["atempo"]})
    for f in W.iterdir():
        if f.name in ("dubbed_24k.wav", "dubbed_16k.wav") or f.name.endswith(".manifest.json"): continue
        (O / f.name).unlink(missing_ok=True); (O / f.name).symlink_to(f)
    sf.write(str(O / "dubbed_24k.wav"), z, sr, subtype="FLOAT")
    import librosa
    sf.write(str(O / "dubbed_16k.wav"), librosa.resample(z, orig_sr=sr, target_sr=16000), 16000, subtype="PCM_16")
    man["r8_close_inner"] = {"pieces": rep, "tmin": a.tmin, "closed_s": round(sum(c[1] for r in rep for c in r["cuts"]), 2)}
    (O / mp.name).write_text(json.dumps(man, indent=1, ensure_ascii=False)); print(json.dumps({"pieces": len(rep), "closed_s": man["r8_close_inner"]["closed_s"]}))


if __name__ == "__main__":
    main()
