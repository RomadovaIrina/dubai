#!/usr/bin/env python3
"""r7 v3b: the EXISTING English take of a unit aligned to the Russian speech bursts (user: "the source mouth must not move while the dub is
silent"). On the r6 timeline (before lip_place), per unit with the face visible over >= half of its source speech:
  1. the unit's source speech cut into pieces at every Silero pause >= 0.4 s inside its bursts (retake.sub_bursts);
  2. the take's words located in the dub track (CTC alignment of the unit's own text, the TTS QA's wav2vec2);
  3. a DP splits the English words into consecutive chunks, each laid on a run of consecutive source pieces: the chunk starts when its first
     piece starts and is tempo-fitted (0.92-1.08, ffmpeg atempo, pitch kept) to end when its last piece ends; a source piece may stay without
     dub. Cost = face-visible source speech left without dub (x2 for a whole piece without dub) + 0.3 x dub over source silence + 0.8 s per cut that is not a pb1 boundary
     (punctuation / clause start) + 0.5 s per one-word chunk + 1.0 x |1 - tempo| x seconds; a lone function word, an overlap with the next
     unit's take and an overlap between chunks are refused;
  4. kept when it beats the take as placed by r6 by >= --min-gain s; the old take's span is faded out (20-ms ramps), the chunks are cut between
     words (tts_align.cut_parts) and added. Words, voice and level are the take's own: only timing and tempo change.
    python scripts/r7/burst_align.py --work <r6 or r7v3src work dir> --out <new work dir> --face <mediapipe npz>"""
import argparse, json, pathlib, re, subprocess, sys, tempfile
import numpy as np, soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent
for p in (HERE.parent / "pilot", HERE.parent / "quality", HERE): sys.path.insert(0, str(p))
FR = 0.02


def atempo(seg, sr, t):
    """pitch-preserving tempo change (ffmpeg atempo; t > 1 = faster)"""
    if abs(t - 1) < 1e-3: return seg
    with tempfile.TemporaryDirectory() as d:
        a, b = pathlib.Path(d) / "a.wav", pathlib.Path(d) / "b.wav"; sf.write(str(a), seg, sr, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(a), "-filter:a", f"atempo={t:.5f}", str(b)], check=True)
        return sf.read(str(b), dtype="float32")[0]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); ap.add_argument("--out", required=True); ap.add_argument("--face")
    ap.add_argument("--tmin", type=float, default=0.92); ap.add_argument("--tmax", type=float, default=1.08); ap.add_argument("--min-gain", type=float, default=0.2)
    ap.add_argument("--min-face", type=float, default=0.5); a = ap.parse_args()
    import lip_place, retake
    from tts_qa import TtsQA, norm
    from tts_align import CtcAligner, cut_parts
    import r4_pause_logic as pl
    W, O = pathlib.Path(a.work).resolve(), pathlib.Path(a.out); O.mkdir(parents=True, exist_ok=True)
    mp = next(W.glob("*_r4a.manifest.json")); man = json.loads(mp.read_text()); D = man["source"]["duration_s"]
    y, sr = sf.read(str(W / "dubbed_24k.wav"), dtype="float32"); y = y if y.ndim == 1 else y.mean(1); n = int(D / FR) + 1
    fv, _ = lip_place.face_mask(a.face, n, D); S = np.zeros(n, bool)
    for s in man["speech_intervals"]: S[int(s["start"] / FR):int(np.ceil(s["end"] / FR))] = True
    words_all = [w for sg in man["asr"]["segments"] for w in sg.get("words") or []]
    qa = TtsQA(); aligner = CtcAligner.from_qa(qa); units = man["units"]; z = y.copy(); rep = {"tmin": a.tmin, "tmax": a.tmax, "units": []}
    fs = lambda t0, t1: float((S & fv)[int(t0 / FR):int(t1 / FR)].sum() * FR)   # face-visible source speech in [t0, t1)
    sil = lambda t0, t1: float((~S)[int(t0 / FR):int(t1 / FR)].sum() * FR)

    def span_of(u):
        ps = [p for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed")]
        return (min(p["placed"][0] for p in ps), max(p["placed"][1] for p in ps)) if ps else None

    def cost_of(placed, B):   # placed: [(t0, t1)]; the same axes as the DP, measured on frames
        Dm = np.zeros(n, bool)
        for t0, t1 in placed: Dm[int(t0 / FR):int(t1 / FR)] = True
        lo, hi = int(B[0][0] / FR), int(min(D, max(B[-1][1], max(t1 for _, t1 in placed))) / FR)
        return float((S & fv & ~Dm)[lo:hi].sum() * FR) + 0.3 * float((Dm & ~S)[lo:hi].sum() * FR)

    for i, u in enumerate(units):
        ps = [p for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed")]
        if not ps: continue
        B, _ = retake.sub_bursts(ps, man, words_all, u["text"]); lo, hi = int(B[0][0] / FR), int(B[-1][1] / FR)
        if hi <= lo or float(fv[lo:hi].mean()) < a.min_face: continue
        a0, a1 = span_of(u); prev_end = max([span_of(v)[1] for v in units[:i] if span_of(v)] or [0.0]) + 0.1
        win = min([span_of(v)[0] for v in units[i + 1:] if span_of(v)] or [D]) - 0.15
        text = " ".join(p["text"] for p in ps); i0, i1 = int(max(0, a0 - 0.03) * sr), int(min(D, a1 + 0.03) * sr)
        segs = []   # the take without r6's placement gaps: its pieces back to back, a gap kept up to 0.15 s
        for q, p in enumerate(ps):
            b0 = p["placed"][0] - (0.03 if q == 0 else min(0.075, (p["placed"][0] - ps[q - 1]["placed"][1]) / 2))
            b1 = p["placed"][1] + (0.03 if q + 1 == len(ps) else min(0.075, (ps[q + 1]["placed"][0] - p["placed"][1]) / 2))
            segs.append(y[int(max(0, b0) * sr):int(min(D, b1) * sr)])
        reg = np.concatenate(segs)
        spans = aligner.align(reg, sr, text); words = text.split()
        rec = {"id": u["id"], "en": text, "bursts": [[round(b0, 2), round(b1, 2)] for b0, b1 in B], "old_cost": round(cost_of([tuple(p["placed"]) for p in ps], B), 2)}
        tc = [len(norm(w_)) for w_ in words]   # aligner tokens per written word ("twenty-eight" = 2)
        if not spans or sum(tc) != len(spans): rec["skipped"] = "align"; rep["units"].append(rec); print(json.dumps(rec, ensure_ascii=False), flush=True); continue
        ws_, q_ = [], 0
        for c_ in tc:
            if c_: ws_.append({"start": spans[q_]["start"], "end": spans[q_ + c_ - 1]["end"]}); q_ += c_
            else: ws_.append(None)
        keep_w = [k_ for k_, x in enumerate(ws_) if x]; words_tok = tc; spans_w = [ws_[k_] for k_ in keep_w]; words = [words[k_] for k_ in keep_w]; tc = [tc[k_] for k_ in keep_w]
        nw, K = len(words), len(B)
        okcut = [j == 0 or j == nw or pl.classify((words[j - 1], None, False), (words[j],))[1] > 0 for j in range(nw + 1)]
        func = lambda j0, j1: j1 - j0 == 1 and re.sub(r"[^\w']", "", words[j0].lower()) in retake.FUNC

        def group(j0, j1, k0, k1, start_min):   # words [j0, j1) on pieces [k0, k1): (cost, tempo, t0, t1)
            d = spans_w[j1 - 1]["end"] - spans_w[j0]["start"] + 0.14; T = B[k1 - 1][1] - B[k0][0]
            t = min(a.tmax, max(a.tmin, d / T)); L = d / t; t0 = max(B[k0][0], start_min); t1 = t0 + L
            limit = B[k1][0] - 0.1 if k1 < K else win
            if t1 > limit + 0.02: return None
            c = max(0.0, fs(B[k0][0], B[k1 - 1][1]) - fs(t0, t1)) + 0.3 * sil(t0, t1) + 1.0 * abs(1 - t) * L
            return c, t, t0, t1

        INF = (1e9, None, 0.0); dp = {(0, 0): (0.0, None, prev_end)}   # (pieces done, words done) -> (cost, back, end of the last chunk)
        for k in range(K):
            for j in range(nw + 1):
                if (k, j) not in dp: continue
                c0, _, le = dp[(k, j)]
                v = (c0 + 2.0 * fs(*B[k]), ((k, j), None), le)   # piece k without dub: a whole talking burst in silence (x2: the mouth the user hates)
                if v[0] < dp.get((k + 1, j), INF)[0]: dp[(k + 1, j)] = v
                if j == nw: continue
                for k1 in range(k + 1, K + 1):
                    for j1 in range(j + 1, nw + 1):
                        if func(j, j1) and nw > 1: continue   # a lone "The" / "and" at a pause
                        g = group(j, j1, k, k1, le + (0.12 if j else 0.0))
                        if g is None: continue
                        cc = c0 + g[0] + (0.8 if j1 < nw and not okcut[j1] else 0.0) + (0.5 if j1 - j == 1 and nw > 1 else 0.0)
                        if cc < dp.get((k1, j1), INF)[0]: dp[(k1, j1)] = (cc, ((k, j), (j, j1, k1, g[3], g[1], g[2])), g[3])
        if (K, nw) not in dp: rec["skipped"] = "no fit"; rep["units"].append(rec); continue
        chunks = []; st = (K, nw)
        while dp[st][1]:
            back, g = dp[st][1]
            if g: chunks.append({"j0": g[0], "j1": g[1], "k0": back[0], "k1": g[2], "t0": g[5], "t1": g[3], "tempo": g[4]})
            st = back
        chunks.reverse()
        placed = [(c["t0"], c["t1"]) for c in chunks]; new_cost = cost_of(placed, B) + 0.8 * sum(1 for c in chunks[:-1] if not okcut[c["j1"]])
        rec.update({"new_cost": round(new_cost, 2), "chunks": [{"en": " ".join(words[c["j0"]:c["j1"]]), "src": [round(B[c["k0"]][0], 2), round(B[c["k1"] - 1][1], 2)],
                                                                "placed": [round(c["t0"], 2), round(c["t1"], 2)], "tempo": round(c["tempo"], 3)} for c in chunks]})
        rec["accepted"] = new_cost <= rec["old_cost"] - a.min_gain
        if rec["accepted"]:
            counts = [sum(tc[c["j0"]:c["j1"]]) for c in chunks]   # cut_parts counts aligner tokens
            if sum(counts) != len(spans): rec["accepted"] = False; rec["skipped"] = "count"; rep["units"].append(rec); continue
            parts = cut_parts(reg, sr, spans, counts)
            k = int(0.02 * sr); g_ = np.zeros(i1 - i0); g_[:k] = 1 - retake.ramp(k); g_[-k:] = retake.ramp(k); z[i0:i1] = z[i0:i1] * g_
            newp = []
            for c, (seg, *_) in zip(chunks, parts):
                s_ = atempo(seg, sr, c["tempo"]); f = int(0.01 * sr); s_[:f] *= retake.ramp(f); s_[-f:] *= retake.ramp(f)[::-1]; j0 = int(c["t0"] * sr)
                z[j0:j0 + len(s_)] += s_[:max(0, len(z) - j0)]
                newp.append({"burst": [B[c["k0"]][0], B[c["k1"] - 1][1]], "window": [B[c["k0"]][0], B[c["k1"] - 1][1]], "phrases": [len(newp)], "group_s": round(len(seg) / sr, 3),
                             "ratio": round(len(s_) / sr / max(0.05, B[c["k1"] - 1][1] - B[c["k0"]][0]), 3), "atempo": round(c["tempo"], 3), "placed": [round(c["t0"], 3), round(c["t0"] + len(s_) / sr, 3)],
                             "track_start": round(c["t0"], 3), "text": " ".join(words[c["j0"]:c["j1"]]), "source_words": "", "r7_burst_align": True})
            u["alignment"]["placements"] = newp; u["alignment"]["track_start"] = newp[0]["placed"][0]; u["r7_burst_align"] = {k_: v for k_, v in rec.items()}
        rep["units"].append(rec); print(json.dumps({k_: v for k_, v in rec.items()}, ensure_ascii=False), flush=True)
    for f in W.iterdir():
        if f.name in ("dubbed_24k.wav", "dubbed_16k.wav", "retake.json") or f.name.endswith(".manifest.json"): continue
        (O / f.name).unlink(missing_ok=True); (O / f.name).symlink_to(f.resolve())
    sf.write(str(O / "dubbed_24k.wav"), z, sr, subtype="PCM_16")
    import librosa
    sf.write(str(O / "dubbed_16k.wav"), librosa.resample(z, orig_sr=sr, target_sr=16000), 16000, subtype="PCM_16")
    man["r7_burst_align"] = {"changed": [r["id"] for r in rep["units"] if r.get("accepted")], "tmin": a.tmin, "tmax": a.tmax}
    (O / mp.name).write_text(json.dumps(man, indent=1, ensure_ascii=False)); (O / "burst_align.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
