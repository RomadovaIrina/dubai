#!/usr/bin/env python3
"""r8 v4: one Chatterbox take PER CLAUSE for a sentence whose source speech spans pauses (05 u6 "We are testing all possible variants | that we
can come up with." ended 4 s before the speaker; one whole take has no quiet point at "variants | that", so a cut there is a cut word). On the
pre-placement timeline (after retake), per unit whose source speech left without dub is >= --short-s and whose source pieces (Silero pauses
>= 0.4 s inside its bursts, retake.sub_bursts) are >= 2:
  * the unit's CURRENT English (no re-translation: same words, same judge score, no padding) is split only at natural boundaries: after punctuation,
    before a clause word (and / but / that / which / ...) or before a preposition where the source pauses >= 0.5 s; never after a function word,
    every clause >= 2 words;
  * each layout (<= 3 cuts) maps its clauses onto consecutive runs of source pieces (DP on the take's own word timing); every clause is synthesised
    as its OWN take (same speaker reference, cfg 0.5 / 0.3 = deliberate, the QA ladder: 3 seeds, Whisper must hear it), with continuation punctuation
    (a non-final clause ends with ",", the next starts lower-case), trimmed by CTC with 0.12 s of the take's own silence at both ends: natural ends;
  * clause k starts at its run's source onset (or after clause k-1 + 0.12 s), never later than clause k-1's end + the source pause + 0.35 s,
    tempo 1.0; it must end before the next unit's take;
  * cost = source speech left uncovered + 0.3 x dub over source silence (retake.py's cost); kept when it beats the current placement by >= --min-gain.
The old take is faded out (20-ms ramps), the clauses are level-matched to it (speech p95) and added; u["r8_clause"] = the record (burst_align
skips such units). Every other unit is bit-identical.
    python scripts/r8/clause_takes.py --work W --out W2 --src16 S16 [--face npz]"""
import argparse, itertools, json, os, pathlib, re, sys, time, zlib
import numpy as np, soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent
for p in (HERE.parent / "pilot", HERE.parent / "quality", HERE.parent / "r7"): sys.path.insert(0, str(p))
FR = 0.02
TEMPI = (1.0, 0.95, 0.9)
QUANT = {"all", "some", "every", "each", "any", "no", "many", "much", "more", "most", "such", "both", "few", "several", "other", "another"}
INTRO = {"today", "now", "here", "so", "well", "then", "also", "yesterday", "tomorrow", "anyway", "first", "next", "later"}   # a sentence-opening adverbial may stand alone: "Today, / we are starting"
PREP = {"on", "in", "at", "for", "with", "from", "about", "among", "around", "into", "over", "under", "after", "before", "during"}   # not "of" / "to": "kind | of", "want | to"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); ap.add_argument("--out", required=True); ap.add_argument("--src16", required=True)
    ap.add_argument("--face"); ap.add_argument("--short-s", type=float, default=0.4); ap.add_argument("--min-gain", type=float, default=0.2); ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--max-cuts", type=int, default=3); ap.add_argument("--tmin", type=float, default=0.9); a = ap.parse_args()
    import retake, r4_pause_logic as pl
    W, O = pathlib.Path(a.work).resolve(), pathlib.Path(a.out); O.mkdir(parents=True, exist_ok=True)
    mp = next(W.glob("*_r4a.manifest.json")); man = json.loads(mp.read_text()); D = man["source"]["duration_s"]
    y, sr = sf.read(str(W / "dubbed_24k.wav"), dtype="float32"); y = y if y.ndim == 1 else y.mean(1); z = y.copy(); n = int(D / FR) + 1; h = int(FR * sr)
    S = np.zeros(n, bool)
    for s in man["speech_intervals"]: S[int(s["start"] / FR):min(n, int(np.ceil(s["end"] / FR)))] = True
    words_all = [w for sg in man["asr"]["segments"] for w in sg.get("words") or []]; units = man["units"]
    span_of = lambda u: (lambda ps: (min(p["placed"][0] for p in ps), max(p["placed"][1] for p in ps)) if ps else None)([p for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed")])
    srcsil = lambda t0, t1: float((~S)[int(t0 / FR):max(int(t0 / FR), int(t1 / FR))].sum() * FR)

    def cost_of(placed, lo_t, hi_t):
        Dm = np.zeros(n, bool)
        for t0, t1 in placed: Dm[int(t0 / FR):min(n, int(t1 / FR))] = True
        lo, hi = int(lo_t / FR), min(n, int(max(hi_t, max(t1 for _, t1 in placed)) / FR))
        unc = float((S & ~Dm)[lo:hi].sum() * FR); sil = float((Dm & ~S)[lo:hi].sum() * FR)
        return unc + 0.3 * sil, {"uncovered_s": round(unc, 2), "over_silence_s": round(sil, 2)}

    rows = []
    for i, u in enumerate(units):
        ps = [p for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed")]
        if not ps: continue
        B, _ = retake.sub_bursts(ps, man, words_all, u["text"], min_piece=0.2)
        if len(B) < 2: continue
        base, det = cost_of([tuple(p["placed"]) for p in ps], B[0][0], B[-1][1])
        if det["uncovered_s"] < a.short_s: continue
        text = " ".join(p["text"] for p in ps).strip(); words = text.split()
        if len(words) < 4: continue
        nxt = next((span_of(v)[0] for v in units[i + 1:] if span_of(v)), D); rows.append({"u": u, "i": i, "B": B, "base": base, "det": det, "words": words, "win": nxt - 0.15})
    print("selected", [(r["u"]["id"], r["det"], len(r["B"])) for r in rows], flush=True)
    rep = []
    if rows:
        os.environ.setdefault("DUB_QWEN_SIZE", "14b")
        import torch, run_clean_pipeline_05 as rp
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        from chatterbox_fp16 import install_speech_token_guard
        from tts_qa import TtsQA
        from tts_align import CtcAligner, cut_parts, hush_gaps
        m = ChatterboxMultilingualTTS.from_local(HERE.parents[1] / "models" / "chatterbox", "cuda", t3_model="v3"); install_speech_token_guard(m)
        qa = TtsQA(); aligner = CtcAligner.from_qa(qa)
        refs = rp.speaker_references(units, man["diarization"]["turns"], pathlib.Path(a.src16), O); cur = None; cache = {}
    for r in rows:
        u, B, words = r["u"], r["B"], r["words"]; t_start = time.perf_counter(); nw = len(words)
        pause_after = lambda k: B[k + 1][0] - B[k][1] if k + 1 < len(B) else 0.0
        def natural(j):   # boundary before words[j]
            prev, nxt = re.sub(r"[^\w']", "", words[j - 1].lower()), re.sub(r"[^\w']", "", words[j].lower())
            if prev in retake.NOCUT or prev in retake.FUNC or prev in QUANT: return False   # not "from all | around the world"
            if re.search(r"[,.;:!?—]$", words[j - 1]) or nxt in pl.CLAUSE: return True
            return nxt in PREP and max(pause_after(k) for k in range(len(B))) >= 0.5
        cuts_ok = [j for j in range(2, nw - 1) if natural(j)]
        if nw > 3 and re.sub(r"[^\w']", "", words[0].lower()) in INTRO: cuts_ok = [1] + cuts_ok   # r8 v4: 04 u2 "Today, / we are starting ..." (the speaker says "сегодня" alone)
        best = None
        if u["speaker"] != cur:
            ref = refs[u["speaker"]]["file"]
            if ref: m.prepare_conditionals(ref)
            cur = u["speaker"]
        def take(txt, cfg):   # one clause take: QA ladder, CTC trim with 0.12 s of the take's own silence at both ends, gap murmur hushed
            key = (txt, cfg)
            if key in cache: return cache[key]
            res = None
            for soff, temp in ((0, 0.8), (1, 0.8), (2, 0.6)):
                torch.manual_seed(a.seed + 100 * u["id"] + 7919 * soff + zlib.crc32(txt.encode()) % 1000)
                with torch.autocast("cuda", dtype=torch.float16):
                    wav = m.generate(txt, language_id="en", temperature=temp, cfg_weight=cfg)
                wv = wav.detach().float().cpu()[0].numpy().astype(np.float32); q = qa.assess(txt, wv, m.sr, "en")
                if q["bad"]: continue
                spans = aligner.align(wv, m.sr, txt)
                if not spans: continue
                wv, _ = hush_gaps(wv, m.sr, spans); seg = cut_parts(wv, m.sr, spans, [len(spans)], pre_s=0.12, post_s=0.12)[0][0]; res = (seg, q["whisper"][:120]); break
            cache[key] = res; return res
        for k in range(1, min(a.max_cuts, len(cuts_ok)) + 1):
            for cs in itertools.combinations(cuts_ok, k):
                bnd = [0, *cs, nw]; cl = [" ".join(words[bnd[q]:bnd[q + 1]]) for q in range(len(bnd) - 1)]
                if any(len(c.split()) < 2 and not (q == 0 and re.sub(r"[^\w']", "", c.lower()) in INTRO) for q, c in enumerate(cl)): continue
                say = [(c[0].lower() + c[1:] if q and not re.match(r"(I\b|I'|[A-Z]{2})", c) else c) for q, c in enumerate(cl)]
                say = [s if q == len(say) - 1 or re.search(r"[,.;:!?]$", s) else s + "," for q, s in enumerate(say)]
                nc = len(cl)
                if nc > len(B): continue
                for cfg in (0.5, 0.3):
                    segs = [take(s, cfg) for s in say]
                    if any(s is None for s in segs): continue
                    lens = [len(s[0]) / m.sr for s in segs]
                    # runs of source pieces per clause: contiguous partition of B into nc runs, closest run lengths to the clause lengths
                    bestrun = None
                    for st in itertools.combinations(range(len(B)), nc):   # clause q starts on source piece st[q]; leading pieces may stay without dub
                        rb = [*st, len(B)]; runs = [(B[rb[q]][0], B[rb[q + 1] - 1][1]) for q in range(nc)]   # (04 u1: "Город" 0.3 s, then "Сочи", "28 августа")
                        for tem in itertools.product(TEMPI, repeat=nc):   # per clause tempo 1.0 / 0.95 / 0.9 (slower only where it covers more than it costs)
                            placed, t = [], None
                            for (r0, r1), L, tp in zip(runs, lens, tem):
                                t0 = r0 if t is None else max(t + 0.12, min(r0, t + srcsil(t, r0) + 0.35))
                                placed.append((t0, t0 + L / tp)); t = t0 + L / tp
                            if t > r["win"] + 0.05:   # r8 v4: the last clause may start up to 0.25 s before its burst to end before the next sentence (04 u1 "on August twenty-eight." vs u2 at 18.8)
                                sh_ = t - (r["win"] + 0.05); lo_ = placed[-2][1] + 0.12 if len(placed) > 1 else 0.0
                                if sh_ > 0.25 or placed[-1][0] - sh_ < lo_: continue
                                placed[-1] = (placed[-1][0] - sh_, placed[-1][1] - sh_); t = placed[-1][1]
                            c_, d_ = cost_of(placed, B[0][0], B[-1][1])
                            c_ += sum(abs(1 - tp_) * (q[1] - q[0]) for tp_, q in zip(tem, placed))   # tempo cost as burst_align
                            if bestrun is None or c_ < bestrun[0]: bestrun = (c_, placed, d_, list(tem))
                    if bestrun and (best is None or bestrun[0] < best[0]): best = (bestrun[0], bestrun[1], bestrun[2], cl, say, segs, cfg, bestrun[3])
        rec = {"id": u["id"], "en": " ".join(words), "old_cost": round(r["base"], 2), "old_detail": r["det"], "natural_cuts": [words[j - 1] + " | " + words[j] for j in cuts_ok]}
        ok = best is not None and (best[0] <= r["base"] - a.min_gain or (r["det"]["uncovered_s"] - best[2]["uncovered_s"] >= 0.3 and best[0] <= r["base"]))   # a real coverage gain at no extra cost (04 u2 "Today,")
        rec["accepted"] = bool(ok)
        if best: rec.update({"tempo": best[7], "new_cost": round(best[0], 2), "new_detail": best[2], "clauses": best[3], "said": best[4], "cfg": best[6], "placed": [[round(p, 2) for p in q] for q in best[1]],
                             "whisper": [s[1] for s in best[5]]})
        if ok:
            _, placed, _, cl, say, segs, cfg, tem = best; import burst_align
            ps = [p for p in u["alignment"]["placements"] if p.get("placed")]; a0, a1 = span_of(u); i0, i1 = int(max(0, a0 - 0.03) * sr), int(min(D, a1 + 0.03) * sr); old = z[i0:i1].copy()
            Lv = lambda x: float(np.percentile(20 * np.log10(np.sqrt((x[:len(x) // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-9), 95))
            gain = 10 ** ((Lv(old) - Lv(np.concatenate([s[0] for s in segs]))) / 20)
            k = int(0.02 * sr); g = np.zeros(i1 - i0); g[:k] = 1 - retake.ramp(k); g[-k:] = retake.ramp(k); z[i0:i1] = old * g
            newp = []
            for c, (seg, _), (t0, t1), tp in zip(cl, segs, placed, tem):
                j0 = int(t0 * sr); s_ = burst_align.atempo(seg, sr, tp) * gain; f = int(0.01 * sr); s_[:f] *= retake.ramp(f); s_[-f:] *= retake.ramp(f)[::-1]; z[j0:j0 + len(s_)] += s_[:max(0, len(z) - j0)]
                newp.append({"burst": [round(t0, 3), round(t1, 3)], "window": [round(t0, 3), round(t1, 3)], "phrases": [len(newp)], "group_s": round(len(seg) / sr, 3),
                             "placed": [round(t0, 3), round(t0 + len(s_) / sr, 3)], "track_start": round(t0, 3), "text": c, "source_words": "", "r7_retake": True, "r8_clause": True, "atempo": tp})
            u["alignment"]["placements"] = newp; u["alignment"]["track_start"] = newp[0]["placed"][0]; u["tts_parts"] = [{"text": c} for c in cl]
            u["r8_clause"] = {k_: v for k_, v in rec.items() if k_ != "whisper"}; rec["level_gain_db"] = round(20 * np.log10(gain), 1)
        rec["seconds"] = round(time.perf_counter() - t_start, 1); rep.append(rec); print(json.dumps(rec, ensure_ascii=False), flush=True)
    for f in W.iterdir():
        if f.name in ("dubbed_24k.wav", "dubbed_16k.wav") or f.name.endswith(".manifest.json"): continue
        (O / f.name).unlink(missing_ok=True); (O / f.name).symlink_to(f.resolve())
    sf.write(str(O / "dubbed_24k.wav"), z, sr, subtype="FLOAT")
    import librosa
    sf.write(str(O / "dubbed_16k.wav"), librosa.resample(z, orig_sr=sr, target_sr=16000), 16000, subtype="PCM_16")
    man["r8_clause_takes"] = {"changed": [r_["id"] for r_ in rep if r_["accepted"]], "selected": [r_["id"] for r_ in rep]}
    (O / mp.name).write_text(json.dumps(man, indent=1, ensure_ascii=False)); (O / "clause_takes.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
