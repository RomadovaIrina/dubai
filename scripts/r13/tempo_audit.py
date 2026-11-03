#!/usr/bin/env python3
"""r13 spec 7.3 audit: the EFFECTIVE tempo of every dubbed phrase against the spec corridor x0.75-1.25, measured, not reconstructed.

  natural  the unit's raw TTS take: the CTC word spans the pipeline stored (units[].tts.word_spans on tts_raw_file), or, where the r8 chain
           re-voiced the unit (clause takes / retake: no spans), a fresh CTC alignment of the unit's final text on its raw take file when it exists
  final    the same words in the FINAL dub track (dubbed_16k.wav): CTC forced alignment of each placement's text inside its placed interval
  tempo    natural duration / final duration of the matched words (> 1 = sped up), per placement (first .. last matched word) and per 4-word
           sliding window (a local speed-up inside a phrase: "the voice speeds up in one place")
Also listed: the placement's recorded atempo, the r8 inner-pause closing factors, the clause-take tempo. librosa time-stretch: not used by
the current chain (ffmpeg atempo only; grep), reported as such.
    python scripts/r13/tempo_audit.py --run $A/r9_<id>/<id> [--out audit.json]"""
import argparse, difflib, json, pathlib, sys
import numpy as np, soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "pilot"))
LO, HI = 0.75, 1.25


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--out"); ap.add_argument("--win", type=int, default=4)
    a = ap.parse_args(); D = pathlib.Path(a.run)
    mp = next(iter(sorted(D.glob("*_r4b.manifest.json")))); man = json.loads(mp.read_text())
    from tts_align import CtcAligner
    from tts_qa import norm
    al = CtcAligner.standalone("cpu")
    dub, sr = sf.read(str(D / "dubbed_16k.wav"), dtype="float32")
    rows, wins, rates = [], [], []
    for u in man["units"]:
        pls = [p for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed") and p.get("text")]
        if not pls: continue
        spans = (u.get("tts") or {}).get("word_spans") or []; nat = [(s["word"], s["start"], s["end"]) for s in spans]; src = "stored_spans"
        raw = u.get("tts_raw_file")
        full = " ".join(p["text"] for p in pls)
        if raw and pathlib.Path(raw).exists():
            # the stored spans belong to the raw take's text; if the final text differs (r8 re-voice), align the final text on the raw take
            if not nat or difflib.SequenceMatcher(None, [w for w, _, _ in nat], norm(full)).ratio() < 0.8:
                y, s0 = sf.read(raw, dtype="float32"); sp = al.align(y if y.ndim == 1 else y.mean(1), s0, full)
                if sp: nat = [(s["word"], s["start"], s["end"]) for s in sp]; src = "realigned_raw_take"
        for p in pls:
            p0, p1 = p["placed"]; a0 = max(0.0, p0 - 0.25); seg = dub[int(a0 * sr):int((p1 + 0.25) * sr)]
            fin = al.align(seg, sr, p["text"]) if len(seg) > sr // 5 else None
            rec = {"unit": u["id"], "placed": [round(p0, 2), round(p1, 2)], "text": p["text"][:80], "atempo_recorded": p.get("atempo"),
                   "close_inner": p.get("r8_close_inner"), "clause_tempo": (u.get("r8_clause") or {}).get("tempo"), "natural_src": src}
            if not fin or not nat:
                rec["tempo"] = None; rec["why"] = "no CTC alignment (final)" if not fin else "no natural reference"; rows.append(rec); continue
            fw = [(w["word"], a0 + w["start"], a0 + w["end"]) for w in fin]
            from dur_translate import syllables   # English syllable heuristic of the production duration-aware translation
            fsd = sum(e - b for _, b, e in fw)
            rec["rate_sps"] = round(sum(syllables(w) for w, _, _ in fw) / fsd, 2) if fsd > 0.3 else None   # final speech rate, syllables / s of speech
            for i in range(len(fw) - a.win + 1):
                q = fw[i:i + a.win]; d_ = sum(e - b for _, b, e in q)
                if d_ > 0.3: rates.append({"unit": u["id"], "sps": round(sum(syllables(w) for w, _, _ in q) / d_, 2), "t0": round(q[0][1], 2), "t1": round(q[-1][2], 2), "words": " ".join(w for w, _, _ in q)})
            sm = difflib.SequenceMatcher(None, [w for w, _, _ in fw], [w for w, _, _ in nat], autojunk=False); pairs = []
            for op, i1, i2, j1, j2 in sm.get_opcodes():
                if op == "equal": pairs += [(fw[i1 + k], nat[j1 + k]) for k in range(i2 - i1)]
            rec["matched_words"] = f"{len(pairs)}/{len(fw)}"
            if len(pairs) < 3:
                rec["tempo"] = None; rec["why"] = "too few matched words"; rows.append(rec); continue
            fd = pairs[-1][0][2] - pairs[0][0][1]; nd = pairs[-1][1][2] - pairs[0][1][1]
            rec["tempo_span"] = round(nd / fd, 3) if fd > 0.1 else None   # first..last word, pauses included (pause insertion / closing moves it)
            fs = sum(x[0][2] - x[0][1] for x in pairs); ns = sum(x[1][2] - x[1][1] for x in pairs)
            rec["tempo"] = round(ns / fs, 3) if fs > 0.1 else None          # SPEECH only (sum of word durations): what atempo does to the voice
            loc = []
            for i in range(len(pairs) - a.win + 1):
                q = pairs[i:i + a.win]; f_ = sum(x[0][2] - x[0][1] for x in q); n_ = sum(x[1][2] - x[1][1] for x in q)
                if f_ > 0.3 and n_ > 0.3: loc.append((round(n_ / f_, 3), round(q[0][0][1], 2), round(q[-1][0][2], 2), " ".join(x[0][0] for x in q)))
            if loc:
                hi = max(loc); lo = min(loc); rec["local_max"] = hi[:3] + (hi[3],); rec["local_min"] = lo[:3] + (lo[3],)
                wins += [{"unit": u["id"], "tempo": t, "t0": t0, "t1": t1, "words": w} for t, t0, t1, w in loc]
            rows.append(rec)
    ok = [r for r in rows if r.get("tempo")]
    viol = [r for r in ok if not LO <= r["tempo"] <= HI]
    lviol = sorted([w for w in wins if not LO <= w["tempo"] <= HI], key=lambda w: -abs(np.log(w["tempo"])))
    rec_v = [r for r in rows if r.get("atempo_recorded") and not LO <= r["atempo_recorded"] <= HI]
    res = {"run": str(D), "corridor": [LO, HI], "placements": len(rows), "measured": len(ok),
           "tempo_p5_p50_p95": np.round(np.percentile([r["tempo"] for r in ok], [5, 50, 95]), 3).tolist() if ok else None,
           "phrase_violations": viol, "span_violations": [r for r in ok if r.get("tempo_span") and not LO <= r["tempo_span"] <= HI], "local_violations": lviol[:20], "local_windows": len(wins), "recorded_atempo_violations": rec_v,
           "librosa_time_stretch": "not used by the current chain (ffmpeg atempo only)", "rows": rows}
    pr = [r["rate_sps"] for r in rows if r.get("rate_sps")]; med = float(np.median(pr)) if pr else None
    if med:   # where the voice is FASTEST, whatever the cause (atempo, a fast Chatterbox take, closed pauses): 4-word windows vs the video median
        res["rate_median_sps"] = round(med, 2); res["rate_peaks"] = sorted(rates, key=lambda w: -w["sps"])[:10]
        res["phrase_rate"] = sorted([{"unit": r["unit"], "placed": r["placed"], "sps": r["rate_sps"], "x_median": round(r["rate_sps"] / med, 2), "tempo": r.get("tempo"), "text": r["text"]}
                                     for r in rows if r.get("rate_sps")], key=lambda r: -r["sps"])
    if a.out: pathlib.Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    if med:
        print(f"   speech rate median {med:.2f} syll/s; fastest phrases: " + "; ".join(f"u{r['unit']} {r['placed'][0]}s {r['sps']} (x{r['x_median']}, atempo-tempo {r['tempo']})" for r in res["phrase_rate"][:4]))
        print("   fastest 4-word windows: " + "; ".join(f"{w['t0']}-{w['t1']}s {w['sps']} {w['words']!r}" for w in res["rate_peaks"][:5]))
    print(f"== {D.name}: {len(ok)}/{len(rows)} placements measured, tempo p5/p50/p95 {res['tempo_p5_p50_p95']}; outside x{LO}-{HI}: "
          f"{len(viol)} phrases, {len(lviol)} local 4-word windows (of {len(wins)}); recorded atempo outside: {len(rec_v)}")
    for r in viol: print(f"   PHRASE u{r['unit']} {r['placed']} speech tempo {r['tempo']} span {r.get('tempo_span')} (recorded atempo {r['atempo_recorded']}) {r['text'][:60]!r}")
    sv = [r for r in ok if r.get("tempo_span") and not LO <= r["tempo_span"] <= HI]
    for r in sv: print(f"   SPAN   u{r['unit']} {r['placed']} span tempo {r['tempo_span']} (speech {r['tempo']}): pauses changed {r['text'][:60]!r}")
    for w in lviol[:8]: print(f"   LOCAL  u{w['unit']} {w['t0']}-{w['t1']} s tempo {w['tempo']} {w['words']!r}")


if __name__ == "__main__":
    main()
