#!/usr/bin/env python3
"""r13 spec 9.2 check: do the subtitle words follow the STRETCHED dub (after atempo / placement), not the raw TTS clip?

Reference = faster-whisper large-v3 word timestamps on the final dub track (dubbed_16k.wav, the voice only). The subtitle words are
r7_subs.dub_words(manifest) (placed interval + CTC word spans mapped through the stretch). Baseline 'naive' = the same words WITHOUT the
stretch: each phrase starts at its placed start but keeps the raw TTS clip durations (= what you get if 9.2 is skipped).
Words are matched by text (difflib on normalised tokens); reported: matched share, median / p90 |start delta|, share within 150 / 300 ms.
    python scripts/r13/subs_check.py --manifest M --dub16 dubbed_16k.wav --out check.json [--lang en]"""
import argparse, difflib, json, pathlib, re, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "r7"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot"))
import r7_subs  # noqa: E402


def norm(w): return re.sub(r"[^\w']", "", w.lower())


def naive_words(man):
    """the stretch NOT applied: words at their raw TTS clip offsets from the placed start"""
    out = []
    for u in man["units"]:
        al = u.get("alignment") or {}; tparts = u.get("tts_parts") or []; spans = (u.get("tts") or {}).get("word_spans") or []
        for pl in al.get("placements", []):
            p0 = pl["placed"][0]; off = 0.0
            for pi in pl.get("phrases", []):
                if pi >= len(tparts): continue
                a, b = tparts[pi]["clip_span"]; words = tparts[pi]["text"].split(); sp = [s for s in spans if a - 0.02 <= s["start"] < b]
                if len(sp) == len(words):
                    out += [(w, p0 + off + s["start"] - a, p0 + off + s["end"] - a) for w, s in zip(words, sp)]
                else:
                    tot = sum(len(w) + 1 for w in words); t = p0 + off
                    for w in words: d = (b - a) * (len(w) + 1) / tot; out.append((w, t, t + d)); t += d
                off += b - a
    return sorted(out, key=lambda x: x[1])


def asr_words(wav, lang):
    import torch  # noqa: F401
    from faster_whisper import WhisperModel
    import run_clean_pipeline_05 as P
    m = WhisperModel(str(P.MODELS / "whisper-large-v3"), device="cuda", compute_type="int8_float16")
    segs, _ = m.transcribe(str(wav), word_timestamps=True, beam_size=5, vad_filter=False, language=lang)
    return [(w.word.strip(), float(w.start), float(w.end)) for s in segs for w in (s.words or []) if w.word.strip()]


def score(sub, ref):
    a = [norm(w[0]) for w in sub]; b = [norm(w[0]) for w in ref]; d = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal": d += [abs(sub[i1 + k][1] - ref[j1 + k][1]) for k in range(i2 - i1)]
    d = np.array(d) if d else np.array([np.nan])
    return {"sub_words": len(sub), "ref_words": len(ref), "matched": int(np.isfinite(d).sum()), "matched_share": round(float(np.isfinite(d).sum()) / max(1, len(sub)), 3),
            "median_ms": round(float(np.nanmedian(d)) * 1000), "p90_ms": round(float(np.nanpercentile(d, 90)) * 1000),
            "within_150ms": round(float(np.nanmean(d <= 0.15)), 3), "within_300ms": round(float(np.nanmean(d <= 0.30)), 3)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); ap.add_argument("--dub16", required=True)
    ap.add_argument("--lang", default="en"); ap.add_argument("--out", required=True); a = ap.parse_args()
    man = json.load(open(a.manifest)); ref = asr_words(a.dub16, a.lang)
    sub = r7_subs.dub_words(man); nv = naive_words(man)
    ats = [pl.get("atempo", 1.0) for u in man["units"] for pl in (u.get("alignment") or {}).get("placements", []) if pl.get("placed")]
    r = {"subs": score(sub, ref), "naive_no_stretch": score(nv, ref), "placements": len(ats),
         "atempo": {"min": round(min(ats), 3), "max": round(max(ats), 3), "share_ne_1": round(float(np.mean([abs(x - 1) > 0.02 for x in ats])), 3)} if ats else None}
    pathlib.Path(a.out).write_text(json.dumps(r, indent=1)); print(json.dumps(r))


if __name__ == "__main__":
    main()
