#!/usr/bin/env python3
"""r4 audio v2 (user on v1: "the pauses are illogical"): where do the dub's pauses fall in the ENGLISH text?
Whisper (large-v3, word timestamps) on the dub voice -> word times; script tokens (the synthesised unit texts, with punctuation) aligned to the
Whisper words (difflib on normalised tokens); every gap > --min-gap s between two consecutive aligned words is classified by the script:
  sentence  the previous word ends a unit or ends with . ? ! (strength 3)
  strong    ; : — – (2)        comma  , (1)        clause   the next word starts a clause (and / but / so / which / because / that ...) (0.5)
  INSIDE    anything else = a pause inside a phrase (article / preposition / adjective + noun, inside a name or number ...)
    python scripts/quality/r4_pause_logic.py --run RUN_MIX_DIR --manifest M.json [--json out.json]"""
import argparse, difflib, json, pathlib, re, sys
import numpy as np
PICK = __import__("os").environ.get("DUB_PAUSE_PICK", "r6")   # r4 = the round-4 word pick (kept for A/B of the metric)
CLAUSE = frozenset("and but so because that which who whom whose where when while if or although though since unless until whether then".split())


def norm(w): return re.sub(r"[^a-z0-9']", "", w.lower().replace("’", "'"))


def script_tokens(man):
    toks = []   # (word with punctuation, unit id, last-of-unit)
    for u in man["units"]:
        t = (u.get("tts_text_norm") or {}).get("spoken") or u.get("translation_fitted") or u.get("translation") or ""
        ws = [w for w in t.split() if norm(w)]
        toks += [(w, u["id"], i == len(ws) - 1) for i, w in enumerate(ws)]
    return toks


def whisper_words(wav):
    import torch  # noqa: F401
    from faster_whisper import WhisperModel
    root = pathlib.Path(__file__).resolve().parents[2]; mdl = str(root / "models" / "whisper-large-v3")
    m = WhisperModel(mdl, device="cuda", compute_type="int8_float16")
    segs, _ = m.transcribe(str(wav), language="en", beam_size=5, word_timestamps=True, vad_filter=False, condition_on_previous_text=False)
    return [(w.word.strip(), float(w.start), float(w.end)) for s in segs for w in (s.words or []) if norm(w.word)]


def classify(prev, nxt):
    w, _, last = prev
    if last or re.search(r"[.?!…]['\"»)]*$", w): return "sentence", 3
    if re.search(r"[;:—–]['\"»)]*$", w) or w.endswith("-"): return "strong", 2
    if re.search(r",['\"»)]*$", w): return "comma", 1
    if nxt and norm(nxt[0]) in CLAUSE: return "clause", 0.5
    return "INSIDE", 0


def analyse(run, man, min_gap=0.12):
    W = whisper_words(pathlib.Path(run) / "dubbed_16k.wav"); S = script_tokens(man)
    sm = difflib.SequenceMatcher(None, [norm(s[0]) for s in S], [norm(w[0]) for w in W], autojunk=False); w2s = {}
    for a, b, n in sm.get_matching_blocks():
        for k in range(n): w2s[b + k] = a + k
    import soundfile as sf
    y, sr = sf.read(str(pathlib.Path(run) / "dubbed_16k.wav"), dtype="float32"); y = y if y.ndim == 1 else y.mean(1)
    h = int(0.01 * sr); fr = 20 * np.log10(np.sqrt((y[: len(y) // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-9)
    sil = fr < np.percentile(fr[fr > -80], 90) - 35 if (fr > -80).any() else fr < -60          # silent 10 ms frames (35 dB below the voice)
    d = np.diff(np.concatenate([[0], sil.astype(np.int8), [0]])); gaps = [(a * 0.01, b * 0.01) for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)) if (b - a) * 0.01 > min_gap]
    starts = np.array([w[1] for w in W]); rows = []
    for g0, g1 in gaps:
        if PICK == "r4":
            j = int(np.searchsorted(starts, g1 - 0.05)) - 1     # the last word that starts before the gap ends (Whisper stretches word ENDS over silence)
        else:   # r6: Whisper also pulls the NEXT word's start back into the silence; the r4 pick then classified (next word | word after) = INSIDE.
            # The word before a gap is the last one that starts before the gap STARTS (+ 50 ms tolerance for a start on the gap edge).
            j = int(np.searchsorted(starts, g0 + 0.05)) - 1
        if j < 0 or j + 1 >= len(W) or j not in w2s or j + 1 not in w2s or w2s[j + 1] != w2s[j] + 1: continue
        i = w2s[j]; cls, st = classify(S[i], S[i + 1] if i + 1 < len(S) else None)
        rows.append({"t": round(g0, 2), "gap_s": round(g1 - g0, 3), "class": cls, "strength": st, "before": S[i][0], "after": S[i + 1][0] if i + 1 < len(S) else "", "unit": S[i][1]})
    inside = [r for r in rows if r["class"] == "INSIDE"]
    by = {c: [r["gap_s"] for r in rows if r["class"] == c] for c in ("sentence", "strong", "comma", "clause", "INSIDE")}
    return {"words_whisper": len(W), "aligned": len(w2s), "pauses": len(rows), "pauses_inside_phrase": len(inside), "inside_phrase_s": round(sum(r["gap_s"] for r in inside), 2),
            "median_gap_by_class": {c: (round(float(np.median(v)), 3) if v else None) for c, v in by.items()}, "count_by_class": {c: len(v) for c, v in by.items()},
            "intra_sentence_pauses": sum(1 for r in rows if r["class"] != "sentence"), "rows": rows}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--json", default=None)
    ap.add_argument("--min-gap", type=float, default=0.12); a = ap.parse_args()
    r = analyse(a.run, json.loads(pathlib.Path(a.manifest).read_text()), a.min_gap)
    if a.json: pathlib.Path(a.json).write_text(json.dumps(r, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in r.items() if k != "rows"}))
    for x in r["rows"]:
        if x["class"] == "INSIDE": print(f"  INSIDE {x['t']:7.2f}s {x['gap_s']:.2f}s  ...{x['before']} | {x['after']}...")


if __name__ == "__main__":
    main()
