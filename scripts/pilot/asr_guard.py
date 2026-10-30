"""r7 spec 6.2 / 5.2: Whisper hallucination guard and Silero-VAD segmentation of the ASR input (run_clean_pipeline_05.run_asr).

faster-whisper drops a 30-s window only when no_speech_prob > 0.6 AND avg_logprob < -1.0: a CONFIDENT hallucination over music / room tone
("Субтитры сделал DimaTorzok", "Продолжение следует", a looped phrase) survives into the transcript. guard() drops what Whisper flags
(no_speech_prob > 0.6, a stock subtitle / outro phrase, a phrase looped >= 3 times) only where Silero VAD hears little speech
(coverage < --asr-hallu-cov of the span), so real speech under music is never dropped; everything flagged but kept is recorded.

vad_chunks(): ASR input split into 45-90 s pieces, cut only in the middle of a Silero pause >= 0.5 s whose level is <= -40 dBFS
(fallbacks recorded). A file <= max_s is one chunk = the whole-file call, bit-identical to the unsegmented path."""
from __future__ import annotations

import re

import numpy as np

PHRASES = [r"субтитр\w* (сделал|созда|подготов|делал|редакт|корректор)\w*", r"редактор субтитров", r"корректор\b", r"продолжение следует",
           r"спасибо за (просмотр|внимание)", r"подписыва\w* на (канал|наш)", r"ставьте лайк", r"до (новых встреч|скорой встречи)",
           r"dimatorzok", r"amara\.org", r"thanks? (you )?for watching", r"subtitles? (by|made)", r"please subscribe"]
PHRASE_RE = re.compile("|".join(PHRASES), re.I)


def coverage(s: float, e: float, speech: list[dict]) -> float:
    if e <= s: return 0.0
    return sum(max(0.0, min(e, v["end"]) - max(s, v["start"])) for v in speech) / (e - s)


def _norm(w: str) -> str:
    return re.sub(r"[^\w']+", "", w.lower())


def loops(words: list[dict], min_rep: int = 3, max_n: int = 8) -> set[int]:
    """indices of the words that repeat an n-gram (n <= max_n) which already occurred right before, in runs of >= min_rep copies (the first copy stays)."""
    t = [_norm(w["word"]) for w in words]; drop: set[int] = set(); i = 0
    while i < len(t):
        best = None
        for n in range(1, max_n + 1):
            g = t[i:i + n]
            if len(g) < n or not any(g): break
            k = 1
            while t[i + k * n:i + (k + 1) * n] == g: k += 1
            if k >= min_rep and (best is None or k * n > best[0] * best[1]): best = (k, n)
        if best:
            k, n = best; drop.update(range(i + n, i + k * n)); i += k * n
        else:
            i += 1
    return drop


def guard(segments: list[dict], speech: list[dict], nsp: float = 0.6, cov_max: float = 0.5) -> tuple[list[dict], dict]:
    """segments carry no_speech_prob / avg_logprob / compression_ratio (run_asr). Returns (kept segments, report)."""
    rep = {"rules": {"no_speech_prob_gt": nsp, "vad_coverage_lt": cov_max, "phrases": len(PHRASES), "loop_min_repeats": 3}, "dropped": [], "trimmed": [], "flagged_kept": []}
    out = []
    # a whole segment that repeats the previous kept one word for word (the cross-segment loop) counts as a loop
    for sg in segments:
        cov = coverage(sg["start"], sg["end"], speech); why = []
        if sg.get("no_speech_prob", 0.0) > nsp: why.append("no_speech")
        if PHRASE_RE.search(sg["text"]): why.append("phrase")
        if out and sg["text"] and _norm(sg["text"]) == _norm(out[-1]["text"]) and len(sg["text"].split()) >= 1: why.append("repeat_segment")
        rec = {"id": sg["id"], "start": sg["start"], "end": sg["end"], "text": sg["text"], "no_speech_prob": sg.get("no_speech_prob"),
               "avg_logprob": sg.get("avg_logprob"), "compression_ratio": sg.get("compression_ratio"), "vad_coverage": round(cov, 3), "why": why}
        if why and cov < cov_max:
            rep["dropped"].append(rec); continue
        if why: rep["flagged_kept"].append(rec)
        ws = sg.get("words") or []; li = loops(ws)
        if li:
            lw = [i for i in sorted(li) if coverage(ws[i]["start"], ws[i]["end"], speech) < cov_max]
            if lw:
                keep = [w for i, w in enumerate(ws) if i not in set(lw)]
                rep["trimmed"].append({**rec, "why": ["loop"], "removed": "".join(ws[i]["word"] for i in lw).strip()})
                if not keep: continue
                sg = {**sg, "words": keep, "text": "".join(w["word"] for w in keep).strip(), "start": keep[0]["start"], "end": keep[-1]["end"]}
            else:
                rep["flagged_kept"].append({**rec, "why": ["loop"]})
        out.append(sg)
    return out, rep


def vad_chunks(audio: np.ndarray, sr: int, speech: list[dict], min_s: float = 45.0, max_s: float = 90.0, pause_s: float = 0.5, db: float = -40.0) -> tuple[list[tuple[float, float]], list[dict]]:
    """[(t0, t1)] covering the file; cut ONLY at the midpoint of a Silero pause >= pause_s whose RMS (over the pause) <= db dBFS: the longest such
    pause in [t0 + min_s, t0 + max_s] that leaves >= min_s for the rest, else in [t0 + min_s, t0 + max_s], else the first one after t0 + max_s
    (a piece > max_s), else no further cut (01: voice-over on music, no pause >= 0.5 s from 45 to 90 s). Every cut / miss is in the report."""
    D = len(audio) / sr; gaps = []; prev = 0.0
    for v in speech + [{"start": D, "end": D}]:
        if v["start"] - prev > 0: gaps.append((prev, v["start"]))
        prev = max(prev, v["end"])

    def lvl(a, b):
        x = audio[int(a * sr):int(b * sr)].astype(np.float64)
        return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9) if len(x) else -120.0

    chunks, cuts, t0 = [], [], 0.0
    while D - t0 > max_s:
        lo, hi = t0 + min_s, t0 + max_s; hi_bal = min(hi, D - min_s)   # hi_bal: the rest stays >= min_s
        inwin = lambda l, h: [(a, b) for a, b in gaps if l <= (a + b) / 2 <= h and a > 0 and b < D]
        ok = lambda g: g[1] - g[0] >= pause_s and lvl(*g) <= db
        pick = None   # the pause rule outranks the piece length: a long piece beats a cut in a loud / short pause (faster-whisper handles any length)
        for kind, cand in (("pause", [g for g in inwin(lo, hi_bal) if ok(g)]), ("pause_short_tail", [g for g in inwin(lo, hi) if ok(g)])):
            if cand: pick = (kind, max(cand, key=lambda g: g[1] - g[0])); break
        if not pick:
            late = [g for g in inwin(hi, D) if ok(g)]
            if late: pick = ("pause_long_piece", min(late))
        if not pick:
            cuts.append({"t": None, "kind": "no_pause", "searched_from": round(lo, 3)}); break
        kind, (a, b) = pick; t = round((a + b) / 2, 3)
        cuts.append({"t": t, "kind": kind, "pause": [round(a, 3), round(b, 3)], "pause_s": round(b - a, 3), "pause_db": round(lvl(a, b), 1),
                     "in_speech": coverage(t - 0.05, t + 0.05, speech) > 0})
        chunks.append((t0, t)); t0 = t
    chunks.append((t0, round(D, 3)))
    return chunks, cuts
