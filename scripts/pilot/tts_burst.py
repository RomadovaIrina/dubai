"""Burst-mode TTS session: synthesis of every part with duration-aware fitting (Fix 3) and per-part intelligibility QA + retry (Fix 4).

Order of preference when a part does not fit its burst window (first_res review: atempo above ~1.15-1.2 measurably hurts intelligibility, 1.3-1.6 was the
normal path before):  concise duration-aware re-translation of THAT part  ->  regenerate TTS for that part  ->  only then a moderate atempo (caps in PlaceOpts:
soft 1.15, hard 1.2).  Every TTS part goes through  expected text -> TTS -> independent ASR -> score  and is retried alone (new seed, then new seed + lower
temperature) when it is not intelligible; the whole video is never re-synthesised.

The caller (run_clean_pipeline_05.py) owns the translation stage; this module receives the live Qwen `llm` for the re-translation and the plans/units of the burst path.
"""
from __future__ import annotations

import pathlib
import re
import sys
import time

import numpy as np
import soundfile as sf

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "quality"))
from e1_burst_align import SR16, burst_fit_ratios, ff, vad_intervals  # noqa: E402

TTS_SR = 24000
REWRITE_SYSTEM = "You are a professional dubbing script adapter. You answer with the rewritten phrase only, without quotes or explanations."


def words_of(text: str) -> int:
    return len(re.findall(r"[\w']+", text))


def energy_onset(y: np.ndarray, sr: int, rel_db: float = -28.0, frames: int = 3, hop_s: float = 0.01) -> float:
    """First time (s) where the signal stays within rel_db of the clip's loud level (90th percentile of 10 ms RMS) for `frames` consecutive frames."""
    h = int(hop_s * sr); n = len(y) // h
    if n < frames + 1:
        return 0.0
    e = 20 * np.log10(np.array([np.sqrt(np.mean(y[i * h:(i + 1) * h] ** 2) + 1e-12) for i in range(n)]) + 1e-9)
    ok = e > np.percentile(e, 90) + rel_db
    for i in range(n - frames):
        if ok[i:i + frames].all():
            return i * hop_s
    return 0.0


def speech_span(y: np.ndarray, sr: int, work: pathlib.Path, tag: str, preroll_max: float = 0.25, onset_margin: float = 0.03) -> tuple[float, float]:
    """Speech span of one TTS part: Silero VAD first start - 40 ms .. last end + 60 ms (clamped), with onset protection (Fix 5): Silero often reports the start
    of a word late (first_res review: 50-60 ms of the first consonant were cut in 20 of 77 parts, 160 ms = "Ba" of "Balcony." in one), so the start is moved
    back to the energy onset (- onset_margin) when that is earlier, by at most preroll_max. The post-roll is NOT changed (extending it helped nothing)."""
    t16 = work / f"tts16_{tag}.wav"
    sf.write(str(work / f"_span_{tag}.wav"), y, sr, subtype="PCM_16")
    ff(["-i", str(work / f"_span_{tag}.wav"), "-ar", str(SR16), "-ac", "1", "-c:a", "pcm_s16le", str(t16)])
    raw = vad_intervals(t16) or [(0.0, len(y) / sr)]
    t16.unlink(); (work / f"_span_{tag}.wav").unlink()
    s_vad = max(0.0, raw[0][0] - 0.04)
    s_ = max(0.0, s_vad - preroll_max, min(s_vad, energy_onset(y, sr) - onset_margin))
    return s_, min(len(y) / sr, raw[-1][1] + 0.06)


def edge_fade(seg: np.ndarray, sr: int, in_ms: float, out_ms: float) -> np.ndarray:
    """Linear fade-in / fade-out of a cut part (an inner cut that lands on a still-voiced sound gets a 20 ms fade-out instead of a click / hard stop)."""
    seg = np.array(seg, dtype=np.float32, copy=True)
    for ms, head in ((in_ms, True), (out_ms, False)):
        k = min(len(seg) // 2, int(ms / 1000 * sr))
        if k > 0:
            ramp = np.linspace(0.0, 1.0, k, dtype=np.float32)
            if head: seg[:k] *= ramp
            else: seg[-k:] *= ramp[::-1]
    return seg


def flow_merge_bursts(p: dict, flow_gap: float) -> int:
    """--sentence-flow-gap (2026-10-13, "предложения не связаны, размазаны по времени"): inside ONE sentence, source bursts separated by a
    pause shorter than flow_gap s become one burst (union window), so the translated sentence is spoken as one connected phrase instead of
    being cut at every hesitation of the source (04 "test how our | 1.1 s | application"). Longer pauses stay cut points (still placed by the
    cut DP, at natural boundaries with --pause-marks). Returns the number of merges; p["bursts"/"groups"/"group_counts"] are updated."""
    if flow_gap <= 0 or len(p["bursts"]) < 2:
        return 0
    b, g, c = [list(p["bursts"][0])], [p["groups"][0]], [p["group_counts"][0]]; merged = 0
    for bs, gs, cs in zip(p["bursts"][1:], p["groups"][1:], p["group_counts"][1:]):
        if bs[0] - b[-1][1] < flow_gap:
            b[-1][1] = bs[1]; g[-1] = g[-1] + " " + gs; c[-1] += cs; merged += 1
        else:
            b.append(list(bs)); g.append(gs); c.append(cs)
    p["bursts"] = [tuple(x) for x in b]; p["groups"] = g; p["group_counts"] = c
    return merged


def adaptive_fade_ms(y: np.ndarray, sr: int, a: float, b: float, inner_head: bool, inner_tail: bool, ref_db: float) -> tuple[float, float]:
    """--cut-fade adaptive (2026-10-13): 5 ms declick fades, but an INNER cut that still lands in voiced sound (the 10 ms next to the cut within
    20 dB of the clip's speech level: CTC word ends are early, "Well, | and" is one breath) gets a 25 ms fade-out / 8 ms fade-in: the word
    decays instead of stopping dead. Sentence starts / ends keep 5 ms (onsets are protected separately)."""
    def lvl(t0, t1):
        i0, i1 = max(0, int(t0 * sr)), min(len(y), int(t1 * sr))
        return 20 * np.log10(np.sqrt(np.mean(y[i0:i1] ** 2)) + 1e-9) if i1 > i0 else -120.0
    fin = 8.0 if inner_head and lvl(a, a + 0.01) > ref_db - 20 else 5.0
    fout = 25.0 if inner_tail and lvl(b - 0.01, b) > ref_db - 20 else 5.0
    return fin, fout


def gap_voice(y: np.ndarray, sr: int, spans: list[dict] | None, min_s: float = 0.10, rel_db: float = -15.0, pre: float = 0.04, post: float = 0.10) -> list[tuple]:
    """--tts-gap-qa (2026-10-13, "посторонние звуки"): speech-level sound BETWEEN the CTC-aligned words of a generation, i.e. a syllable / drawl /
    filler Chatterbox inserted that is not in the text (04 u03: "popular with <0.45 s held vowel> bloggers", CTC hears "with the bloggers",
    Whisper normalises it away so the ASR QA scores it 0.09 = fine). Returns [(start_s, end_s, level_db_rel_speech)] of runs >= min_s that lie
    outside [word start - pre, word end + post] of every word and inside the first / last word, within rel_db of the clip's speech level."""
    if not spans or len(spans) < 2:
        return []
    h = int(0.01 * sr); nf = len(y) // h
    if nf < 3:
        return []
    fdb = 20 * np.log10(np.sqrt(np.mean(y[: nf * h].reshape(nf, h) ** 2, axis=1)) + 1e-9); voiced = fdb > -60
    ref = float(np.percentile(fdb[voiced], 90)) if voiced.any() else -20.0
    cov = np.zeros(nf, bool)
    for w in spans:
        cov[max(0, int((w["start"] - pre) / 0.01)):min(nf, int((w["end"] + post) / 0.01) + 1)] = True
    lo, hi = int(spans[0]["start"] / 0.01), min(nf, int(spans[-1]["end"] / 0.01))
    loud = (fdb > ref + rel_db) & ~cov; out = []; i = lo
    while i < hi:
        if loud[i]:
            j = i
            while j < hi and loud[j]: j += 1
            if (j - i) * 0.01 >= min_s: out.append((round(i * 0.01, 2), round(j * 0.01, 2), round(float(fdb[i:j].max() - ref), 1)))
            i = j
        else: i += 1
    return out


def concise_rewrite(llm, src_name: str, tgt_name: str, source_group: str, text: str, max_words: int, seconds: float) -> str | None:
    """Ask Qwen for a shorter natural spoken version of ONE part. Returns None when the answer is unusable (empty, not shorter, loses a number)."""
    from burst_aware_audio import missing_numbers
    prompt = (f"The {src_name} phrase \"{source_group}\" is spoken in {seconds:.1f} seconds. Its current {tgt_name} dubbing text is:\n\"{text}\"\n\n"
              f"Rewrite the {tgt_name} text as a concise, natural, easy-to-pronounce spoken phrase of at most {max_words} words that keeps the meaning and every number. "
              f"Do not add information. Answer with the phrase only.")
    r = llm.create_chat_completion([{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": prompt}], max_tokens=96, temperature=0.0)
    new = " ".join(r["choices"][0]["message"]["content"].split()).strip().strip('"«»“”')
    if not new or "\n" in new or words_of(new) > max_words or words_of(new) >= words_of(text) or missing_numbers(text, new):
        return None
    return new


PAUSE_SYSTEM = "You mark natural pause points in spoken English. You answer with the marked sentence only."
PAUSE_SHOTS = [   # generic examples, deliberately NOT taken from the five test videos (no leakage into the evaluation)
    ("After the meeting we can finally discuss the new budget for next year.", "After the meeting | we can finally discuss | the new budget for next year."),
    ("My older brother bought a red bicycle yesterday.", "My older brother | bought a red bicycle | yesterday."),
    ("If the weather stays warm, we will go to the lake on Sunday.", "If the weather stays warm, | we will go to the lake | on Sunday."),
    ("The children quickly put on their winter jackets and ran outside.", "The children quickly put on their winter jackets | and ran outside."),
    ("This little shop sells fresh bread every morning.", "This little shop | sells fresh bread | every morning.")]


def pause_marks(llm, text: str) -> set[int] | None:
    """--pause-marks qwen (2026-10-13, "логичная расстановка пауз"): ask Qwen where a speaker can naturally pause inside the sentence (between
    phrases / clauses, never inside a noun phrase, between a verb and its object / particle or inside a fixed expression). Returns the 0-based
    indices of the words AFTER which a pause is natural, or None when the answer changed the words (then the function-word rule is used).
    The function-word list missed "has a small | refrigerator", "and possible | opportunities", "here | comes", "already | allow" (q0 run)."""
    words = text.split()
    if len(words) < 3:
        return set()
    msgs = [{"role": "system", "content": PAUSE_SYSTEM}]
    for a, b in PAUSE_SHOTS:
        msgs += [{"role": "user", "content": f"Mark with | where a speaker can naturally pause:\n{a}"}, {"role": "assistant", "content": b}]
    msgs.append({"role": "user", "content": f"Mark with | where a speaker can naturally pause:\n{text}"})
    t0 = time.perf_counter()
    try:
        out = llm.create_chat_completion(msgs, max_tokens=len(words) * 4 + 32, temperature=0.0)["choices"][0]["message"]["content"]
    except Exception:
        return None
    finally:
        JUDGE_STATS["pause_marks_s"] = JUDGE_STATS.get("pause_marks_s", 0.0) + time.perf_counter() - t0
    toks = out.replace("|", " | ").split(); got, marks = [], set()
    for t in toks:
        if t == "|":
            if got: marks.add(len(got) - 1)
        else:
            got.append(t)
    if [w.strip('"«»“”') for w in got] != [w.strip('"«»“”') for w in words]:
        return None
    marks.discard(len(words) - 1)
    return marks


JUDGE_STATS = {"calls": 0, "seconds": 0.0}   # Qwen judge cost of the fit-up path (reset per run_tts_phrase call)
FILLER = re.compile(r"\b(um+|uh+|er+|erm|you know|basically|actually|literally|i mean|sort of)\b", re.I)
JUDGE_SYSTEM = "You are a strict bilingual reviewer of dubbing scripts. You answer with JSON only."


_APPEND_FUNC = frozenset("a an the and or but so to of for in on at by with from as is are was were be been it its this that these those i we you he she they me us him them my our your their will would can could do does did have has had not no very just also there here".split())


def appended_clause(old: str, new: str) -> str | None:
    """Deterministic fit-up guard (2026-10-12 review, no Qwen call): a fuller rewrite that APPENDS a new clause at the end ("... video shooting
    to see how it works", "With proper use" -> "When used correctly, it is.") or ends on a dangling auxiliary ("..., it is.") is padding, not a
    fuller translation. Returns the reason or None. Insertions inside the sentence (spelled-out subjects, full verb forms) are not affected."""
    import difflib
    tok = lambda t: re.findall(r"[a-z0-9']+", t.lower().replace("’", "'"))
    o, n = tok(old), tok(new)
    if not o or not n:
        return None
    tag, i1, i2, j1, j2 = difflib.SequenceMatcher(None, o, n, autojunk=False).get_opcodes()[-1]
    if tag in ("insert", "replace") and j2 == len(n) and (tag == "insert" or i2 == len(o)):
        run = n[j1:j2]; novel = [w for w in run if w not in o and w not in _APPEND_FUNC]
        if len(run) >= 2 and len(novel) >= 2 and (j2 - j1) - (i2 - i1) >= 2:
            return f"appended clause '{' '.join(run)}'"
    if re.search(r"\b(it|that|this|there)\s+(is|was|does|did|will|can)[.!…]*\s*$", new.strip().lower()) and not re.search(r"\b(is|was|does|did|will|can)[.!…]*\s*$", old.strip().lower()):
        return "dangling auxiliary ending"
    return None


def faithful_check(llm, src_name: str, tgt_name: str, source_text: str, old_text: str, new_text: str) -> tuple[bool, list[str]]:
    """Fit-UP guard: Qwen lists every fact / detail / quality / outcome of the proposed fuller wording that the source does not state or clearly imply.
    Returns (faithful, added_items); an unparsable answer counts as NOT faithful (the rewrite is then rejected, never silently accepted)."""
    prompt = (f"{src_name} source sentence: \"{source_text}\"\nCurrent {tgt_name} translation: \"{old_text}\"\nProposed fuller {tgt_name} translation: \"{new_text}\"\n\n"
              f"Does the proposed translation state any NEW fact, event, outcome, quality or claim that the {src_name} source does not say? "
              f"Synonyms, paraphrases, a more literal rendering of the source words, spelled-out pronouns and natural connectors are NOT additions. "
              f"Answer with JSON only: {{\"added\": [\"...\"]}} quoting each genuinely new element, or {{\"added\": []}} if nothing new is stated.")
    tj = time.perf_counter()
    r = llm.create_chat_completion([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}], max_tokens=200, temperature=0.0)
    JUDGE_STATS["calls"] += 1; JUDGE_STATS["seconds"] += time.perf_counter() - tj
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", r["choices"][0]["message"]["content"].strip())
    m_ = re.search(r"\{.*\}", txt, re.S)
    try:
        import json as _json
        obj = _json.loads(m_.group(0) if m_ else txt); added = [str(x) for x in (obj.get("added") or []) if str(x).strip()]
        return (len(added) == 0), added
    except Exception:
        return False, [f"judge answer unparsable: {txt[:120]}"]


def fuller_rewrite(llm, src_name: str, tgt_name: str, source_text: str, text: str, target_words: int, source_seconds: float, tts_seconds: float, min_words: int | None = None, attempts: int = 3,
                   guard: bool = False) -> tuple[str | None, str]:
    """Fit-UP (2026-10-02): ask Qwen for a FULLER natural spoken version of ONE sentence whose synthesized speech is much shorter than the source speech.
    Same meaning, every number / name / fact kept, same style, no added information, no filler words. The word budget is explicit (min..max) and a
    too-short / too-long answer is fed back once. Returns (text | None, reason)."""
    from burst_aware_audio import missing_numbers
    nw_old = words_of(text); lo = max(nw_old + 1, int(min_words or round(0.85 * target_words))); hi = target_words + 4
    prompt = (f"The {src_name} sentence \"{source_text}\" is spoken in {source_seconds:.1f} seconds in a video. Its current {tgt_name} dubbing translation is:\n\"{text}\"\n\n"
              f"Spoken aloud this translation lasts only {tts_seconds:.1f} seconds ({nw_old} words), too short for the speaker's lip movement, because it compresses the {src_name} sentence. "
              f"Write a fuller {tgt_name} translation of the SAME {src_name} sentence in natural spoken style, {lo} to {hi} words (it MUST be longer than the current {nw_old} words): "
              f"render every word and clause of the {src_name} sentence (nothing the speaker says may be dropped or shortened), keep every number, name and fact and the same tone, "
              f"spell out pronouns and subjects, use full verb forms instead of contractions. Do NOT add any fact, outcome, quality or idea that the speaker does not say, "
              f"and do not use filler words (um, you know, basically, actually). Answer with the sentence only.")
    msgs = [{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": prompt}]; why = "no answer"
    for _ in range(attempts + 1):
        r = llm.create_chat_completion(msgs, max_tokens=220, temperature=0.0)
        raw = r["choices"][0]["message"]["content"]; new = " ".join(raw.split()).strip().strip('"«»“”'); nw_new = words_of(new)
        if not new: why = "empty"
        elif new.lower() == text.lower(): why = "identical"
        elif nw_new <= nw_old: why = f"not longer ({nw_new} <= {nw_old} words)"
        elif nw_new > hi + 2: why = f"too long ({nw_new} > {hi + 2} words)"
        elif missing_numbers(text, new) or missing_numbers(source_text, new): why = "number lost"
        elif FILLER.search(new): why = f"filler word '{FILLER.search(new).group(0)}'"
        elif guard and appended_clause(text, new): why = appended_clause(text, new) + " (translate the source fully instead of adding a clause at the end)"
        else:
            ok, added = faithful_check(llm, src_name, tgt_name, source_text, text, new)
            if ok:
                return new, "ok"
            why = "added information: " + "; ".join(added)[:160]
        if why.startswith("added information"):
            fb = (f"Your translation states something the {src_name} speaker does not say: {'; '.join(added)[:200]}. Remove it. Reach {lo} to {hi} words only by translating every "
                  f"{src_name} word and clause fully (spelled-out subjects and pronouns, full verb forms, no contractions), never by adding ideas. Answer with the sentence only.")
        else:
            fb = (f"That answer has {nw_new} words; it must have between {lo} and {hi} words and keep the same meaning. Rewrite it again, answer with the sentence only."
                  if ("longer" in why or "too long" in why or why == "identical") else f"That is invalid ({why}). Rewrite it again within {lo}-{hi} words, keep every number and name, no filler words; answer with the sentence only.")
        msgs = msgs + [{"role": "assistant", "content": raw}, {"role": "user", "content": fb}]
    return None, why


def run_tts_burst(units: list, plans: list, refs: dict, tgt_lang: str, work: pathlib.Path, seed: int, times: dict, *, opts, llm=None, src_name: str = "", tgt_name: str = "",
                  fit: dict | None = None, qa_cfg: dict | None = None, min_part_words: int = 3, log=print) -> dict:
    """Synthesise every part of every unit (speaker-grouped like the slot path), fit / retry per part, write tts_u<id>_p<k>.wav and u['tts_parts'] / u['tts'].
    fit = {"enabled": bool, "rounds": int, "target_ratio": float}; qa_cfg = {"enabled": bool, "retries": int, "low_temp": float} (None / disabled = single plain attempt)."""
    import torch
    import torchaudio
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
    from chatterbox_fp16 import install_speech_token_guard
    if tgt_lang not in SUPPORTED_LANGUAGES:
        raise RuntimeError(f"target language {tgt_lang!r} not supported by Chatterbox")
    fit = fit or {"enabled": False}; qa_cfg = qa_cfg or {"enabled": False}
    t0 = time.perf_counter(); m = ChatterboxMultilingualTTS.from_local(pathlib.Path(__file__).resolve().parents[2] / "models" / "chatterbox", "cuda", t3_model="v3")
    builtin = m.conds; guard = install_speech_token_guard(m); times["tts_load"] = round(time.perf_counter() - t0, 3)
    prev_end_of = {u["id"]: plans[i - 1]["slot"][1] for i, u in enumerate(units) if i > 0}    # dry-run estimate of the previous speech end (the aligner uses the real one)
    qa = None
    if qa_cfg.get("enabled"):
        from tts_qa import TtsQA
        t0 = time.perf_counter(); qa = TtsQA(); times["qa_load"] = round(time.perf_counter() - t0, 3)
    stats = {"parts": 0, "attempts": 0, "retried_parts": 0, "retry_success": 0, "final_bad": 0, "fit_parts_rewritten": 0, "fit_parts_merged": 0, "fit_rounds_used": 0, "fit_rewrite_rejected": 0, "gen_seconds": 0.0, "qa_seconds": 0.0}

    def cast(dtype):
        for mod in (m.t3, m.s3gen, m.ve):
            mod.to(dtype)
        if m.conds is not None:
            m.conds = m.conds.to(device="cuda")

    def gen_part(u, k: int, text: str, dtype, salt: int = 0) -> dict:
        """One TTS part with the QA retry ladder: attempt 1 production settings, 2 new seed, 3 new seed + lower temperature. Keeps the best attempt."""
        base = seed + 100 * u["id"] + k + 100003 * salt
        ladder = [(0, 0.8)] + ([(1, 0.8), (2, qa_cfg.get("low_temp", 0.6))][: qa_cfg.get("retries", 2)] if qa is not None else [])
        attempts, best = [], None
        for ai, (soff, temp) in enumerate(ladder):
            s = base + 7919 * soff
            torch.manual_seed(s)
            tg = time.perf_counter()
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                wav = m.generate(text, language_id=tgt_lang, temperature=temp)
            wav = wav.detach().float().cpu()
            if not torch.isfinite(wav).all() or wav.abs().max() < 1e-4:
                raise RuntimeError(f"non-finite or silent TTS for unit {u['id']} part {k}")
            stats["gen_seconds"] += time.perf_counter() - tg; stats["attempts"] += 1
            rec = {"attempt": ai + 1, "seed": s, "temperature": temp, "seconds": round(wav.shape[-1] / m.sr, 3)}
            if qa is not None:
                tq = time.perf_counter(); r = qa.assess(text, wav[0].numpy(), m.sr, tgt_lang); stats["qa_seconds"] += time.perf_counter() - tq
                rec.update({"score": r["score"], "bad": r["bad"], "reasons": r["reasons"], "ctc": r["ctc"][:120], "whisper": r["whisper"][:120]})
            attempts.append(rec)
            if best is None or (qa is not None and (rec["bad"], rec["score"]) < (best[1]["bad"], best[1]["score"])):
                best = (wav, rec)
            if qa is None or not rec["bad"]:
                break
        wav, rec = best
        f = work / (f"tts_u{u['id']:02d}_p{k}.wav" if not salt else f"tts_u{u['id']:02d}_p{k}_r{salt}.wav"); torchaudio.save(str(f), wav, m.sr)
        bad_final = bool(rec.get("bad", False))
        stats["parts"] += 0 if salt else 1
        if len(attempts) > 1 and not salt:
            stats["retried_parts"] += 1; stats["retry_success"] += 0 if bad_final else 1
        return {"file": str(f), "seconds": round(wav.shape[-1] / m.sr, 3), "text": text, "attempts": attempts, "qa_bad_final": bad_final, "qa_score": rec.get("score"), "seed_used": rec["seed"], "temperature_used": rec["temperature"]}

    def chunk_of(tp: dict, tag: str):
        y, sr = sf.read(tp["file"], dtype="float32"); assert sr == TTS_SR
        if y.ndim > 1: y = y.mean(axis=1)
        s_, e_ = speech_span(y, sr, work, tag); tp["speech_span"] = [round(s_, 3), round(e_, 3)]
        return y[int(s_ * sr):int(e_ * sr)]

    def synth_all(dtype) -> None:
        cast(dtype); by_spk: dict = {}
        for u, p in zip(units, plans):
            by_spk.setdefault(u["speaker"], []).append((u, p))
        for spk, ups in by_spk.items():
            ref = refs[spk]["file"]
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                if ref: m.prepare_conditionals(ref)
                else: m.conds = builtin.to(device="cuda"); guard.use_builtin()
            for u, p in ups:
                u["parts_original"] = list(u["parts"]); u["tts_parts"] = []; u["fit"] = []
                for k, text in enumerate(u["parts"]):
                    u["tts_parts"].append(gen_part(u, k, text, dtype))
                chunks = [chunk_of(tp, f"u{u['id']:02d}_p{k}") for k, tp in enumerate(u["tts_parts"])]
                if fit.get("enabled"):
                    prev_end = prev_end_of.get(u["id"])
                    for rnd in range(fit.get("rounds", 4)):
                        need = burst_fit_ratios(chunks, p["bursts"], p["slot"], p["next_start"], opts, prev_end=prev_end)
                        over = [(k, r) for k, r in enumerate(need) if r is not None and r > opts.atempo_cap + 1e-3]
                        if not over:
                            break
                        k, r = max(over, key=lambda kr: kr[1])
                        rec = {"round": rnd + 1, "part": k, "ratio_before": r, "old": u["parts"][k], "parts_in_unit": len(u["parts"])}
                        stats["fit_rounds_used"] += 1
                        if len(u["parts"]) > 1:
                            # 1) meaning-preserving: merge the over-long part with the neighbour that has more slack -> ONE natural phrase in the union of both bursts
                            nb = [j for j in (k - 1, k + 1) if 0 <= j < len(u["parts"])]
                            j = min(nb, key=lambda j_: need[j_] if need[j_] is not None else 0.0)
                            a_, b_ = sorted((k, j)); text = u["parts"][a_] + " " + u["parts"][b_]
                            rec.update(action="merge_parts", with_part=j, new=text)
                            u["parts"][a_:b_ + 1] = [text]; p["bursts"][a_:b_ + 1] = [(p["bursts"][a_][0], p["bursts"][b_][1])]
                            p["groups"][a_:b_ + 1] = [p["groups"][a_] + " " + p["groups"][b_]]; p["group_counts"][a_:b_ + 1] = [p["group_counts"][a_] + p["group_counts"][b_]]
                            u["tts_parts"][a_:b_ + 1] = [gen_part(u, a_, text, dtype, salt=rnd + 1)]; chunks[a_:b_ + 1] = [chunk_of(u["tts_parts"][a_], f"u{u['id']:02d}_p{a_}r{rnd + 1}")]
                            stats["fit_parts_merged"] += 1
                        else:
                            # 2) a whole sentence that is still far too long for its window: concise re-translation of the sentence (never of a fragment), then regenerate
                            nw = words_of(u["parts"][0]); target = max(int(nw * fit.get("target_ratio", 1.05) / r), min_part_words + 1)
                            if llm is None or not fit.get("rewrite", True) or r <= opts.hard_cap + 0.05 or nw < 6 or target >= nw:
                                rec["action"] = "kept"; rec["result"] = "short phrase or marginal overflow: lead / spill / moderate atempo instead"; u["fit"].append(rec); break
                            new = concise_rewrite(llm, src_name, tgt_name, p["groups"][0], u["parts"][0], target, u["tts_parts"][0]["seconds"] / r)
                            rec.update(action="concise_rewrite", max_words=target)
                            if new is None:
                                rec["result"] = "rewrite_rejected"; stats["fit_rewrite_rejected"] += 1; u["fit"].append(rec); break
                            u["parts"][0] = new; rec.update(new=new, result="rewritten"); stats["fit_parts_rewritten"] += 1
                            u["tts_parts"][0] = gen_part(u, 0, new, dtype, salt=rnd + 1); chunks[0] = chunk_of(u["tts_parts"][0], f"u{u['id']:02d}_p0r{rnd + 1}")
                        u["fit"].append(rec)
                    need = burst_fit_ratios(chunks, p["bursts"], p["slot"], p["next_start"], opts, prev_end=prev_end)
                    u["fit_final_ratios"] = need
                if u["parts"] != u["parts_original"]:
                    u["translation_fitted"] = " ".join(u["parts"])
                stats["final_bad"] += sum(tp["qa_bad_final"] for tp in u["tts_parts"])
                u["tts"] = {"file": None, "parts": len(u["parts"]), "seconds": round(sum(t["seconds"] for t in u["tts_parts"]), 3), "sr": m.sr, "reference": refs[spk]["source"], "dtype": str(dtype).replace("torch.", "")}
                log(f"  [tts] u{u['id']:02d} {spk} {len(u['parts'])} parts {u['tts']['seconds']:.2f}s" + (f" fit-ops {len(u['fit'])}" if u['fit'] else "") +
                    (f" retried {sum(len(t['attempts']) > 1 for t in u['tts_parts'])}" if qa is not None else ""))

    used = None
    t0 = time.perf_counter()
    for dtype in (torch.float16, torch.float32):
        try:
            for k_ in stats:
                stats[k_] = 0 if not isinstance(stats[k_], float) else 0.0
            synth_all(dtype); used = dtype; break
        except Exception as e:
            log(f"  [tts] {dtype} failed: {type(e).__name__}: {str(e)[:200]} -> retrying next dtype")
    if used is None:
        raise RuntimeError("Chatterbox generation failed in every dtype")
    times["tts"] = round(time.perf_counter() - t0, 3)
    sr = m.sr
    del m
    if qa is not None:
        qa.close()
    torch.cuda.empty_cache()
    stats["gen_seconds"] = round(stats["gen_seconds"], 2); stats["qa_seconds"] = round(stats["qa_seconds"], 2)
    stats["retry_success_rate"] = round(stats["retry_success"] / stats["retried_parts"], 3) if stats["retried_parts"] else None
    return {"model": "ChatterboxMultilingualTTS t3=v3 (models/chatterbox)", "sr": sr, "dtype": str(used).replace("torch.", ""), "language_id": tgt_lang,
            "voice_reference": "speaker's own diarized speech via audio_prompt_path/prepare_conditionals",
            "speech_token_guard": {"enabled": True, "flow_calls": guard.calls, "violations": guard.violations}, "qa": {"enabled": qa is not None, **qa_cfg}, "fit": {**fit}, "stats": stats, "references": refs}


def distribute_words(spans: list[dict], raw_words: list[str], bursts: list, norm_fn, gap_bonus: float = 0.5, punct_bonus: float = 0.25) -> list[int]:
    """Priority 2 (2026-10-02): distribute the ALIGNED words of one synthesized sentence over the source bursts so that the cumulative TTS speech
    time follows the cumulative source burst duration. Cuts fall only on raw-word boundaries (never inside a word; a raw word such as "28th" may
    be several aligned tokens), preferring a boundary with punctuation or a longer inter-word pause near the proportional point.
    Returns the number of ALIGNED tokens per burst (len == len(bursts), every entry >= 1)."""
    n = len(spans); M = len(bursts)
    tok_per_raw = [len(norm_fn(w)) for w in raw_words]
    ends = []; acc = 0                                   # token index after each raw word (the only legal cut positions)
    for w, t in zip(raw_words, tok_per_raw):
        acc += t
        if t:
            ends.append((acc, bool(re.search(r"[,.;:!?…]['\"»)]*$", w))))
    if acc != n or M <= 1:
        return [n] if M <= 1 else None
    t0 = spans[0]["start"]; S = max(spans[-1]["end"] - t0, 1e-3)
    bd = [max(e - s, 0.05) for s, e in bursts]; B = sum(bd); counts = []; prev = 0
    for j in range(M - 1):
        target = sum(bd[: j + 1]) / B * S
        cands = [(k, pu) for k, pu in ends if prev < k <= n - (M - 1 - j)]
        if not cands:
            return None
        def score(k, pu):
            cum = spans[k - 1]["end"] - t0; gap = spans[k]["start"] - spans[k - 1]["end"] if k < n else 0.0
            return abs(cum - target) - (punct_bonus if pu else 0.0) - gap_bonus * min(gap, 0.5)
        k = min(cands, key=lambda c: score(*c))[0]; counts.append(k - prev); prev = k
    counts.append(n - prev)
    return counts if all(c >= 1 for c in counts) else None


# English words a spoken phrase must not END on before a pause (articles, prepositions, conjunctions, auxiliaries, pronouns, determiners):
# "smooth and | will bring us the | result" was the dominant "voice breaks off mid-phrase" defect of the timing split (2026-10-12 review).
CUT_FUNC_WORDS = frozenset(("a an the and or but nor so to of for in on at by with from as into onto than that which who whom whose this these those my our your his "
                            "her its their i we you he she it they me us him them will would can could shall should may might must is are was were be been being am "
                            "have has had do does did not no very just also let's such all each every some any both either neither another other more most less few "
                            "many much several").split())


# words a phrase must not end on before a LONG source pause (w_func_long): the cut list without object pronouns ("with you |", "send it |" are
# natural phrase ends) plus wh-words / subordinators ("in general what | I want")
LONG_PAUSE_FUNC = (CUT_FUNC_WORDS - frozenset("you it me us him them her".split())) | frozenset("what where when why how if because about like whether".split())


# two-word expressions a pause must not split (the first word is not always a function word): 03 "moving | forward" (1.5 s source pause)
FIXED_PHRASES = frozenset([("moving", "forward"), ("go", "ahead"), ("right", "now"), ("thank", "you"), ("come", "on"), ("by", "the"), ("in", "general"), ("of", "course"),
                           ("at", "least"), ("as", "well"), ("each", "other"), ("so", "that"), ("in", "order"), ("let's", "go"), ("you", "know")])


# --phrase-cut acoustic (2026-10-12 offline sweep on the five baseline clips): cuts inside still-voiced sound 14/25 -> 4/8 (> -15 / -20 dB of the
# speech level), timing deviation 0.132 -> 0.146 s mean, max unchanged 0.55 s; the function-word rule of "syntax" is off (it cost coverage).
# + w_fill / w_single (cand4): parts < 40 % of their burst 9 -> 6, one-word parts 11 -> 4, timing deviation 0.151 s (cand3: "that" alone on a 0.7 s burst of 05).
ACOUSTIC_CUT = {"w_func": 0.0, "w_voiced": 1.0, "w_punct": 0.25, "w_tight": 0.0, "w_gap": 0.5, "w_fill": 1.0, "w_single": 0.5}


def distribute_words_syntax(spans: list[dict], raw_words: list[str], bursts: list, norm_fn, tol: float = 0.4, w_punct: float = 0.6, w_gap: float = 1.0,
                            w_func: float = 0.8, w_tight: float = 0.35, w_merge: float = 0.9, func_words=CUT_FUNC_WORDS, y: np.ndarray | None = None, sr: int = TTS_SR,
                            w_voiced: float = 0.5, steep: float = 4.0, w_fill: float = 0.0, w_single: float = 0.0, w_func_long: float = 0.0, long_lo: float = 0.3,
                            long_hi: float = 0.8, long_grow: float = 0.0, long_late_tol: float = 0.0, allowed_after: set | None = None) -> list[int] | None:
    """Syntax-aware variant of distribute_words (2026-10-12): the cut positions are chosen jointly (dynamic programming) with
      cost = 0.5 per `tol` s of deviation from the proportional burst point (+ steep per s beyond `tol`: a cut moved by ~1 s for a comma left a 2.1 s
             source burst of 05 with 0.3 s of dub -> coverage loss, 2026-10-12 A/B)  - w_punct at punctuation  - w_gap x inter-word pause (<= 0.4 s)
             + w_func after a function word ("the | result", "smooth and | will")  + w_tight for a cut with < 60 ms pause and no punctuation
             + w_func after a number ("20-25 | years")
    and a boundary may also be left uncut (cost w_merge + 0.3 x the source pause, <= 2 s): the two bursts then form ONE window and the phrase runs
    through the short source pause instead of breaking after "and". With the clip `y`, a cut whose quietest 5 ms in the inter-word window is still
    within 15 dB of the speech level costs up to w_voiced more (CTC word ends are early: "video | as" with a 100 ms CTC pause was still voiced).
    w_fill / w_single (acoustic mode): a part whose speech covers < 40 % of its source burst costs up to w_fill (05 u06: "that" = 0.14 s on a
    0.7 s burst), a one-word part (no punctuation) costs w_single.
    w_func_long (2026-10-13, "заметные паузы"): a cut after a function word / number / inside a fixed phrase ("in | general", "moving | forward")
    costs up to w_func_long more, scaled by the SOURCE pause at that boundary (0 below long_lo s, full at long_hi s): the dub reproduces the
    source pause there, and a 1-2 s silence after "to our |" or "Probably more in |" is what sounds unnatural; short pauses keep the plain
    acoustic rule (the syntax mode's global function-word rule moved cuts at every boundary and cost coverage).
    Returns aligned tokens per burst; 0 = merged into the next burst."""
    import functools
    n = len(spans); M = len(bursts)
    ends = []; acc = 0; end_raw = {}   # aligned-token index of a cut -> index of the raw word it follows
    for wi, w in enumerate(raw_words):
        t = len(norm_fn(w)); acc += t
        if t:
            end_raw[acc] = wi
            ends.append((acc, bool(re.search(r"[,.;:!?…]['\"»)]*$", w)), re.sub(r"[^\w']", "", w.lower().replace("’", "'")), bool(re.search(r"\d", w))))
    nxt = {}   # aligned-token index of a cut -> the next raw word (lower-case), for the fixed-phrase rule
    wl = [re.sub(r"[^\w']", "", w.lower().replace("’", "'")) for w in raw_words if len(norm_fn(w))]
    for i, (k, *_r) in enumerate(ends):
        nxt[k] = wl[i + 1] if i + 1 < len(wl) else ""
    if acc != n or M <= 1:
        return [n] if M <= 1 else None
    t0 = spans[0]["start"]; S = max(spans[-1]["end"] - t0, 1e-3)
    bd = [max(e - s, 0.05) for s, e in bursts]; B = sum(bd); tgt = [sum(bd[: j + 1]) / B * S for j in range(M - 1)]

    voiced = {}
    if y is not None and w_voiced > 0:
        h = max(1, int(0.005 * sr)); nf = len(y) // h
        r = 20 * np.log10(np.sqrt(np.mean(y[: nf * h].reshape(nf, h) ** 2, axis=1) + 1e-12)); ref = float(np.percentile(r, 90))
        for k, *_ in ends:
            if k < n:
                a_, b_ = int((spans[k - 1]["end"] - 0.02) * sr) // h, int((spans[k]["start"] + 0.02) * sr) // h + 1
                v = float(r[max(0, a_):min(nf, b_)].min()) - ref if b_ > a_ else 0.0
                voiced[k] = w_voiced * min(1.0, max(0.0, (v + 30.0) / 15.0))

    def cut_cost(j, k, pu, lw, num):
        gap = spans[k]["start"] - spans[k - 1]["end"] if k < n else 0.0
        ds = spans[k - 1]["end"] - t0 - tgt[j]; d = abs(ds); tj = tol
        sp_after = max(0.0, bursts[j + 1][0] - bursts[j][1]) if j + 1 < len(bursts) else 0.0
        if long_late_tol and ds > 0:   # a LATER cut runs into the source pause after burst j: tolerated up to +min(long_late_tol, pause/2) s (02 "I will | 9 s | show")
            tj = tol + min(long_late_tol, 0.5 * sp_after)
        elif allowed_after is not None and ds > 0 and sp_after >= 1.0 and end_raw.get(k) in allowed_after:
            # targeted: only a move LATER onto a natural (Qwen-marked) boundary, only before a >= 1 s source pause (the extra words run into the pause)
            tj = tol + min(0.6, 0.5 * sp_after)
        c = 0.5 * d / max(tj, 1e-3) + steep * max(0.0, d - tj) - (w_punct if pu else 0.0) - w_gap * min(gap, 0.4)
        if not pu:
            c += (w_func if lw in func_words or num else 0.0) + (w_tight if gap < 0.06 else 0.0)
            weak = (end_raw.get(k) not in allowed_after) if allowed_after is not None else (lw in LONG_PAUSE_FUNC or num or (lw, nxt.get(k, "")) in FIXED_PHRASES)
            if w_func_long and weak:
                sp = bursts[j + 1][0] - bursts[j][1]
                # full weight at long_hi, growing to 2x at long_hi + 2 s: a 9 s silence after "I will |" is worse than a 0.8 s one (02 u05)
                c += w_func_long * (min(1.0, max(0.0, (sp - long_lo) / max(long_hi - long_lo, 1e-3))) + long_grow * min(1.0, max(0.0, (sp - long_hi) / 2.0)))
        return c + voiced.get(k, 0.0)

    words_end = {k: i for i, (k, *_r) in enumerate(ends)}

    def part_cost(j, prev, k, pu):
        """Part j = aligned tokens prev..k-1 placed on burst j (only the parts that END at a cut; the merge path is not charged)."""
        if not (w_fill or w_single):
            return 0.0
        dur = spans[k - 1]["end"] - spans[prev]["start"]; fill = dur / max(bursts[j][1] - bursts[j][0], 0.05)
        c = w_fill * max(0.0, 0.4 - fill) / 0.4
        raw_words_in_part = words_end[k] - (words_end[prev] if prev in words_end else -1)
        if raw_words_in_part <= 1 and not pu:
            c += w_single
        return c

    @functools.lru_cache(maxsize=None)
    def best(j, prev):
        if j == M - 1:
            return 0.0, ()
        sp_j = max(bursts[j + 1][0] - bursts[j][1], 0.0); c, rest = best(j + 1, prev)
        # with the long-pause rule a merge over a long source pause costs at least a weak cut there: otherwise the DP dodges "how our |" by
        # running the phrase through the pause, and the source keeps talking after the dub has finished (05 u03: +1.7 s source without dub)
        mg = w_func_long * min(1.0, max(0.0, (sp_j - long_lo) / max(long_hi - long_lo, 1e-3))) if w_func_long else 0.0
        opt = (c + w_merge + 0.3 * min(sp_j, 2.0) + mg, (None,) + rest)
        for k, pu, lw, num in ends:
            if prev < k < n:
                c, rest = best(j + 1, k); c += cut_cost(j, k, pu, lw, num) + part_cost(j, prev, k, pu)
                if c < opt[0]:
                    opt = (c, (k,) + rest)
        return opt
    counts = []; prev = 0
    for k in best(0, 0)[1]:
        counts.append(0 if k is None else k - prev); prev = prev if k is None else k
    counts.append(n - prev)
    return counts if counts[-1] >= 1 else None


def run_tts_phrase(units: list, plans: list, refs: dict, tgt_lang: str, work: pathlib.Path, seed: int, times: dict, *, opts, llm=None, src_name: str = "", tgt_name: str = "",
                   fit: dict | None = None, qa_cfg: dict | None = None, fit_up: dict | None = None, split_retries: int = 2, min_part_words: int = 1, soft_part_words: int = 2,
                   part_max_words: int = 16, phrase_split: str = "timing", phrase_cut: str = "timing", long_pause_func: float = 0.0, long_pause_grow: float = 0.0, cut_fade: str = "fixed", gap_qa: bool = False, gap_qa_retries: int = 2, pause_mark_mode: str = "off", sentence_flow_gap: float = 0.0, log=print) -> dict:
    """Phrase-first TTS (2026-10-02, priorities 1-3 of the coverage fix). Per unit:
      ONE Chatterbox generation of the whole natural translated sentence (QA retry ladder as in run_tts_burst)
      -> fit-DOWN (concise rewrite when the sentence cannot fit its total window at atempo_cap, rule as before)
      -> fit-UP (fuller rewrite when the synthesized speech is shorter than fit_up["ratio"] x the source speech of the unit; accepted only when the
         duration moves closer to the target AND the QA score does not get worse; <= fit_up["rounds"] rounds)
      -> Qwen cuts the FINAL text into one part per source burst (lossless, as before; one-word parts merged into a neighbour)
      -> CTC forced alignment of the expected text inside the generated clip (tts_align.CtcAligner, the wav2vec2 of the TTS QA) and cuts BETWEEN words
      -> a span that would still need more than atempo_cap takes the neighbouring burst too (union window, no re-synthesis)
      -> u["tts_parts"] = the spans (tts_u<id>_p<k>.wav), so align_units_burst / place_groups / build_dubbed_track work unchanged.
    No separate Chatterbox call per burst, no 1-2-word prompts, no cut inside a word. Alignment fallback (non-English target / alignment failure):
    Silero speech span + energy-valley cuts proportional to the part word counts, recorded per unit (`cut`)."""
    import torch
    import torchaudio
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
    from chatterbox_fp16 import install_speech_token_guard
    from burst_aware_audio import consolidate_parts, norm_tokens, proportional_split, qwen_split
    from e1_burst_align import valley_split
    from tts_align import CtcAligner, cut_parts
    from tts_qa import norm
    if tgt_lang not in SUPPORTED_LANGUAGES:
        raise RuntimeError(f"target language {tgt_lang!r} not supported by Chatterbox")
    fit = fit or {"enabled": False}; qa_cfg = qa_cfg or {"enabled": False}; fit_up = fit_up or {"enabled": False}
    import model_cache
    def _load_tts():
        m_ = ChatterboxMultilingualTTS.from_local(pathlib.Path(__file__).resolve().parents[2] / "models" / "chatterbox", "cuda", t3_model="v3")
        return m_, m_.conds
    t0 = time.perf_counter(); m, builtin = model_cache.get("chatterbox_v3", _load_tts)
    guard = install_speech_token_guard(m); times["tts_load"] = round(time.perf_counter() - t0, 3)
    qa = None; aligner = None; qa_obj = None
    if qa_cfg.get("enabled") or tgt_lang == "en":
        from tts_qa import TtsQA
        t0 = time.perf_counter(); qa_obj = model_cache.get("tts_qa", TtsQA); times["qa_load"] = round(time.perf_counter() - t0, 3)
        qa = qa_obj if qa_cfg.get("enabled") else None
        if tgt_lang == "en":
            aligner = CtcAligner.from_qa(qa_obj)
    prev_end_of = {u["id"]: plans[i - 1]["slot"][1] for i, u in enumerate(units) if i > 0}
    stats = {"parts": 0, "attempts": 0, "retried_parts": 0, "retry_success": 0, "final_bad": 0, "fit_parts_rewritten": 0, "fit_parts_merged": 0, "fit_rounds_used": 0, "fit_rewrite_rejected": 0,
             "fit_up_units": 0, "fit_up_rounds": 0, "fit_up_accepted": 0, "fit_up_rejected": 0, "fit_up_candidates": 0, "units_single_burst": 0, "units_qwen_split": 0, "units_single_fallback": 0,
             "spans": 0, "span_merges": 0, "align_ok": 0, "align_fallback": 0, "units_timing_split": 0, "units_proportional_fallback": 0, "gen_seconds": 0.0, "qa_seconds": 0.0, "align_seconds": 0.0,
             "fitup_qwen_seconds": 0.0, "fitdown_qwen_seconds": 0.0, "fitup_gen_qa_seconds": 0.0}

    def cast(dtype):
        for mod in (m.t3, m.s3gen, m.ve):
            mod.to(dtype)
        if m.conds is not None:
            m.conds = m.conds.to(device="cuda")

    def gen_sentence(u, text: str, dtype, salt: int = 0) -> tuple:
        """One whole-sentence generation with the QA retry ladder of run_tts_burst (same seed rule, part index 0). Returns (wav float32 1-D, record)."""
        base = seed + 100 * u["id"] + 100003 * salt
        ladder = [(0, 0.8)] + ([(1, 0.8), (2, qa_cfg.get("low_temp", 0.6))][: qa_cfg.get("retries", 2)] if qa is not None else [])
        attempts, best = [], None
        for ai, (soff, temp) in enumerate(ladder):
            s = base + 7919 * soff; torch.manual_seed(s); tg = time.perf_counter()
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                wav = m.generate(text, language_id=tgt_lang, temperature=temp)
            wav = wav.detach().float().cpu()
            if not torch.isfinite(wav).all() or wav.abs().max() < 1e-4:
                raise RuntimeError(f"non-finite or silent TTS for unit {u['id']}")
            stats["gen_seconds"] += time.perf_counter() - tg; stats["attempts"] += 1
            rec = {"attempt": ai + 1, "seed": s, "temperature": temp, "seconds": round(wav.shape[-1] / m.sr, 3)}
            if qa is not None:
                tq = time.perf_counter(); r = qa.assess(text, wav[0].numpy(), m.sr, tgt_lang); stats["qa_seconds"] += time.perf_counter() - tq
                rec.update({"score": r["score"], "bad": r["bad"], "reasons": r["reasons"], "ctc": r["ctc"][:160], "whisper": r["whisper"][:160]})
            attempts.append(rec)
            if best is None or (qa is not None and (rec["bad"], rec["score"]) < (best[1]["bad"], best[1]["score"])):
                best = (wav, rec)
            if qa is None or not rec["bad"]:
                break
        wav, rec = best
        return wav[0].numpy().astype(np.float32), {"text": text, "attempts": attempts, "qa_bad_final": bool(rec.get("bad", False)), "qa_score": rec.get("score"), "seed_used": rec["seed"],
                                                   "temperature_used": rec["temperature"], "seconds": round(len(wav[0]) / m.sr, 3)}

    def align(y, text):
        if aligner is None:
            return None
        ta = time.perf_counter(); sp = aligner.align(y, m.sr, text); stats["align_seconds"] += time.perf_counter() - ta
        return sp

    def speech_seconds(y, spans) -> float:
        if spans:
            return max(0.05, spans[-1]["end"] - spans[0]["start"])
        s_, e_ = speech_span(y, m.sr, work, "tmp_phrase"); return max(0.05, e_ - s_)

    def fallback_cut(y, counts: list[int]) -> list[tuple]:
        """No word alignment: Silero speech span + energy-valley cuts proportional to the part word counts (not used for English with a working aligner)."""
        s_, e_ = speech_span(y, m.sr, work, "tmp_fb"); tot = max(sum(counts), 1); bounds = [s_]; acc = 0
        for c in counts[:-1]:
            acc += c; bounds.append(s_ + (e_ - s_) * acc / tot)
        bounds.append(e_); out = []
        for k in range(len(counts)):
            a, b = bounds[k], bounds[k + 1]
            if k > 0:
                sub = valley_split(y, m.sr, (bounds[k - 1], b), 2)
                if len(sub) == 2:
                    a = sub[1][0]; out[-1] = (out[-1][0], a)
            out.append((a, b))
        return out

    def synth_all(dtype) -> None:
        cast(dtype); by_spk: dict = {}
        for u, p in zip(units, plans):
            by_spk.setdefault(u["speaker"], []).append((u, p))
        for spk, ups in by_spk.items():
            ref = refs[spk]["file"]
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                if ref: m.prepare_conditionals(ref)
                else: m.conds = builtin.to(device="cuda"); guard.use_builtin()
            for u, p in ups:
                text = u["translation"]; u["parts_original"] = [text]; u["fit"] = []; u["fit_up"] = []
                y, rec = gen_sentence(u, text, dtype); spans = align(y, text); sp_s = speech_seconds(y, spans)
                src_words = " ".join(p["groups"]); target = sum(e - s for s, e in p["bursts"]); prev_end = prev_end_of.get(u["id"])
                total_win = max(0.2, p["next_start"] - opts.gap - p["bursts"][0][0])
                # ---- fit-DOWN: the whole sentence does not fit its total window at the cap -> concise rewrite (rules as in run_tts_burst)
                for rnd in range(fit.get("rounds", 4) if fit.get("enabled") else 0):
                    r = sp_s / total_win
                    if r <= opts.atempo_cap + 1e-3:
                        break
                    # short sentences (4-5 words) too: 02 "Relaxation zone: lying areas." overflowed its window by 0.4 s at the hard cap and was cut (2026-10-12)
                    nw = words_of(text); overflow = sp_s / opts.hard_cap - total_win; short_ok = nw >= fit.get("short_min_words", 4) and overflow > fit.get("short_overflow_s", 0.3)
                    # a short sentence is rewritten at most ONCE: 02 u16 (2026-10-13) went "The relaxation zone also includes lying down areas." -> "Relaxation zone:
                    # lying areas." -> "Lying areas." (meaning lost, 2-word TTS unintelligible: QA bad x3, Whisper hears "Thank you, Iris")
                    short_ok = short_ok and not any(x.get("result") == "rewritten" for x in u["fit"])
                    tgt_w = max(int(nw * fit.get("target_ratio", 1.05) / r), 4 if nw >= 6 else 3); stats["fit_rounds_used"] += 1
                    recf = {"round": rnd + 1, "ratio_before": round(r, 3), "old": text, "action": "concise_rewrite", "max_words": tgt_w, "overflow_at_hard_cap_s": round(overflow, 3)}
                    if llm is None or r <= opts.hard_cap + 0.05 or (nw < 6 and not short_ok) or tgt_w >= nw:
                        recf.update(action="kept", result="short phrase or marginal overflow: lead / spill / moderate atempo instead"); u["fit"].append(recf); break
                    tq = time.perf_counter(); new = concise_rewrite(llm, src_name, tgt_name, src_words, text, tgt_w, sp_s / r); stats["fitdown_qwen_seconds"] += time.perf_counter() - tq
                    if new is None:
                        recf["result"] = "rewrite_rejected"; stats["fit_rewrite_rejected"] += 1; u["fit"].append(recf); break
                    y2, rec2 = gen_sentence(u, new, dtype, salt=10 + rnd); sp2 = align(y2, new); s2 = speech_seconds(y2, sp2)
                    recf.update(new=new, result="rewritten", seconds_before=round(sp_s, 3), seconds_after=round(s2, 3)); u["fit"].append(recf); stats["fit_parts_rewritten"] += 1
                    text, y, rec, spans, sp_s = new, y2, rec2, sp2, s2
                # ---- fit-UP: synthesized speech much shorter than the source speech of the unit -> fuller rewrite; accepted only if closer AND not less intelligible
                if fit_up.get("enabled") and llm is not None and target > 0.5 and words_of(text) >= 3 and sp_s / target < fit_up.get("ratio", 0.65):
                    stats["fit_up_candidates"] += 1
                    for rnd in range(fit_up.get("rounds", 2)):
                        ratio = sp_s / target; rate = words_of(text) / sp_s
                        tgt_w = max(words_of(text) + 1, int(round(rate * target * fit_up.get("aim", 1.0)))); lo_w = max(words_of(text) + 1, int(round(rate * target * fit_up.get("min_ratio", 0.85)))); stats["fit_up_rounds"] += 1
                        recu = {"round": rnd + 1, "ratio_before": round(ratio, 3), "old": text, "target_words": tgt_w, "min_words": lo_w, "source_seconds": round(target, 3), "tts_seconds_before": round(sp_s, 3)}
                        tq = time.perf_counter(); new, why = fuller_rewrite(llm, src_name, tgt_name, src_words, text, tgt_w, target, sp_s, min_words=lo_w, guard=fit_up.get("guard", False)); stats["fitup_qwen_seconds"] += time.perf_counter() - tq
                        if new is None:
                            recu["result"] = "rewrite_rejected"; recu["reason"] = why; stats["fit_up_rejected"] += 1; u["fit_up"].append(recu)
                            if rnd + 1 < fit_up.get("rounds", 2) and "too long" in why:   # ask once more with a smaller budget
                                continue
                            break
                        tq = time.perf_counter(); y2, rec2 = gen_sentence(u, new, dtype, salt=20 + rnd); sp2 = align(y2, new); s2 = speech_seconds(y2, sp2); r2 = s2 / target
                        stats["fitup_gen_qa_seconds"] += time.perf_counter() - tq
                        closer = abs(1 - r2) < abs(1 - ratio) - 0.02
                        not_worse = (qa is None) or ((not rec2["qa_bad_final"]) and (rec2["qa_score"] is None or rec["qa_score"] is None or rec2["qa_score"] <= rec["qa_score"] + 0.05))
                        recu.update(new=new, tts_seconds_after=round(s2, 3), ratio_after=round(r2, 3), qa_before=rec.get("qa_score"), qa_after=rec2.get("qa_score"), closer=closer, intelligible=not_worse)
                        if closer and not_worse:
                            recu["result"] = "accepted"; stats["fit_up_accepted"] += 1; u["fit_up"].append(recu)
                            text, y, rec, spans, sp_s = new, y2, rec2, sp2, s2
                            if sp_s / target >= fit_up.get("ratio", 0.65):
                                break
                        else:
                            recu["result"] = "rejected (not closer)" if not closer else "rejected (less intelligible)"; stats["fit_up_rejected"] += 1; u["fit_up"].append(recu); break
                    if any(x.get("result") == "accepted" for x in u["fit_up"]):
                        stats["fit_up_units"] += 1
                # ---- gap QA: an inserted syllable / filler between the aligned words -> new seeds for the FINAL text (keep the cleanest, never a QA-bad one)
                if gap_qa and spans is not None:
                    gv = gap_voice(y, m.sr, spans); u["gap_qa"] = {"found": gv, "retries": []}
                    for rnd in range(gap_qa_retries if gv else 0):
                        tq = time.perf_counter(); y2, rec2 = gen_sentence(u, text, dtype, salt=40 + rnd); sp2 = align(y2, text)
                        gv2 = gap_voice(y2, m.sr, sp2) if sp2 is not None else gv; stats["gap_qa_seconds"] = stats.get("gap_qa_seconds", 0.0) + time.perf_counter() - tq
                        u["gap_qa"]["retries"].append({"found": gv2, "qa_bad": rec2["qa_bad_final"], "seconds": round(speech_seconds(y2, sp2), 3)})
                        if sp2 is not None and not rec2["qa_bad_final"] and sum(b - a for a, b, _ in gv2) < sum(b - a for a, b, _ in gv):
                            y, rec, spans, sp_s, gv = y2, rec2, sp2, speech_seconds(y2, sp2), gv2; u["gap_qa"]["accepted"] = rnd + 1
                        if not gv:
                            break
                    stats["gap_qa_units"] = stats.get("gap_qa_units", 0) + (1 if u["gap_qa"]["found"] else 0); stats["gap_qa_left"] = stats.get("gap_qa_left", 0) + (1 if gv else 0)
                if text != u["translation"]:
                    u["translation_fitted"] = text
                # ---- sentence flow: short source pauses inside the sentence are not reproduced (one connected phrase over the union window)
                fm = flow_merge_bursts(p, sentence_flow_gap)
                if fm: u["flow_merges"] = fm; stats["flow_merges"] = stats.get("flow_merges", 0) + fm
                # ---- split the FINAL text into one part per source burst (lossless): timing distribution of the aligned words (default) or Qwen
                n = len(p["bursts"]); raw_words = text.split(); counts = None; parts = None
                if n == 1:
                    parts = [text]; u["split"] = {"method": "single"}; stats["units_single_burst"] += 1
                else:
                    slog = {"method": None, "attempts": []}
                    if phrase_split == "qwen" or spans is None:
                        parts, slog = qwen_split(llm, p["groups"], text, src_name, tgt_name, split_retries) if llm is not None else (None, slog)
                        if parts is not None:
                            stats["units_qwen_split"] += 1
                            parts, p["bursts"], p["groups"], p["group_counts"], merges = consolidate_parts(parts, p["bursts"], p["groups"], p["group_counts"], min_part_words, soft_part_words, part_max_words)
                            if merges:
                                slog["merges"] = merges; stats["fit_parts_merged"] += len(merges)
                    if parts is None and spans is not None:
                        if phrase_cut == "syntax" and tgt_lang == "en":
                            counts = distribute_words_syntax(spans, raw_words, p["bursts"], norm, y=y, sr=m.sr)
                        elif phrase_cut == "acoustic":   # timing split + quiet cut points only (no function-word rule): swallowed word ends without the coverage cost
                            marks = pause_marks(llm, text) if (pause_mark_mode == "qwen" and long_pause_func and llm is not None) else None
                            if pause_mark_mode == "qwen": u["pause_marks"] = sorted(marks) if marks is not None else None
                            counts = distribute_words_syntax(spans, raw_words, p["bursts"], norm, y=y, sr=m.sr, **ACOUSTIC_CUT, w_func_long=long_pause_func, long_grow=long_pause_grow, allowed_after=marks)
                        else:
                            counts = distribute_words(spans, raw_words, p["bursts"], norm)
                        if counts is not None and 0 in counts:     # syntax split left a boundary uncut: merge that burst into the next one (union window)
                            for j in range(len(counts) - 2, -1, -1):
                                if counts[j] == 0:
                                    p["bursts"][j:j + 2] = [(p["bursts"][j][0], p["bursts"][j + 1][1])]; p["groups"][j:j + 2] = [p["groups"][j] + " " + p["groups"][j + 1]]
                                    p["group_counts"][j:j + 2] = [p["group_counts"][j] + p["group_counts"][j + 1]]; counts[j:j + 2] = [counts[j + 1]]
                            slog["burst_merges"] = slog.get("burst_merges", 0) + 1; stats["syntax_burst_merges"] = stats.get("syntax_burst_merges", 0) + 1
                        if counts is not None:
                            parts = []; k0 = 0; acc = 0; idx = 0
                            for c in counts:                           # raw words whose aligned tokens fall into this part
                                words_here = []
                                while idx < len(raw_words) and acc < k0 + c:
                                    acc += len(norm(raw_words[idx])); words_here.append(raw_words[idx]); idx += 1
                                parts.append(" ".join(words_here)); k0 += c
                            slog["method"] = ("timing" if phrase_split != "qwen" else "timing_fallback") + ({"syntax": "+syntax" if tgt_lang == "en" else "", "acoustic": "+acoustic"}.get(phrase_cut, "")); stats["units_timing_split"] += 1
                    if parts is None:                                   # lossless proportional cut by source word counts (never a single lump over the pauses)
                        parts = proportional_split(text, p["group_counts"]); slog["method"] = "proportional_fallback"; stats["units_proportional_fallback"] += 1
                        if any(not norm_tokens(x) for x in parts):
                            parts = [text]; slog["method"] = "single_fallback"; stats["units_single_fallback"] += 1
                            p["bursts"] = [(p["bursts"][0][0], p["bursts"][-1][1])]; p["groups"] = [" ".join(p["groups"])]; p["group_counts"] = [sum(p["group_counts"])]
                    u["split"] = slog
                assert norm_tokens(" ".join(parts)) == norm_tokens(text), "split lost words"
                # ---- cut the ONE clip between words
                counts = [len(norm(x)) for x in parts]; pieces = None; method = None
                if spans is not None and sum(counts) == len(spans):
                    try:
                        cp = cut_parts(y, m.sr, spans, counts, onset_s=energy_onset(y, m.sr)); pieces = [(a, b) for _, a, b, _ in cp]; method = "ctc_forced_align"; stats["align_ok"] += 1
                    except Exception as e:
                        pieces = None; method = f"align_cut_failed: {type(e).__name__}"
                if pieces is None:
                    pieces = fallback_cut(y, counts); method = (method + "; " if method else "") + "silero_valley_fallback"; stats["align_fallback"] += 1
                # ---- a span that still needs more than atempo_cap takes its neighbour's burst too (union window, no re-synthesis)
                def chunks_of(pcs):
                    return [y[int(a * m.sr):int(b * m.sr)] for a, b in pcs]
                need = burst_fit_ratios(chunks_of(pieces), p["bursts"], p["slot"], p["next_start"], opts, prev_end=prev_end); merged_spans = 0
                while len(pieces) > 1:
                    over = [(k, r) for k, r in enumerate(need) if r is not None and r > opts.atempo_cap + 1e-3]
                    if not over:
                        break
                    k, _ = max(over, key=lambda kr: kr[1]); nb = [j for j in (k - 1, k + 1) if 0 <= j < len(pieces)]
                    j = min(nb, key=lambda j_: need[j_] if need[j_] is not None else 0.0); a_, b_ = sorted((k, j))
                    pieces[a_:b_ + 1] = [(pieces[a_][0], pieces[b_][1])]; parts[a_:b_ + 1] = [parts[a_] + " " + parts[b_]]
                    p["bursts"][a_:b_ + 1] = [(p["bursts"][a_][0], p["bursts"][b_][1])]; p["groups"][a_:b_ + 1] = [p["groups"][a_] + " " + p["groups"][b_]]; p["group_counts"][a_:b_ + 1] = [p["group_counts"][a_] + p["group_counts"][b_]]
                    merged_spans += 1; need = burst_fit_ratios(chunks_of(pieces), p["bursts"], p["slot"], p["next_start"], opts, prev_end=prev_end)
                stats["span_merges"] += merged_spans; u["fit_final_ratios"] = need
                u["parts"] = parts; u["tts_parts"] = []
                _fr = 20 * np.log10(np.sqrt(np.mean(y[: len(y) // 240 * 240].reshape(-1, 240) ** 2, axis=1)) + 1e-9) if len(y) >= 240 else np.array([-60.0])
                ref_db = float(np.percentile(_fr[_fr > -60], 90)) if (_fr > -60).any() else -20.0   # speech level of the clip (10 ms frames)
                raw_f = work / f"tts_u{u['id']:02d}_raw.wav"   # the final (post QA / fit) whole-sentence generation before any cut: stage-by-stage audio diagnosis
                torchaudio.save(str(raw_f), torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0), m.sr); u["tts_raw_file"] = str(raw_f)
                for k, ((a, b), ptxt) in enumerate(zip(pieces, parts)):
                    seg = y[int(a * m.sr):int(b * m.sr)]
                    if phrase_cut == "syntax":   # the cut parts were saved without any fade (cut_parts' faded copies are not used); part of the syntax candidate
                        seg = edge_fade(seg, m.sr, 5.0, 20.0 if k + 1 < len(pieces) else 10.0)
                    elif phrase_cut == "acoustic":   # the 5 ms declick fades cut_parts was meant to apply
                        if cut_fade == "adaptive":
                            seg = edge_fade(seg, m.sr, *adaptive_fade_ms(y, m.sr, a, b, k > 0, k + 1 < len(pieces), ref_db))
                        else:
                            seg = edge_fade(seg, m.sr, 5.0, 5.0)
                    f = work / f"tts_u{u['id']:02d}_p{k}.wav"; torchaudio.save(str(f), torch.from_numpy(np.ascontiguousarray(seg)).unsqueeze(0), m.sr)
                    u["tts_parts"].append({"file": str(f), "seconds": round(len(seg) / m.sr, 3), "text": ptxt, "clip_span": [round(a, 3), round(b, 3)], "speech_span": [0.0, round(len(seg) / m.sr, 3)],
                                           "attempts": rec["attempts"], "qa_bad_final": rec["qa_bad_final"], "qa_score": rec["qa_score"], "seed_used": rec["seed_used"], "temperature_used": rec["temperature_used"],
                                           "words": len(norm(ptxt)), "cut": method})
                stats["parts"] += 1; stats["spans"] += len(pieces)
                if len(rec["attempts"]) > 1:
                    stats["retried_parts"] += 1; stats["retry_success"] += 0 if rec["qa_bad_final"] else 1
                stats["final_bad"] += int(rec["qa_bad_final"])
                u["tts"] = {"file": None, "parts": len(parts), "seconds": round(len(y) / m.sr, 3), "speech_seconds": round(sp_s, 3), "sr": m.sr, "reference": refs[spk]["source"], "dtype": str(dtype).replace("torch.", ""),
                            "mode": "phrase", "cut": method, "source_speech_seconds": round(target, 3), "ratio_tts_to_source_speech": round(sp_s / target, 3) if target else None, "word_spans": spans}
                log(f"  [tts/phrase] u{u['id']:02d} {spk} 1 gen -> {len(pieces)} spans ({method}) tts {sp_s:.2f}s / source speech {target:.2f}s = {sp_s / max(target, 1e-6):.2f}" +
                    (f" fit-up {[x.get('result') for x in u['fit_up']]}" if u["fit_up"] else "") + (f" fit-down {[x.get('result') for x in u['fit']]}" if u["fit"] else "") +
                    (f" retried ({len(rec['attempts'])} attempts)" if len(rec["attempts"]) > 1 else "") + (" QA-BAD" if rec["qa_bad_final"] else "") + (f" span-merges {merged_spans}" if merged_spans else ""))

    used = None; t0 = time.perf_counter()
    for dtype in (torch.float16, torch.float32):
        try:
            for k_ in stats:
                stats[k_] = 0 if not isinstance(stats[k_], float) else 0.0
            JUDGE_STATS.update(calls=0, seconds=0.0)
            synth_all(dtype); used = dtype; break
        except Exception as e:
            log(f"  [tts] {dtype} failed: {type(e).__name__}: {str(e)[:200]} -> retrying next dtype")
    if used is None:
        raise RuntimeError("Chatterbox generation failed in every dtype")
    times["tts"] = round(time.perf_counter() - t0, 3); sr = m.sr; del m
    if qa_obj is not None and not model_cache.cached("tts_qa"):
        qa_obj.close()
    torch.cuda.empty_cache()
    stats["judge_calls"] = JUDGE_STATS["calls"]; stats["judge_seconds"] = JUDGE_STATS["seconds"]; stats["pause_marks_seconds"] = round(JUDGE_STATS.pop("pause_marks_s", 0.0), 2)
    for k_ in ("gen_seconds", "qa_seconds", "align_seconds", "fitup_qwen_seconds", "fitdown_qwen_seconds", "fitup_gen_qa_seconds", "judge_seconds", "gap_qa_seconds"):
        if k_ not in stats: continue
        stats[k_] = round(stats[k_], 2)
    stats["retry_success_rate"] = round(stats["retry_success"] / stats["retried_parts"], 3) if stats["retried_parts"] else None
    return {"model": "ChatterboxMultilingualTTS t3=v3 (models/chatterbox)", "sr": sr, "dtype": str(used).replace("torch.", ""), "language_id": tgt_lang, "mode": "phrase",
            "voice_reference": "speaker's own diarized speech via audio_prompt_path/prepare_conditionals", "speech_token_guard": {"enabled": True, "flow_calls": guard.calls, "violations": guard.violations},
            "qa": {"enabled": qa is not None, **qa_cfg}, "fit": {**fit}, "fit_up": {**fit_up}, "alignment": {"method": "ctc_forced_align (wav2vec2-base-960h, torchaudio.forced_align)" if aligner else "silero_valley_fallback", "cuts": "between words only"},
            "split": {"min_part_words": min_part_words, "soft_part_words": soft_part_words, "part_max_words": part_max_words, "retries": split_retries}, "stats": stats, "references": refs}
