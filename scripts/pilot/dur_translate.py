"""r5 duration-aware translation (--dur-translate on): fix the EN/RU length mismatch at the source instead of patching it afterwards.

Round 4 left rushed speech (atempo x1.15), "source talks, dub silent" stretches and long merged silences in pb1; all three come from an English
sentence whose natural TTS length does not match the source speech it replaces, and the reactive fit-up rewrite (one candidate, accepted only when
closer) rarely passed. Per unit, after the plain translation's first TTS take:
  1. dry-run the REAL downstream placement of that take (flow merge -> boundary-only cut DP -> burst_fit_ratios) and cost it in seconds:
     1.5 x speech that needs a speed-up beyond 1.05  +  1.0 x source speech left without dub (part shorter than 0.95 x its burst)
     +  0.3 x dub running over source silence inside its window;
  2. when that cost is above `ok_cost`, Qwen writes N alternative translations with explicit word budgets derived from the source speech the
     sentence must fill and this voice's measured rate on the plain take (syllables / s), with the source pause points marked so the English
     phrase boundaries can land there (pb1 then pauses exactly where the speaker does);
  3. deterministic guards (numbers, names / loanwords, foreign script, filler) + ONE Qwen judge call per candidate that first back-translates
     the candidate into the source language and then scores meaning 1-5 with the missing / added items -> only score >= min_score, nothing added;
  4. TTS of the top-K candidates by predicted duration (syllables / rate), each costed by the same dry run; the cheapest wins (a small fidelity
     term per judge point, and the plain translation wins ties within `switch_margin` s).
Everything is recorded per unit in u["dur_translate"]; STATS collects the Qwen / TTS time of the stage."""
from __future__ import annotations

import copy
import json
import math
import re
import time

STATS = {"units": 0, "triggered": 0, "switched": 0, "candidates": 0, "rejected_guard": 0, "rejected_judge": 0, "tts": 0, "qwen_s": 0.0, "tts_s": 0.0}
SHOTS_SRC_PAUSE = "|"
DUR_SYSTEM = "You are a professional dubbing translator. You write natural spoken translations that fit a given length. You answer with JSON only."
JUDGE_SYSTEM = "You are a strict bilingual reviewer of dubbing translations. You answer with JSON only."
_FILLER = re.compile(r"\b(um+|uh+|er+|erm|you know|basically|actually|literally|i mean|sort of|kind of)\b", re.I)


def syllables(text: str) -> int:
    """English syllables (vowel-group heuristic with silent-e / -le / -ed rules); digits count as their spoken length roughly."""
    n = 0
    for w in re.findall(r"[a-z']+|\d+", text.lower()):
        if w.isdigit():
            n += 2 * len(w); continue
        w = w.replace("'", "")
        g = re.findall(r"[aeiouy]+", w); k = len(g)
        if w.endswith("e") and not w.endswith(("le", "ee", "ye")) and k > 1: k -= 1
        if w.endswith("ed") and not w.endswith(("ted", "ded")) and k > 1: k -= 1
        n += max(1, k)
    return n


def words_of(text: str) -> int:
    return len(re.findall(r"[\w']+", text))


def marked_source(u: dict, p: dict, min_pause: float) -> tuple[str, list[float]]:
    """The source sentence with " | " at every source pause >= min_pause s (bursts of the plan) and the speech seconds of every piece."""
    toks = u["text"].split(); counts = list(p["group_counts"])
    groups = [" ".join(toks[sum(counts[:i]):sum(counts[:i + 1])]) for i in range(len(counts))] if len(toks) == sum(counts) else list(p["groups"])
    pieces, secs = [groups[0]], [p["bursts"][0][1] - p["bursts"][0][0]]
    for prev, cur, g in zip(p["bursts"], p["bursts"][1:], groups[1:]):
        if cur[0] - prev[1] < min_pause: pieces[-1] += " " + g; secs[-1] += cur[1] - prev[1]
        else: pieces.append(g); secs.append(cur[1] - cur[0])
    return " | ".join(pieces), [round(s, 2) for s in secs]


def _chat(llm, msgs, max_tokens: int, temperature: float = 0.0) -> str:
    t = time.perf_counter()
    r = llm.create_chat_completion(msgs, max_tokens=max_tokens, temperature=temperature)
    STATS["qwen_s"] += time.perf_counter() - t
    return r["choices"][0]["message"]["content"]


def _json(txt: str):
    txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip()); m = re.search(r"[\[{].*[\]}]", txt, re.S)
    return json.loads(m.group(0) if m else txt)


def candidates(llm, src_name: str, tgt_name: str, marked: str, piece_s: list[float], plain: str, budgets: list[int], seconds: float) -> list[str]:
    """ONE Qwen call: len(budgets) alternative translations, the i-th with about budgets[i] words. Returns the parsed strings (maybe fewer)."""
    k = len(budgets)
    pause_note = (f" The speaker pauses at every \"{SHOTS_SRC_PAUSE}\" (piece lengths: {', '.join(f'{s:.1f} s' for s in piece_s)}): where it reads naturally, let an English "
                  f"phrase end (comma) or a new clause start at the same place, so the dub can pause there too. Never write \"{SHOTS_SRC_PAUSE}\" in the answer.") if len(piece_s) > 1 else ""
    prompt = (f"{src_name} sentence (spoken in {seconds:.1f} seconds of speech): \"{marked}\"\nCurrent {tgt_name} dubbing translation ({words_of(plain)} words): \"{plain}\"\n\n"
              f"The dub must take the same time as the speaker at a normal speaking pace. Write {k} alternative natural spoken {tgt_name} translations of the SAME sentence, "
              f"with about {', '.join(str(b) for b in budgets)} words respectively.{pause_note} Every version must say everything the speaker says and nothing more: keep every "
              f"number, name and fact, add no new fact, quality or outcome, no filler words (um, you know, basically, actually). Reach a longer length with fuller, more "
              f"explicit wording of the same content (spelled-out subjects, full verb forms, a more literal rendering); reach a shorter length with tighter wording. "
              f"Answer with JSON only: {{\"versions\": [\"...\", ...]}}.")
    msgs = [{"role": "system", "content": DUR_SYSTEM}, {"role": "user", "content": prompt}]
    for attempt in range(2):
        raw = _chat(llm, msgs, max_tokens=80 + 40 * k + 3 * sum(budgets))
        try:
            obj = _json(raw); vs = obj.get("versions") if isinstance(obj, dict) else obj
            out = [" ".join(str(v).split()).strip().strip('"«»“”') for v in vs if isinstance(v, str) and v.strip()]
            if out:
                return [o.replace(SHOTS_SRC_PAUSE, " ").replace("  ", " ") for o in out]
        except Exception:
            pass
        msgs = msgs + [{"role": "assistant", "content": raw}, {"role": "user", "content": "That was not valid JSON of the form {\"versions\": [\"...\"]}. Answer again with JSON only."}]
    return []


def judge(llm, src_name: str, tgt_name: str, source: str, cand: str) -> dict:
    """ONE Qwen call: back-translate the candidate into the source language, then score its meaning against the source (1-5) with missing / added items."""
    prompt = (f"{src_name} source: \"{source}\"\n{tgt_name} dubbing translation: \"{cand}\"\n\n"
              f"Step 1: translate the {tgt_name} text back into {src_name} literally. Step 2: compare that back-translation with the {src_name} source. List every fact, detail "
              f"or claim of the source that the translation drops (\"missing\") and every fact, detail, quality or outcome that the translation states but the source does not "
              f"(\"added\"). Synonyms, paraphrase, word order, spelled-out pronouns, connectors and a more literal or more fluent rendering are NOT differences. Then score the "
              f"meaning: 5 = same meaning, 4 = same meaning with a negligible nuance lost, 3 = a detail missing or added, 2 = meaning partly wrong, 1 = wrong.\n"
              f"Answer with JSON only: {{\"back_translation\": \"...\", \"missing\": [...], \"added\": [...], \"score\": N}}")
    raw = _chat(llm, [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}], max_tokens=320)
    try:
        o = _json(raw)
        return {"score": int(o.get("score", 0)), "back": str(o.get("back_translation", ""))[:300], "missing": [str(x) for x in (o.get("missing") or []) if str(x).strip()][:5],
                "added": [str(x) for x in (o.get("added") or []) if str(x).strip()][:5]}
    except Exception:
        return {"score": 0, "back": "", "missing": [], "added": [], "unparsable": raw[:200]}


def guard(source: str, plain: str, cand: str, ents: list) -> str | None:
    """Deterministic rejections before any judge / TTS call."""
    from burst_aware_audio import missing_numbers
    from tts_burst import dropped_words
    from tts_text import script_defects
    if not cand or "\n" in cand: return "empty"
    if cand.lower().strip(" .") == plain.lower().strip(" ."): return "identical"
    if script_defects(cand): return f"script {script_defects(cand)}"
    if missing_numbers(source, cand) and not missing_numbers(source, plain): return "number lost"
    if dropped_words(plain, cand, source): return f"name / loanword dropped {dropped_words(plain, cand, source)}"
    for e in ents or []:
        if re.search(r"\b" + re.escape(e["english"]) + r"\b", plain, re.I) and not re.search(r"\b" + re.escape(e["english"]) + r"\b", cand, re.I):
            return f"named entity dropped: {e['english']}"
    if _FILLER.search(cand) and not _FILLER.search(plain): return f"filler '{_FILLER.search(cand).group(0)}'"
    if words_of(cand) < 2: return "too short"
    return None


def simulate(y, sr: int, spans, spoken: str, p: dict, opts, prev_end, flow_gap: float, cut_kw: dict, norm) -> dict:
    """Dry run of the downstream placement of ONE take: the same flow merge, boundary-only cut DP and window rule as run_tts_phrase + place_groups.
    Returns the cost in seconds and its parts. y = the take, spans = its CTC word spans (None -> whole take as one part on the merged window)."""
    import numpy as np
    from e1_burst_align import burst_fit_ratios
    from tts_burst import distribute_words_syntax, flow_merge_bursts
    q = copy.deepcopy(p); flow_merge_bursts(q, flow_gap)
    raw = spoken.split(); counts = None
    if spans is not None and len(q["bursts"]) > 1:
        try:
            counts = distribute_words_syntax(spans, raw, q["bursts"], norm, y=y, sr=sr, **cut_kw)
        except Exception:
            counts = None
    if spans is None or counts is None:   # one part over the union of all bursts
        q["bursts"] = [(q["bursts"][0][0], q["bursts"][-1][1])]; counts = [len(spans) if spans else 1]
    if 0 in counts:
        for j in range(len(counts) - 2, -1, -1):
            if counts[j] == 0:
                q["bursts"][j:j + 2] = [(q["bursts"][j][0], q["bursts"][j + 1][1])]; counts[j:j + 2] = [counts[j + 1]]
    durs = []; k0 = 0
    for c in counts:
        if spans:
            durs.append(max(0.05, spans[k0 + c - 1]["end"] - spans[k0]["start"]))
        else:
            durs.append(len(y) / sr)
        k0 += c
    ratios = burst_fit_ratios([np.zeros(int(d * sr), dtype=np.float32) for d in durs], q["bursts"], q["slot"], q["next_start"], opts, sr=sr, prev_end=prev_end)
    over = under = extra = 0.0; rows = []
    for d, r, (bs, be) in zip(durs, ratios, q["bursts"]):
        ideal = be - bs; win = d / r if r else ideal
        o = max(0.0, d - 1.05 * win); un = max(0.0, 0.95 * ideal - d); ex = max(0.0, min(d, win) - 1.05 * ideal)
        over += o; under += un; extra += ex; rows.append([round(d, 2), round(ideal, 2), round(win, 2)])
    cost = 1.5 * over + 1.0 * under + 0.3 * extra
    return {"cost": round(cost, 3), "over_s": round(over, 3), "under_s": round(under, 3), "extra_s": round(extra, 3), "parts": rows, "ideal_s": round(sum(r_[1] for r_ in rows), 3),
            "speech_s": round(sum(durs), 3)}


def select(u: dict, p: dict, plain: str, first: tuple, synth, sim, llm, src_name: str, tgt_name: str, cfg: dict, spoken, log=print) -> tuple:
    """Per unit. first = (y, rec, spans, speech_seconds) of the plain take; synth(text, salt) -> the same tuple for another text; sim(y, spans, text) ->
    simulate() dict. Returns the chosen (text, y, rec, spans, speech_seconds) and writes u["dur_translate"]."""
    STATS["units"] += 1
    y0, rec0, sp0, s0 = first; c0 = sim(y0, sp0, plain)
    info = {"plain": plain, "plain_sim": c0, "chosen": "plain", "rounds": []}; u["dur_translate"] = info
    if cfg.get("judge_plain", True):
        info["plain_judge"] = judge(llm, src_name, tgt_name, u["text"], plain)
    if c0["cost"] <= cfg.get("ok_cost", 0.25) and not rec0.get("qa_bad_final"):
        return plain, y0, rec0, sp0, s0
    STATS["triggered"] += 1
    best = {"text": plain, "y": y0, "rec": rec0, "spans": sp0, "s": s0, "sim": c0, "score": (info.get("plain_judge") or {}).get("score", 5), "plain": True}
    seen = {plain.lower().strip(" .")}; marked, piece_s = marked_source(u, p, cfg.get("mark_pause", 0.45)); salt = 60
    def total(c):   # selection cost: placement seconds + fidelity + QA
        return c["sim"]["cost"] + cfg.get("fid_w", 0.12) * (5 - min(5, max(1, c["score"]))) + (2.0 if c["rec"].get("qa_bad_final") else 0.0) - (cfg.get("switch_margin", 0.15) if c.get("plain") else 0.0)
    for rnd in range(cfg.get("rounds", 2)):
        b = best; spk_b = spoken(b["text"]); rate = syllables(spk_b) / max(b["s"], 0.3)   # this voice's syllables / s on the current best take
        ideal = max(0.4, b["sim"]["ideal_s"]); goal_syl = rate * ideal; wpsyl = words_of(b["text"]) / max(1, syllables(spk_b))
        goal_w = goal_syl * wpsyl
        budgets = sorted({max(3, int(round(goal_w * f))) for f in cfg.get("budget_f", (0.9, 1.0, 1.12))})
        cands = candidates(llm, src_name, tgt_name, marked, piece_s, plain, budgets, ideal)
        rr = {"round": rnd + 1, "rate_syl_s": round(rate, 2), "ideal_s": round(ideal, 2), "goal_words": round(goal_w, 1), "budgets": budgets, "candidates": []}; info["rounds"].append(rr)
        pool = []
        for c in cands:
            key = c.lower().strip(" ."); STATS["candidates"] += 1
            if key in seen: continue
            seen.add(key); why = guard(u["text"], plain, c, u.get("ne") or [])
            pred = syllables(spoken(c)) / max(rate, 0.5); row = {"text": c, "words": words_of(c), "pred_s": round(pred, 2)}; rr["candidates"].append(row)
            if why: row["reject"] = why; STATS["rejected_guard"] += 1; continue
            pool.append((abs(math.log(max(pred, 0.1) / ideal)), c, row))
        pool.sort(key=lambda t: t[0]); tried = 0
        for _, c, row in pool:
            if tried >= cfg.get("tts_top", 3): break
            j = judge(llm, src_name, tgt_name, u["text"], c); row["judge"] = j
            if j["score"] < cfg.get("min_score", 4) or j["added"]:
                row["reject"] = f"judge {j['score']}" + (f" added {j['added']}" if j["added"] else ""); STATS["rejected_judge"] += 1; continue
            tried += 1; t = time.perf_counter(); y, rec, sp, s = synth(c, salt); salt += 1; STATS["tts"] += 1; STATS["tts_s"] += time.perf_counter() - t
            cs = sim(y, sp, c); row.update(sim=cs, tts_s=round(s, 2), qa_bad=bool(rec.get("qa_bad_final")), qa_score=rec.get("qa_score"))
            cand = {"text": c, "y": y, "rec": rec, "spans": sp, "s": s, "sim": cs, "score": j["score"]}
            row["total"] = round(total(cand), 3)
            if total(cand) < total(best): best = cand
        if best["sim"]["cost"] <= cfg.get("ok_cost", 0.25) or not pool:
            break
    info["chosen"] = "plain" if best.get("plain") else best["text"]; info["chosen_sim"] = best["sim"]; info["chosen_score"] = best["score"]
    if not best.get("plain"):
        STATS["switched"] += 1
    log(f"  [dur] u{u['id']:02d} plain cost {c0['cost']:.2f}s (tts {c0['speech_s']:.2f}/{c0['ideal_s']:.2f}s) -> "
        f"{'kept plain' if best.get('plain') else repr(best['text'][:70])} cost {best['sim']['cost']:.2f}s score {best['score']}")
    return best["text"], best["y"], best["rec"], best["spans"], best["s"]
