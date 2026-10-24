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
_FILLER = re.compile(r"\b(um+|uh+|er+|erm|you know|basically|actually|literally|i mean|sort of|kind of|okay|you see|everyone|if you (?:know|understand) what i mean)\b", re.I)


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


CAND_SHOTS = [   # generic examples, deliberately NOT taken from the five test videos (no leakage into the evaluation): (source, reference, words, version)
    ("Сегодня мы | покажем вам нашу новую кухню.", "Today we will show you our new kitchen.", 12, "Today we are going to show all of you what our new kitchen looks like."),
    ("Ну вот, | работа закончена, | можно отдыхать.", "Well, the work is done, we can rest now.", 6, "Well, work's done, time to rest."),
    ("Здесь у нас спальня.", "Here we have the bedroom.", 8, "And over here we have got the bedroom.")]


def _cand_prompt(src_name, tgt_name, marked, plain, n):
    pause = f" The speaker pauses at every \"{SHOTS_SRC_PAUSE}\": where it reads naturally, end an English phrase (comma) or start a clause there." if SHOTS_SRC_PAUSE in marked else ""
    return (f"{src_name}: \"{marked}\"\nReference {tgt_name} translation ({words_of(plain)} words): \"{plain}\"\n"
            f"Write ONE natural spoken {tgt_name} dubbing version of the same sentence with {n} words (between {n - 1} and {n + 1}).{pause} Same meaning as the "
            f"reference: keep every number and name, add no new fact. Lengthen only with fuller natural wording of what is said (full verb forms, explicit "
            f"subjects, idiomatic spoken phrasing such as \"going to\", \"over here\"), shorten with tighter wording. Answer with the version only.")


def candidates(llm, src_name: str, tgt_name: str, marked: str, piece_s: list[float], plain: str, budgets: list[int], seconds: float) -> list[str]:
    """One Qwen conversation per word budget (3 generic shots, temperature 0): a version with budget +-1 words; a miss is fed back once with the
    count it got (Qwen 7B ignores budgets given as a list: 11-15 words asked, 8-10 delivered). Returns the versions (closest attempt per budget)."""
    out = []
    for n in budgets:
        msgs = [{"role": "system", "content": DUR_SYSTEM.replace(" You answer with JSON only.", " You answer with the version only.")}]
        for src, ref, k, ver in CAND_SHOTS:
            msgs += [{"role": "user", "content": _cand_prompt(src_name, tgt_name, src, ref, k)}, {"role": "assistant", "content": ver}]
        msgs.append({"role": "user", "content": _cand_prompt(src_name, tgt_name, marked, plain, n)}); best = None
        for attempt in range(2):
            raw = _chat(llm, msgs, max_tokens=24 + 4 * n)
            v = " ".join(raw.replace(SHOTS_SRC_PAUSE, " ").split()).strip().strip('"«»“”')
            if not v or "\n" in raw.strip(): break
            if best is None or abs(words_of(v) - n) < abs(words_of(best) - n): best = v
            if abs(words_of(v) - n) <= 1: break
            msgs += [{"role": "assistant", "content": raw}, {"role": "user", "content": f"That has {words_of(v)} words; it must have between {n - 1} and {n + 1} words with the same "
                     f"meaning ({'fuller, more explicit wording' if words_of(v) < n else 'tighter wording'}). Answer with the version only."}]
        if best: out.append(best)
    return out


PIECE_SHOTS = [   # generic (not from the test videos): pieces, seconds, word budgets, answer
    (["в прошлом году", "мы с братом поехали", "на море."], [0.8, 1.4, 0.7], [3, 6, 3], ["Last year,", "my brother and I went off", "to the seaside."]),
    (["ну что", "погнали дальше."], [0.6, 1.0], [3, 4], ["Well then,", "let's move on now."]),
    (["большая светлая комната", "и балкончик."], [1.6, 0.9], [7, 4], ["A big, bright and spacious room", "and a small balcony."])]


def _piece_prompt(src_name, tgt_name, pieces, secs, budgets):
    rows = "\n".join(f"{i + 1}. \"{p}\" ({s:.1f} s, about {b} words)" for i, (p, s, b) in enumerate(zip(pieces, secs, budgets)))
    return (f"The {src_name} speaker says these {len(pieces)} pieces with a pause after each:\n{rows}\nTranslate for dubbing: {len(pieces)} {tgt_name} pieces in the same order, "
            f"each saying what the {src_name} piece at the same position says, each about its word budget so it takes the same time, together one natural fluent "
            f"{tgt_name} sentence where a pause after each piece sounds natural (a phrase or clause ends there). Keep every number and name, add no new fact. "
            f"JSON only: {{\"pieces\": [...]}}")


def piece_candidates(llm, src_name: str, tgt_name: str, pieces: list[str], secs: list[float], wps: float, scales=(1.0,)) -> list[tuple[str, list[str]]]:
    """Source-pause-ordered candidates: one English piece per source piece (pause >= mark_pause), each budgeted to its burst (this voice's words / s x
    the piece's speech seconds), joined with a comma where the piece does not end a phrase, so pb1's boundary rule reproduces the source pause there.
    Returns [(joined text, pieces)] (one per scale of the budgets)."""
    from tts_burst import CLAUSE_START
    out = []
    for sc in scales:
        budgets = [max(1, int(round(wps * s_ * sc))) for s_ in secs]
        msgs = [{"role": "system", "content": DUR_SYSTEM}]
        for ps, ss, bs, ans in PIECE_SHOTS:
            msgs += [{"role": "user", "content": _piece_prompt(src_name, tgt_name, ps, ss, bs)}, {"role": "assistant", "content": json.dumps({"pieces": ans})}]
        msgs.append({"role": "user", "content": _piece_prompt(src_name, tgt_name, pieces, secs, budgets)})
        for attempt in range(2):
            raw = _chat(llm, msgs, max_tokens=60 + 6 * sum(budgets))
            try:
                got = [" ".join(str(x).split()).strip().strip('"«»“”') for x in (_json(raw).get("pieces") or [])]
            except Exception:
                got = []
            if len(got) == len(pieces) and all(words_of(g) for g in got):
                break
            msgs += [{"role": "assistant", "content": raw}, {"role": "user", "content": f"It must be JSON with exactly {len(pieces)} non-empty pieces. Answer again."}]; got = []
        if not got: continue
        txt = got[0]
        for g in got[1:]:
            first = re.sub(r"[^\w']", "", g.split()[0].lower()) if g.split() else ""
            if not re.search(r"[,.;:!?…—-]['\"»)]*$", txt) and first not in CLAUSE_START: txt += ","
            txt += " " + g
        txt = txt[0].upper() + txt[1:]
        out.append((txt, got))
    return out


JUDGE_SHOTS = [   # generic calibration examples (not from the test videos)
    ("Ну вот, работа закончена.", "Well then, the work is finished now.", {"back_translation": "Ну вот, работа теперь закончена.", "missing": [], "added": [], "score": 5, "natural": 5}),
    ("Здесь у нас спальня.", "And over here we have got the bedroom.", {"back_translation": "А вот здесь у нас спальня.", "missing": [], "added": [], "score": 5, "natural": 5}),
    ("Мы купили новый стол.", "We bought a new oak table yesterday.", {"back_translation": "Вчера мы купили новый дубовый стол.", "missing": [], "added": ["oak", "yesterday"], "score": 3, "natural": 5}),
    ("Квартира светлая и тихая.", "The apartment is bright.", {"back_translation": "Квартира светлая.", "missing": ["тихая"], "added": [], "score": 3, "natural": 5}),
    ("Мы приедем в пятницу вечером.", "We will arrive, on Friday, in the evening.", {"back_translation": "Мы приедем, в пятницу, вечером.", "missing": [], "added": [], "score": 5, "natural": 3}),
    ("Дом стоит у самого моря.", "The house stands right, by the sea.", {"back_translation": "Дом стоит прямо у моря.", "missing": [], "added": [], "score": 5, "natural": 2})]


def _filter_items(items: list[str], source: str, ref: str | None) -> list[str]:
    """Drop judge items that are not real differences (7B noise, 05 smoke2: "of course" flagged as added for «конечно же»): an item quoted in the
    source language that occurs in the source, or an English item that the trusted reference translation also says."""
    out = []
    for it in items:
        t = it.lower().strip(" '\".")
        if not t: continue
        if re.search(r"[а-яё]", t) and all(w in source.lower() for w in re.findall(r"[а-яё]+", t)): continue
        if ref and re.fullmatch(r"[a-z' ]+", t) and re.search(r"\b" + re.escape(t) + r"\b", ref.lower()): continue
        out.append(it)
    return out


def judge(llm, src_name: str, tgt_name: str, source: str, cand: str, ref: str | None = None) -> dict:
    """ONE Qwen call (4 calibration shots): back-translate the candidate into the source language, then list the FACTS it drops / adds and score the
    meaning 1-5 (wording, particles, connectors and fuller phrasing are free). `ref` is kept for the record only (the shots calibrate the score)."""
    def q(src, c):
        return (f"{src_name} source: \"{src}\"\n{tgt_name} dubbing version: \"{c}\"\nBack-translate the version into {src_name}, then list the facts (things, numbers, "
                f"names, actions, qualities, times) of the source that the version drops (\"missing\") and the facts it adds (\"added\"). Wording, word order, synonyms, "
                f"particles, connectors and fuller phrasing are NOT facts. Score 5 = same facts, 4 = same facts with a nuance changed, 3 = one fact missing or added, "
                f"2 = several, 1 = wrong meaning. Natural 5 = fluent spoken English with commas only where a speaker naturally pauses, 3 = one odd break or awkward phrase, "
                f"1 = broken English. JSON only.")
    msgs = [{"role": "system", "content": JUDGE_SYSTEM}]
    for src, c, o in JUDGE_SHOTS:
        msgs += [{"role": "user", "content": q(src, c)}, {"role": "assistant", "content": json.dumps(o, ensure_ascii=False)}]
    msgs.append({"role": "user", "content": q(source, cand)})
    raw = _chat(llm, msgs, max_tokens=320)
    try:
        o = _json(raw); sc = int(o.get("score", 0))
        miss = _filter_items([str(x) for x in (o.get("missing") or [])], source, None); add = _filter_items([str(x) for x in (o.get("added") or [])], source, ref)
        if sc == 3 and not miss and not add and (o.get("missing") or o.get("added")): sc = 4   # every listed difference was noise
        o["missing"], o["added"], o["score"] = miss, add, sc
        return {"score": int(o.get("score", 0)), "natural": int(o.get("natural", 5)), "back": str(o.get("back_translation", ""))[:300], "missing": [str(x) for x in (o.get("missing") or []) if str(x).strip()][:5],
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
    seen = {plain.lower().strip(" .")}; marked, piece_s = marked_source(u, p, cfg.get("mark_pause", 0.6)); salt = 60
    def total(c):   # selection cost: placement seconds + fidelity + QA
        return c["sim"]["cost"] + cfg.get("fid_w", 0.12) * (5 - min(5, max(1, c["score"]))) + (2.0 if c["rec"].get("qa_bad_final") else 0.0) - (cfg.get("switch_margin", 0.15) if c.get("plain") else 0.0)
    for rnd in range(cfg.get("rounds", 2)):
        b = best; spk_b = spoken(b["text"]); rate = syllables(spk_b) / max(b["s"], 0.3)   # this voice's syllables / s on the current best take
        ideal = max(0.4, b["sim"]["ideal_s"]); goal_syl = rate * ideal; wpsyl = words_of(b["text"]) / max(1, syllables(spk_b))
        goal_w = goal_syl * wpsyl
        budgets = sorted({max(3, int(round(goal_w * f))) for f in cfg.get("budget_f", (0.9, 1.0, 1.12))})
        if rnd and budgets == info["rounds"][-1]["budgets"]: break   # the same budgets give the same (temperature 0) versions
        cands = candidates(llm, src_name, tgt_name, marked, piece_s, plain, budgets, ideal)
        if len(piece_s) > 1 and cfg.get("pieces", True):   # source-pause-ordered candidates (a boundary at every source pause, each piece budgeted to its burst)
            wps = words_of(b["text"]) / max(b["s"], 0.3)
            pc = [t_ for t_, _ in piece_candidates(llm, src_name, tgt_name, marked.split(" | "), piece_s, wps, scales=cfg.get("piece_scales", (1.0, 1.15)))]
            piece_set = {t_.lower().strip(" .") for t_ in pc}; cands += pc
        else:
            piece_set = set()
        log(f"    [dur] u{u['id']:02d} r{rnd + 1} rate {rate:.2f} syl/s ideal {ideal:.2f}s budgets {budgets} -> {len(cands)} versions {[words_of(c) for c in cands]}")
        rr = {"round": rnd + 1, "rate_syl_s": round(rate, 2), "ideal_s": round(ideal, 2), "goal_words": round(goal_w, 1), "budgets": budgets, "candidates": []}; info["rounds"].append(rr)
        pool = []
        for c in cands:
            key = c.lower().strip(" ."); STATS["candidates"] += 1
            if key in seen: continue
            seen.add(key); why = guard(u["text"], plain, c, u.get("ne") or [])
            pred = syllables(spoken(c)) / max(rate, 0.5); row = {"text": c, "words": words_of(c), "pred_s": round(pred, 2)}; rr["candidates"].append(row)
            if why: row["reject"] = why; STATS["rejected_guard"] += 1; log(f"    [dur] u{u['id']:02d} r{rnd + 1} guard: {why} | {c[:70]}"); continue
            if key in piece_set: row["pieces"] = True   # their ideal is the speech of the bursts, not the merged window: always synthesised first
            pool.append((-1.0 if key in piece_set else abs(math.log(max(pred, 0.1) / ideal)), c, row))
        pool.sort(key=lambda t: t[0]); tried = 0
        if rnd == 0 and cfg.get("plain_reseeds", 1):   # the plain text with other seeds: Chatterbox takes vary +-10 % in length, a free faithful lever
            pool = [(0.0, plain, {"text": plain, "reseed": True})] * cfg.get("plain_reseeds", 1) + pool
        for _, c, row in pool:
            if tried >= cfg.get("tts_top", 3) + (cfg.get("plain_reseeds", 1) if rnd == 0 else 0): break
            if row.get("reseed"):
                row = dict(row); rr["candidates"].append(row); j = {"score": best["score"] if best.get("plain") else (info.get("plain_judge") or {}).get("score", 5)}
            else:
                j = judge(llm, src_name, tgt_name, u["text"], c, ref=plain); row["judge"] = j
            if not row.get("reseed") and (j["score"] < cfg.get("min_score", 4) or j.get("natural", 5) < cfg.get("min_natural", 4)):
                row["reject"] = f"judge {j['score']} natural {j.get('natural')}" + (f" added {j['added']}" if j["added"] else ""); STATS["rejected_judge"] += 1
                log(f"    [dur] u{u['id']:02d} r{rnd + 1} {row['reject']} missing {j.get('missing')} | {c[:70]}"); continue
            tried += 1; t = time.perf_counter(); y, rec, sp, s = synth(c, salt); salt += 1; STATS["tts"] += 1; STATS["tts_s"] += time.perf_counter() - t
            cs = sim(y, sp, c); row.update(sim=cs, tts_s=round(s, 2), qa_bad=bool(rec.get("qa_bad_final")), qa_score=rec.get("qa_score"))
            cand = {"text": c, "y": y, "rec": rec, "spans": sp, "s": s, "sim": cs, "score": j["score"], "plain": bool(row.get("reseed"))}
            row["total"] = round(total(cand), 3)
            log(f"    [dur] u{u['id']:02d} r{rnd + 1} {'reseed' if row.get('reseed') else 'piece' if row.get('pieces') else 'cand'} tts {s:.2f}s cost {cs['cost']:.2f} (over {cs['over_s']:.2f} under {cs['under_s']:.2f}) "
                f"total {row['total']:.2f} score {j['score']}{' QA-BAD' if rec.get('qa_bad_final') else ''} | {c[:70]}")
            if total(cand) < total(best): best = cand
        if best["sim"]["cost"] <= cfg.get("ok_cost", 0.25) or not pool:
            break
    # pace: a sentence still much shorter than its source speech (slow speaker: 05 u03 7.6 s of speech, 3.5 syl/s) -> the best text (and the plain one)
    # at Chatterbox's slower deliberate pace (cfg_weight < 0.5; +15-25 % length on a 4 s sentence) instead of an atempo slow-down
    if cfg.get("pace", (0.3, 0.2)) and best["sim"]["under_s"] > cfg.get("pace_under_s", 0.4):
        texts = [best["text"]] + ([plain] if not best.get("plain") and cfg.get("pace_plain", True) else [])
        for cw in cfg.get("pace", (0.3, 0.2)):
            for tx in texts:
                is_plain = tx == plain; sc_ = (info.get("plain_judge") or {}).get("score", 5) if is_plain else best["score"]
                t = time.perf_counter(); y, rec, sp, s = synth(tx, salt, cfg_w=cw); salt += 1; STATS["tts"] += 1; STATS["tts_s"] += time.perf_counter() - t; STATS["pace_takes"] = STATS.get("pace_takes", 0) + 1
                cs = sim(y, sp, tx); cand = {"text": tx, "y": y, "rec": rec, "spans": sp, "s": s, "sim": cs, "score": sc_, "plain": is_plain, "cfg_w": cw}
                row = {"text": tx, "pace_cfg_weight": cw, "sim": cs, "tts_s": round(s, 2), "qa_bad": bool(rec.get("qa_bad_final")), "qa_score": rec.get("qa_score"), "total": round(total(cand), 3)}
                info.setdefault("pace", []).append(row)
                log(f"    [dur] u{u['id']:02d} pace cfg {cw} tts {s:.2f}s cost {cs['cost']:.2f} (under {cs['under_s']:.2f}) total {row['total']:.2f}{' QA-BAD' if rec.get('qa_bad_final') else ''} | {tx[:60]}")
                if total(cand) < total(best): best = cand
            if best["sim"]["under_s"] <= cfg.get("pace_under_s", 0.4): break
    info["cfg_weight"] = best.get("cfg_w", 0.5)
    if best.get("cfg_w"): STATS["paced"] = STATS.get("paced", 0) + 1
    info["chosen"] = ("plain" if best["y"] is y0 else "plain (other seed)") if best.get("plain") else best["text"]; info["chosen_sim"] = best["sim"]; info["chosen_score"] = best["score"]
    if not best.get("plain"):
        STATS["switched"] += 1
    elif best["y"] is not y0:
        STATS["reseeded"] = STATS.get("reseeded", 0) + 1
    log(f"  [dur] u{u['id']:02d} plain cost {c0['cost']:.2f}s (tts {c0['speech_s']:.2f}/{c0['ideal_s']:.2f}s) -> "
        f"{info['chosen'] if best.get('plain') else repr(best['text'][:70])} cost {best['sim']['cost']:.2f}s score {best['score']}")
    return best["text"], best["y"], best["rec"], best["spans"], best["s"]
