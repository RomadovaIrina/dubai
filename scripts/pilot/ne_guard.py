"""Named-entity / proper-noun guard (r4 audio, 2026-10-20; user: 04 «город Сочи» was dubbed as "Solki").

Cause on 04: Qwen translated correctly ("Sochi city"), Chatterbox said "Soky city" (CTC "soky", Whisper "It's OKCITY"); the sentence CER (0.05)
passed the TTS QA, so no retry happened. Generic guard, no per-video lists:
  1. extract(): ONE Qwen call per video lists the proper nouns (people, places, brands) of the source sentences with the English form used in
     the translation; an entity is kept only if that form is really in the translation (transliterated, not translated).
  2. respellings(): pronunciation respellings for Chatterbox, derived from the Russian source form by a phonetic Cyrillic -> English map
     ("Сочи" -> "Sochee", "So-chee") + Qwen's own respelling; the written English form stays in the script, only the synthesised text changes.
  3. heard(): the TTS QA's free Whisper decode must contain every entity word (similarity >= 0.8); a miss marks the attempt bad, the retry
     ladder then synthesises the respellings in turn and keeps the best attempt.
  4. misheard(): the same check on the back-transcription of the final mix -> gate metric "named entities misheard"."""
from __future__ import annotations
import difflib, json, re

_RU = {"а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "yo", "ж": "zh", "з": "z", "и": "ee", "й": "y", "к": "k", "л": "l", "м": "m", "н": "n",
       "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "oo", "ф": "f", "х": "kh", "ц": "ts", "ч": "ch", "ш": "sh", "щ": "sh", "ъ": "", "ы": "i", "ь": "",
       "э": "e", "ю": "yu", "я": "ya"}
_VOW = set("аеёиоуыэюя")
EXTRACT_SYSTEM = "You extract proper nouns from a dubbing script. You answer with JSON only."


def ru_phonetic(word: str, hyphen: bool = False) -> str:
    """Cyrillic word -> English-reader respelling ("Сочи" -> "Sochee"; hyphen=True -> "So-chee": V-CV / VC-CV syllable splits)."""
    w = word.lower(); v = [i for i, ch in enumerate(w) if ch in _VOW]; cuts = []
    if hyphen:
        for i, j in zip(v, v[1:]): cuts.append(i + 1 if j - i - 1 <= 1 else i + 2)
    def tr(i, ch):
        if ch == "е" and (i == 0 or w[i - 1] in _VOW or w[i - 1] in "ьъ"): return "ye"
        return _RU.get(ch, ch)
    chunks, prev = [], 0
    for c in cuts + [len(w)]: chunks.append("".join(tr(i, w[i]) for i in range(prev, c))); prev = c
    s = "-".join(c for c in chunks if c)
    return s[:1].upper() + s[1:]


def _norm(t: str) -> list[str]:
    return re.findall(r"[a-z]+", t.lower())


_TOK = None


def rare(word: str) -> bool:
    """a RARE proper noun = its capitalised form is >= 2 Whisper BPE tokens (Sochi, Krasnodar); frequent names / nationalities / common words are
    one token (Moscow, Dubai, Finnish, Russian, sauna, class) and Chatterbox / Whisper handle them -> not guarded."""
    global _TOK
    if _TOK is None:
        import pathlib
        from tokenizers import Tokenizer
        _TOK = Tokenizer.from_file(str(pathlib.Path(__file__).resolve().parents[2] / "models" / "whisper-large-v3" / "tokenizer.json"))
    return len(_TOK.encode(" " + word[:1].upper() + word[1:].lower(), add_special_tokens=False).ids) >= 2


def guard_words(entity: str, translation: str) -> list[str]:
    """the entity words that are checked / respelled: capitalised in the translation, >= 3 letters, rare."""
    return [w for w in re.findall(r"[A-Za-z]+", entity) if len(w) >= 3 and re.search(r"\b" + w[:1].upper() + re.escape(w[1:]) + r"\b", translation) and rare(w)]


def _words(entity: str) -> list[str]:
    return [w for w in _norm(entity) if len(w) >= 3]


def heard(entity, transcript: str, thr: float = 0.8) -> bool:
    """every guarded word (entity dict: its 'guard' words; str: all words >= 3 letters) has a transcript word (or two joined words) with similarity >= thr."""
    ws = [w.lower() for w in entity["guard"]] if isinstance(entity, dict) else _words(entity)
    hw = _norm(transcript); cand = hw + [a + b for a, b in zip(hw, hw[1:])]
    return all(any(difflib.SequenceMatcher(None, w, h).ratio() >= thr for h in cand) for w in ws)


def respellings(ent: dict) -> list[str]:
    """alternative spellings for TTS, in retry order (deduplicated, never the written form itself): every guarded word is replaced by the phonetic
    respelling of the source word that sounds closest to it (any word order: «город сочи» -> "Sochi city"), plain then hyphenated, then Qwen's."""
    src = re.findall(r"[\u0400-\u04ff]+", ent.get("source", "")); eng = ent["english"]; out = []
    best = {}
    for g in ent.get("guard") or []:
        sc = [(difflib.SequenceMatcher(None, ru_phonetic(w).lower(), g.lower()).ratio(), w) for w in src]
        if sc and max(sc)[0] >= 0.4:
            w = max(sc)[1]
            while len(w) > 3 and w[-1] in _VOW | {"й"} and g[-1].lower() not in "aeiouy": w = w[:-1]   # case ending (Краснодаре -> Краснодар)
            best[g] = w
    for hy in (False, True):
        if best: out.append(re.sub(r"[A-Za-z]+", lambda m: ru_phonetic(best[m.group(0)], hy) if m.group(0) in best else m.group(0), eng))
    if ent.get("respell"): out.append(ent["respell"])
    seen, res = {eng.lower()}, []
    for r in out:
        if r and r.lower() not in seen: seen.add(r.lower()); res.append(r)
    return res


def apply(text: str, ents: list, k: int) -> str:
    """text with every entity replaced by its k-th respelling (k < 0: unchanged)."""
    if k < 0: return text
    for e in ents:
        r = e.get("respellings") or []
        if r: text = re.sub(r"\b" + re.escape(e["english"]) + r"\b", r[min(k, len(r) - 1)], text, flags=re.I)
    return text


def latin_respell(w: str) -> str:
    """a Latin-script name the source already writes in Latin (brands: Neometria, Flora): syllables hyphenated (V-CV / VC-CV), "Ne-o-met-ri-a"."""
    v = [i for i, ch in enumerate(w.lower()) if ch in "aeiouy"]; cuts = [i + 1 if j - i - 1 <= 1 else i + 2 for i, j in zip(v, v[1:])]
    return "-".join(w[a:b] for a, b in zip([0] + cuts, cuts + [len(w)]))


def unit_entities(u: dict) -> list:
    """r8 v4s: the unit's named entities to check by ear = u["ne"] (Qwen's list with its respellings) + rare Latin-script names written the same
    in the source and the translation (brands the list missed: 01 "Neometria", "Flora"), respelled by syllables."""
    ents = [dict(e, guard=e.get("guard") or [w for w in re.findall(r"[A-Za-z]+", e["english"]) if len(w) >= 3]) for e in (u.get("ne") or [])]
    have = {g.lower() for e in ents for g in e["guard"]}
    for w in dict.fromkeys(re.findall(r"\b[A-Z][a-z]{2,}\b", u.get("text", ""))):
        if w.lower() not in have and re.search(r"\b" + w + r"\b", u.get("translation", "")) and rare(w):
            ents.append({"english": w, "guard": [w], "respellings": [latin_respell(w)], "extra": True}); have.add(w.lower())
    return ents


WINDOW_TOKENS, WINDOW_OVERLAP, ANSWER_TOKENS = 3000, 2, 600


def _ntok(llm, text: str) -> int:
    try: return len(llm.tokenize(text.encode("utf-8"), add_bos=False))
    except Exception: return len(text) // 2 + 1   # no tokenizer: conservative for Cyrillic


def _windows(llm, head: str, rows: list) -> list:
    """r15 T (long videos: "Requested tokens (4929) exceed context window of 4096"): the script rows as ONE list when the whole prompt fits
    the model context (minus the answer), else consecutive windows of <= WINDOW_TOKENS tokens (with the instructions), WINDOW_OVERLAP rows shared."""
    try: n_ctx = int(llm.n_ctx())
    except Exception: n_ctx = 4096
    if _ntok(llm, EXTRACT_SYSTEM + head + "\n".join(rows)) <= n_ctx - ANSWER_TOKENS - 64: return [rows]
    h = _ntok(llm, EXTRACT_SYSTEM + head) + 16; out, cur, n = [], [], h
    for r in rows:
        t = _ntok(llm, r) + 1
        if cur and n + t > WINDOW_TOKENS:
            out.append(cur); cur = cur[-WINDOW_OVERLAP:]; n = h + sum(_ntok(llm, x) + 1 for x in cur)
        cur.append(r); n += t
    if cur: out.append(cur)
    return out


def extract(llm, units: list, log=print) -> dict:
    """Qwen lists the proper nouns per unit; kept only when the English form occurs in the unit translation. Sets u['ne'] = [entity ...]."""
    rows = [f'{u["id"]}: RU: {u.get("text", "").strip()} | EN: {u.get("translation", "").strip()}' for u in units if u.get("translation")]
    prompt = ("Below are numbered sentences of a Russian video with their English dubbing translation. List every proper noun (names of people, cities, "
              "districts, streets, companies, brands, products, buildings) that occurs in a sentence. For each give: the sentence number, the Russian words "
              "exactly as written in RU, the English form exactly as written in EN, and a simple English pronunciation respelling of the English form "
              "(how an English reader should say it, e.g. Sochi -> SOH-chee). Do NOT list common nouns, nationalities, languages, or generic objects, rooms, "
              "services or foods, even when they are foreign words. Answer with a JSON list of objects with the keys "
              '"unit", "source", "english", "respell"; [] if there are none.\n\n')
    stats = {"candidates": 0, "kept": 0, "entities": []}
    parts = _windows(llm, prompt, rows)   # r15 T: ONE window (= the unchanged single call) whenever the whole script fits the context
    items = []
    for k, part in enumerate(parts):
        try:
            r = llm.create_chat_completion([{"role": "system", "content": EXTRACT_SYSTEM}, {"role": "user", "content": prompt + "\n".join(part)}], max_tokens=600, temperature=0.0)
            txt = r["choices"][0]["message"]["content"]; m = re.search(r"\[.*\]", txt, re.S); got = json.loads(m.group(0)) if m else []
        except Exception as e:  # never block the dub on the guard
            if len(parts) == 1:
                log(f"  [ne] extraction failed: {type(e).__name__}: {str(e)[:120]}"); return {**stats, "error": str(e)[:200]}
            log(f"  [ne] extraction failed in window {k + 1}/{len(parts)}: {type(e).__name__}: {str(e)[:120]}"); stats.setdefault("window_errors", []).append(str(e)[:200]); continue
        items += got if isinstance(got, list) else []
    if len(parts) > 1:   # overlapping windows: the same (sentence, English form) once
        seen, uniq = set(), []
        for it in items:
            key = (str(it.get("unit")), str(it.get("english", "")).strip().lower()) if isinstance(it, dict) else None
            if key is not None and key not in seen: seen.add(key); uniq.append(it)
        items = uniq; stats["windows"] = len(parts)
        log(f"  [ne] long script: {len(rows)} sentences in {len(parts)} windows (<= {WINDOW_TOKENS} tokens, {WINDOW_OVERLAP} sentences overlap), {len(items)} candidates")
    by_id = {u["id"]: u for u in units}
    for it in items if isinstance(items, list) else []:
        try:
            u = by_id.get(int(it.get("unit"))); eng = str(it.get("english", "")).strip(); src = str(it.get("source", "")).strip()
        except Exception: continue
        stats["candidates"] += 1
        if not u or not eng or not _words(eng) or not re.search(r"\b" + re.escape(eng) + r"\b", u.get("translation", ""), re.I): continue
        if not re.search(r"[Ѐ-ӿ]", src) or src.lower() not in u.get("text", "").lower(): continue
        gw = guard_words(eng, u.get("translation", ""))
        if not gw: log(f"  [ne] u{u['id']:02d} {src} -> {eng}: no rare capitalised word, not guarded"); continue
        e = {"source": src, "english": eng, "guard": gw, "respell": str(it.get("respell", "")).strip()}; e["respellings"] = respellings(e)
        u.setdefault("ne", []).append(e); stats["kept"] += 1; stats["entities"].append({"unit": u["id"], **e})
        log(f"  [ne] u{u['id']:02d} {src} -> {eng} respell {e['respellings']}")
    return stats


def misheard(units: list, transcript: str) -> dict:
    """gate metric: entities of the script whose English form is not heard in the back-transcription of the final mix."""
    ents = [(u["id"], e) for u in units for e in (u.get("ne") or []) if e.get("guard")]
    miss = [f"u{i}:{e['english']}" for i, e in ents if not heard(e, transcript)]
    return {"entities": len(ents), "misheard": len(miss), "list": miss}
