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


def _words(entity: str) -> list[str]:
    return [w for w in _norm(entity) if len(w) >= 3]


def heard(entity: str, transcript: str, thr: float = 0.8) -> bool:
    """every entity word (>= 3 letters) has a transcript word (or two joined words) with similarity >= thr."""
    hw = _norm(transcript); cand = hw + [a + b for a, b in zip(hw, hw[1:])]
    return all(any(difflib.SequenceMatcher(None, w, h).ratio() >= thr for h in cand) for w in _words(entity))


def respellings(ent: dict) -> list[str]:
    """alternative spellings for TTS, in retry order (deduplicated, never the written form itself)."""
    src = [w for w in re.findall(r"[Ѐ-ӿ]+", ent.get("source", ""))]
    eng = ent["english"].split(); out = []
    if src and len(src) == len(eng):   # word-for-word: respell only the words that are transliterations (keep "City", "Park" ...)
        for hy in (False, True):
            ws = [ru_phonetic(s, hy) if difflib.SequenceMatcher(None, ru_phonetic(s).lower(), e.lower()).ratio() >= 0.5 else e for s, e in zip(src, eng)]
            out.append(" ".join(ws))
    elif src:
        out += [" ".join(ru_phonetic(s) for s in src), " ".join(ru_phonetic(s, True) for s in src)]
    if ent.get("respell"): out.append(ent["respell"])
    seen, res = {ent["english"].lower()}, []
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


def extract(llm, units: list, log=print) -> dict:
    """Qwen lists the proper nouns per unit; kept only when the English form occurs in the unit translation. Sets u['ne'] = [entity ...]."""
    rows = [f'{u["id"]}: RU: {u.get("text", "").strip()} | EN: {u.get("translation", "").strip()}' for u in units if u.get("translation")]
    prompt = ("Below are numbered sentences of a Russian video with their English dubbing translation. List every proper noun (names of people, cities, "
              "districts, streets, companies, brands, products, buildings) that occurs in a sentence. For each give: the sentence number, the Russian words "
              "exactly as written in RU, the English form exactly as written in EN, and a simple English pronunciation respelling of the English form "
              "(how an English reader should say it, e.g. Sochi -> SOH-chee). Do NOT list common nouns. Answer with a JSON list of objects with the keys "
              '"unit", "source", "english", "respell"; [] if there are none.\n\n' + "\n".join(rows))
    stats = {"candidates": 0, "kept": 0, "entities": []}
    try:
        r = llm.create_chat_completion([{"role": "system", "content": EXTRACT_SYSTEM}, {"role": "user", "content": prompt}], max_tokens=600, temperature=0.0)
        txt = r["choices"][0]["message"]["content"]; m = re.search(r"\[.*\]", txt, re.S); items = json.loads(m.group(0)) if m else []
    except Exception as e:  # never block the dub on the guard
        log(f"  [ne] extraction failed: {type(e).__name__}: {str(e)[:120]}"); return {**stats, "error": str(e)[:200]}
    by_id = {u["id"]: u for u in units}
    for it in items if isinstance(items, list) else []:
        try:
            u = by_id.get(int(it.get("unit"))); eng = str(it.get("english", "")).strip(); src = str(it.get("source", "")).strip()
        except Exception: continue
        stats["candidates"] += 1
        if not u or not eng or not _words(eng) or not re.search(r"\b" + re.escape(eng) + r"\b", u.get("translation", ""), re.I): continue
        if not re.search(r"[Ѐ-ӿ]", src) or src.lower() not in u.get("text", "").lower(): continue
        e = {"source": src, "english": eng, "respell": str(it.get("respell", "")).strip()}; e["respellings"] = respellings(e)
        u.setdefault("ne", []).append(e); stats["kept"] += 1; stats["entities"].append({"unit": u["id"], **e})
        log(f"  [ne] u{u['id']:02d} {src} -> {eng} respell {e['respellings']}")
    return stats


def misheard(units: list, transcript: str) -> dict:
    """gate metric: entities of the script whose English form is not heard in the back-transcription of the final mix."""
    ents = [(u["id"], e["english"]) for u in units for e in (u.get("ne") or [])]
    miss = [f"u{i}:{e}" for i, e in ents if not heard(e, transcript)]
    return {"entities": len(ents), "misheard": len(miss), "list": miss}
