#!/usr/bin/env python3
"""r13: the mixed-script guard for EVERY target language (the accepted EN chain has its own: tts_text.script_defects, English only).

Qwen 14B code-switches on some targets (LANGS.md: tr / pl / pt / hi / ko / ar / he): Russian words left untranslated, Chinese / Korean /
Arabic leaking in, character loops, English "Note:" asides. defects() finds them; guarded() retries with a strict prompt (twice: greedy,
then temperature 0.3) and keeps the cleanest candidate (fewest defects, then the first).
    from lang_guard import guarded; text, info = guarded(chat, ru_text, "tr", "Turkish")    # chat(messages, temperature) -> str"""
from __future__ import annotations
import re, unicodedata
from collections import Counter

SCRIPT = {"el": "GREEK", "he": "HEBREW", "ar": "ARABIC", "hi": "DEVANAGARI", "zh": "CJK", "ja": "JA", "ko": "HANGUL", "ru": "CYRILLIC", "uk": "CYRILLIC"}
SYSTEM_PROMPT = "You are a professional translator. Return only the translation, with no explanation."


def script_of(ch: str) -> str:
    n = unicodedata.name(ch, "")
    if "CJK" in n: return "CJK"
    if "HIRAGANA" in n or "KATAKANA" in n: return "KANA"
    for s in ("LATIN", "CYRILLIC", "GREEK", "HEBREW", "ARABIC", "DEVANAGARI", "HANGUL"):
        if n.startswith(s) or f" {s} " in f" {n} ": return s
    return "OTHER"


def foreign_share(text: str, lang: str) -> float:
    """letters in a script that is neither the target's nor Latin (names / brands in Latin are fine)"""
    want = SCRIPT.get(lang, "LATIN"); ok = {"LATIN", "CJK", "KANA"} if want == "JA" else {"LATIN", want}
    c = Counter(script_of(ch) for ch in text if ch.isalpha()); tot = sum(c.values()) or 1
    return sum(v for k, v in c.items() if k not in ok and k != "OTHER") / tot


def defects(text: str, lang: str) -> list[str]:
    d = []
    if foreign_share(text, lang) > 0.02: d.append("foreign_script")
    if re.search(r"(.)\1{7,}", text): d.append("loop")
    if re.search(r"(?i)\bnote:|\(note|translation:", text): d.append("note")
    if not text.strip(): d.append("empty")
    return d


def _clean(t: str) -> str:
    t = " ".join(t.split()).strip()
    return t[1:-1].strip() if len(t) >= 2 and t[0] in "\"«“" and t[-1] in "\"»”" else t


def guarded(chat, ru: str, lang: str, lang_name: str, src_name: str = "Russian", first: str | None = None) -> tuple[str, dict]:
    """the production translation (or `first` if already made) + up to 2 strict retries when it has defects"""
    t0 = first if first is not None else _clean(chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Translate from {src_name} to {lang_name}:\n{ru}"}], 0.0))
    cands = [(t0, defects(t0, lang))]
    if cands[0][1]:
        strict = (f"Translate from {src_name} to {lang_name}. Write ONLY in {lang_name}: no {src_name} words, no Chinese, no other languages, no notes "
                  f"or comments. Names and brands may stay in Latin letters. Answer with the {lang_name} translation only.\n{ru}")
        for temp in (0.0, 0.3):
            t = _clean(chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": strict}], temp)); d = defects(t, lang)
            cands.append((t, d))
            if not d: break
    best = min(range(len(cands)), key=lambda i: (len(cands[i][1]), i))
    return cands[best][0], {"first_defects": cands[0][1], "tries": len(cands), "used": best, "final_defects": cands[best][1]}
