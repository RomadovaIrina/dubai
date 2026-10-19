"""Speakable English text for Chatterbox (Agent C r3, 2026-10-19).

Why: the audio-r2 intelligibility guard (reports/experiments/r3-audio) put number / unit sentences among the worst units:
01 "600x600 with a thickness of 9 mm" -> Whisper "600 x 67", CTC "six hundred ixix seventh ... nine milimit"; Chatterbox has no
number verbaliser (its punc_norm only maps "..." -> ", " and similar). The QA then compared the audio with digits it could not say.
The runner applies `speakable()` to the unit translation and every Qwen rewrite BEFORE synthesis, so TTS, QA, CTC alignment, the part
split and the evaluation all see the same words. Idempotent; text without digits / symbols is unchanged except ".." -> ".".
"""
from __future__ import annotations
import re

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth", "twelve": "twelfth"}
UNITS = {  # number + unit -> spoken unit (singular, plural)
    "mm": ("millimeter", "millimeters"), "cm": ("centimeter", "centimeters"), "m": ("meter", "meters"), "km": ("kilometer", "kilometers"),
    "m2": ("square meter", "square meters"), "m²": ("square meter", "square meters"), "sq m": ("square meter", "square meters"), "sq. m": ("square meter", "square meters"),
    "sqm": ("square meter", "square meters"), "m3": ("cubic meter", "cubic meters"), "m³": ("cubic meter", "cubic meters"),
    "kg": ("kilogram", "kilograms"), "g": ("gram", "grams"), "t": ("ton", "tons"), "l": ("liter", "liters"), "ml": ("milliliter", "milliliters"),
    "kw": ("kilowatt", "kilowatts"), "w": ("watt", "watts"), "v": ("volt", "volts"), "h": ("hour", "hours"), "min": ("minute", "minutes"),
    "%": ("percent", "percent"), "°c": ("degree", "degrees"), "°": ("degree", "degrees"),
    "rub": ("ruble", "rubles"), "rub.": ("ruble", "rubles"), "₽": ("ruble", "rubles"), "$": ("dollar", "dollars"), "€": ("euro", "euros"),
}


def num_words(n: int) -> str:
    if n < 20: return _ONES[n]
    if n < 100: return _TENS[n // 10] + ("" if n % 10 == 0 else "-" + _ONES[n % 10])
    if n < 1000: return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + num_words(n % 100))
    for div, name in ((10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand")):
        if n >= div:
            return num_words(n // div) + " " + name + ("" if n % div == 0 else " " + num_words(n % div))
    return str(n)


def year_words(n: int) -> str:
    if 2000 <= n <= 2009: return num_words(n)
    hi, lo = divmod(n, 100)
    return num_words(hi) + (" hundred" if lo == 0 else (" oh " + _ONES[lo] if lo < 10 else " " + num_words(lo)))


def ordinal_words(n: int) -> str:
    w = num_words(n); head, _, last = w.rpartition("-") if "-" in w.split(" ")[-1] else w.rpartition(" ")
    sep = "-" if "-" in w.split(" ")[-1] else " "
    if last in _ORD: last = _ORD[last]
    elif last.endswith("y"): last = last[:-1] + "ieth"
    else: last = last + "th"
    return (head + sep if head else "") + last


def _decimal(intpart: str, frac: str) -> str:
    return num_words(int(intpart)) + " point " + " ".join(_ONES[int(c)] for c in frac)


_UNIT_RE = "|".join(sorted((re.escape(k) for k in UNITS), key=len, reverse=True))


def speakable(text: str) -> str:
    t = text
    t = re.sub(r"(?<!\.)\.\.(?!\.)", ".", t)                                                   # ".." (LLM / ASR artefact) -> "."
    t = re.sub(r"(\d)[  ](?=\d{3}\b)", r"\1", t)                                           # 10 000 -> 10000
    t = re.sub(r"(\d),(?=\d{3}\b)", r"\1", t)                                                   # 10,000 -> 10000
    t = re.sub(r"(\d+(?:\.\d+)?)\s*[x×х]\s*(?=\d)", r"\1 by ", t)                                # 600x600 -> 600 by 600
    t = re.sub(r"(\d)\s*-\s*(?=\d)", r"\1 to ", t)                                              # 10-15 -> 10 to 15
    t = re.sub(r"([$€₽])\s*(\d+(?:\.\d+)?)", lambda m: f"{m.group(2)} {m.group(1)}", t)          # $5 -> 5 $
    def unit(m):
        num, u = m.group(1), m.group(2).lower()
        sing, plur = UNITS[u]
        return f"{num} {sing if num == '1' else plur}"
    t = re.sub(rf"(\d+(?:\.\d+)?)\s*({_UNIT_RE})(?=[\s.,;:!?)]|$)", unit, t, flags=re.I)
    t = re.sub(r"\b(\d+)(st|nd|rd|th)\b", lambda m: ordinal_words(int(m.group(1))), t)
    t = re.sub(r"\b(\d+)\.(\d+)\b", lambda m: _decimal(m.group(1), m.group(2)), t)
    t = re.sub(r"\b(1[1-9]\d\d|20\d\d)\b(?!\s*(?:" + "|".join(p for s, p in UNITS.values()) + r"))", lambda m: year_words(int(m.group(1))), t)
    t = re.sub(r"\b\d+\b", lambda m: num_words(int(m.group(0))), t)
    t = t.replace("&", " and ").replace("#", " number ")
    return re.sub(r"\s+", " ", t).strip()


_FOREIGN = re.compile(r"[\u0400-\u04ff\u0590-\u06ff\u0e00-\u0e7f\u1100-\u11ff\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\uf900-\ufaff\uff00-\uffef]")
_BRACKET = re.compile(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}|<[^>]*>")


def script_defects(text: str) -> list[str]:
    """Qwen drift in an ENGLISH output (r3 2026-10-19, Q5_K_M on 03: "...will proceed顺利地", "...getting to know [it better]"; pilot 0.9:
    Q8_0 switched to Chinese mid-sentence on a long input): characters of another script, bracketed asides. [] -> clean."""
    out = []
    f = _FOREIGN.findall(text)
    if f: out.append("foreign script: " + "".join(f)[:20])
    b = _BRACKET.findall(text)
    if b: out.append("bracketed: " + " ".join(b)[:40])
    return out


def strip_defects(text: str) -> str:
    t = _BRACKET.sub(" ", text); t = _FOREIGN.sub("", t)
    return re.sub(r"\s+([,.!?;:])", r"\1", re.sub(r"\s+", " ", t)).strip()


if __name__ == "__main__":
    for s in ["Bathroom... Matte wide-format ceramic tile 600x600 with a thickness of 9 mm..", "Such wallpapers withstand up to 10 repaintings.. Floor..",
              "Today is Friday, August 28th; area 45 m2, 0.5 % moisture, 1 m, built in 2019, price 3,500,000 rubles, 10-15 minutes, $20, room #5."]:
        print(s, "\n  ->", speakable(s))
