"""Sentence-aware translation units (2026-10-02, priorities B1 + C1 of the coverage fix).

Whisper segments of a punctuation-less monologue are not semantic units ("...bring us all the desired | the result we wished..."). This module
  1. asks Qwen (the already loaded llama.cpp model) to add punctuation / sentence boundaries to the transcript WITHOUT changing a word
     (validated: normalized token sequence identical; retry with the error fed back; fallback = whisper segments, recorded);
  2. builds units = sentences (word timestamps kept; sentences longer than max_unit_s are split at the widest inter-word gap);
  3. merges tiny units (< min_words words OR < min_s seconds) into the neighbour with the smaller pause when the speaker is the same, the pause
     is <= max_gap_s and the merged unit stays <= max_merged_s (the "вперед" -> "forward" case: a one-word Chatterbox prompt hallucinates).
Every decision is recorded (segmentation report) so the manifest shows what was merged / where the fallback was used.
"""
from __future__ import annotations

import re

SENT_SYSTEM = "You are a careful transcript editor. You answer with the edited text only."
_SENT_END = re.compile(r"(?<=[.!?…])\s+")


def norm_tokens(s: str) -> list[str]:
    return re.findall(r"[\w']+", s.lower().replace("’", "'"))


def clean_words(segments: list[dict]) -> list[dict]:
    """All whisper words in time order, standalone punctuation tokens dropped (e.g. a lone '.' whisper emitted inside a segment)."""
    ws = []
    for sg in segments:
        for w in sg.get("words") or []:
            if norm_tokens(w["word"]):
                ws.append({**w, "word": w["word"].strip(), "segment": sg["id"]})
    ws.sort(key=lambda w: (w["start"], w["end"]))
    return ws


def _chunks(words: list[dict], max_words: int) -> list[list[dict]]:
    """Consecutive word chunks of <= max_words, cut only at whisper segment boundaries (a segment longer than max_words stays whole)."""
    out: list[list[dict]] = []; cur: list[dict] = []
    for w in words:
        if cur and w["segment"] != cur[-1]["segment"] and len(cur) + 1 > max_words:
            out.append(cur); cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def _tokens_with_punct(text: str) -> list[tuple[str, bool]]:
    """(normalized token, ends_a_sentence) for every word of a punctuated text."""
    out = []
    for raw in text.split():
        toks = norm_tokens(raw)
        if not toks:
            if out:
                out[-1] = (out[-1][0], out[-1][1] or bool(re.search(r"[.!?…]", raw)))
            continue
        end = bool(re.search(r"[.!?…]['\"»)]*$", raw))
        for k, t in enumerate(toks):
            out.append((t, end and k == len(toks) - 1))
    return out


def map_sentence_ends(orig_tokens: list[str], punct_text: str, max_change_frac: float = 0.05, max_changes: int = 4) -> tuple[list[int] | None, dict]:
    """Sentence-end positions (index of the LAST word of each sentence in the ORIGINAL word sequence) taken from Qwen's punctuated text, mapped back
    through a token alignment. The original words are never replaced: a synonym 'correction' by the model ("дальше" -> "далее") only has to be
    small enough (<= max_changes changed tokens and <= max_change_frac) to trust the boundaries; otherwise None."""
    import difflib
    out = _tokens_with_punct(punct_text); hyp = [t for t, _ in out]
    sm = difflib.SequenceMatcher(None, orig_tokens, hyp, autojunk=False); changed = 0; ends = set(); info = {"replaced": [], "inserted": [], "deleted": []}
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for k in range(j2 - j1):
                if out[j1 + k][1]:
                    ends.add(i1 + k)
        elif tag == "replace":
            changed += max(i2 - i1, j2 - j1); info["replaced"].append((orig_tokens[i1:i2], hyp[j1:j2]))
            if any(out[j][1] for j in range(j1, j2)):
                ends.add(i2 - 1)
        elif tag == "insert":
            changed += j2 - j1; info["inserted"].append(hyp[j1:j2])
            if any(out[j][1] for j in range(j1, j2)) and i1 > 0:
                ends.add(i1 - 1)
        elif tag == "delete":
            changed += i2 - i1; info["deleted"].append(orig_tokens[i1:i2])
    info["changed_tokens"] = changed; info["ratio"] = round(sm.ratio(), 4)
    if changed > max_changes or changed > max_change_frac * max(len(orig_tokens), 1):
        return None, info
    ends.add(len(orig_tokens) - 1)
    return sorted(e for e in ends if 0 <= e < len(orig_tokens)), info


def punctuate_chunk(llm, text: str, lang_name: str, retries: int = 1) -> tuple[list[int] | None, list[dict]]:
    """Ask Qwen for punctuation / sentence boundaries; returns sentence-end word indices of `text` (original words) or None."""
    toks = norm_tokens(text)
    prompt = (f"Below is a {lang_name} speech transcript without punctuation. Add punctuation and capitalization and split it into sentences. "
              f"Do not change, add, remove or reorder any word; do not correct words; do not translate. Return the edited text only.\n\n{text}")
    msgs = [{"role": "system", "content": SENT_SYSTEM}, {"role": "user", "content": prompt}]; log = []
    for attempt in range(retries + 1):
        r = llm.create_chat_completion(msgs, max_tokens=1024, temperature=0.0)
        out = " ".join(r["choices"][0]["message"]["content"].split()).strip().strip('"«»')
        ends, info = map_sentence_ends(toks, out)
        log.append({"attempt": attempt, "ok": ends is not None, "changed_tokens": info.get("changed_tokens"), "ratio": info.get("ratio"), "diff": {k: v for k, v in info.items() if k in ("replaced", "inserted", "deleted") and v}, "out": out[:400]})
        if ends is not None:
            return ends, log
        msgs = msgs + [{"role": "assistant", "content": out}, {"role": "user", "content": "That changed too many words. Keep every word exactly as given, only add punctuation and capitalization. Return the text only."}]
    return None, log


def sentence_units(segments: list[dict], llm, lang_name: str, duration: float, max_unit_s: float = 20.0, chunk_words: int = 120, retries: int = 1) -> tuple[list[dict], dict]:
    """Units from punctuated sentences. Returns (units, report); units have the build_units() fields (asr_segment, start, end, text, words, id) plus
    `sentence` (punctuated text) and `source` ('sentence' | 'whisper_fallback')."""
    words = clean_words(segments); rep = {"method": "sentence", "chunks": [], "fallback_chunks": 0, "sentences": 0, "split_long": 0}
    units: list[dict] = []
    for ch in _chunks(words, chunk_words):
        text = " ".join(w["word"] for w in ch)
        ends, log = punctuate_chunk(llm, text, lang_name, retries) if llm is not None else (None, [])
        rep["chunks"].append({"words": len(ch), "ok": ends is not None, "attempts": log})
        if ends is None:   # fallback: this chunk's whisper segments stay the units
            rep["fallback_chunks"] += 1
            by_seg: dict = {}
            for w in ch:
                by_seg.setdefault(w["segment"], []).append(w)
            for sid, ws in by_seg.items():
                units.append({"asr_segment": sid, "words_list": ws, "sentence": " ".join(w["word"] for w in ws), "source": "whisper_fallback"})
            continue
        pos = 0
        for e in ends:
            ws = ch[pos:e + 1]; pos = e + 1
            if not ws:
                continue
            sent = " ".join(w["word"] for w in ws); sent = sent[0].upper() + sent[1:] + "."     # original words, sentence-final punctuation only
            units.append({"asr_segment": ws[0]["segment"], "words_list": ws, "sentence": sent, "source": "sentence"}); rep["sentences"] += 1
    # long sentences -> split at the widest inter-word gap (same rule as build_units)
    def _split(ws: list[dict]) -> list[list[dict]]:
        if ws[-1]["end"] - ws[0]["start"] <= max_unit_s or len(ws) < 4:
            return [ws]
        n = len(ws); i = max(range(2, n - 1), key=lambda k: (ws[k]["start"] - ws[k - 1]["end"]) - 0.01 * abs(k - n / 2))
        return _split(ws[:i]) + _split(ws[i:])
    out = []
    for u in units:
        pieces = _split(u["words_list"])
        if len(pieces) > 1:
            rep["split_long"] += 1
        for k, ws in enumerate(pieces):
            text = u["sentence"] if len(pieces) == 1 else " ".join(w["word"] for w in ws)
            out.append({"asr_segment": u["asr_segment"], "start": round(max(0.0, ws[0]["start"]), 3), "end": round(min(duration, ws[-1]["end"]), 3), "text": text,
                        "words": len(ws), "sentence": u["sentence"], "source": u["source"] + ("" if len(pieces) == 1 else f"_split{k}")})
    out = [u for u in out if u["text"] and u["end"] > u["start"]]
    out.sort(key=lambda u: u["start"])
    for i, u in enumerate(out):
        u["id"] = i
    rep["units"] = len(out)
    return out, rep


def merge_tiny_units(units: list[dict], min_words: int = 3, min_s: float = 1.0, max_gap_s: float = 2.0, max_merged_s: float = 25.0) -> tuple[list[dict], list[dict]]:
    """Merge units with < min_words words or < min_s seconds into the neighbour with the smaller pause (same speaker, pause <= max_gap_s,
    merged duration <= max_merged_s). Returns (units, merges); unit ids are renumbered."""
    us = [dict(u) for u in units]; merges = []
    changed = True
    while changed and len(us) > 1:
        changed = False
        for i, u in enumerate(us):
            tiny = u["words"] < min_words or (u["end"] - u["start"]) < min_s
            if not tiny:
                continue
            cands = []
            for j in (i - 1, i + 1):
                if 0 <= j < len(us) and us[j].get("speaker") == u.get("speaker"):
                    gap = (u["start"] - us[j]["end"]) if j < i else (us[j]["start"] - u["end"])
                    merged = max(u["end"], us[j]["end"]) - min(u["start"], us[j]["start"])
                    if gap <= max_gap_s and merged <= max_merged_s:
                        cands.append((gap, j))
            if not cands:
                u["tiny_unmerged"] = {"words": u["words"], "seconds": round(u["end"] - u["start"], 3), "reason": "no neighbour with same speaker / pause <= max_gap / merged <= max_merged"}
                continue
            gap, j = min(cands); a, b = (j, i) if j < i else (i, j)
            A, B = us[a], us[b]
            text = (A["text"].rstrip() + " " + B["text"].lstrip()).strip()
            merged = {**A, "start": A["start"], "end": B["end"], "text": text, "words": A["words"] + B["words"], "sentence": (A.get("sentence", A["text"]) + " " + B.get("sentence", B["text"])).strip(),
                      "source": f"{A.get('source', '?')}+{B.get('source', '?')}", "merged_from": [{k: x[k] for k in ("start", "end", "text", "words")} for x in (A, B)]}
            merged.pop("tiny_unmerged", None)
            merges.append({"tiny": {k: u[k] for k in ("start", "end", "text", "words")}, "into": {k: us[j][k] for k in ("start", "end", "text")}, "gap_s": round(gap, 3), "merged": [merged["start"], merged["end"]]})
            us[a:b + 1] = [merged]; changed = True
            break
    for i, u in enumerate(us):
        u["id"] = i
    return us, merges
