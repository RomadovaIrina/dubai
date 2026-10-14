"""--pause-policy thought (2026-10-14): one complete thought first, the pause after it.

Human feedback on the 2026-10-12 pipeline (01 / 03 / 04): the dub reproduced every source pause INSIDE a sentence on whatever English word was
there ("Sochi city | 2.8 s | today, | 2.1 s | August 28", "test how our | 1.1 s | application", "magical application which | 2.4 s | will
make life easier") - the burst-aware placement cut one natural English sentence into one piece per source burst. 10 of 11 long dub pauses on
04 and 16 of 19 on 03 were inside a sentence. The reviewer's rule: "не «день ……… сегодня хороший», а «день сегодня хороший ………»"; the dub may
cover less of the source speech time, but must not be sped up / slowed down to fill it.

Policy: every sentence (unit) is ONE part, spoken at natural speed from the first source burst of the sentence; all the silence goes after it.
When the English sentence is longer than its window (first burst -> next sentence's first burst - gap), the `overflow` tools (A/B; the reviewer
wants to compare them by quality AND time):
  rewrite  concise Qwen rewrite already at a 3 % overflow (TTS stage, run_tts_phrase fit-down; costs Qwen + one more TTS take)
  split    cut the sentence at a punctuation boundary (, ; : — . ! ?) onto a source pause: each piece is still a whole clause (cut stage, free)
  shift    the sentence runs on into the next one's window and the next sentence starts later (<= max_shift_s; placement, free)
  tempo    atempo up to tempo_cap (1.1) (placement, free)
Whatever overflow is left after the chosen tool is resolved by the safety ladder (tempo <= 1.1, shift <= 0.5 s, tempo <= hard 1.2) and counted,
so every variant shows how often it needed help. Words are never cut.

Edges ("в конце фразы слова на согласной он либо резко сплёвывает, либо пропадает со щелчком"): the old cut ended a part 80 ms after the CTC end
of its last word, and CTC word ends are early (plosive releases / final fricatives come later), so final consonants were cut or their release
moved to the next part. Here a part ends where the clip's energy has really decayed (stays >= 38 dB below the speech level for 60 ms), searched
up to 0.45 s after the CTC end, with a 20 ms cosine fade-out; an inner cut lies in the middle of the longest really silent run between the
words (else in the later half of the gap, so a release stays with its word)."""
from __future__ import annotations

import pathlib
import re
import subprocess

import numpy as np

PUNCT_END = re.compile(r"[,;:—–.!?…]['\"»)]*$")


def _ff(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-nostdin", *args], check=True)


# ------------------------------------------------------------------------------------------------------------------------------------ edges
def frame_db(y: np.ndarray, sr: int, hop_s: float = 0.005) -> tuple[np.ndarray, float]:
    h = max(1, int(hop_s * sr)); n = len(y) // h
    if n == 0:
        return np.array([-120.0]), hop_s
    return 20 * np.log10(np.sqrt(np.mean(y[: n * h].reshape(n, h) ** 2, axis=1)) + 1e-9), h / sr


def speech_level_db(y: np.ndarray, sr: int) -> float:
    d, _ = frame_db(y, sr, 0.01); v = d[d > -60]
    return float(np.percentile(v, 90)) if len(v) else -20.0


def tail_end(y: np.ndarray, sr: int, word_end: float, ref_db: float, rel_db: float = -38.0, hold_s: float = 0.06, max_ext: float = 0.45,
             min_ext: float = 0.06, limit: float | None = None) -> float:
    """End of a sentence-final / part-final word: first point after the CTC word end where the energy stays below ref_db + rel_db for hold_s
    (+ 20 ms), within [word_end + min_ext, word_end + max_ext] and before `limit` (next word start / clip end)."""
    d, hop = frame_db(y, sr); dur = len(y) / sr; lim = min(dur, limit if limit is not None else dur)
    i0 = int(word_end / hop); i1 = min(len(d), int(min(lim, word_end + max_ext) / hop)); hold = max(1, int(hold_s / hop)); thr = ref_db + rel_db
    end = None
    for i in range(i0, i1):
        if np.all(d[i:i + hold] < thr):
            end = i * hop + 0.02; break
    if end is None:
        end = min(lim, word_end + max_ext)
    return float(min(lim, max(end, min(lim, word_end + min_ext))))


def inner_cut(y: np.ndarray, sr: int, e_prev: float, s_next: float, ref_db: float, rel_db: float = -38.0) -> float:
    """Cut point between two words: centre of the longest run below ref_db + rel_db inside [e_prev, s_next] (>= 20 ms), else the energy
    minimum of the LATER half of the gap (a plosive release / final consonant after an early CTC end stays with its word)."""
    if s_next - e_prev < 0.03:
        return 0.5 * (e_prev + s_next)
    d, hop = frame_db(y, sr); a, b = int(e_prev / hop), max(int(e_prev / hop) + 1, int(s_next / hop))
    q = d[a:b] < ref_db + rel_db; best = (0, 0); i = 0
    while i < len(q):
        if q[i]:
            j = i
            while j < len(q) and q[j]: j += 1
            if j - i > best[1] - best[0]: best = (i, j)
            i = j
        else:
            i += 1
    if (best[1] - best[0]) * hop >= 0.02:
        return (a + 0.5 * (best[0] + best[1])) * hop
    m = a + (b - a) // 2
    return (m + int(np.argmin(d[m:b]))) * hop + hop / 2 if b > m else 0.5 * (e_prev + s_next)


def cosine_fade(seg: np.ndarray, sr: int, in_ms: float, out_ms: float) -> np.ndarray:
    seg = seg.copy(); n_in = min(len(seg) // 2, int(in_ms / 1000 * sr)); n_out = min(len(seg) // 2, int(out_ms / 1000 * sr))
    if n_in > 0: seg[:n_in] *= (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n_in))).astype(np.float32)
    if n_out > 0: seg[-n_out:] *= (0.5 + 0.5 * np.cos(np.linspace(0, np.pi, n_out))).astype(np.float32)
    return seg


def cut_thought(y: np.ndarray, sr: int, spans: list[dict], counts: list[int], onset_s: float | None, pre_s: float = 0.06,
                preroll_max: float = 0.25) -> tuple[list[tuple[float, float]], dict]:
    """Bounds of the parts (seconds in the clip): head = the onset rule of tts_align.cut_parts (CTC start - 60 ms, moved back to the energy
    onset - 30 ms by at most 250 ms), inner cuts = inner_cut(), tail = tail_end() of the last word."""
    ref = speech_level_db(y, sr); dur = len(y) / sr
    s0 = max(0.0, spans[0]["start"] - pre_s)
    if onset_s is not None:
        s0 = max(0.0, max(s0 - preroll_max, min(s0, onset_s - 0.03)))
    cuts = []; i = 0
    for c in counts[:-1]:
        i += c; cuts.append(inner_cut(y, sr, spans[i - 1]["end"], spans[i]["start"], ref))
    e = tail_end(y, sr, spans[-1]["end"], ref, limit=dur)
    bounds = [s0] + cuts + [e]
    return [(bounds[k], bounds[k + 1]) for k in range(len(counts))], {"ref_db": round(ref, 1), "tail_ext_s": round(e - spans[-1]["end"], 3)}


# ------------------------------------------------------------------------------------------------------------------------------------ split
def split_thought(spans: list[dict], raw_words: list[str], bursts: list, next_anchor: float, norm_fn, gap: float, inner_gap: float = 0.12,
                  tol: float = 0.03, tail_s: float = 0.12, max_parts: int = 3) -> tuple[list[int], list[list[int]]] | None:
    """--overflow split: cut the sentence only at punctuation onto a source pause so that every piece fits its window at natural speed.
    Returns (aligned-token counts per part, burst indices per part) or None when no punctuation split improves the fit."""
    tok_end = []; acc = 0
    for w in raw_words:
        acc += len(norm_fn(w)); tok_end.append(acc)
    n = len(spans)
    if acc != n or len(bursts) < 2:
        return None
    cand = [tok_end[w] for w in range(len(raw_words) - 1) if PUNCT_END.search(raw_words[w]) and 0 < tok_end[w] < n]

    def need(k0, k1, b0, b1, limit):   # tokens [k0, k1) placed from burst b0 to `limit`
        d = spans[k1 - 1]["end"] - spans[k0]["start"] + tail_s
        return d / max(limit - bursts[b0][0], 0.05)

    def solve(k0, b0, parts_left):
        whole = need(k0, n, b0, len(bursts) - 1, next_anchor - gap)
        best = (whole, [n - k0], [list(range(b0, len(bursts)))])
        if whole <= 1 + tol or parts_left <= 1:
            return best
        for k in (c for c in cand if c > k0):
            for j in range(b0, len(bursts) - 1):
                r1 = need(k0, k, b0, j, bursts[j + 1][0] - inner_gap)
                if r1 > 1 + tol:
                    continue
                r2, c2, g2 = solve(k, j + 1, parts_left - 1)
                score = max(r1, r2)
                pause = bursts[j + 1][0] - bursts[j][1]
                if score < best[0] - 1e-6 or (abs(score - best[0]) < 1e-6 and pause > 0.3):
                    best = (score, [k - k0] + c2, [list(range(b0, j + 1))] + g2)
        return best

    r, counts, groups = solve(0, 0, max_parts)
    whole = need(0, n, 0, len(bursts) - 1, next_anchor - gap)
    if len(counts) < 2 or r >= whole - 0.03:
        return None
    return counts, groups


def cover_split(spans: list[dict], raw_words: list[str], bursts: list, next_anchor: float, norm_fn, gap: float, inner_gap: float = 0.12,
                tol: float = 0.03, tail_s: float = 0.12, max_parts: int = 3, min_gain: float = 1.0, min_pause: float = 0.5, min_words: int = 2,
                split_cost: float = 0.3) -> tuple[list[int], list[list[int]]] | None:
    """--underfill clause (A/B variant): the English sentence is much shorter than the source speaker's, who keeps talking after it (04 u2: 6.4 s
    of dub on a 16 s sentence = 9.5 s of closed mouth while the speaker talks). Cut it ONLY at punctuation (a clause end) onto a source pause
    >= min_pause, every piece at natural speed inside its window, when that covers >= min_gain s more of the source speech. The pause then
    comes after a clause, never on a weak word. Returns (aligned-token counts, burst indices per part) or None."""
    tok_end = []; acc = 0
    for w in raw_words:
        acc += len(norm_fn(w)); tok_end.append(acc)
    n = len(spans); M = len(bursts)
    if acc != n or M < 2:
        return None
    wcount = {tok_end[w]: w + 1 for w in range(len(raw_words))}
    cand = [tok_end[w] for w in range(len(raw_words) - 1) if PUNCT_END.search(raw_words[w]) and 0 < tok_end[w] < n]
    NEG = float("-inf")

    def dur(k0, k1):
        return spans[k1 - 1]["end"] - spans[k0]["start"] + tail_s

    def cover(a, d, b0, b1):
        return sum(max(0.0, min(a + d, be) - max(a, bs)) for bs, be in bursts[b0:b1 + 1])

    def solve(k0, b0, parts_left):
        d = dur(k0, n); a = bursts[b0][0]
        if d > (next_anchor - gap - a) * (1 + tol):
            return NEG, None, None
        best = (cover(a, d, b0, M - 1), [n - k0], [list(range(b0, M))])
        if parts_left <= 1:
            return best
        for k in cand:
            if k <= k0 or wcount[k] - wcount.get(k0, 0) < min_words or len(raw_words) - wcount[k] < min_words:
                continue
            d1 = dur(k0, k)
            for j in range(b0, M - 1):
                if bursts[j + 1][0] - bursts[j][1] < min_pause or d1 > (bursts[j + 1][0] - inner_gap - a) * (1 + tol):
                    continue
                c2, cn2, g2 = solve(k, j + 1, parts_left - 1)
                if c2 == NEG:
                    continue
                tot = cover(a, d1, b0, j) + c2 - split_cost
                if tot > best[0]:
                    best = (tot, [k - k0] + cn2, [list(range(b0, j + 1))] + g2)
        return best

    whole = cover(bursts[0][0], dur(0, n), 0, M - 1)
    c, counts, groups = solve(0, 0, max_parts)
    if counts is None or len(counts) < 2 or c - whole < min_gain:
        return None
    return counts, groups


# -------------------------------------------------------------------------------------------------------------------------------- placement
class ThoughtOpts:
    def __init__(self, overflow: set, gap: float = 0.2, inner_gap: float = 0.12, tempo_cap: float = 1.1, hard_cap: float = 1.2, max_shift: float = 1.0,
                 safety_shift: float = 0.5, tol: float = 0.03, underfill: str = "none"):
        self.overflow, self.gap, self.inner_gap, self.tempo_cap, self.hard_cap = set(overflow), gap, inner_gap, tempo_cap, hard_cap
        self.max_shift, self.safety_shift, self.tol, self.underfill = max_shift, safety_shift, tol, underfill

    def as_dict(self) -> dict:
        return {"overflow": sorted(self.overflow), "underfill": self.underfill, "gap": self.gap, "inner_gap": self.inner_gap, "tempo_cap": self.tempo_cap, "hard_cap": self.hard_cap,
                "max_shift": self.max_shift, "safety_shift": self.safety_shift, "tol": self.tol}


def _atempo(seg: np.ndarray, sr: int, tempo: float, tmp: pathlib.Path) -> np.ndarray:
    import soundfile as sf
    if abs(tempo - 1.0) < 1e-3:
        return seg
    a, b = tmp.with_suffix(".in.wav"), tmp.with_suffix(".out.wav"); sf.write(str(a), seg, sr, subtype="PCM_16")
    _ff(["-i", str(a), "-af", f"atempo={tempo:.6f}", "-ar", str(sr), "-ac", "1", "-c:a", "pcm_s16le", str(b)])
    out, _ = sf.read(str(b), dtype="float32"); a.unlink(); b.unlink()
    return out


def align_units_thought(units: list[dict], plans: list[dict], work: pathlib.Path, o: ThoughtOpts, duration: float, sr: int = 24000) -> dict:
    """Sequential placement of the thought parts on the original timeline: part k of unit i starts at max(its source anchor, cursor) at natural
    speed; overflow is resolved by the unit's tools then the safety ladder; never overlapping, never cut."""
    import soundfile as sf
    cursor = 0.0; used: dict = {}; shifts = []; tempos = []
    flat = [(i, k) for i, u in enumerate(units) for k in range(len(u["tts_parts"]))]
    for idx, (i, k) in enumerate(flat):
        u, p = units[i], plans[i]; t = u["tts_parts"][k]
        seg, sr_ = sf.read(t["file"], dtype="float32"); assert sr_ == sr
        if seg.ndim > 1: seg = seg.mean(axis=1)
        anchor = p["bursts"][k][0]; last_of_unit = k + 1 == len(u["tts_parts"]); is_last = idx + 1 == len(flat)
        if not last_of_unit:
            limit = p["bursts"][k + 1][0] - o.inner_gap
        elif i + 1 < len(units):
            limit = plans[i + 1]["bursts"][0][0] - o.gap
        else:
            limit = duration - 0.02
        start = max(anchor, cursor); dur = len(seg) / sr; room = max(limit - start, 0.05)
        tempo = 1.0; flags = []
        if dur > room + 1e-3:
            eff = room + (0.0 if is_last else (o.max_shift if "shift" in o.overflow else 0.0))
            if "tempo" in o.overflow:
                tempo = min(dur / room, o.tempo_cap)
            if dur / tempo > eff + 1e-3 and tempo < o.tempo_cap:              # safety 1: mild tempo
                tempo = min(dur / eff, o.tempo_cap); flags.append("safety_tempo")
            if dur / tempo > eff + 1e-3 and not is_last and "shift" not in o.overflow:   # safety 2: short shift of what follows
                eff += o.safety_shift; flags.append("safety_shift")
            if dur / tempo > eff + 1e-3:                                      # safety 3: hard tempo (words are never cut)
                tempo = max(tempo, min(dur / eff, o.hard_cap)); flags.append("safety_hard")
        how = "natural" if dur <= room + 1e-3 else ("tempo" if "tempo" in o.overflow and tempo > 1.001 else "")
        seg2 = _atempo(seg, sr, tempo, work / f"thought_u{u['id']:02d}_p{k}")
        end = start + len(seg2) / sr
        if end > limit + 1e-3 and how != "natural":
            how = (how + "+" if how else "") + ("shift" if "shift" in o.overflow else "overrun")
        how = "+".join([x for x in [how] if x] + flags) or "natural"
        if end > duration:   # only at the very end of the video
            seg2 = seg2[: max(0, int((duration - start) * sr))]; end = duration; how += "+cut_at_video_end"
        for key in how.split("+"):
            used[key] = used.get(key, 0) + 1
        t["placed_audio"] = seg2; t["placed_start"] = start; t["placed_end"] = end; t["atempo"] = round(tempo, 3); t["how"] = how
        t["shift_s"] = round(start - anchor, 3); t["window"] = [round(start, 3), round(limit, 3)]
        shifts.append(start - anchor); tempos.append(tempo)
        cursor = end + (o.inner_gap if not last_of_unit else o.gap)
    report = []
    for u, p in zip(units, plans):
        parts = u["tts_parts"]; a0 = parts[0]["placed_start"]; a1 = max(t["placed_end"] for t in parts)
        track = np.zeros(int(round((a1 - a0) * sr)) + 1, dtype=np.float32)
        placements = []
        for k, t in enumerate(parts):
            o0 = int(round((t["placed_start"] - a0) * sr)); seg2 = t.pop("placed_audio"); b0 = min(len(track), o0 + len(seg2)); track[o0:b0] += seg2[: b0 - o0]
            bs, be = p["bursts"][k]
            placements.append({"burst": [round(bs, 3), round(be, 3)], "window": t["window"], "phrases": [k], "group_s": t["seconds"],
                               "ratio": round(t["seconds"] / max(t["window"][1] - t["window"][0], 0.05), 3), "atempo": t["atempo"],
                               "placed": [round(t["placed_start"], 3), round(t["placed_end"], 3)], "overflow_into_next_burst_s": 0.0, "cut_at_slot_end_s": 0.0,
                               "spilled_past_slot_s": round(max(0.0, t["placed_end"] - p["slot"][1]), 3), "lead_s": 0.0, "track_start": round(a0, 3),
                               "shift_s": t["shift_s"], "how": t["how"], "text": t["text"], "source_words": p["groups"][k] if k < len(p["groups"]) else ""})
        aligned = work / f"aligned_u{u['id']:02d}.wav"; sf.write(str(aligned), track, sr, subtype="PCM_16")
        slot_s, slot_e = p["slot"]
        u["slot"] = {"start": round(slot_s, 3), "end": round(slot_e, 3), "seconds": round(slot_e - slot_s, 3)}
        u["alignment"] = {"tts_seconds": u["tts"]["seconds"], "ratio_tts_to_slot": round(u["tts"]["seconds"] / max(slot_e - slot_s, 1e-6), 4), "mode": "thought",
                          "atempo": max(pl["atempo"] for pl in placements), "bursts": len(p["bursts"]), "bursts_without_words": p.get("bursts_without_words", 0),
                          "placements": placements, "aligned_seconds": round(len(track) / sr, 3), "file": str(aligned), "track_start": round(a0, 3)}
        report.append({"id": u["id"], "placements": placements})
    sh = np.array(shifts or [0.0]); tp = np.array(tempos or [1.0])
    return {"mode": "thought", "opts": o.as_dict(), "used": used, "parts": len(flat), "shift_max_s": round(float(sh.max()), 3), "shift_mean_s": round(float(sh.mean()), 3),
            "atempo_max": round(float(tp.max()), 3), "atempo_gt_1": int((tp > 1.001).sum()), "units": report}
