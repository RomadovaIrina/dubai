#!/usr/bin/env python3
"""Layer-1 "spread" of the dub inside its source burst (Agent C, 2026-10-16; used by recut_offline.py --spread).

After place_groups a part shorter than its burst starts at the burst start, is slowed to the --fill-slowdown floor and leaves ONE tail gap:
the speaker keeps talking, the dub is silent, LatentSync shows a closed mouth (03 24.5 s, 34.3 s). The burst-level shortfall cannot be
filled by tempo (every such part already sits at x0.90, P0), so this step only decides WHERE the leftover silence goes: a short late start
(head, <= head_max) and pauses (<= pause_max each) at natural boundaries INSIDE the part -- punctuation, before a clause word, an existing
quiet TTS pause after a content word -- so every resulting gap is short (rule of thumb: < 0.3-0.4 s reads as natural); the tail keeps the rest.
Never after a function word / inside a fixed phrase (tts_burst.CUT_FUNC_WORDS / FIXED_PHRASES), never in voiced sound, never closer than
min_seg s of speech to another cut or the part edge, never past the source burst end. The rendered (already time-stretched) part is taken
from the unit's aligned track, so tempo and fades are exactly those of the run; only its position / internal pauses change.

modes: off | centre (head + tail only) | pauses (head + internal pauses + tail)
"""
from __future__ import annotations
import math, os, re
import numpy as np

PREPOSITIONS = frozenset("about above across after against along among around before behind below beneath beside between beyond during except inside like near "
                         "off outside over past since through throughout toward towards under underneath until upon via within without".split())
CLAUSE_NEXT = frozenset("and but so because that which who whom whose where when while if or although though since unless until whether".split())


def _lvl_db(y, sr, t0, t1):
    i0, i1 = max(0, int(t0 * sr)), min(len(y), int(t1 * sr))
    return 20 * np.log10(np.sqrt(np.mean(y[i0:i1] ** 2)) + 1e-9) if i1 > i0 else -120.0


def _quietest(y, sr, t0, t1, win=0.01):
    """Centre of the quietest win-s window in [t0, t1] of y (seconds) and its level (dB)."""
    h = max(1, int(win * sr / 2)); i0, i1 = max(0, int(t0 * sr)), min(len(y), int(t1 * sr))
    if i1 - i0 < 2 * h: return 0.5 * (t0 + t1), _lvl_db(y, sr, t0, t1)
    e = np.convolve(y[i0:i1] ** 2, np.ones(2 * h) / (2 * h), mode="valid"); k = int(np.argmin(e))
    return (i0 + k + h) / sr, 10 * np.log10(float(e[k]) + 1e-18)


def candidates(u: dict, k: int, y_raw: np.ndarray, sr: int, norm_fn, func_words, fixed_phrases, quiet_rel_db: float = -25.0, punct_rel_db: float = -18.0) -> list[dict]:
    """Word boundaries inside part k of unit u where a pause may be lengthened: [{"t_raw", "quality", "word", "next", "tts_gap"}]."""
    spans = u["tts"]["word_spans"]; text = u.get("translation_fitted") or u["translation"]; raw = text.split()
    parts = u["tts_parts"]; tok0 = sum(len(norm_fn(p["text"])) for p in parts[:k]); tok1 = tok0 + len(norm_fn(parts[k]["text"]))
    if not spans or tok1 > len(spans): return []
    fr = 20 * np.log10(np.sqrt(np.mean(y_raw[: len(y_raw) // 240 * 240].reshape(-1, 240) ** 2, axis=1)) + 1e-9); ref = float(np.percentile(fr[fr > -60], 90)) if (fr > -60).any() else -20.0
    out = []; acc = 0
    for i, w in enumerate(raw):
        acc += len(norm_fn(w))
        if not (tok0 < acc < tok1): continue
        lw = re.sub(r"[^\w']", "", w.lower().replace("’", "'")); nw = re.sub(r"[^\w']", "", raw[i + 1].lower().replace("’", "'")) if i + 1 < len(raw) else ""
        punct = bool(re.search(r"[,.;:!?…—-]['\"»)]*$", w))
        if not punct and (lw in func_words or lw in PREPOSITIONS or (lw, nw) in fixed_phrases or re.search(r"\d", lw)): continue
        a, b = spans[acc - 1]["end"], spans[acc]["start"]; gap = b - a
        t, lv = _quietest(y_raw, sr, a - 0.015, b + 0.015)
        rel = lv - ref
        if rel > (punct_rel_db if punct else quiet_rel_db): continue      # still voiced at the quietest 10 ms (CTC word ends are early): a cut would be heard
        # 1 = Chatterbox itself paused after a content word AND a function word follows (a phonological-phrase boundary: "laminate | of",
        # "work | entails" no, "high-density | MDF" no -- adjective + noun / verb + object stay together)
        q = 3 if punct else 2 if nw in CLAUSE_NEXT else 1 if (gap >= 0.12 and (nw in func_words or nw in PREPOSITIONS)) else 0
        if q == 0 or (q == 1 and os.environ.get("DUB_PAUSE_BOUNDARY_ONLY", "0") == "1"): continue   # r4 v2: no pause after a content word without punctuation
        out.append({"t_raw": t, "quality": q, "word": w, "next": raw[i + 1] if i + 1 < len(raw) else "", "tts_gap": round(gap, 3), "cut_rel_db": round(rel, 1)})
    return out


def allocate(L: float, caps: list[float], ok: float, head_max: float) -> tuple[float, list[float], float, int]:
    """Split leftover L (s) into head + pauses + tail: the shortest prefix of `caps` (per-boundary max inserted s, priority order) whose
    water-filled tail is <= ok (else all); water level x with min(x, head_max) + sum(min(x, cap_i)) + x = L. Returns (head, pauses, tail, n)."""
    def fill(cs):
        lo, hi = 0.0, L
        for _ in range(50):
            x = 0.5 * (lo + hi)
            if min(x, head_max) + sum(min(x, c) for c in cs) + x > L: hi = x
            else: lo = x
        x = lo; ps = [min(x, c) for c in cs]; return min(x, head_max), ps, L - min(x, head_max) - sum(ps)
    for n in range(0, len(caps) + 1):
        h, ps, t = fill(caps[:n])
        if t <= ok + 1e-6: return h, ps, t, n
    h, ps, t = fill(caps); return h, ps, t, len(caps)


def pick(cands: list[dict], seg_len: float, pos_of, min_seg: float) -> list[dict]:
    """Usable boundaries in PRIORITY order (best quality first, then the most central): >= min_seg s of speech between cuts / part edges."""
    chosen = []
    for c in sorted(cands, key=lambda c: (-c["quality"], abs(pos_of(c) - seg_len / 2))):
        p = pos_of(c)
        if p < min_seg or seg_len - p < min_seg or any(abs(p - pos_of(d)) < min_seg for d in chosen): continue
        chosen.append(c)
    return chosen


def spread_unit(u: dict, track: np.ndarray, sr: int, y_raw: np.ndarray, cfg: dict, norm_fn, func_words, fixed_phrases) -> tuple[np.ndarray, list]:
    """Re-place every part of unit u inside its burst (cfg: mode, ok, head_max, pause_max, min_seg). Returns (new aligned track, log)."""
    al = u["alignment"]; ts = al.get("track_start", u["slot"]["start"]); pls = al["placements"]; log = []
    if cfg["mode"] == "off" or any(pl.get("mode") == "slot_fallback" for pl in pls): return track, log
    new = np.zeros_like(track); fade = max(1, int(0.004 * sr))
    for k, pl in enumerate(pls):
        if not pl.get("placed"): continue
        t0, t1 = pl["placed"]; a0 = int(round((t0 - ts) * sr)); b0 = min(len(track), int(round((t1 - ts) * sr)) + int(0.002 * sr))
        seg = track[a0:b0].copy(); nz = np.nonzero(np.abs(seg) > 1e-5)[0]; seg = seg[: nz[-1] + 1] if len(nz) else seg
        limit = min(pl["burst"][1], pl["window"][1]) if pl.get("window") else pl["burst"][1]
        L = limit - (t0 + len(seg) / sr)
        # a late start that already exists (the source spoke before the planned burst start: VAD interval starts earlier than the slot / cursor)
        onset = next((s_ for s_, e_ in cfg.get("src_ivs", []) if s_ <= t0 + 0.02 < e_), t0); head_cap = max(0.0, cfg["head_max"] - max(0.0, t0 - onset))
        if L <= cfg["ok"] or k >= len(u.get("tts_parts", [])):
            new[a0:a0 + len(seg)] += seg; continue
        cs = []
        chosen, pauses, pos_of = [], [], None
        if cfg["mode"] == "pauses":
            tp = u["tts_parts"][k]; c0, c1 = tp["clip_span"]; eff = (c1 - c0) / max(len(seg) / sr, 1e-3)   # effective tempo of the rendered part
            pos_of = lambda c: (c["t_raw"] - c0) / eff
            cs = [c for c in candidates(u, k, y_raw, sr, norm_fn, func_words, fixed_phrases) if c0 < c["t_raw"] < c1]
            for c in cs: c["cap"] = min(cfg["pause_max"], cfg["pause_total_max"] - c["tts_gap"] / eff)   # silence there = TTS pause + inserted <= pause_total_max
            cs = [c for c in pick(cs, len(seg) / sr, pos_of, cfg["min_seg"]) if c["cap"] >= 0.05]
            head, pauses, tail, n = allocate(L, [c["cap"] for c in cs], cfg["ok"], head_cap); chosen = cs[:n]
            order = sorted(range(n), key=lambda i: pos_of(chosen[i])); chosen = [chosen[i] for i in order]; pauses = [pauses[i] for i in order]
        else:
            head, pauses, tail, n = allocate(L, [], cfg["ok"], head_cap)
        # render: head offset, then the part split at the chosen boundaries with silences in between
        pieces = []; prev = 0
        for c in chosen:
            p = int(round(pos_of(c) * sr)); pieces.append(seg[prev:p]); prev = p
        pieces.append(seg[prev:])
        out = []
        for i, pc in enumerate(pieces):
            pc = pc.copy()
            if i > 0 and len(pc) > 2 * fade: pc[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            if i + 1 < len(pieces) and len(pc) > 2 * fade: pc[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
            out.append(pc)
            if i < len(pauses) and i + 1 < len(pieces): out.append(np.zeros(int(round(pauses[i] * sr)), np.float32))
        y = np.concatenate(out); s0 = a0 + int(round(head * sr)); e0 = min(len(new), s0 + len(y)); new[s0:e0] += y[: e0 - s0]
        nt0 = ts + s0 / sr; nt1 = ts + e0 / sr
        pl["spread"] = {"tail_before_s": round(L, 3), "late_start_before_s": round(max(0.0, t0 - onset), 3), "head_s": round(head, 3), "pauses": [{"after": c["word"], "s": round(pz, 3), "quality": c["quality"], "tts_gap": c["tts_gap"], "cut_rel_db": c["cut_rel_db"],
                        "t": round(nt0 + (pos_of(c) + sum(pauses[:j])) , 3)} for j, (c, pz) in enumerate(zip(chosen, pauses))], "tail_after_s": round(limit - nt1, 3),
                        "placed_before": [t0, t1]}
        pl["placed"] = [round(nt0, 3), round(nt1, 3)]
        log.append({"unit": u["id"], "part": k, **pl["spread"]})
    return new, log
