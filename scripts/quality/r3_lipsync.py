#!/usr/bin/env python3
"""Round 3 (track A) generator-side helpers.

Dub-silent mouth ("silence mix"): where the SOURCE speaker talks but no dub part sounds (03 24-25 s / 35-36 s class), LatentSync driven by
the silent dub closes the mouth while the jaw / cheeks outside the mask keep moving -> a frozen mouth inside a talking face. Here the
whisper conditioning of those frames is moved toward the SOURCE speech by `lam`:
    chunk = dub + w(t) * lam * (src - dub)
w(t) = 0 / 1 per master frame (source speaks, no placed dub part, run >= min_s; or a residual-gap JSON from track C), eased with cosine ramps
of `ramp` frames. lam < 1 keeps the articulation small (a relaxed mouth that follows the jaw, not a mimed sentence); the conditioning is
continuous in time, so there is no snap at a window start. Same definition of the stretches as track B's temporal_mouth_stabilize.py.
"""
from __future__ import annotations
import json, math, pathlib, tempfile
import numpy as np
FPS = 25; SR = 16000


def load_gaps(path) -> list:
    """[start_s, end_s] list, or {"gaps"|"residual_gaps": [...]} with [s, e] / {start, end} / {start_s, end_s} items (track B's format)."""
    d = json.loads(pathlib.Path(path).read_text()); items = d.get("gaps", d.get("residual_gaps", d)) if isinstance(d, dict) else d; out = []
    for it in items:
        if isinstance(it, dict):
            if it.get("class") == "natural": continue   # track C residual.json: "natural" = shorter than 0.35 s, not a residual gap
            out.append((float(it.get("start", it.get("start_s", it.get("t0")))), float(it.get("end", it.get("end_s", it.get("t1"))))))
        else: out.append((float(it[0]), float(it[1])))
    return sorted(out)


def silent_mask(man: dict, n: int, min_s: float = 0.3, merge_frames: int = 4, gaps: list | None = None) -> np.ndarray:
    """Per master frame: source speaks (manifest speech_intervals) and no placed dub part sounds, runs >= min_s (gaps closed over <= merge_frames)."""
    t = np.arange(n) / FPS
    if gaps is not None:
        m = np.zeros(n, bool)
        for g0, g1 in gaps:
            if g1 - g0 >= min_s: m |= (t >= g0) & (t < g1)
        return m
    src = np.zeros(n, bool); dub = np.zeros(n, bool)
    for iv in man.get("speech_intervals", []): src |= (t >= iv["start"]) & (t < iv["end"])
    for u in man.get("units", []):
        for p in (u.get("alignment") or {}).get("placements", []) or []:
            if p.get("placed"): dub |= (t >= p["placed"][0]) & (t < p["placed"][1])
    m = src & ~dub; runs = []; i = 0
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]: j += 1
            runs.append([i, j]); i = j
        else: i += 1
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] <= merge_frames: merged[-1][1] = r[1]
        else: merged.append(r)
    out = np.zeros(n, bool)
    for a, b in merged:
        if (b - a) / FPS >= min_s: out[a:b] = True
    return out


def ramp(mask: np.ndarray, k: int) -> np.ndarray:
    w = np.zeros(len(mask), np.float32); i = 0; n = len(mask)
    while i < n:
        if not mask[i]: i += 1; continue
        j = i
        while j < n and mask[j]: j += 1
        L = j - i; kk = min(k, L // 2)
        for q in range(L):
            d = min(q + 1, L - q); w[i + q] = 1.0 if kk == 0 or d > kk else 0.5 - 0.5 * math.cos(math.pi * d / (kk + 1))
        i = j
    return w


class SilenceMix:
    """Per-segment whisper-chunk hook. `w`: per master frame weight (absolute frame index); `frame_offset`: absolute index of the
    frame the caller calls 0 (window runs)."""

    def __init__(self, src_wav, w: np.ndarray, lam: float, frame_offset: int = 0):
        import soundfile as sf
        y, sr = sf.read(str(src_wav), dtype="float32")
        if y.ndim > 1: y = y.mean(1)
        assert sr == SR, sr
        self.src, self.w, self.lam, self.off = y, np.asarray(w, np.float32), float(lam), int(frame_offset)
        self.frames_mixed = 0

    def hook(self, start: int, n: int):
        a = self.off + start; ww = np.zeros(n, np.float32); k = max(0, min(n, len(self.w) - a)); ww[:k] = self.w[a:a + k]
        if not (ww > 0).any(): return None
        def fn(audio_encoder, chunks):
            import soundfile as sf, torch
            s0 = int(round(a / FPS * SR)); seg = self.src[s0:s0 + int(round(n / FPS * SR))]
            seg = np.concatenate([seg, np.zeros(int(round(n / FPS * SR)) - len(seg), np.float32)])
            with tempfile.NamedTemporaryFile(suffix=".wav") as tf:
                sf.write(tf.name, seg, SR, subtype="PCM_16")
                sc = audio_encoder.feature2chunks(feature_array=audio_encoder.audio2feat(tf.name), fps=FPS)
            out = list(chunks)
            for i in range(min(len(out), len(sc), n)):
                if ww[i] > 0:
                    out[i] = chunks[i] + float(ww[i] * self.lam) * (sc[i].to(chunks[i]) - chunks[i]); self.frames_mixed += 1
            return out
        return fn
