"""Word boundaries inside ONE synthesized TTS clip (phrase-first TTS, 2026-10-02): CTC forced alignment of the KNOWN expected text with
wav2vec2-base-960h (the acoustic model the TTS QA already uses) via torchaudio.functional.forced_align, then safe cut points BETWEEN words
(energy minimum inside the inter-word gap, zero-crossing snapped, short fades) so a whole-sentence generation can be distributed over the
source speech bursts without ever cutting inside a word and without a separate Chatterbox call per burst.

    al = CtcAligner(processor, model)              # or CtcAligner.from_qa(TtsQA)
    spans = al.align(y24, 24000, "Today is Friday, August 28th.")   # [{"word","start","end","score"}] in seconds, or None
    parts = cut_parts(y24, 24000, spans, [3, 2])   # [(y_part, start_s, end_s, words)] for parts of 3 and 2 words
"""
from __future__ import annotations

import os
import numpy as np

SR16 = 16000


class CtcAligner:
    def __init__(self, processor, model, device: str = "cuda"):
        import torch
        self.torch, self.proc, self.w2v, self.device = torch, processor, model, device
        self.blank = int(getattr(model.config, "pad_token_id", 0) or 0)
        vocab = processor.tokenizer.get_vocab()
        self.delim = vocab.get("|", 4)

    @classmethod
    def from_qa(cls, qa) -> "CtcAligner":
        return cls(qa.proc, qa.w2v, qa.device)

    def align(self, y: np.ndarray, sr: int, expected: str) -> list[dict] | None:
        """Word spans (seconds, in the clip's own time base) of `expected` inside `y`; None when the alignment is impossible/implausible."""
        import librosa
        import torchaudio.functional as taF
        from tts_qa import norm
        torch = self.torch
        words = norm(expected)
        if not words or len(y) < sr // 10:
            return None
        y16 = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=SR16) if sr != SR16 else y.astype(np.float32)
        text = " ".join(words).upper()
        ids = self.proc.tokenizer(text, add_special_tokens=False).input_ids
        unk = self.proc.tokenizer.unk_token_id
        if not ids or any(i == unk for i in ids):
            return None
        x = self.proc(y16, sampling_rate=SR16, return_tensors="pt").input_values.to(self.device)
        with torch.no_grad():
            logits = self.w2v(x).logits[0].float()
        logp = torch.log_softmax(logits, dim=-1).cpu()
        T = logp.shape[0]
        if T < len(ids):
            return None
        frame_s = len(y16) / SR16 / T
        try:
            ali, scores = taF.forced_align(logp.unsqueeze(0), torch.tensor([ids], dtype=torch.int32), blank=self.blank)
        except Exception:
            return None
        spans = taF.merge_tokens(ali[0], scores[0].exp(), blank=self.blank)
        out = []; cur = []
        for sp in spans:
            if sp.token == self.delim:
                if cur:
                    out.append(cur); cur = []
                continue
            cur.append(sp)
        if cur:
            out.append(cur)
        if len(out) != len(words):
            return None
        res = []
        for w, chars in zip(words, out):
            s = chars[0].start * frame_s; e = (chars[-1].end) * frame_s
            res.append({"word": w, "start": round(float(s), 3), "end": round(float(e), 3), "score": round(float(np.mean([c.score for c in chars])), 3)})
        for a, b in zip(res, res[1:]):
            if b["start"] < a["end"] - 1e-6 or a["end"] <= a["start"]:
                return None
        return res


def _rms_min_point(y: np.ndarray, sr: int, t0: float, t1: float, hop_s: float = 0.005) -> float:
    a, b = int(t0 * sr), int(t1 * sr)
    if b - a < int(2 * hop_s * sr):
        return 0.5 * (t0 + t1)
    hop = max(1, int(hop_s * sr)); seg = y[a:b]; n = len(seg) // hop
    rms = np.array([np.sqrt(np.mean(seg[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n)])
    k = int(np.argmin(rms)); c = a + k * hop + hop // 2
    lo, hi = max(a, c - hop), min(b - 1, c + hop)
    zc = [i for i in range(lo, hi) if (y[i] <= 0 < y[i + 1]) or (y[i + 1] <= 0 < y[i])]
    if zc:
        c = min(zc, key=lambda i: abs(i - c))
    return c / sr


def cut_points(spans: list[dict], counts: list[int], y: np.ndarray, sr: int) -> list[float]:
    """Cut times (seconds) between consecutive parts given the per-part word counts (len(counts)-1 cuts); each cut lies in the gap
    between the last word of part k and the first word of part k+1 (energy minimum, zero-crossing snapped)."""
    assert sum(counts) == len(spans), (sum(counts), len(spans))
    cuts = []; i = 0
    for c in counts[:-1]:
        i += c
        e_prev, s_next = spans[i - 1]["end"], spans[i]["start"]
        if os.environ.get("DUB_CUT_SNAP", "1") == "1":   # r3 D (user: "abrupt phrase cuts"): CTC word ends are early -> a gap < 30 ms used to be cut at
            # its midpoint, i.e. inside the still-voiced end of the previous word. Search the energy minimum over a wider window that reaches past
            # the CTC end of the previous word (+60 ms) but stays out of the next word's body (its first 20 ms at most).
            lo = e_prev - 0.01; hi = max(s_next + 0.02, e_prev + 0.06)
            cuts.append(_rms_min_point(y, sr, lo, hi))
        else:
            cuts.append(_rms_min_point(y, sr, e_prev, s_next) if s_next - e_prev >= 0.03 else 0.5 * (e_prev + s_next))
    return cuts


def _fade(seg: np.ndarray, sr: int, ms: float = 5.0) -> np.ndarray:
    n = min(len(seg) // 2, int(ms / 1000 * sr))
    if n <= 0:
        return seg
    seg = seg.copy(); ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
    seg[:n] *= ramp; seg[-n:] *= ramp[::-1]
    return seg


def hush_gaps(y: np.ndarray, sr: int, spans: list[dict], min_gap: float = 0.25, pre_s: float = 0.06, post_s: float = 0.08,
              floor_db: float = -30.0, ramp_s: float = 0.02, onset_db: float = 8.0, onset_hf_db: float = 6.0,
              after: set[int] | None = None) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Duck what the TTS put between two aligned words (a voiced hesitation murmur the dub gate keeps: 05 r7v3a 31.45-32.0 s, the user's "fart")
    to floor_db with sin^2 ramps: every inter-word gap >= min_gap, from post_s after the word to the next word's acoustic onset - 0.03 s.
    The onset is the first 10-ms frame within onset_db of that word's median level or onset_hf_db of its median > 2.5 kHz level (a fricative
    head), searched up to its CTC start + 0.05 (CTC starts a word before the murmur ends: "m-filming" -> "coming in"); never before start - pre_s.
    after: only the gaps after these word indices (a phrase boundary: ducking inside a phrase makes a pause there). Returns (audio, hushed spans)."""
    from scipy.signal import butter, sosfilt
    y = y.copy(); g = 10 ** (floor_db / 20); r = max(1, int(ramp_s * sr)); h = max(1, int(0.01 * sr)); done = []
    lv = lambda x: 20 * np.log10(np.sqrt((x[:len(x) // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-9)
    db, hf = lv(y), lv(sosfilt(butter(4, 2500, "highpass", fs=sr, output="sos"), y))
    for n, (w0, w1) in enumerate(zip(spans, spans[1:])):
        if w1["start"] - w0["end"] < min_gap or (after is not None and n not in after): continue
        a = w0["end"] + post_s; b = w1["start"] - pre_s
        k0, k1 = int(w1["start"] * 100), max(int(w1["start"] * 100) + 1, int(w1["end"] * 100))
        if k1 <= len(db):
            ref, ref_hf = np.median(db[k0:k1]), np.median(hf[k0:k1])
            k = int(b * 100)
            while k < min(len(db), int((w1["start"] + 0.05) * 100)) and db[k] < ref - onset_db and hf[k] < ref_hf - onset_hf_db: k += 1
            b = max(b, k / 100 - 0.03)
        if b - a < 2 * ramp_s: continue
        i, j = int(a * sr), int(b * sr); env = np.full(j - i, g)
        q = np.sin(np.linspace(0, np.pi / 2, r)) ** 2; env[:r] = 1 - (1 - g) * q; env[-r:] = g + (1 - g) * q
        y[i:j] *= env; done.append((round(a, 3), round(b, 3)))
    return y, done


def cut_parts(y: np.ndarray, sr: int, spans: list[dict], counts: list[int], pre_s: float = 0.06, post_s: float = 0.08,
              onset_s: float | None = None, preroll_max: float = 0.25, tight_inner: bool = False) -> list[tuple[np.ndarray, float, float, list[dict]]]:
    """Slice the clip into len(counts) consecutive parts at safe inter-word cut points. The first part starts pre_s before its first word
    (moved back to `onset_s` - 0.03 when an energy onset is earlier, by at most preroll_max: the onset-protection rule of tts_burst.speech_span),
    the last part ends post_s after its last word; leading/trailing silence of the clip is dropped. tight_inner: the inner edges too (a part ends
    post_s after its own last word, the next starts pre_s before its first), so what the TTS put in the gap (a hesitation murmur: 05 r7v3a 31.5 s,
    cut off into silence once the parts are placed apart) stays out. Returns (audio, start_s, end_s, word spans)."""
    cuts = cut_points(spans, counts, y, sr); dur = len(y) / sr
    s0 = max(0.0, spans[0]["start"] - pre_s)
    if onset_s is not None:
        s0 = max(0.0, max(s0 - preroll_max, min(s0, onset_s - 0.03)))
    e_last = min(dur, spans[-1]["end"] + post_s)
    bounds = [s0] + cuts + [e_last]; out = []; i = 0
    for k, c in enumerate(counts):
        a, b = bounds[k], bounds[k + 1]
        if tight_inner:
            if k: a = max(a, spans[i]["start"] - pre_s)
            if k < len(counts) - 1: b = min(b, spans[i + c - 1]["end"] + post_s)
        seg = _fade(y[int(a * sr):int(b * sr)], sr)
        out.append((seg, round(a, 3), round(b, 3), spans[i:i + c])); i += c
    return out
