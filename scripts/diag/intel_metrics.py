#!/usr/bin/env python3
"""Intelligibility metrics for synthesized speech chunks (diagnostic tooling, no pipeline code touched).

Two independent ASR decoders, neither of which is told the expected text:
  * faster-whisper large-v3 (free decode, temperature 0, no previous-text conditioning, chunk padded with silence) - language-model heavy, may "repair" mumbling
  * wav2vec2-base-960h greedy CTC - acoustic only, no language model: sensitive to swallowed endings / mush
Scores vs the expected text: WER, CER, insertions/deletions/substitutions, final-word similarity, missing-final-word, word-count mismatch, repetition.
Sound-level flags: stretched phonemes (CTC frames of one symbol), unexplained voiced sound (energy where CTC says blank), tail energy.
Forced alignment is NOT used as evidence of clarity (it would align bad pronunciation too); it is not used at all here.
"""
from __future__ import annotations
import difflib, re
import numpy as np

SR = 24000
_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def num_words(n: int) -> str:
    if n < 20: return _ONES[n]
    if n < 100: return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000: return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + num_words(n % 100))
    if n < 10000: return num_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + num_words(n % 1000))
    return str(n)


def norm(text: str) -> list[str]:
    t = text.lower().replace("’", "'").replace("%", " percent ").replace("×", " x ")
    t = re.sub(r"(\d+)\.(\d+)", lambda m: f"{num_words(int(m.group(1)))} point {' '.join(num_words(int(c)) for c in m.group(2))}", t)
    t = re.sub(r"\d+", lambda m: num_words(int(m.group(0))), t)
    t = re.sub(r"\bmm\b", "millimeters", t)
    return re.findall(r"[a-z']+", t)


def edit_ops(ref: list, hyp: list) -> tuple[int, int, int, int]:
    """(S, D, I, distance) via Levenshtein with backtrace over arbitrary token lists."""
    n, m = len(ref), len(hyp); d = np.zeros((n + 1, m + 1), dtype=int); d[:, 0] = range(n + 1); d[0, :] = range(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, S, D, I = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            S += ref[i - 1] != hyp[j - 1]; i -= 1; j -= 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            D += 1; i -= 1
        else:
            I += 1; j -= 1
    return S, D, I, int(d[n, m])


def score(expected: str, recognized: str) -> dict:
    r, h = norm(expected), norm(recognized); S, D, I, dist = edit_ops(r, h)
    rc, hc = list("".join(r)), list("".join(h)); cs, cd, ci, cdist = edit_ops(rc, hc) if rc else (0, 0, len(hc), len(hc))
    last = r[-1] if r else ""; cand = h[-2:] if h else []
    last_sim = max((difflib.SequenceMatcher(None, last, c).ratio() for c in cand), default=0.0)
    # last-word char-suffix proxy: how much of the expected final word's last 3 letters survive in the last recognized word
    suf = last[-3:]; suf_ok = bool(h) and (h[-1].endswith(suf) or any(c.endswith(suf) for c in h[-2:]))
    rep = 0
    for n_ in (1, 2, 3):
        for i in range(len(h) - 2 * n_ + 1):
            if h[i:i + n_] == h[i + n_:i + 2 * n_] and not (n_ == 1 and h[i] in ("that", "had")):
                rep += 1
    return {"wer": round(dist / max(len(r), 1), 3), "cer": round(cdist / max(len(rc), 1), 3), "sub": S, "del": D, "ins": I, "ref_words": len(r), "hyp_words": len(h),
            "word_count_mismatch": len(h) - len(r), "final_word_sim": round(last_sim, 2), "final_word_missing": last_sim < 0.6, "final_suffix_kept": suf_ok, "repetitions": rep}


class Asr:
    def __init__(self, device="cuda"):
        import torch
        from faster_whisper import WhisperModel
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        import glob
        self.torch = torch
        self.wh = WhisperModel("/workspace/dub/dubai/models/whisper-large-v3", device=device, compute_type="float16")
        p = glob.glob("/workspace/dub/dubai/models/hf/models--facebook--wav2vec2-base-960h/snapshots/*")[0]
        self.proc = Wav2Vec2Processor.from_pretrained(p); self.w2v = Wav2Vec2ForCTC.from_pretrained(p).to(device).eval(); self.device = device

    @staticmethod
    def to16(y: np.ndarray, sr: int = SR) -> np.ndarray:
        import librosa
        return librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 else y.astype(np.float32)

    def whisper(self, y16: np.ndarray, pad_s: float = 0.4, language="en") -> dict:
        pad = np.zeros(int(pad_s * 16000), dtype=np.float32); x = np.concatenate([pad, y16, pad])
        segs, info = self.wh.transcribe(x, language=language, beam_size=5, temperature=0.0, condition_on_previous_text=False, word_timestamps=True, vad_filter=False,
                                        no_speech_threshold=0.95, log_prob_threshold=None, compression_ratio_threshold=None)
        segs = list(segs); words = [{"w": w.word.strip(), "s": round(w.start - pad_s, 3), "e": round(w.end - pad_s, 3), "p": round(w.probability, 3)} for s in segs for w in (s.words or [])]
        return {"text": " ".join(s.text.strip() for s in segs).strip(), "words": words, "avg_logprob": round(float(np.mean([s.avg_logprob for s in segs])), 3) if segs else None,
                "no_speech": round(float(np.max([s.no_speech_prob for s in segs])), 3) if segs else None}

    def ctc(self, y16: np.ndarray) -> dict:
        torch = self.torch; x = self.proc(y16, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        with torch.no_grad():
            lg = self.w2v(x).logits[0].float(); pr = lg.softmax(-1).cpu().numpy()
        ids = pr.argmax(-1); vocab = {v: k for k, v in self.proc.tokenizer.get_vocab().items()}; blank = self.proc.tokenizer.pad_token_id
        txt = self.proc.decode(ids); frame_s = len(y16) / 16000 / len(ids)
        # stretched symbol: longest run of the SAME non-blank symbol (frames) in the raw argmax path, allowing 1-frame blank dropouts
        runs = []; i = 0
        while i < len(ids):
            if ids[i] == blank or vocab.get(int(ids[i])) == "|": i += 1; continue
            j = i
            while j + 1 < len(ids) and (ids[j + 1] == ids[i] or (ids[j + 1] == blank and j + 2 < len(ids) and ids[j + 2] == ids[i])): j += 1
            runs.append((vocab.get(int(ids[i])), i, j)); i = j + 1
        longest = max(runs, key=lambda r: r[2] - r[1], default=(None, 0, 0))
        return {"text": txt.lower().strip(), "frame_s": frame_s, "max_same_symbol_s": round((longest[2] - longest[1] + 1) * frame_s, 3), "max_symbol": longest[0],
                "blank_prob": pr[:, blank].astype(np.float32), "conf_mean": round(float(pr.max(-1).mean()), 3)}


def unexplained_sound(y16: np.ndarray, ctc: dict, rel_db: float = -32.0, min_s: float = 0.25) -> dict:
    """Voiced/energetic sound where the acoustic model sees no speech symbol: longest run of frames with energy within rel_db of the
    chunk's loud level while CTC blank prob > 0.9 (breath, hum, extended 'aaa' that decodes to nothing, vocal noise)."""
    fs = ctc["frame_s"]; n = len(ctc["blank_prob"]); hop = int(round(fs * 16000)); e = np.array([np.sqrt(np.mean(y16[i * hop:(i + 1) * hop] ** 2) + 1e-12) for i in range(n)])
    ref = np.percentile(e, 95) + 1e-9; loud = 20 * np.log10(e / ref + 1e-9) > rel_db; bl = ctc["blank_prob"] > 0.9; m = loud & bl
    best = (0, 0, 0); i = 0
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]: j += 1
            if j - i > best[2] - best[1]: best = (1, i, j)
            i = j + 1
        else: i += 1
    return {"unexplained_s": round((best[2] - best[1] + 1) * fs, 3) if best[0] else 0.0, "at_s": round(best[1] * fs, 3) if best[0] else None}


def voiced_stable_run(y16: np.ndarray) -> dict:
    """Longest run of voiced, spectrally stable frames (sustained vowel / hum): pyin voicing + low MFCC flux."""
    import librosa
    hop = 160; f0, vf, vp = librosa.pyin(y16, fmin=70, fmax=400, sr=16000, frame_length=1024, hop_length=hop)
    mf = librosa.feature.mfcc(y=y16, sr=16000, n_mfcc=13, hop_length=hop, n_fft=1024)[1:]; flux = np.r_[0, np.linalg.norm(np.diff(mf, axis=1), axis=0)]
    n = min(len(vf), mf.shape[1]); vf = vf[:n]; flux = flux[:n]; thr = np.percentile(flux, 40)
    m = vf & (np.convolve(flux, np.ones(5) / 5, mode="same") < thr); best = 0; bi = 0; i = 0
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]: j += 1
            if j - i + 1 > best: best, bi = j - i + 1, i
            i = j + 1
        else: i += 1
    return {"stable_voiced_s": round(best * hop / 16000, 3), "stable_at_s": round(bi * hop / 16000, 3)}


def tail_energy(y: np.ndarray, sr: int = SR) -> float:
    """dBFS (RMS) of an audio array, -120 for silence."""
    return round(20 * np.log10(np.sqrt(np.mean(y.astype(np.float64) ** 2)) + 1e-6), 1) if len(y) else -120.0


def held_vowel_tail(y: np.ndarray, sr: int = SR) -> dict:
    """Length of a sustained, spectrally stationary, LOUD sound at the very END of a chunk (the 'aaa' tail: a held vowel as loud as the loudest
    speech, no natural decay): walk backwards from the end (after trailing digital silence) while the log-mel frame-to-frame change stays below
    65 % of the chunk's median change and the frame energy is within 12 dB of the chunk's 90th-percentile loudness.  1-frame dropouts are bridged."""
    import librosa
    y16 = librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 else y.astype(np.float32); hop = 160
    mel = np.log(librosa.feature.melspectrogram(y=y16, sr=16000, n_fft=512, hop_length=hop, n_mels=40) + 1e-6); flux = np.r_[0, np.abs(np.diff(mel, axis=1)).mean(0)]
    rms = librosa.feature.rms(y=y16, frame_length=400, hop_length=hop)[0]; n = min(len(flux), len(rms)); flux, db = flux[:n], 20 * np.log10(rms[:n] + 1e-6); body = np.percentile(db, 90)
    med = np.median(flux[flux > 0]) if (flux > 0).any() else 1.0; ok = (flux < 0.65 * med) & (db > body - 12); i = n - 1
    while i > 0 and db[i] < body - 40: i -= 1
    i -= 1                                                        # last frame's flux is the boundary artefact
    end = i
    while i > 0 and (ok[i] or ok[i - 1]): i -= 1
    return {"held_tail_s": round((end - i) * hop / 16000, 3), "held_tail_db": round(float(np.mean(db[i:end + 1])), 1) if end > i else None, "body_db": round(float(body), 1)}
