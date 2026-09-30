"""Independent-ASR intelligibility QA for synthesized TTS parts (round trip: expected text -> TTS -> ASR -> score).

Two decoders that are never told the expected text: faster-whisper large-v3 (free decode, temperature 0, silence-padded) and wav2vec2-base-960h greedy CTC
(acoustic only, no language model, so it does not "repair" mumbling). A finite / non-silent waveform is NOT evidence of intelligible speech: the fp16
speech-token corruption (see chatterbox_fp16.py) produced finite, loud, unintelligible audio.  Whisper invents text on 1-2 s fragments, therefore the
verdict uses the CTC CER alone for parts of <= 2 words and the mean of both CERs otherwise; stray um/ah/oh/ha tokens always count.

    qa = TtsQA(device="cuda");  r = qa.assess("Some expected text.", wav24k_float32)  ->  r["bad"], r["score"], r["reasons"], ...
"""
from __future__ import annotations

import difflib
import glob
import pathlib
import re

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
TTS_SR = 24000
BAD_SCORE = 0.30            # CER (0..1) above which a part is "not intelligible enough" (clean fp16-fixed parts: mean 0.09, see the diagnosis report)
_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
INTERJECTION = re.compile(r"\b(um+|uh+|ah+|aah+|oh+|hmm+|mm+|erm?|eh+|ha+|huh|aha)\b")


def num_words(n: int) -> str:
    if n < 20: return _ONES[n]
    if n < 100: return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000: return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + num_words(n % 100))
    if n < 10000: return num_words(n // 1000) + " thousand" + ("" if n % 1000 == 0 else " " + num_words(n % 1000))
    return str(n)


def norm(text: str) -> list[str]:
    t = text.lower().replace("’", "'").replace("%", " percent ").replace("×", " x ")
    t = re.sub(r"(\d+)[.,](\d+)", lambda m: f"{num_words(int(m.group(1)))} point {' '.join(num_words(int(c)) for c in m.group(2))}", t)
    t = re.sub(r"\d+", lambda m: num_words(int(m.group(0))), t)
    t = re.sub(r"\bmm\b", "millimeters", t)
    return re.findall(r"[a-z']+", t)


def _distance(ref: list, hyp: list) -> int:
    n, m = len(ref), len(hyp)
    d = list(range(m + 1))
    for i in range(1, n + 1):
        prev, d[0] = d[0], i
        for j in range(1, m + 1):
            cur = d[j]
            d[j] = min(d[j] + 1, d[j - 1] + 1, prev + (ref[i - 1] != hyp[j - 1]))
            prev = cur
    return d[m]


def cer(expected: str, recognized: str) -> float:
    r, h = list("".join(norm(expected))), list("".join(norm(recognized)))
    return _distance(r, h) / max(len(r), 1)


def wer(expected: str, recognized: str) -> float:
    r, h = norm(expected), norm(recognized)
    return _distance(r, h) / max(len(r), 1)


def stray_interjections(recognized: str, expected: str) -> list[str]:
    exp = set(norm(expected))
    return [w for w in INTERJECTION.findall(recognized.lower()) if w not in exp]


def final_word_missing(expected: str, recognized: str) -> bool:
    r, h = norm(expected), norm(recognized)
    if not r:
        return False
    return max((difflib.SequenceMatcher(None, r[-1], c).ratio() for c in h[-2:]), default=0.0) < 0.6


class TtsQA:
    def __init__(self, device: str = "cuda", whisper_dir: str | pathlib.Path | None = None, w2v_dir: str | pathlib.Path | None = None):
        import torch
        from faster_whisper import WhisperModel
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        self.torch, self.device = torch, device
        self.wh = WhisperModel(str(whisper_dir or MODELS / "whisper-large-v3"), device=device, compute_type="float16")
        p = str(w2v_dir or MODELS / "wav2vec2-base-960h")
        self.proc = Wav2Vec2Processor.from_pretrained(p)
        self.w2v = Wav2Vec2ForCTC.from_pretrained(p).to(device).eval()

    @staticmethod
    def to16(y: np.ndarray, sr: int = TTS_SR) -> np.ndarray:
        import librosa
        return librosa.resample(y.astype(np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 else y.astype(np.float32)

    def whisper(self, y16: np.ndarray, language: str = "en", pad_s: float = 0.4) -> str:
        pad = np.zeros(int(pad_s * 16000), dtype=np.float32)
        segs, _ = self.wh.transcribe(np.concatenate([pad, y16, pad]), language=language, beam_size=5, temperature=0.0, condition_on_previous_text=False,
                                     vad_filter=False, no_speech_threshold=0.95, log_prob_threshold=None, compression_ratio_threshold=None)
        return " ".join(s.text.strip() for s in segs).strip()

    def ctc(self, y16: np.ndarray) -> str:
        torch = self.torch
        x = self.proc(y16, sampling_rate=16000, return_tensors="pt").input_values.to(self.device)
        with torch.no_grad():
            ids = self.w2v(x).logits[0].argmax(-1).cpu()
        return self.proc.decode(ids).lower().strip()

    def assess(self, expected: str, y: np.ndarray, sr: int = TTS_SR, language: str = "en") -> dict:
        y16 = self.to16(y, sr)
        words = len(norm(expected))
        if len(y16) < 1600 or float(np.abs(y16).max()) < 1e-4:
            return {"bad": True, "score": 1.0, "reasons": ["silent_or_empty"], "ctc": "", "whisper": "", "ctc_cer": 1.0, "w_cer": 1.0, "interjections": [], "words": words}
        c_txt, w_txt = self.ctc(y16), self.whisper(y16, language)
        c_cer, w_cer = cer(expected, c_txt), cer(expected, w_txt)
        ints = stray_interjections(c_txt, expected) + stray_interjections(w_txt, expected)
        score = c_cer if words <= 2 else 0.5 * (c_cer + w_cer)
        dur = len(y) / sr
        reasons = []
        if score > BAD_SCORE: reasons.append(f"cer {score:.2f}")
        if ints: reasons.append("interjection " + ",".join(sorted(set(ints))))
        if words <= 2 and dur / max(words, 1) > 1.2: reasons.append(f"stretched {dur:.1f}s for {words} words")
        return {"bad": bool(reasons), "score": round(float(score), 3), "reasons": reasons, "ctc": c_txt, "whisper": w_txt, "ctc_cer": round(c_cer, 3), "w_cer": round(w_cer, 3),
                "interjections": ints, "words": words, "final_word_missing": final_word_missing(expected, c_txt) and final_word_missing(expected, w_txt)}

    def close(self):
        import gc
        del self.wh, self.w2v
        gc.collect(); self.torch.cuda.empty_cache()
