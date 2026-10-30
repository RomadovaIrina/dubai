"""r9 speed (audio lever W, DUB_QA_BATCH=1): the TTS QA's Whisper (large-v3, beam 5) of many takes in one batch, with faster-whisper's own logic.

The QA hears every take with WhisperModel.transcribe (tts_qa.TtsQA.whisper): ~0.1-0.2 s per take, one take at a time. Here each take's own
transcribe() runs as a cooperative worker (scripts/r9/coop.py), and the two GPU calls inside it are parked and made for all takes together:
WhisperModel.encode (the mel features, every take padded to Whisper's 30-s window anyway) and the ctranslate2 generate (beam search, one
prompt per take). Segment splitting, the no-speech skip, a second window, the text join: unchanged per take. Differences to one-by-one calls
are the batched GEMM rounding only.
    texts = whisper_many(qa, [y16, ...]) == [qa.whisper(y16) for y16 in ...]"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import coop  # noqa: E402

STATS = {"takes": 0, "enc_batches": 0, "gen_batches": 0}


class _H:   # one take's encoder output: row i of a batched StorageView (+ its features, to re-encode a row set that is not one whole batch)
    __slots__ = ("sv", "i", "feat")

    def __init__(self, sv, i, feat):
        self.sv, self.i, self.feat = sv, i, feat


class _Model:   # the ctranslate2 Whisper model with generate() parked; everything else passes through
    def __init__(self, m, co):
        self._m, self._co = m, co

    def generate(self, enc, prompts, **kw):
        return self._co.call("gen", (enc, prompts, kw))

    def __getattr__(self, k):
        return getattr(self._m, k)


def whisper_many(qa, y16s, language: str = "en"):
    import numpy as np
    from faster_whisper.transcribe import get_ctranslate2_storage
    wh = qa.wh; m0 = wh.model
    def enc_h(feats):   # [features (n_mels, frames)] -> handles into ONE batched encode
        fs = [f if f.ndim == 2 else f[0] for f in feats]
        sv = m0.encode(get_ctranslate2_storage(np.stack(fs)), to_cpu=m0.device == "cuda" and len(m0.device_index) > 1); STATS["enc_batches"] += 1
        return [_H(sv, i, f) for i, f in enumerate(fs)]
    def gen_h(ps):   # [(handle, [prompt], kwargs)] -> [[result]]: one generate per distinct kwargs over the rows of one encoder batch
        out = [None] * len(ps); groups = {}
        for k, (h, pr, kw) in enumerate(ps): groups.setdefault(repr(sorted(kw.items())), []).append(k)
        for ks in groups.values():
            hs = [ps[k][0] for k in ks]
            if all(h.sv is hs[0].sv for h in hs) and [h.i for h in hs] == list(range(len(hs))) and len(hs) == hs[0].sv.shape[0]:
                sv = hs[0].sv
            else:   # not exactly one whole encoder batch in order: encode these rows together again (same features)
                sv = m0.encode(get_ctranslate2_storage(np.stack([h.feat for h in hs])), to_cpu=m0.device == "cuda" and len(m0.device_index) > 1)
            res = m0.generate(sv, [ps[k][1][0] for k in ks], **ps[ks[0]][2]); STATS["gen_batches"] += 1
            for k, r in zip(ks, res): out[k] = [r]
        return out
    co = coop.Coop({"enc": enc_h, "gen": gen_h}, order=("enc", "gen"))
    wh.model = _Model(m0, co); wh.encode = lambda features: co.call("enc", features)
    try:
        out = co.map(lambda y: qa.whisper(y, language), list(y16s))
    finally:
        wh.model = m0; del wh.encode
    STATS["takes"] += len(out)
    return out


def prefetch(qa, wavs, sr, language: str = "en"):
    """the Whisper texts of these takes into qa's cache (TtsQA.assess then uses them): the same 16-kHz input assess would make"""
    import numpy as np
    ys = []
    for w in wavs:
        if w is None: continue
        y = w.detach().float().cpu()[0].numpy().astype(np.float32) if hasattr(w, "detach") else np.asarray(w, dtype=np.float32)
        y16 = qa.to16(y, sr)
        if len(y16) < 1600 or float(np.abs(y16).max()) < 1e-4: continue   # assess never calls Whisper on these
        ys.append(y16)
    if not ys: return
    for y16, t in zip(ys, whisper_many(qa, ys, language)): qa.wcache[(language, y16.shape[0], y16.tobytes())] = t
