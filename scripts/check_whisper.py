"""faster-whisper large-v3, compute_type=int8_float16 (runs as float16 on sm_120), word timestamps."""
from smoke_common import *
import torch  # noqa: maps libcublas before ctranslate2 loads
import ctranslate2
from faster_whisper import WhisperModel
print("ct2 supported cuda compute types:", ctranslate2.get_supported_compute_types("cuda"))
with Timer() as tl:
    m = WhisperModel(str(MODELS / "whisper-large-v3"), device="cuda", compute_type="int8_float16")
print(f"model loaded in {tl.s:.1f}s; effective compute type: {m.model.compute_type}")
with Timer() as t:
    segs, info = m.transcribe(str(DEMO_WAV), word_timestamps=True, beam_size=5, vad_filter=False)
    segs = list(segs)
print(f"language={info.language} p={info.language_probability:.2f}, duration={info.duration:.1f}s, transcribe {t.s:.1f}s")
words = [w for s in segs for w in (s.words or [])]
for s in segs[:3]: print(f"  [{s.start:6.2f}-{s.end:6.2f}] {s.text.strip()}")
print(f"  {len(words)} words, first: " + " ".join(f"{w.word.strip()}@{w.start:.2f}" for w in words[:8]))
if not words: fail("no words with timestamps")
print("PASS faster-whisper", gpu_mem())
