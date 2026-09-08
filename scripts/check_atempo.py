"""Duration change: librosa.effects.time_stretch and ffmpeg atempo (CPU)."""
from smoke_common import *
import subprocess, librosa, soundfile as sf, numpy as np
y, sr = librosa.load(str(DEMO_WAV), sr=None, mono=True)
rate = 1.25
with Timer() as t:
    ys = librosa.effects.time_stretch(y, rate=rate)
out1 = REPORTS / "atempo_librosa.wav"; sf.write(out1, ys, sr)
print(f"librosa: {len(y)/sr:.2f}s -> {len(ys)/sr:.2f}s (rate {rate}) in {t.s:.2f}s -> {out1.name}")
out2 = REPORTS / "atempo_ffmpeg.wav"
subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(DEMO_WAV), "-af", f"atempo={rate}", str(out2)], check=True)
d2 = librosa.get_duration(path=str(out2))
print(f"ffmpeg atempo: {len(y)/sr:.2f}s -> {d2:.2f}s -> {out2.name}")
if abs(len(ys)/sr - len(y)/sr/rate) > 0.1 or abs(d2 - len(y)/sr/rate) > 0.1: fail("stretched duration off by >0.1s")
print("PASS librosa/atempo")
