"""Silero VAD on the demo speech file."""
from smoke_common import *
import torch
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
torch.set_num_threads(THREADS)
model = load_silero_vad()
wav = read_audio(str(DEMO_WAV), sampling_rate=16000)
with Timer() as t:
    ts = get_speech_timestamps(wav, model, sampling_rate=16000, return_seconds=True)
print(f"audio {len(wav)/16000:.1f}s, {len(ts)} speech segments in {t.s:.2f}s")
for s in ts[:5]: print("  ", s)
if not ts: fail("no speech detected in demo audio")
print("PASS silero-vad")
