"""r9: qa_batch.whisper_many vs TtsQA.whisper one by one on real Chatterbox takes (texts identical?, time), and the CTC share of assess"""
import pathlib, sys, time
import numpy as np, torch
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE))
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import t3_batch, s3_batch, qa_batch
from tts_qa import TtsQA
m = ChatterboxMultilingualTTS.from_local(HERE.parents[1] / "models" / "chatterbox", "cuda", t3_model="v3")
for mod in (m.t3, m.s3gen, m.ve): mod.to(torch.float16)
with torch.autocast("cuda", dtype=torch.float16): m.prepare_conditionals(sys.argv[1])
T = ["I found out about this app from my friends and I want to express huge gratitude to the developers.",
     "Everything is simple, everything is easy, accessible, and understandable. Thank you very much.",
     "This will perfectly suit moms who are on maternity leave and prefer to work online.", "Hello, my name is Maria.",
     "After sitting at home for three whole years, I finally understood that I could not continue living this way anymore."]
items = [dict(text=t, seed=7 + k, temperature=0.8, cfg_weight=0.5, language_id="en") for k, t in enumerate(T * 4)]
with torch.autocast("cuda", dtype=torch.float16):
    wavs = s3_batch.vocode_batch(m, items, t3_batch.decode_batch(m, items))
qa = TtsQA(); ys = [qa.to16(w[0].numpy(), m.sr) for w in wavs]
qa.whisper(ys[0]); qa_batch.whisper_many(qa, ys[:2])
t0 = time.perf_counter(); one = [qa.whisper(y) for y in ys]; t1 = time.perf_counter(); bat = qa_batch.whisper_many(qa, ys); t2 = time.perf_counter()
c = [qa.ctc(y) for y in ys]; t3 = time.perf_counter()
print(f"whisper one by one {(t1 - t0) / len(ys):.3f} s/take, batched {(t2 - t1) / len(ys):.3f} s/take; ctc {(t3 - t2) / len(ys):.3f} s/take; identical {sum(a == b for a, b in zip(one, bat))}/{len(ys)}")
for a, b in zip(one, bat):
    if a != b: print("  DIFF one :", a); print("       bat :", b)
print(qa_batch.STATS)
