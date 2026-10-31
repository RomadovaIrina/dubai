"""r9: s3_batch.vocode_batch vs t3_batch.vocode one by one on the same decoded takes (wave diff, time)"""
import pathlib, sys, time
import numpy as np, torch
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE))
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import t3_batch, s3_batch
m = ChatterboxMultilingualTTS.from_local(HERE.parents[1] / "models" / "chatterbox", "cuda", t3_model="v3")
for mod in (m.t3, m.s3gen, m.ve): mod.to(torch.float16)
with torch.autocast("cuda", dtype=torch.float16): m.prepare_conditionals(sys.argv[1])
T = ["I found out about this app from my friends and I want to express huge gratitude to the developers.",
     "Everything is simple, everything is easy, accessible, and understandable. Thank you very much.",
     "This will perfectly suit moms who are on maternity leave and prefer to work online.", "Hello, my name is Maria.",
     "After sitting at home for three whole years, I finally understood that I could not continue living this way anymore."]
items = [dict(text=t, seed=7 + k, temperature=0.8, cfg_weight=0.5, language_id="en") for k, t in enumerate(T * 3)]
ac = lambda: torch.autocast("cuda", dtype=torch.float16)
with ac(): decs = t3_batch.decode_batch(m, items); t3_batch.vocode(m, items[0], decs[0]); s3_batch.vocode_batch(m, items[:2], decs[:2])
def sync(): torch.cuda.synchronize(); return time.perf_counter()
with ac():
    t0 = sync(); one = [t3_batch.vocode(m, it, d) for it, d in zip(items, decs)]; t1 = sync()
    bat = s3_batch.vocode_batch(m, items, decs); t2 = sync()
    b1 = s3_batch.vocode_batch(m, items[:1], decs[:1])
print(f"one by one {(t1 - t0) / len(items):.3f} s/take, batched {(t2 - t1) / len(items):.3f} s/take")
print("batch of 1 identical:", torch.equal(b1[0], one[0]))
for k, (a, b) in enumerate(zip(one, bat)):
    a, b = a[0].numpy(), b[0].numpy(); n = min(len(a), len(b))
    print(k, len(a), len(b), "max|d| %.5f" % np.abs(a[:n] - b[:n]).max(), "rel rms %.5f" % (np.sqrt(np.mean((a[:n] - b[:n]) ** 2)) / np.sqrt(np.mean(a[:n] ** 2))))
