"""r9: where S3Gen's per-take time goes (flow CFM / HiFiGAN / watermark), batch 1, fp16 autocast as in the pipeline"""
import pathlib, sys, time
import torch
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot"))
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import t3_batch
m = ChatterboxMultilingualTTS.from_local(HERE.parents[1] / "models" / "chatterbox", "cuda", t3_model="v3")
for mod in (m.t3, m.s3gen, m.ve): mod.to(torch.float16)
with torch.autocast("cuda", dtype=torch.float16): m.prepare_conditionals(sys.argv[1])
T = {"flow": 0.0, "hift": 0.0, "wm": 0.0}
def timed(obj, name, key):
    f = getattr(obj, name)
    def g(*a, **k):
        torch.cuda.synchronize(); t = time.perf_counter(); r = f(*a, **k); torch.cuda.synchronize(); T[key] += time.perf_counter() - t; return r
    setattr(obj, name, g)
timed(m.s3gen, "flow_inference", "flow"); timed(m.s3gen, "hift_inference", "hift"); timed(m.watermarker, "apply_watermark", "wm")
items = [dict(text=t, seed=7 + k, temperature=0.8, cfg_weight=0.5, language_id="en") for k, t in enumerate([
    "I found out about this app from my friends and I want to express huge gratitude to the developers.",
    "Everything is simple, everything is easy, accessible, and understandable. Thank you very much.",
    "This will perfectly suit moms who are on maternity leave and prefer to work online."] * 3)]
with torch.autocast("cuda", dtype=torch.float16):
    decs = t3_batch.decode_batch(m, items)
    t3_batch.vocode(m, items[0], decs[0])
for k in T: T[k] = 0.0
t0 = time.perf_counter()
with torch.autocast("cuda", dtype=torch.float16):
    for it, d in zip(items, decs): t3_batch.vocode(m, it, d)
torch.cuda.synchronize(); tot = time.perf_counter() - t0
print({k: round(v / len(items), 3) for k, v in T.items()}, "vocode", round(tot / len(items), 3), "tokens", [len(d[0]) for d in decs])
