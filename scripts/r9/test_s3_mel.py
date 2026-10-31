"""r9: is the batched CFM's difference padding leakage or rounding? mel of take k alone / in a same-length batch (no padding) / padded"""
import pathlib, sys
import torch
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE))
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
import t3_batch, s3_batch
m = ChatterboxMultilingualTTS.from_local(HERE.parents[1] / "models" / "chatterbox", "cuda", t3_model="v3")
for mod in (m.t3, m.s3gen, m.ve): mod.to(torch.float16)
with torch.autocast("cuda", dtype=torch.float16): m.prepare_conditionals(sys.argv[1])
T = ["Everything is simple, everything is easy, accessible, and understandable. Thank you very much.", "Hello, my name is Maria.",
     "After sitting at home for three whole years, I finally understood that I could not continue living this way anymore."]
items = [dict(text=t, seed=7 + k, temperature=0.8, cfg_weight=0.5, language_id="en") for k, t in enumerate(T)]
with torch.autocast("cuda", dtype=torch.float16):
    decs = t3_batch.decode_batch(m, items); ps = [s3_batch._prep(m, it, d) for it, d in zip(items, decs)]
    for fp in ("fp16",):
        a = s3_batch._solve(m, [ps[0]])[0]; aa = s3_batch._solve(m, [ps[0], ps[0]]); pad = s3_batch._solve(m, [ps[0], ps[2]])[0]
        a4 = s3_batch._solve(m, [ps[0]] * 4)
        d = lambda x, y: f"max {float((x - y).abs().max()):.4f} mean {float((x - y).abs().mean()):.5f}"
        print("alone vs alone again:", d(a, s3_batch._solve(m, [ps[0]])[0]))
        print("alone vs batch-2 same take (no padding):", d(a, aa[0]), "| row 0 vs row 1:", d(aa[0], aa[1]))
        print("alone vs batch-4 same take:", d(a, a4[0]))
        print("alone vs batch with a longer take (padded):", d(a, pad), "| T", ps[0]["mu"].shape[2], ps[2]["mu"].shape[2])
        print("mel scale: mean |mel|", float(a.abs().mean()))
