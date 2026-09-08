"""Chatterbox Multilingual v3 from local weights, fp16 (user-applied cast; upstream runs fp32), EN + RU sample."""
from smoke_common import *
import torch, torchaudio, importlib.metadata as md
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
ckpt = MODELS / "chatterbox"
print("chatterbox-tts", md.version("chatterbox-tts"), "| weights:", ckpt, "| t3=v3")
with Timer() as tl:
    m = ChatterboxMultilingualTTS.from_local(ckpt, "cuda", t3_model="v3")
print(f"loaded fp32 in {tl.s:.1f}s, sr={m.sr}")
TEXTS = [("en", "Reproducible environments make pilot testing boring, which is exactly the point."),
         ("ru", "Воспроизводимое окружение делает пилотное тестирование скучным, и это правильно.")]

def cast(model, dtype):
    for mod in (model.t3, model.s3gen, model.ve):
        mod.to(dtype)
    if model.conds is not None:
        model.conds = model.conds.to(device="cuda")  # tensors inside are matched by chatterbox at generate time
    return model

def run(dtype, tag):
    cast(m, dtype)
    outs = []
    for lang, text in TEXTS:
        with Timer() as t, torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            wav = m.generate(text, language_id=lang)
        wav = wav.detach().float().cpu()
        if not torch.isfinite(wav).all() or wav.abs().max() < 1e-4:
            raise RuntimeError(f"{tag}: non-finite or silent output for {lang}")
        p = REPORTS / f"tts_{lang}_{tag}.wav"; torchaudio.save(str(p), wav, m.sr)
        print(f"  [{tag}] {lang}: {wav.shape[-1]/m.sr:.2f}s audio in {t.s:.1f}s -> {p.name}")
        outs.append(p)
    return outs

used = None
for dtype, tag in ((torch.float16, "fp16"), (torch.bfloat16, "bf16"), (torch.float32, "fp32")):
    try:
        torch.cuda.reset_peak_memory_stats(); run(dtype, tag); used = tag; break
    except Exception as e:
        print(f"  {tag} failed: {type(e).__name__}: {str(e)[:200]}")
if used is None: fail("chatterbox generation failed in every dtype")
print(f"PASS chatterbox ({used}, requested fp16)", gpu_mem())
