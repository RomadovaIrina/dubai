#!/usr/bin/env python3
"""r13 distillation pilot, eval: the PRODUCTION face generator with the LoRA student merged into the LatentSync UNet, 4 steps, no CFG
(guidance 1.0), no DeepCache; nothing edited (r4_video.sh runs ${R10_GEN_PY}; the UNet loader is wrapped at runtime).
    R10_GEN_PY=scripts/r13/distill/student_gen.py R13_STUDENT_LORA=<dir>/lora.pt bash scripts/r9/e2e.sh ...   (the rest: the k5 composite as usual)"""
import os, pathlib, runpy, sys
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "pilot")); sys.path.insert(0, str(ROOT / "scripts" / "quality")); sys.path.insert(0, str(ROOT / "third_party" / "latentsync"))
import torch
from latentsync.models.unet import UNet3DConditionModel

LORA = os.environ["R13_STUDENT_LORA"]; orig = UNet3DConditionModel.from_pretrained.__func__


def from_pretrained(cls, *a, **k):
    unet, info = orig(cls, *a, **k)
    if LORA == "none":   # the control: the same 4-step / no-CFG sampler with the ORIGINAL weights
        print("[student] control: no LoRA (original weights, 4 steps, no CFG)", flush=True); return unet, info
    d = torch.load(LORA, map_location="cpu"); sd = d["lora"]; merged = 0
    mods = dict(unet.named_modules())
    for name in d["names"]:   # W <- W + B @ A (alpha = rank: scale 1)
        lin = mods[name]; A = sd[f"{name}.A"].float(); B = sd[f"{name}.B"].float()
        with torch.no_grad(): lin.weight.data = (lin.weight.data.float() + B @ A).to(lin.weight.dtype)
        merged += 1
    print(f"[student] merged LoRA ({merged} projections, step {d.get('step')}) from {LORA}", flush=True)
    return unet, info


UNet3DConditionModel.from_pretrained = classmethod(from_pretrained)
target = ROOT / "scripts" / "quality" / "candidate_video_stage.py"
sys.argv[0] = str(target); N = os.environ.get("R13_STUDENT_STEPS", "4"); sys.argv += ["--steps", N, "--steps-silent", N, "--guidance", "1.0", "--no-deepcache", "--bridge-steps", "1"] + os.environ.get("R13_STUDENT_EXTRA", "").split()   # r14: extra generator flags, e.g. --ref-mode shift32 (overlap: LS_STRIDE / LS_OVERLAP env)
runpy.run_path(str(target), run_name="__main__")
