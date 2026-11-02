"""r11 FACE: a minimal LoRA for the LatentSync UNet (no peft dependency). Adapters on the attention / motion-module projections
(the modules LatentSync's own stage-2 recipe trains: 'attentions.' and 'motion_modules.'); merged into the base weights at load, so inference
costs nothing extra.
    add_lora(unet, rank, alpha) -> list of trainable params;  save_lora(unet, path);  merge_lora(unet, path) (in place, then the adapters are gone)
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn

TARGET_LEAVES = ("to_q", "to_k", "to_v", "to_out.0")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base; self.rank = rank; self.scale = alpha / rank; self.enabled = True
        self.A = nn.Parameter(torch.zeros(rank, base.in_features, dtype=torch.float32, device=base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=base.weight.device))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))   # B = 0: the adapter starts as the identity

    def forward(self, x):
        y = self.base(x)
        if not self.enabled: return y
        return y + ((x.to(self.A.dtype) @ self.A.t()) @ self.B.t() * self.scale).to(y.dtype)


def _targets(unet, which: str = "all"):
    """all = attentions (attn1 self + attn2 audio cross) + motion modules (LatentSync's stage-2 set);
    self = the spatial SELF-attention (attn1) only: appearance, the audio -> mouth mapping and the temporal motion stay the base model's."""
    for name, mod in unet.named_modules():
        if not isinstance(mod, nn.Linear): continue
        if not ("attentions." in name or "motion_modules." in name): continue
        if not any(name.endswith(t) for t in TARGET_LEAVES): continue
        if which == "self" and ("motion_modules." in name or ".attn1." not in name): continue
        yield name


def _set(unet, name, new):
    parent = unet; parts = name.split(".")
    for p in parts[:-1]: parent = getattr(parent, p)
    setattr(parent, parts[-1], new)


def _get(unet, name):
    m = unet
    for p in name.split("."): m = getattr(m, p)
    return m


def add_lora(unet, rank: int = 16, alpha: float = 16.0, which: str = "all"):
    params = []
    for name in list(_targets(unet, which)):
        lin = _get(unet, name); w = LoRALinear(lin, rank, alpha); _set(unet, name, w); params += [w.A, w.B]
    return params


def set_enabled(unet, on: bool):
    """switch every adapter on / off (off = the frozen base model, for distillation targets)"""
    for mod in unet.modules():
        if isinstance(mod, LoRALinear): mod.enabled = on


def save_lora(unet, path):
    sd = {}
    for name, mod in unet.named_modules():
        if isinstance(mod, LoRALinear): sd[name] = {"A": mod.A.detach().cpu(), "B": mod.B.detach().cpu(), "scale": mod.scale}
    torch.save(sd, path); return len(sd)


@torch.no_grad()
def merge_lora(unet, path, strength: float = 1.0):
    """W += strength * scale * B @ A for every saved adapter (the UNet keeps its plain Linear layers)."""
    sd = torch.load(path, map_location="cpu"); n = 0
    for name, v in sd.items():
        lin = _get(unet, name)
        if isinstance(lin, LoRALinear): lin = lin.base
        d = (v["B"].float() @ v["A"].float()) * v["scale"] * strength
        lin.weight.add_(d.to(lin.weight.device, lin.weight.dtype)); n += 1
    return n
