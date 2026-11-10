"""r16c K2: KeySync noise sharing across windows / keyframes (the beard / face "springing" = texture boil, Irina 08-09.11, q04 / q05 / q03).
KeySync draws FRESH Gaussian noise for every keyframe batch (dubbing_pipeline_raw.py: video = torch.randn(shape) in sample_keyframes) and every
14-frame interpolation window (the same in sample_interpolation) + the student's re-noise between its steps (randn_like in the LCM / gate sampler),
so the high-frequency texture (beard, pores) is re-rolled every window. KS_NOISE (ks_fast.py installs it after the gate):
  static        one fixed noise frame (seed KS_NOISE_SEED, default 1234) for EVERY frame of every keyframe batch / window / re-noise
  window        one fixed (T, C, H, W) noise block reused by every batch / window (frame j of each window gets the same noise)
  blend:R       sqrt(R) * static + sqrt(1 - R) * fresh (R in 0..1: the share of shared noise variance; fresh noise keeps the per-frame freedom)
Only tensors of the latent shape (..., C, H, W of the first latent seen) inside sample_keyframes / sample_interpolation are replaced; everything
else (conditioning, audio) is untouched. Prints KS_NOISE lines."""
import math, os
import torch

MODE = os.environ.get("KS_NOISE", "")
SEED = int(os.environ.get("KS_NOISE_SEED", "1234"))
ST = {"base": None, "block": None, "calls": 0}
_randn, _randn_like = torch.randn, torch.randn_like


def _shared(shape, device, dtype):
    C, H, W = shape[-3:]; T = int(math.prod(shape[:-3])) if len(shape) > 3 else 1
    if ST["base"] is None or ST["base"].shape[-3:] != (C, H, W):
        g = torch.Generator(device="cpu").manual_seed(SEED)
        ST["base"] = _randn((1, C, H, W), generator=g); ST["block"] = _randn((64, C, H, W), generator=g)
    if MODE == "window":
        b = ST["block"]
        if T > len(b): b = b.repeat(math.ceil(T / len(b)), 1, 1, 1)
        n = b[:T]
    else:
        n = ST["base"].expand(T, C, H, W)
    n = n.reshape(shape).to(device=device, dtype=dtype)
    if MODE.startswith("blend:"):
        r = float(MODE.split(":")[1]); n = math.sqrt(r) * n + math.sqrt(1 - r) * _randn(shape, device=device, dtype=dtype)
    ST["calls"] += 1
    return n


def _is_latent(shape):
    return len(shape) >= 4 and shape[-3] in (4, 8) and shape[-1] >= 16 and shape[-2] >= 16


def randn(*size, **kw):
    shape = tuple(size[0]) if len(size) == 1 and isinstance(size[0], (tuple, list, torch.Size)) else tuple(size)
    if _is_latent(shape) and "generator" not in kw:
        return _shared(shape, kw.get("device", "cpu"), kw.get("dtype", torch.float32))
    return _randn(*size, **kw)


def randn_like(x, **kw):
    if isinstance(x, torch.Tensor) and _is_latent(tuple(x.shape)):
        return _shared(tuple(x.shape), kw.get("device", x.device), kw.get("dtype", x.dtype))
    return _randn_like(x, **kw)


def _wrap(fn):
    def w(*a, **k):
        torch.randn, torch.randn_like = randn, randn_like
        try: return fn(*a, **k)
        finally: torch.randn, torch.randn_like = _randn, _randn_like
    return w


def install(d):
    if not MODE or MODE == "off": return
    d.sample_keyframes = _wrap(d.sample_keyframes); d.sample_interpolation = _wrap(d.sample_interpolation)
    print(f"KS_NOISE {MODE} (seed {SEED}): keyframe batches + interpolation windows + re-noise share their noise", flush=True)
    import atexit
    atexit.register(lambda: print(f"KS_NOISE replaced {ST['calls']} latent noise draws", flush=True))
