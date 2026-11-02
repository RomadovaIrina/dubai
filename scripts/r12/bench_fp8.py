#!/usr/bin/env python3
"""r12 LS: LatentSync UNet full-call time at the production shape (window batch 2 x CFG 2 = 4 rows, 13 x 16 x 64 x 64, audio 64 x 50 x 384),
fp16 vs torchao FP8 (dynamic activation + weight) on the nn.Linear layers, eager and torch.compile, + the output difference vs fp16.
    flock /workspace/dub/gpu.lock python scripts/r12/bench_fp8.py [--rows 4] [--iters 20] [--min-in 320]"""
import argparse, copy, pathlib, sys, time
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "optim")); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot"))
import latentsync_accel  # noqa: E402,F401  (puts third_party/latentsync on sys.path)
from resident_vram_probe import load_latentsync   # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--rows", type=int, default=4); ap.add_argument("--iters", type=int, default=20); ap.add_argument("--min-in", type=int, default=320)
ap.add_argument("--compile", choices=["off", "on"], default="off"); a = ap.parse_args()
pipe = load_latentsync(); unet = pipe.unet.eval(); dev = "cuda"; dt = torch.float16
g = torch.Generator(device=dev).manual_seed(0)
x = torch.randn(a.rows, 13, 16, 64, 64, device=dev, dtype=dt, generator=g); au = torch.randn(a.rows * 16, 50, 384, device=dev, dtype=dt, generator=g)
t = torch.tensor(500, device=dev)


def bench(m, name):
    f = torch.compile(m, mode="default") if a.compile == "on" else m
    with torch.no_grad():
        for _ in range(3): y = f(x, t, encoder_hidden_states=au).sample
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(a.iters): y = f(x, t, encoder_hidden_states=au).sample
        torch.cuda.synchronize(); ms = 1000 * (time.perf_counter() - t0) / a.iters
    print(f"== {name} compile={a.compile}: {ms:.1f} ms / full call ({a.rows} rows)", flush=True)
    return y.float()


ref = bench(unet, "fp16")
from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight   # noqa: E402
u8 = copy.deepcopy(unet); n = [0]


def filt(mod, fqn):
    ok = isinstance(mod, torch.nn.Linear) and mod.in_features >= a.min_in and mod.out_features >= a.min_in and mod.in_features % 16 == 0 and mod.out_features % 16 == 0
    n[0] += ok; return ok


quantize_(u8, float8_dynamic_activation_float8_weight(), filter_fn=filt)
print(f"   fp8 linears: {n[0]}", flush=True)
y8 = bench(u8, "fp8-linear")
d = (y8 - ref).abs(); print(f"== diff vs fp16: mean {d.mean().item():.4f} max {d.max().item():.3f} (ref std {ref.std().item():.3f})", flush=True)
