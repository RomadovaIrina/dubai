#!/usr/bin/env python
"""0.3 — FlashAttention readiness check.

Answers four questions, by measurement rather than by assumption:

  1. what GPU/torch/CUDA is this, and is the `flash-attn` package installed at all;
  2. which SDPA backends torch will actually accept for this stack's attention shapes
     (torch ships its own fused FlashAttention kernels — the `flash-attn` package is a
     *separate* implementation and is not required to get flash attention);
  3. does each backend produce correct numbers here, and how fast is it;
  4. a machine-readable dump so the identical run on the RTX 5090 (sm_120) can be diffed
     against this one instead of argued about.

    source scripts/env.sh
    python scripts/pilot/check_flashattention.py
    python scripts/pilot/check_flashattention.py --json reports/pilot/flashattn_4090.json

This script installs nothing and changes nothing. `constraints.txt` forbids xformers;
`flash-attn` is not pinned either way, so whether to add it is a decision for a human —
the report ends with what that would and would not buy.
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT, gpu_info, write_json

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

MIB = 1024 * 1024

# (name, batch, heads, q_len, kv_len, head_dim, causal) — shapes this pilot actually runs
SHAPES = [
    ("latentsync-unet self-attn 512², 1 frame", 1, 8, 4096, 4096, 64, False),
    ("latentsync-unet self-attn, 16-frame batch", 16, 8, 1024, 1024, 64, False),
    ("latentsync audio cross-attn (kv=50)", 16, 8, 4096, 50, 64, False),
    ("chatterbox t3 decoder (causal)", 1, 16, 1024, 1024, 64, True),
]
BACKENDS = [
    ("FLASH_ATTENTION", SDPBackend.FLASH_ATTENTION),
    ("EFFICIENT_ATTENTION", SDPBackend.EFFICIENT_ATTENTION),
    ("CUDNN_ATTENTION", SDPBackend.CUDNN_ATTENTION),
    ("MATH", SDPBackend.MATH),
]


def pkg(name: str):
    try:
        return md.version(name)
    except Exception:
        return None


def bench(backend, q, k, v, causal, iters=20, warmup=5):
    """Return (ms_per_call, peak_torch_mib) or raise if the backend refuses the shape."""
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    with sdpa_kernel(backend):
        for _ in range(warmup):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / iters
    return dt * 1000, torch.cuda.max_memory_allocated() / MIB, out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=str(PILOT / "flashattn.json"))
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args()
    dtype = getattr(torch, a.dtype)

    if not torch.cuda.is_available():
        print("FAIL: no CUDA device")
        return 1

    g = gpu_info()
    rep: dict = {
        "gpu": g,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "dtype": a.dtype,
        "packages": {n: pkg(n) for n in
                     ("flash-attn", "xformers", "triton", "torch", "transformers", "diffusers")},
        "sdpa_flags": {
            "flash": torch.backends.cuda.flash_sdp_enabled(),
            "mem_efficient": torch.backends.cuda.mem_efficient_sdp_enabled(),
            "math": torch.backends.cuda.math_sdp_enabled(),
            "cudnn": torch.backends.cuda.cudnn_sdp_enabled(),
        },
        "shapes": [],
    }

    print("=" * 78)
    print("0.3  FlashAttention check")
    print("=" * 78)
    print(f"GPU            {g.get('name')}  {g.get('capability')}  "
          f"{g.get('total_vram_gib')} GiB  ({g.get('multi_processor_count')} SM)")
    print(f"torch          {torch.__version__}  (cuda {torch.version.cuda}, "
          f"cudnn {torch.backends.cudnn.version()})")
    print(f"arch list      {', '.join(g.get('arch_list') or [])}")
    print()
    print("packages")
    for n, v in rep["packages"].items():
        print(f"  {n:<14} {v if v else 'NOT INSTALLED'}")
    print()
    print("torch SDPA backends enabled:",
          ", ".join(k for k, v in rep["sdpa_flags"].items() if v) or "none")

    # does torch itself think flash is usable for a representative shape?
    try:
        from torch.backends.cuda import SDPAParams, can_use_flash_attention
        q = torch.randn(1, 8, 1024, 64, device="cuda", dtype=dtype)
        params = SDPAParams(q, q, q, None, 0.0, False, False)
        rep["can_use_flash_attention"] = bool(can_use_flash_attention(params, False))
    except Exception as e:
        rep["can_use_flash_attention"] = f"probe failed: {type(e).__name__}: {e}"
    print("can_use_flash_attention (1×8×1024×64):", rep["can_use_flash_attention"])
    print()

    ok_all = True
    for name, B, H, Lq, Lkv, D, causal in SHAPES:
        q = torch.randn(B, H, Lq, D, device="cuda", dtype=dtype)
        k = torch.randn(B, H, Lkv, D, device="cuda", dtype=dtype)
        v = torch.randn(B, H, Lkv, D, device="cuda", dtype=dtype)
        print(f"--- {name}")
        print(f"    q={tuple(q.shape)} k={tuple(k.shape)} causal={causal} dtype={a.dtype}")

        # fp32 math reference for correctness
        with sdpa_kernel(SDPBackend.MATH):
            ref = F.scaled_dot_product_attention(q.float(), k.float(), v.float(),
                                                 is_causal=causal)
        entry = {"name": name, "B": B, "H": H, "q_len": Lq, "kv_len": Lkv,
                 "head_dim": D, "causal": causal, "backends": {}}
        for bname, backend in BACKENDS:
            try:
                ms, peak, out = bench(backend, q, k, v, causal, iters=a.iters)
                err = (out.float() - ref).abs().max().item()
                entry["backends"][bname] = {"ok": True, "ms": round(ms, 3),
                                            "peak_torch_mib": round(peak, 1),
                                            "max_abs_err_vs_fp32_math": err}
                print(f"      {bname:<20} {ms:8.3f} ms   peak {peak:8.1f} MiB   "
                      f"max|Δ| vs fp32 {err:.2e}")
            except Exception as e:
                msg = f"{type(e).__name__}: {str(e).splitlines()[0][:110]}"
                entry["backends"][bname] = {"ok": False, "error": msg}
                print(f"      {bname:<20} unavailable — {msg}")
                if bname == "FLASH_ATTENTION":
                    ok_all = False
            finally:
                torch.cuda.empty_cache()
        # which backend does torch pick when left alone?
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(a.iters):
            F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        torch.cuda.synchronize()
        entry["default_ms"] = round((time.perf_counter() - t0) / a.iters * 1000, 3)
        print(f"      {'(torch default)':<20} {entry['default_ms']:8.3f} ms")
        rep["shapes"].append(entry)
        del q, k, v, ref
        torch.cuda.empty_cache()
        print()

    fastest = {}
    for e in rep["shapes"]:
        cands = {b: d["ms"] for b, d in e["backends"].items() if d.get("ok")}
        if cands:
            fastest[e["name"]] = min(cands, key=cands.get)
    rep["fastest_backend_per_shape"] = fastest

    print("=" * 78)
    print("verdict")
    print("=" * 78)
    print(f"  flash-attn package        : {'installed ' + rep['packages']['flash-attn'] if rep['packages']['flash-attn'] else 'NOT installed'}")
    print(f"  torch fused flash kernels : {'usable' if ok_all else 'NOT usable for every shape'}")
    print("  fastest backend per shape :")
    for n, b in fastest.items():
        print(f"      {b:<20} {n}")
    print()
    print("  for the 5090 (sm_120): run this exact script there and diff the JSON.")
    print("  Compare in this order — (1) can_use_flash_attention, (2) whether")
    print("  FLASH_ATTENTION reports 'unavailable' for any shape, (3) ms per shape,")
    print("  (4) max|Δ| vs fp32. Those four cover 'does it run' and 'is it worth it'.")
    print()

    write_json(pathlib.Path(a.json), rep)
    print(f"-> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
