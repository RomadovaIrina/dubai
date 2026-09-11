#!/usr/bin/env python3

import sys
import torch
import traceback


def main():
    print("=" * 80)
    print("FlashAttention environment check")
    print("=" * 80)

    print(f"Python: {sys.version}")
    print(f"Torch: {torch.__version__}")
    print(f"Torch CUDA: {torch.version.cuda}")

    if not torch.cuda.is_available():
        print("CUDA unavailable")
        return

    device = torch.device("cuda")

    props = torch.cuda.get_device_properties(0)

    print("\nGPU:")
    print(props.name)

    print(f"Compute capability: {props.major}.{props.minor}")
    print(f"VRAM GB: {props.total_memory / 1024**3:.2f}")

    print("\nCUDA arch list:")
    print(torch.cuda.get_arch_list())


    print("\nChecking flash_attn import")

    try:
        import flash_attn

        print("flash_attn import: OK")
        print(
            "version:",
            getattr(flash_attn, "__version__", "unknown")
        )

    except Exception as e:
        print("flash_attn import: FAIL")
        print(e)

        return


    print("\nRunning flash attention kernel test")

    try:
        from flash_attn import flash_attn_func

        batch = 1
        seqlen = 128
        heads = 8
        dim = 64

        q = torch.randn(
            batch,
            seqlen,
            heads,
            dim,
            device=device,
            dtype=torch.float16,
        )

        k = torch.randn_like(q)
        v = torch.randn_like(q)

        torch.cuda.synchronize()

        out = flash_attn_func(
            q,
            k,
            v
        )

        torch.cuda.synchronize()

        print("Kernel execution: PASS")
        print("Output shape:", out.shape)

    except Exception:

        print("Kernel execution: FAIL")

        traceback.print_exc()


if __name__ == "__main__":
    main()