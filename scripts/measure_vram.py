#!/usr/bin/env python3

import gc
import torch
import subprocess
import time


def nvidia_smi_memory():

    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ]
    )

    return int(result.decode().strip())


def cuda_memory():

    return {
        "allocated_MB":
            torch.cuda.memory_allocated()
            /1024**2,

        "reserved_MB":
            torch.cuda.memory_reserved()
            /1024**2,

        "peak_MB":
            torch.cuda.max_memory_allocated()
            /1024**2,
    }



def measure(
    name,
    load_fn,
    infer_fn=None
):

    print("\n")
    print("="*60)
    print(name)

    torch.cuda.empty_cache()
    gc.collect()

    torch.cuda.reset_peak_memory_stats()


    print("Before:")
    print(cuda_memory())


    start=time.time()

    model = load_fn()

    torch.cuda.synchronize()

    load_time=time.time()-start


    print("\nAfter load:")
    print(cuda_memory())


    if infer_fn:

        torch.cuda.reset_peak_memory_stats()

        start=time.time()

        infer_fn(model)

        torch.cuda.synchronize()

        infer_time=time.time()-start


        print("\nAfter inference:")
        print(cuda_memory())

        print(
            "Inference sec:",
            infer_time
        )


    peak=torch.cuda.max_memory_allocated()/1024**3


    print(
        f"\n{name} peak VRAM: {peak:.2f} GB"
    )


    del model

    gc.collect()
    torch.cuda.empty_cache()

    print("\nAfter unload:")
    print(cuda_memory())



def main():

    print(torch.cuda.get_device_name())

    # сюда потом подключаются реальные модели

    print(
        """
Add model loaders:

measure(
    "LatentSync",
    load_latentsync,
    run_latentsync
)
"""
    )


if __name__=="__main__":
    main()