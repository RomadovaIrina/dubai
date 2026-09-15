#!/usr/bin/env python3
"""Fetch one Qwen2.5-7B GGUF quantization for pilot 0.9.

Downloads ONE quant at a time so the 50 GB dev disk is not filled accidentally.
"""
from __future__ import annotations
import argparse, pathlib, sys

PAT={"q5_k_m":"qwen2.5-7b-instruct-q5_k_m*.gguf",
     "q8_0":"qwen2.5-7b-instruct-q8_0*.gguf"}
EST={"q5_k_m":"~5–6 GB","q8_0":"~8 GB"}

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument("quant",choices=PAT)
    ap.add_argument("--root",default="/workspace/dub/models/qwen2.5-7b-instruct-gguf")
    ap.add_argument("--yes",action="store_true")
    a=ap.parse_args()
    print(f"Qwen2.5-7B {a.quant}: expected download {EST[a.quant]}")
    print(f"destination: {a.root}")
    if not a.yes:
        print("Refusing automatic multi-GB download without --yes.")
        return 2
    from huggingface_hub import snapshot_download
    p=snapshot_download(repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",local_dir=a.root,
                        allow_patterns=[PAT[a.quant]])
    files=sorted(pathlib.Path(a.root).glob(PAT[a.quant]))
    print("downloaded:"); [print(" ",x, f"{x.stat().st_size/2**30:.2f} GiB") for x in files]
    return 0 if files else 1
if __name__=="__main__": raise SystemExit(main())
