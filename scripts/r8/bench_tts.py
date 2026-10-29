#!/usr/bin/env python3
"""r8 batching groundwork: where a Chatterbox take spends its time (T3 token decode vs S3Gen vocoder vs the rest), plain and with t3_fast.
    python scripts/r8/bench_tts.py --ref <speaker wav>"""
import argparse, pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
for p in (HERE.parent / "pilot", HERE.parent / "quality", HERE.parent / "r7"): sys.path.insert(0, str(p))
TEXTS = ["Let's make a video with the glasses, let's see, we'll test it again,", "City of Sochi on August twenty-eight.",
         "Today we are starting the work processes to create our magical app which is going to make life a lot easier for people.",
         "So here we have a really wonderful video, and the weather is just fantastic,"]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ref", required=True); ap.add_argument("--fast", action="store_true"); a = ap.parse_args()
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from chatterbox_fp16 import install_speech_token_guard
    m = ChatterboxMultilingualTTS.from_local(HERE.parents[1] / "models" / "chatterbox", "cuda", t3_model="v3"); install_speech_token_guard(m)
    if a.fast:
        from t3_fast import install_fast_t3
        install_fast_t3(m)
    m.prepare_conditionals(a.ref)
    T = {"t3": 0.0, "s3gen": 0.0}; o3, os_ = m.t3.inference, m.s3gen.inference
    def t3i(*x, **k):
        torch.cuda.synchronize(); t = time.perf_counter(); r = o3(*x, **k); torch.cuda.synchronize(); T["t3"] += time.perf_counter() - t; return r
    def s3i(*x, **k):
        torch.cuda.synchronize(); t = time.perf_counter(); r = os_(*x, **k); torch.cuda.synchronize(); T["s3gen"] += time.perf_counter() - t; return r
    m.t3.inference = t3i; m.s3gen.inference = s3i
    for txt in TEXTS[:1]:   # warm-up
        with torch.autocast("cuda", dtype=torch.float16): m.generate(txt, language_id="en", temperature=0.8, cfg_weight=0.5)
    T.update(t3=0.0, s3gen=0.0); t0 = time.perf_counter(); dur = 0.0
    for k, txt in enumerate(TEXTS):
        torch.manual_seed(7 + k)
        with torch.autocast("cuda", dtype=torch.float16): w = m.generate(txt, language_id="en", temperature=0.8, cfg_weight=0.5)
        dur += w.shape[-1] / m.sr
    tot = time.perf_counter() - t0
    print(f"fast={a.fast} takes={len(TEXTS)} audio={dur:.1f}s wall={tot:.2f}s t3={T['t3']:.2f}s s3gen={T['s3gen']:.2f}s other={tot - T['t3'] - T['s3gen']:.2f}s "
          f"(x realtime {dur / tot:.2f})")


if __name__ == "__main__":
    main()
