#!/usr/bin/env python3
"""r8 batch check: speech tokens + wavs of generate_batch vs one-by-one generate (upstream T3, and t3_fast), same seeds; timing.
    python scripts/r8/test_t3_batch.py --ref <speaker wav> [--n 6]"""
import argparse, pathlib, sys, time
HERE = pathlib.Path(__file__).resolve().parent
for p in (HERE.parent / "pilot", HERE.parent / "quality", HERE.parent / "r7"): sys.path.insert(0, str(p))
TEXTS = ["Let's make a video with the glasses, let's see, we'll test it again,", "City of Sochi on August twenty-eight.",
         "Today we are starting the work processes to create our magical app which is going to make life a lot easier for people.",
         "So here we have a really wonderful video, and the weather is just fantastic,", "We are testing all possible variants",
         "that we can come up with.", "I'm just walking around, filming", "and saying something.", "After all, these modern phones already give us the ability to do this.",
         "Well, I just want to see how our app performs in this format.", "Here we have a Finnish sauna,", "now I'll show you a Russian banya."]


def common(a, b):
    k = 0
    while k < min(len(a), len(b)) and a[k] == b[k]: k += 1
    return k


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ref", required=True); ap.add_argument("--n", type=int, default=6); a = ap.parse_args()
    import numpy as np, torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from chatterbox_fp16 import install_speech_token_guard
    import t3_batch
    m = ChatterboxMultilingualTTS.from_local(HERE.parents[1] / "models" / "chatterbox", "cuda", t3_model="v3"); install_speech_token_guard(m); m.prepare_conditionals(a.ref)
    items = [dict(text=TEXTS[k % len(TEXTS)], seed=7 + 100 * k, temperature=0.8 if k % 3 else 0.6, cfg_weight=0.5 if k % 2 == 0 else 0.3) for k in range(a.n)]
    rec = {}; orig = m.t3.inference
    def wrap(tag):
        def f(*x, **k):
            r = orig_cur[0](*x, **k); rec.setdefault(tag, []).append(r[0].tolist()); return r
        return f
    def seq(tag):
        m.t3.inference = wrap(tag); wavs = []; torch.cuda.synchronize(); t0 = time.perf_counter()
        for it in items:
            torch.manual_seed(it["seed"])
            with torch.autocast("cuda", dtype=torch.float16): wavs.append(m.generate(it["text"], language_id="en", temperature=it["temperature"], cfg_weight=it["cfg_weight"]))
        torch.cuda.synchronize(); return wavs, time.perf_counter() - t0
    orig_cur = [orig]
    with torch.autocast("cuda", dtype=torch.float16): m.generate(TEXTS[0], language_id="en")   # warm-up
    w_up, t_up = seq("upstream")
    from t3_fast import install_fast_t3
    install_fast_t3(m); orig_cur[0] = m.t3.inference
    with torch.autocast("cuda", dtype=torch.float16): m.generate(TEXTS[0], language_id="en")
    w_fa, t_fa = seq("fast")
    m.t3.inference = orig_cur[0]
    with torch.autocast("cuda", dtype=torch.float16): t3_batch.generate_batch(m, items[:2])   # warm-up (graphs)
    torch.cuda.synchronize(); t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16): w_ba = t3_batch.generate_batch(m, items)
    torch.cuda.synchronize(); t_ba = time.perf_counter() - t0; tb = t3_batch.LAST_TOKENS
    print(f"N={a.n}  time upstream {t_up:.2f}s  fast {t_fa:.2f}s  batch {t_ba:.2f}s")
    for k, it in enumerate(items):
        up = [t for t in rec["upstream"][k]]; fa = rec["fast"][k]; ba = tb[k]
        same_wav = w_up[k].shape == w_ba[k].shape and bool(np.allclose(w_up[k].numpy(), w_ba[k].numpy(), atol=1e-4)) if w_ba[k] is not None else False
        print(f"  {k} cfg {it['cfg_weight']} T {it['temperature']} len up {len(up)} fast {len(fa)} batch {len(ba)} | common prefix up-fast {common(up, fa)} up-batch {common(up, ba)} fast-batch {common(fa, ba)} | wav up==batch {same_wav}")


if __name__ == "__main__":
    main()
