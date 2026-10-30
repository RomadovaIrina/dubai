#!/usr/bin/env python3
"""r9 speed L4 (overlap): the source-only rigid head motion that fps_composite.py computes (and caches as <source_lm stem>_rigid.npz) on its
first run, computed ahead of time from the same inputs (decoded source frames + the --source-lm landmarks with the --max-hold drop), so that
it runs on the CPU while the GPU is busy with LatentSync. Same code path: fps_composite.decode_source + anchor_filters.rigid_motion.
    python scripts/r9/r9_source_rigid.py --src SRC --source-lm X_source_lm_F.npz [--max-hold 2]"""
from __future__ import annotations
import argparse, pathlib, sys, time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "quality")); sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "optim"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--source-lm", required=True); ap.add_argument("--max-hold", type=int, default=2)
    a = ap.parse_args(); t0 = time.perf_counter()
    import fps_composite as fc, anchor_filters as af
    S, _, _, _ = fc.decode_source(a.src); sl = pathlib.Path(a.source_lm)
    lm_s = np.load(sl)["lm"][:len(S)].astype(np.float64)
    if len(lm_s) != len(S): raise SystemExit(f"--source-lm has {len(lm_s)} frames, source {len(S)}")
    rawp = sl.with_name(sl.stem + "_mp.npz")
    if rawp.exists():   # = fps_composite.py --max-hold
        raw_ok = np.isfinite(np.load(rawp)["lm"][:len(S), 0, 0]); held = np.isfinite(lm_s[:, 0, 0]) & ~raw_ok
        for s_, e_ in af.runs(held):
            if e_ - s_ > a.max_hold: lm_s[s_:e_] = np.nan
    rp = sl.with_name(sl.stem + "_rigid.npz"); tmp = rp.with_name(".tmp_" + rp.name)
    np.savez_compressed(tmp, H=af.rigid_motion(S, lm_s)); tmp.rename(rp)
    print(f"== source_rigid {rp.name}: {len(S)} frames, {time.perf_counter() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
