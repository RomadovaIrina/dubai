#!/usr/bin/env python3
"""r14: the r2 vs r3 student tables (markdown) from what the night chain left on disk: the training val (ckpt/<run>/step_*.json +
ckpt_val.json) and the E2E evals (eval/<tag>_<id>: sync_speech, flicker_metric, mouth_open, dabai_job GPU seconds).
    python scripts/r14/night_table.py [--root /workspace/dub/r14d] > table.md"""
import argparse, glob, json, pathlib, re

TAGS = ["q2", "s4", "s4p1500", "s4p1500c8", "s4p1500c8rf", "r3", "r3c8", "r3c8rf", "r3fin", "r3f8", "r3b"]; IDS = ["old03", "m03", "m18"]


def jl(p):
    try: return json.load(open(p))
    except Exception: return None


def f(v, n=3):
    return "–" if v is None else (f"{v:.{n}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="/workspace/dub/r14d"); a = ap.parse_args(); R = pathlib.Path(a.root); E = R / "eval"
    print("### Training val (EMA, 24 held-out windows; teacher = 20 steps + CFG)\n")
    print("| run | step | latent MSE | mouth pix L1 | jitter | stable-SyncNet sim | opening / teacher |\n|---|---|---|---|---|---|---|")
    for run in ("r2_s4", "r3_s4", "r3b_s4"):
        cv = {r["ckpt"]: r for r in (jl(R / "ckpt" / run / "ckpt_val.json") or {}).get("by_ckpt", [])}
        for p in sorted(glob.glob(str(R / "ckpt" / run / "step_*[0-9].json"))):
            r = jl(p) or {}; s = r.get("step"); c = cv.get(f"step_{s:05d}_ema.pt", {}) if s else {}; pix = c.get("pix"); op = r.get("val_open_ratio_ema") or c.get("open_ratio")
            print(f"| {run} | {s} | {f(r.get('val_mse_ema'), 4)} | {f(pix, 4)} | {f(r.get('val_jitter_ema'), 4)} | {f(r.get('val_sync_ema'))} | {f(op)} |")
    print("\n### End-to-end on the held-out videos (dabai_job, q2 + F3 composite in every arm)\n")
    print("| video | arm | sync_speech | flicker mean | flicker pops (src p99) | mouth opening vs q2 | near-closed voiced | GPU s / video-min |\n|---|---|---|---|---|---|---|---|")
    for id_ in IDS:
        for t in TAGS:
            n = f"{t}_{id_}"; st = (E / f"{n}.sync.txt"); sy = re.search(r": ([-0-9.]+) over", st.read_text()) if st.exists() else None
            lg = E / f"{n}.log"; gs = None
            if lg.exists():
                m = re.search(r'"video_minutes": ([0-9.]+).*?"gpu_busy_s": ([0-9.]+)', lg.read_text(), re.S)
                if m: gs = float(m.group(2)) / float(m.group(1))
            if not sy and gs is None: continue
            fl = jl(E / f"{('s4' if t == 'q2' else t)}_{id_}.flicker.json") or {}; fv = fl.get("videos", {}).get(t, {})
            mo = jl(E / f"{('s4' if t == 'q2' else t)}_{id_}.mouth.json") or {}; mv = mo.get(t, {})
            print(f"| {id_} | {t} | {sy.group(1) if sy else '–'} | {f(fv.get('mean'), 4)} | {f(fv.get('pops'))} | {f(mv.get('vs_q2', {}).get('mean')) if t != 'q2' else '1'} | "
                  f"{f(mv.get('closed_share'), 2)} | {f(gs, 0)} |")


if __name__ == "__main__":
    main()
