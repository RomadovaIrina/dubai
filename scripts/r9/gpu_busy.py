#!/usr/bin/env python3
"""r9: wall vs GPU-busy seconds per stage of one timed audio (+ sep) run.
    python scripts/r9/gpu_busy.py --nv nv.csv --tl timeline.jsonl --dur <video s> [--json out.json]
nv.csv = nvidia-smi --query-gpu=timestamp,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 100 (GPU utilization = the share of the
last sample period with a kernel running); timeline = scripts/r9/timeline.py intervals ([category, t0, t1], wall clock). Every 100-ms sample is
given to the INNERMOST open foreground interval (the latest-starting one); GPU-busy s = sum of util/100 x dt. Background categories (bg:...: the
MediaPipe track, the htdemucs stems / remix of the sep overlap) run alongside the foreground: reported apart, their samples are not exclusive."""
import argparse, collections, datetime, json


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nv", required=True); ap.add_argument("--tl", required=True); ap.add_argument("--dur", type=float, required=True)
    ap.add_argument("--json"); a = ap.parse_args()
    S = []
    for ln in open(a.nv):
        p = [x.strip() for x in ln.split(",")]
        try: S.append((datetime.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp(), float(p[1])))
        except Exception: continue
    T = [json.loads(l) for l in open(a.tl) if l.strip()]
    run = [t for t in T if t[0] == "run"]; t0, t1 = (run[0][1], run[0][2]) if run else (min(x[1] for x in T), max(x[2] for x in T))
    fg = [t for t in T if not t[0].startswith("bg:") and t[0] != "run"]; bg = [t for t in T if t[0].startswith("bg:")]
    wall, busy, cnt = collections.Counter(), collections.Counter(), collections.Counter()
    bw, bb = collections.Counter(), collections.Counter()
    tot_busy = 0.0
    for k, (ts, u) in enumerate(S):
        if not (t0 <= ts <= t1): continue
        dt = min(0.2, (S[k + 1][0] - ts) if k + 1 < len(S) else 0.1)
        open_ = [t for t in fg if t[1] <= ts < t[2]]
        cat = max(open_, key=lambda t: t[1])[0] if open_ else "other (process start / imports / bash / waits)"
        wall[cat] += dt; busy[cat] += u / 100 * dt; tot_busy += u / 100 * dt
        for t in bg:
            if t[1] <= ts < t[2]: bw[t[0]] += dt; bb[t[0]] += u / 100 * dt
    vm = a.dur / 60.0; rows = []
    print(f"run {t1 - t0:.1f} s wall = {(t1 - t0) / vm:.1f} s/vmin; GPU busy {tot_busy:.1f} s = {tot_busy / vm:.1f} s/vmin ({100 * tot_busy / max(1e-9, t1 - t0):.0f} % of the wall)")
    print(f"| stage | wall s | GPU-busy s | busy % | wall s/vmin | GPU-busy s/vmin |\n|---|---:|---:|---:|---:|---:|")
    for cat in sorted(wall, key=lambda c: -wall[c]):
        rows.append({"stage": cat, "wall_s": round(wall[cat], 1), "busy_s": round(busy[cat], 1)})
        print(f"| {cat} | {wall[cat]:.1f} | {busy[cat]:.1f} | {100 * busy[cat] / max(1e-9, wall[cat]):.0f} | {wall[cat] / vm:.1f} | {busy[cat] / vm:.1f} |")
    for cat in sorted(bw):
        rows.append({"stage": cat, "wall_s": round(bw[cat], 1), "busy_s_shared": round(bb[cat], 1), "background": True})
        print(f"| {cat} (background, overlapped) | {bw[cat]:.1f} | ({bb[cat]:.1f} shared) | | {bw[cat] / vm:.1f} | |")
    if a.json: json.dump({"wall_s": round(t1 - t0, 1), "busy_s": round(tot_busy, 1), "dur_s": a.dur, "rows": rows}, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()
