#!/usr/bin/env python3
"""r8 F: SyncNet AV-offset sweep per window from the gate's per-frame distance matrices (<id>_<arm>.sync_perframe.npz: d[frame, 31 offsets],
25 fps, offset index 15 = 0; upstream sign: offset = 15 - argmin). Per --win s window with >= --min-frames tracked frames and dub speech
(sync_perframe.json per_second dub_act >= 0.5 on >= half its seconds): best offset, conf_best = median - min of the window-mean curve,
conf_at0 = median - d(0). A constant / drifting best offset = a timing problem; offset ~0 with lower conf = weaker articulation.
    python scripts/r8/r8_sync_windows.py --ids 03 05 04 01 --arms m11r8 m15 m16dfb m17 m18 [--win 5]
"""
import argparse, json, pathlib, numpy as np
B = pathlib.Path("/workspace/dub/artifacts/r4m")
ap = argparse.ArgumentParser(); ap.add_argument("--ids", nargs="+"); ap.add_argument("--arms", nargs="+"); ap.add_argument("--win", type=float, default=5.0)
ap.add_argument("--min-frames", type=int, default=40); ap.add_argument("--out", default="/workspace/dub/artifacts/r8f/sync_windows.json")
a = ap.parse_args(); W = int(a.win * 25); R = {}
for id_ in a.ids:
    R[id_] = {}
    for arm in a.arms:
        p = B / arm / f"{id_}_{arm}.sync_perframe.npz"
        if not p.exists(): continue
        d = np.load(p)["d"]; js = json.loads((B / arm / f"{id_}_{arm}.sync_perframe.json").read_text()).get("per_second", [])
        act = {int(x.get("s", x.get("second", i))): (x.get("dub_act") or 0) for i, x in enumerate(js)}
        rows = []
        for w0 in range(0, len(d) - W // 2, W):
            blk = d[w0:w0 + W]; ok = np.isfinite(blk).all(1)
            secs = range(w0 // 25, (w0 + W) // 25); sp = np.mean([act.get(s, 0) >= 0.5 for s in secs]) if secs else 0
            if ok.sum() < a.min_frames or sp < 0.5: continue
            m = blk[ok].mean(0); med = float(np.median(m)); i = int(np.argmin(m))
            rows.append({"t0": round(w0 / 25, 1), "off": 15 - i, "conf_best": round(med - float(m[i]), 2), "conf0": round(med - float(m[15]), 2), "n": int(ok.sum())})
        offs = [r["off"] for r in rows]; c0 = [r["conf0"] for r in rows]
        R[id_][arm] = {"windows": rows, "n": len(rows), "median_off": float(np.median(offs)) if offs else None, "nonzero_off": sum(abs(o) >= 2 for o in offs),
                       "conf0_mean": round(float(np.mean(c0)), 2) if c0 else None, "conf_best_mean": round(float(np.mean([r["conf_best"] for r in rows])), 2) if rows else None,
                       "off_drift_corr_t": round(float(np.corrcoef([r["t0"] for r in rows], offs)[0, 1]), 2) if len(rows) > 3 and np.std(offs) > 0 else 0.0}
        s = R[id_][arm]; print(f"== {id_} {arm:8s} windows {s['n']:2d} | median offset {s['median_off']} | |off|>=2 in {s['nonzero_off']} | conf@0 {s['conf0_mean']} | conf@best {s['conf_best_mean']} | drift r {s['off_drift_corr_t']}")
        print("     offsets: " + " ".join(f"{r['t0']:.0f}:{r['off']:+d}/{r['conf0']:.1f}" for r in rows))
pathlib.Path(a.out).write_text(json.dumps(R, indent=1))
