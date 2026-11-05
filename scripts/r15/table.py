#!/usr/bin/env python3
"""r15 W: the results table (markdown) from /workspace/dub/r15w/eval: SyncNet (<arm>_<id>.sync.txt), flicker (flicker_<id>.json: mean, pops),
tremble (tremble_<id>.json: mouth / chin landmark jitter vs q2, scripts/r15/tremble.py), mouth opening vs q2 (mouth_<id>.json), face GPU-busy
s / video-min (<arm>_<id>/timing.json; q2 = its E2E gpu_time.json face_gen + face_comp, else the r15 arm "q2r") and the E2E estimate = face +
the baseline's audio stage (gpu_time.json "audio"; old03's sampler did not run -> ~52, the m03 / m18 mean).
    python scripts/r15/table.py [--arms q2 s4c8 ...] [--ids old03 m03 m18]"""
import argparse, json, pathlib, re
X = pathlib.Path("/workspace/dub/r15w/eval")
ap = argparse.ArgumentParser(); ap.add_argument("--arms", nargs="+", default="q2 s4c8 tkf2".split()); ap.add_argument("--ids", nargs="+", default=["old03", "m03", "m18"])
a = ap.parse_args()


def js(p):
    try: return json.loads(p.read_text())
    except Exception: return None


def gpu(arm, i):
    t = js(X / f"{arm}_{i}" / "timing.json")
    if t: return t["total"].get("gpu_busy_s_per_vmin")
    if arm == "q2":
        g = js(X / f"q2_{i}" / "e2e" / "gpu_time.json")
        if g: return round(sum(g["stages"].get(s, {}).get("gpu_busy_s_per_vmin", 0) for s in ("face_gen", "face_comp")), 1)
        return gpu("q2r", i) if (X / f"q2r_{i}").exists() else None


def audio(i):
    g = js(X / f"q2_{i}" / "e2e" / "gpu_time.json")
    return (g["stages"].get("audio", {}).get("gpu_busy_s_per_vmin"), "") if g else (52.0, "~")


print("| video | arm | SyncNet | flicker mean | pops (src p99) | tremble mouth / chin vs q2 | mouth opening vs q2 (mean / speed) | near-closed voiced | face GPU s / vmin | E2E est. |")
print("|---|---|---|---|---|---|---|---|---|---|")
for i in a.ids:
    fl = (js(X / f"flicker_{i}.json") or {}).get("videos", {}); mo = js(X / f"mouth_{i}.json") or {}; tr = js(X / f"tremble_{i}.json") or {}
    au, mark = audio(i)
    for arm in a.arms:
        s = (X / f"{arm}_{i}.sync.txt"); m_ = re.search(r":\s*([-0-9.]+) over", s.read_text()) if s.exists() else None
        f = fl.get(arm, {}); o = mo.get(arm, {}) if isinstance(mo, dict) else {}; t = tr.get(arm) or {}; g = gpu(arm, i)
        if not (m_ or f or o or g): continue
        vs = next((v for k, v in o.items() if k.startswith("vs_")), None) or ({"mean": 1.0, "speed": 1.0} if arm.startswith("q2") else {}); cs = o.get("closed_share"); fm = f.get("mean")
        tq = [k for k in (t.get("mouth") or {}) if k.startswith("vs_")]
        trs = f"{t['mouth'][tq[0]]:.2f} / {t['chin'][tq[0]]:.2f}" if tq else "–"
        e2e = f"{mark}{g + au:.0f}" if g is not None and au is not None else "–"
        print(f"| {i} | {arm} | {m_.group(1) if m_ else '–'} | {'%.4f' % fm if fm is not None else '–'} | {f.get('pops', '–')} | {trs} | "
              f"{vs.get('mean', '–')} / {vs.get('speed', '–')} | {'%.3f' % cs if cs is not None else '–'} | {g or '–'} | {e2e} |")
