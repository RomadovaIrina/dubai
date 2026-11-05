#!/usr/bin/env python3
"""r15 W: the results table (markdown) from /workspace/dub/r15w/eval: SyncNet (<arm>_<id>.sync.txt), flicker (flicker_<id>.json: mean, pops,
mean at the eye spots), mouth opening vs q2 (mouth_<id>.json), face GPU-busy s / video-min (<arm>_<id>/timing.json; q2 = its E2E gpu_time.json
face_gen + face_comp, else the r15 arm "q2r").
    python scripts/r15/table.py [--arms q2 s4c8 tb tbw bon3 kf2 kf3 tkf2] [--ids old03 m03 m18]"""
import argparse, json, pathlib, re
X = pathlib.Path("/workspace/dub/r15w/eval")
ap = argparse.ArgumentParser(); ap.add_argument("--arms", nargs="+", default="q2 s4c8 tb tbw bon3 kf2 kf3 tkf2".split()); ap.add_argument("--ids", nargs="+", default=["old03", "m03", "m18"])
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


print("| video | arm | SyncNet | flicker mean | pops (src p99) | flicker @ eye spots | mouth opening vs q2 (mean / speed) | near-closed voiced | face GPU s / vmin |")
print("|---|---|---|---|---|---|---|---|---|")
for i in a.ids:
    fl = (js(X / f"flicker_{i}.json") or {}).get("videos", {}); mo = js(X / f"mouth_{i}.json") or {}
    if isinstance(mo, dict) and "videos" in mo: mo = mo["videos"]
    for arm in a.arms:
        s = (X / f"{arm}_{i}.sync.txt"); m_ = re.search(r":\s*([-0-9.]+) over", s.read_text()) if s.exists() else None
        f = fl.get(arm, {}); o = mo.get(arm, {}) if isinstance(mo, dict) else {}
        if not (m_ or f or o or gpu(arm, i)): continue
        at = f.get("at") or {}; ats = sum(at.values()) / len(at) if at else None
        vs = o.get("vs_q2") or {}; cs = o.get("closed_share"); fm = f.get("mean")
        print(f"| {i} | {arm} | {m_.group(1) if m_ else '–'} | {'%.4f' % fm if fm is not None else '–'} | {f.get('pops', '–')} | {'%.4f' % ats if ats is not None else '–'} | "
              f"{vs.get('mean', 1.0 if arm == 'q2' else '–')} / {vs.get('speed', 1.0 if arm == 'q2' else '–')} | {'%.3f' % cs if cs is not None else '–'} | {gpu(arm, i) or '–'} |")
