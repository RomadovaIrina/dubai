#!/usr/bin/env python3
"""r15 Z final table: every arm with a final in eval/ (alone and + agent F's colour lock <arm>lk): SyncNet, F lip_flicker excess blinks / 10 s,
the QA gate's flashes (r3_flash: frames / strong = mc_ratio >= 2 and mc_vs_src >= 2, the m03 "snaps"), face GPU-busy s / vmin.
    python scripts/r15/zreport_table.py > ZTABLE_FINAL.md"""
import json, pathlib, re
X = pathlib.Path("/workspace/dub/r15w/eval")
ORDER = ["q2d", "q2v2", "q2as", "q2c8", "te2", "te2k", "q2s4", "q2c8rf", "s4c8", "q2ct3", "q2ct5", "q2ct7", "q2g15", "q2g125", "q2rs7", "q2rs10",
         "q2ts25", "q2ts33", "q2cfa", "q2nb", "q2nr", "q2ss0", "q2all"]
NOTE = {"q2d": "production q2 + F3 (+ LS_DUMP_DIR)", "q2v2": "prod + audit default fixes", "q2as": "LS_AFFINE_SMOOTH=9", "q2c8": "stride 8 continue",
        "te2": "c8 + 2-noise latent ensemble", "te2k": "te2 + keyframes/2", "q2s4": "stride 4 continue", "q2c8rf": "c8 + fixed reference",
        "s4c8": "student 4-step c8", "q2ct3": "CFG trunc 3/10", "q2ct5": "CFG trunc 5/10", "q2ct7": "CFG trunc 7/10", "q2g15": "guidance 1.5 (prod 2.0)",
        "q2g125": "guidance 1.25", "q2rs7": "CFG rescale 0.7", "q2rs10": "CFG rescale 1.0", "q2ts25": "LS_CFG_TSMOOTH=0.25", "q2ts33": "LS_CFG_TSMOOTH=0.333",
        "q2cfa": "CF_ALIGN_FIT=lsq + CF_MOUTH=rigid", "q2nb": "AF_UPPER=nobrow", "q2nr": "LS_NOSE_RIGID=1", "q2ss0": "--steps-silent 0", "q2all": "cfa + nb + nr + vm"}


def last(fn):
    d = {}
    for l in (X / fn).read_text().splitlines() if (X / fn).exists() else []:
        try: r = json.loads(l); d[r["label"]] = r
        except Exception: pass
    return d


LF, FL = last("lipflick.jsonl"), last("flash.jsonl")
def sync(n):
    p = X / f"{n}.sync.txt"; m = re.search(r":\s*([-0-9.]+) over", p.read_text()) if p.exists() else None
    return m.group(1) if m else "–"
def gpu(a, i):
    try: return round(json.loads((X / f"{a}_{i}" / "timing.json").read_text())["total"]["gpu_busy_s_per_vmin"])
    except Exception: return "–"
def fl(n):
    r = FL.get(n); return f"{r['flash_frames']} / {r['strong']}" if r and r.get("flash_frames") is not None else "–"
def bl(n):
    r = LF.get(n); return r.get("excess_per_10s", "–") if r else "–"


print("| video | arm | what | SyncNet | + lock SyncNet | blinks / 10 s | + lock blinks | flashes / strong | + lock flashes / strong | face GPU s/vmin |")
print("|---|---|---|---|---|---|---|---|---|---|")
for i in ("m03", "m18", "old03"):
    for a in ORDER:
        if not (X / f"{a}_{i}" / f"{a}_{i}.mp4").exists(): continue
        print(f"| {i} | {a} | {NOTE.get(a, '')} | {sync(f'{a}_{i}')} | {sync(f'{a}lk_{i}')} | {bl(f'{a}_{i}')} | {bl(f'{a}lk_{i}')} | {fl(f'{a}_{i}')} | {fl(f'{a}lk_{i}')} | {gpu(a, i)} |")
