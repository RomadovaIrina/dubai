#!/usr/bin/env python3
"""Round 2: SyncNet on head-stabilised vs as-is clips, full window and sub-windows (is a SyncNet gap on shaky footage a SyncNet-crop
artifact?).   python scripts/quality/r2_syncstab.py --id 04 --w w1 --t0 25 --arms clean0 flow4nr32 ... [--sub 2.2] [--jobs 3]"""
import argparse, json, os, pathlib, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot")); os.environ.setdefault("DUB_ROOT", str(HERE.parents[1]))
R2 = pathlib.Path("/workspace/dub/artifacts/lipsync-temporal/r2"); O = R2 / "syncstab"


def job(spec):
    from syncnet_common import eval_syncnet
    vid, w, t0, arm, start, dur, stab = spec
    tag = f"{vid}_{w}_{arm}_s{start:g}_d{dur:g}_{'stab' if stab else 'raw'}"; out = O / f"{tag}.json"
    if out.exists(): return json.loads(out.read_text())
    clip = O / f"{tag}.mp4"
    cmd = [sys.executable, str(HERE / "stabilise_clip.py"), "--clip", str(R2 / f"win/{vid}_{w}_{arm}_dubonly.mp4"), "--id", vid, "--t0", str(t0), "--out", str(clip),
           "--start", str(start), "--dur", str(dur)] + ([] if stab else ["--no-stab"])
    subprocess.run(cmd, check=True)
    try: x = eval_syncnet(clip); r = {"conf": x["confidence"], "av": x["av_offset_frames"]}
    except Exception as e: r = {"error": str(e)[-200:]}
    r.update({"arm": arm, "start": start, "dur": dur, "stab": stab}); out.write_text(json.dumps(r)); clip.unlink(missing_ok=True)
    print(tag, r, flush=True); return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--id", required=True); ap.add_argument("--w", required=True); ap.add_argument("--t0", type=float, required=True)
    ap.add_argument("--dur", type=float, required=True); ap.add_argument("--arms", nargs="+", required=True); ap.add_argument("--sub", type=float, default=2.2); ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args(); O.mkdir(parents=True, exist_ok=True)
    specs = []
    for arm in a.arms:
        for stab in (True, False):
            specs.append((a.id, a.w, a.t0, arm, 0.0, a.dur, stab))
            s = 0.0
            while s + a.sub <= a.dur + 1e-6:
                specs.append((a.id, a.w, a.t0, arm, round(s, 2), a.sub, stab)); s += a.sub
    with ThreadPoolExecutor(a.jobs) as ex: res = list(ex.map(job, specs))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
