#!/usr/bin/env python3
"""Round 2: (re)compute missing SyncNet (isolated eval dirs, N parallel) and common-metric v4 results for every arm in r2/win.
    python scripts/quality/r2_fill_evals.py [--jobs 3] [--skip-wrap]"""
import argparse, json, pathlib, subprocess, sys, re
from concurrent.futures import ThreadPoolExecutor
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot"))
import os; os.environ.setdefault("DUB_ROOT", str(HERE.parents[1]))
W = pathlib.Path("/workspace/dub/artifacts/lipsync-temporal/r2/win")
WIN = {"01_w0": (0, 10.4), "03_w1": (46, 56.4), "03_w2": (60, 68), "03_w3": (21.6, 37.6), "04_w1": (25, 31.6), "04_w2": (39, 47.6), "05_w1": (42, 49.36), "05_w2": (27.5, 34.2)}


def sync(base):
    from syncnet_common import eval_syncnet
    out = pathlib.Path(str(base) + ".sync.json")
    if out.exists(): return
    r = {}
    for k, p in (("dub_only", pathlib.Path(str(base) + "_dubonly.mp4")), ("mix", pathlib.Path(str(base) + ".mp4"))):
        try: x = eval_syncnet(p); r[k] = {"conf": x["confidence"], "av_offset": x["av_offset_frames"]}
        except Exception as e: r[k] = {"error": str(e)[-300:]}
    out.write_text(json.dumps(r)); print("sync", base.name, r, flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=3); ap.add_argument("--skip-wrap", action="store_true"); ap.add_argument("--no-v4", action="store_true")
    a = ap.parse_args()
    bases = []
    for m in sorted(W.glob("*.manifest.json")):
        if m.name.endswith(".v4man.json"): continue
        base = pathlib.Path(str(m)[:-len(".manifest.json")]); arm = re.match(r"\d\d_w\d_(.+)", base.name).group(1)
        if a.skip_wrap and arm.startswith("wrap"): continue
        bases.append(base)
    with ThreadPoolExecutor(a.jobs) as ex: list(ex.map(sync, bases))
    if a.no_v4: return
    for base in bases:
        if pathlib.Path(str(base) + ".v4.json").exists(): continue
        k = base.name[:5]; t0, t1 = WIN[k]; arm = base.name[6:]
        subprocess.run(["/workspace/dub/dubai/scripts/quality/r2_eval.sh", k[:2], str(t0), str(t1), k[3:], arm], check=False)


if __name__ == "__main__":
    main()
