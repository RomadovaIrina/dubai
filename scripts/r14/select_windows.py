#!/usr/bin/env python3
"""r14: the training window set = all own windows (dub audio; held-out clips never captured) + the external windows that pass the sync filter
(filter_windows.py scores): 16-frame windows with stable-SyncNet sim >= --thr, minus whole clips whose source is out of sync (--drop-jobs),
plus the 8-frame bridge windows a kept 16-frame window covers. Symlinks into --out.
    python scripts/r14/select_windows.py --filter filter.json --own DIR --ext DIR --out DIR [--thr 0.6] [--drop-jobs x_c01 x_c05]"""
import argparse, glob, json, os, pathlib
import torch


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--filter", required=True); ap.add_argument("--own", required=True); ap.add_argument("--ext", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--thr", type=float, default=0.6); ap.add_argument("--drop-jobs", nargs="*", default=[]); a = ap.parse_args()
    S = json.loads(pathlib.Path(a.filter).read_text())["scores"]; out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    spans, keep = {}, []
    for f in sorted(glob.glob(str(pathlib.Path(a.ext) / "*.pt"))):
        v = S.get(pathlib.Path(f).name)
        if v and v["job"] not in a.drop_jobs and v["sim"] >= a.thr:
            keep.append(f); spans.setdefault(v["job"], []).append(v["start"])
    nb = 0
    for f in sorted(glob.glob(str(pathlib.Path(a.ext) / "*.pt"))):
        if pathlib.Path(f).name in S: continue
        d = torch.load(f); j, s = d.get("job"), d.get("start")
        if j not in a.drop_jobs and s is not None and any(s0 is not None and s0 <= s < s0 + 16 for s0 in spans.get(j, [])): keep.append(f); nb += 1
    own = sorted(glob.glob(str(pathlib.Path(a.own) / "*.pt")))
    for f in own + keep:
        l = out / pathlib.Path(f).name
        if not l.exists(): os.symlink(f, l)
    rep = {"own": len(own), "ext_16": len(keep) - nb, "ext_bridge_8": nb, "total": len(own) + len(keep), "thr": a.thr, "drop_jobs": a.drop_jobs,
           "by_job": {j: len(v) for j, v in spans.items()}}
    (out.parent / (out.name + ".json")).write_text(json.dumps(rep, indent=1)); print(json.dumps(rep))


if __name__ == "__main__":
    main()
