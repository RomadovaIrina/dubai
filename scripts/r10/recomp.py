#!/usr/bin/env python3
"""r10 FACE: re-run fps_composite.py with a finished composite's exact args (its .fps_composite.json 'args') and named / key=value changes.
    python scripts/r10/recomp.py --args-json '<json>' --set nosharp|nearest|rigidreg|notex|noregion|key=value[,key=value...] --out X.mp4 [--extra "..."]"""
import argparse, json, pathlib, subprocess, sys
NAMED = {"nosharp": {"sharpen_auto": 0, "ls_sharpen": 0, "ls_sharpen_lips": 0}, "nearest": {"retime": "nearest_rigid"}, "rigidreg": {"register": "rigid"},
         "notex": {"skin_texture": "off"}, "noregion": {"mouth_region": 0}, "nocolour": {"colour_match": "off"}, "noreg": {"register": "off"}, "base": {}}
ap = argparse.ArgumentParser(); ap.add_argument("--args-json", required=True); ap.add_argument("--set", default="base"); ap.add_argument("--out", required=True)
ap.add_argument("--extra", default=""); ap.add_argument("--threads", type=int, default=16)
a = ap.parse_args(); args = json.loads(a.args_json)
for item in a.set.split("+"):
    if item in NAMED: args.update(NAMED[item])
    else:
        for kv in item.split(","):
            if "=" in kv: k, v = kv.split("=", 1); args[k.replace("-", "_")] = v
args["out"] = a.out; args["threads"] = a.threads
cmd = [sys.executable, str(pathlib.Path(__file__).resolve().parents[1] / "quality" / "fps_composite.py")]
for k, v in args.items():
    if v is None or v == "" or v is False: continue
    cmd += [f"--{k.replace('_', '-')}", str(v)]
cmd += a.extra.split()
print(" ".join(cmd), flush=True); sys.exit(subprocess.run(cmd).returncode)
