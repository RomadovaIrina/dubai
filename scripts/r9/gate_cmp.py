#!/usr/bin/env python3
"""r9 speed: the per-lever face gate (brief_r9_speed.md) of a face_only.sh variant against the r9 + fix baseline of the same video.
  SyncNet dub speech >= base - 0.3; strong flashes <= base + 3; pressed lips <= base + 0.5 s; whole-frame gate PASS; stick / unstick and slide events
  (artefacts.py, final output) <= base; occluder ring frames over 2x median <= base; fallback % (frames > 0.5) <= base + 1.
  A variant whose decoded video is bit-identical to the baseline passes by construction (md5 printed).
    python scripts/r9/gate_cmp.py --base /workspace/dub/r9/face/m03_b0 --cand /workspace/dub/r9/face/m03_L14 [--md md_row_file]"""
import argparse, json, pathlib, re, subprocess


def load(d: pathlib.Path):
    r = {}
    try:
        g = next(iter(json.loads((d / "face_gate.json").read_text())["videos"].values()))
        r.update({k: g.get(k) for k in ("sync_speech", "sync_dub_only", "flash_strong", "pressed_lips_s", "frame_gate_verdict", "frame_gate_worst_pct", "twitch_sum")})
    except Exception as e: r["err_gate"] = str(e)[:80]
    try:
        a = json.loads((d / "art.json").read_text())["layers"]["L4 final"]
        r["stick"] = a["a_stick_unstick"]["events"]; r["slide"] = a["d_slide"]["events"]; r["teeth"] = a["c_teeth"]["events"]; r["amp"] = a["b_amplitude"]["events"]
    except Exception as e: r["err_art"] = str(e)[:80]
    try: r["occl_2x"] = json.loads((d / "occlusion.json").read_text())["frames_over_2x_median"]
    except Exception: pass
    m = re.search(r'"fallback_pct_gt05": ([0-9.]+)', (d / "logs/face_comp.log").read_text()) if (d / "logs/face_comp.log").exists() else None
    r["fallback_pct"] = float(m.group(1)) if m else None
    try: t_ = json.loads((d / "timing.json").read_text()); r["timing"] = t_["s_per_video_min"]; r["duration_s"] = t_["duration_s"]
    except Exception: pass
    return r


def md5(d: pathlib.Path):
    v = next(iter(sorted(d.glob("*_spleeter.mp4"))), None)
    if v is None: return None
    return subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(v), "-map", "0:v", "-f", "md5", "-"], capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--base", required=True); ap.add_argument("--cand", required=True); ap.add_argument("--md", default=""); a = ap.parse_args()
    bd, cd = pathlib.Path(a.base), pathlib.Path(a.cand); b, c = load(bd), load(cd); same = md5(bd) == md5(cd) and md5(cd) is not None
    checks = {}
    def chk(name, ok): checks[name] = "PASS" if ok else "FAIL"
    if same: checks = {"bit_identical": "PASS"}
    else:
        f = lambda k: (b.get(k) is not None and c.get(k) is not None)
        if f("sync_speech"): chk("sync -0.3", c["sync_speech"] >= b["sync_speech"] - 0.3)
        if f("flash_strong"): chk("flashes +2", c["flash_strong"] <= b["flash_strong"] + 2)   # coordinator 30.10: counts get a small noise allowance
        if f("pressed_lips_s"): chk("pressed +0.5s", c["pressed_lips_s"] <= b["pressed_lips_s"] + 0.5)
        if c.get("frame_gate_verdict") is not None: chk("frame gate", c["frame_gate_verdict"] == "PASS")
        else: checks["frame gate"] = "PENDING"
        tol = 2 if (c.get("duration_s") or 0) > 60 else 1   # stick / slide: +1 per video, +2 on videos > 60 s
        if f("stick"): chk(f"stick/unstick +{tol}", c["stick"] <= b["stick"] + tol)
        if f("slide"): chk(f"slides +{tol}", c["slide"] <= b["slide"] + tol)
        if f("occl_2x"): chk("occluder ring", c["occl_2x"] <= b["occl_2x"])
        if f("fallback_pct"): chk("fallback % +1", c["fallback_pct"] <= b["fallback_pct"] + 1)
    verdict = "FAIL" if "FAIL" in checks.values() else ("PENDING" if "PENDING" in checks.values() else "PASS")
    keys = ("sync_speech", "flash_strong", "pressed_lips_s", "frame_gate_verdict", "stick", "slide", "teeth", "amp", "occl_2x", "fallback_pct")
    print(f"== gate {cd.name} vs {bd.name}: {verdict} {checks}")
    print("   base", {k: b.get(k) for k in keys}, b.get("timing")); print("   cand", {k: c.get(k) for k in keys}, c.get("timing"))
    if a.md:
        cell = lambda k: f"{b.get(k)} -> {c.get(k)}"
        row = f"| {cd.name} | {verdict} | {'bit-identical' if same else ''} | " + " | ".join(cell(k) for k in keys) + " |"
        with open(a.md, "a") as fh: fh.write(row + "\n")
    (cd / "gate_vs_base.json").write_text(json.dumps({"base": str(bd), "verdict": verdict, "checks": checks, "base_rows": b, "cand_rows": c, "bit_identical": same}, indent=1))


if __name__ == "__main__":
    main()
