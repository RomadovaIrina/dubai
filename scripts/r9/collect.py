#!/usr/bin/env python3
"""r9 E2E: manifest.json + timing.json of one scripts/r9/e2e.sh run (status, flags, outputs vs the source format, per-stage wall time,
s/video-min, VRAM peak per stage, and the gate rows the stages already wrote: face score / frame gate, audio coverage / cuts / pacing / gaps / faith / WER)."""
from __future__ import annotations
import argparse, glob, json, pathlib, subprocess


def jl(p):
    try: return json.loads(pathlib.Path(p).read_text())
    except Exception: return None


def vinfo(p):
    try:
        r = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(p)], capture_output=True, text=True).stdout)
        v = next(s for s in r["streams"] if s["codec_type"] == "video"); a = next((s for s in r["streams"] if s["codec_type"] == "audio"), {})
        n, d = v["avg_frame_rate"].split("/"); rot = next((int(round(float(sd["rotation"]))) for sd in v.get("side_data_list", []) if "rotation" in sd), 0)
        w, h = int(v["width"]), int(v["height"]); w, h = (h, w) if abs(rot) % 180 == 90 else (w, h)
        return {"width": w, "height": h, "fps": round(float(n) / float(d), 3) if float(d) else 0, "duration_s": round(float(r["format"]["duration"]), 3),
                "a_duration_s": round(float(a.get("duration", 0) or 0), 3), "vcodec": v["codec_name"], "size_mb": round(int(r["format"]["size"]) / 1e6, 2)}
    except Exception as e:
        return {"error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    for k in ("id", "out", "status", "flags", "errors", "stages", "nvsmi", "precheck", "audio", "htd", "face", "input"): ap.add_argument("--" + k, default="")
    a = ap.parse_args(); out = pathlib.Path(a.out); pc = jl(a.precheck) or {}
    dur = (pc.get("probe") or {}).get("duration_s") or 0.0; vmin = dur / 60 if dur else None
    rows = [l.rstrip("\n").split("\t") for l in open(a.stages) if "\t" in l]
    stages = {}
    for (n, t0), (_, t1) in zip(rows, rows[1:]): stages[n] = stages.get(n, 0.0) + float(t1) - float(t0)
    total = float(rows[-1][1]) - float(rows[0][1]) if len(rows) > 1 else 0.0
    vram, peak = {}, None
    try:   # nvsmi: "timestamp, used MiB, util"; stage peaks by wall clock
        import datetime as dt
        samp = []
        for l in open(a.nvsmi):
            p = [x.strip() for x in l.split(",")]
            if len(p) >= 2 and p[1].isdigit(): samp.append((dt.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").timestamp(), int(p[1])))
        peak = max((m for _, m in samp), default=None)
        for (n, t0), (_, t1) in zip(rows, rows[1:]):
            m = max((v for t, v in samp if float(t0) <= t <= float(t1)), default=None)
            if m is not None: vram[n] = max(vram.get(n, 0), m)
    except Exception as e:
        vram["error"] = str(e)
    src = vinfo(pathlib.Path(a.input)); outs = {}
    for n in ("demucs", "spleeter"):
        p = out / f"{a.id}_DabAI_r9_{n}.mp4"
        if p.exists():
            i = vinfo(p); i["file"] = p.name
            i["matches_source"] = all(abs((i.get(k) or 0) - (src.get(k) or 0)) < (0.05 if k == "fps" else 0.5) for k in ("width", "height", "fps"))
            outs[n] = i
    flags = [f for f in a.flags.split() if f]
    if outs and not all(o.get("matches_source") for o in outs.values()): flags.append("output_format_differs_from_source")
    # gate rows the stages wrote (best effort: missing files = null)
    fd = pathlib.Path(a.face); ad = pathlib.Path(a.audio); hd = pathlib.Path(a.htd); i_ = a.id
    fm = jl(fd / f"{i_}_{fd.name}.manifest.json") or {}
    metrics = {
        "face_score": jl(next(iter(glob.glob(str(fd / f"{i_}*score*.json"))), "")) if glob.glob(str(fd / f"{i_}*score*.json")) else None,
        "face_r4v": fm.get("r4v"), "lipsync_by_action": (fm.get("lipsync") or {}).get("by_action"),
        "audio_metrics": jl(ad / f"{i_}.metrics.json"), "gaps_spleeter": jl(ad / f"{i_}.gaps.json"), "gaps_demucs": jl(hd / f"{i_}.gaps.json"),
        "coverage": {pathlib.Path(f).name: jl(f) for f in sorted(glob.glob(str(ad / "cov" / f"*_{i_}*.json")))},
        "face_gate": ((jl(out / "work" / "face_gate.json") or {}).get("videos") or {}).get(i_),
        "occlusion": {k: v for k, v in (jl(out / "work" / "occlusion.json") or {}).items() if k != "top"} or None,
        "closed_mouth": ((jl(out / "work" / "closed_mouth.json") or {}).get(i_) or {}).get(fd.name),
        "fallback": next((json.loads(l.split("== fallback ", 1)[1]) for l in open(out / "logs" / "face_comp.log") if l.startswith("== fallback ")), None) if (out / "logs" / "face_comp.log").exists() else None,
        "demucs_cov": {pathlib.Path(f).name: jl(f) for f in sorted(glob.glob(str(hd / "cov" / "*.json")))},
    }
    timing = {"id": i_, "duration_s": dur, "video_min": round(vmin, 4) if vmin else None, "stage_wall_s": {k: round(v, 1) for k, v in stages.items()},
              "total_wall_s": round(total, 1), "s_per_video_min": {k: round(v / vmin, 1) for k, v in stages.items()} if vmin else None,
              "total_s_per_video_min": round(total / vmin, 1) if vmin else None, "vram_peak_mib": peak, "vram_peak_mib_by_stage": vram}
    man = {"id": i_, "input": a.input, "status": a.status, "flags": flags, "errors": [e for e in a.errors.split("\n") if e.strip()], "source": src, "precheck": pc, "outputs": outs, "timing": timing, "metrics": metrics,
           "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()}
    (out / "timing.json").write_text(json.dumps(timing, indent=1)); (out / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    print(f"== {i_} {a.status} flags {flags}; total {timing['total_wall_s']} s = {timing['total_s_per_video_min']} s/video-min; VRAM peak {peak} MiB; "
          + "; ".join(f"{k} {v}" for k, v in (timing['s_per_video_min'] or {}).items()))


if __name__ == "__main__":
    main()
