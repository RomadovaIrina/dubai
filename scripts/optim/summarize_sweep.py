#!/usr/bin/env python3
"""Collect bench_latentsync.py result files into one sweep table (JSON + Markdown).

    python scripts/optim/summarize_sweep.py --baseline reports/pilot/optim/latentsync_baseline.json \
        --runs reports/pilot/optim/sweep_b*.json reports/pilot/optim/deepcache_ab_*.json \
        --json reports/pilot/optim/batch_sweep.json --md reports/pilot/optim/batch_sweep.md
"""
from __future__ import annotations

import argparse
import glob
import json
import pathlib
import time

VRAM_TOTAL_MIB = 32607


def row_of(rep: dict) -> dict:
    c = rep["config"]; st = rep.get("steady_state") or {}; cold = rep.get("cold") or {}
    last = rep["runs"][-1] if rep.get("runs") else {}
    peak = max([r.get("peak_process_mib") or 0 for r in rep.get("runs", [])] + [0]) or None
    tpeak = max([r.get("torch_peak_reserved_mib") or 0 for r in rep.get("runs", [])] + [0]) or None
    return {"label": rep["label"], "mode": c["mode"], "batch": c["window_batch_size"], "deepcache": c["deepcache"], "compile": c["compile_backend"],
            "sdpa": c["sdpa_backend"], "status": rep["status"], "oom": any(r.get("oom") for r in rep.get("runs", [])),
            "cold_wall_s": cold.get("wall_s"), "warm_wall_s": st.get("wall_s"), "s_per_video_min": st.get("s_per_video_min") or cold.get("s_per_video_min"),
            "peak_process_mib": peak, "torch_peak_reserved_mib": tpeak, "output_valid": bool(last.get("output_valid")), "output_frames": rep.get("output_frames"),
            "duration_s": (last.get("output_probe") or {}).get("duration_s"), "psnr_vs_baseline": rep.get("psnr_vs_ref"),
            "compile_setup_s": rep.get("compile_setup_s"), "error": last.get("error")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    ap.add_argument("--title", default="LatentSync window-batch sweep (test_videos/04.mp4)")
    a = ap.parse_args()
    base = json.load(open(a.baseline)); files = sorted({f for p in a.runs for f in glob.glob(p)})
    rows = [row_of(base)] + [row_of(json.load(open(f))) for f in files]
    base_warm = (base.get("steady_state") or {}).get("wall_s")
    for r in rows:
        w = r["warm_wall_s"] or r["cold_wall_s"]
        r["speedup_vs_baseline_warm"] = round(base_warm / w, 3) if (base_warm and w) else None
    ok = [r for r in rows if r["mode"] == "accel" and r["compile"] == "none" and r["deepcache"] and r["status"] == "PASS" and r["output_valid"]]
    safe = max(ok, key=lambda r: r["batch"]) if ok else None
    fastest = min(ok, key=lambda r: r["warm_wall_s"] or r["cold_wall_s"]) if ok else None
    rep = {"title": a.title, "source_duration_s": base["source"]["duration_s"], "baseline_warm_wall_s": base_warm, "rows": rows,
           "largest_safe_batch": safe["batch"] if safe else None, "fastest_batch": fastest["batch"] if fastest else None,
           "fastest_warm_wall_s": fastest["warm_wall_s"] if fastest else None,
           "oom_batches": [r["batch"] for r in rows if r["oom"]], "vram_total_mib": VRAM_TOTAL_MIB, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    pathlib.Path(a.json).write_text(json.dumps(rep, indent=2))
    md = [f"# {a.title}", "", f"Source: {base['source']['duration_s']:.2f} s. Baseline = upstream sequential pipeline, DeepCache, fp16, 20 steps, seed 1247.  ",
          f"Largest SAFE batch (PASS, valid output, no OOM): **{rep['largest_safe_batch']}** · fastest measured batch: **{rep['fastest_batch']}** · OOM at: {rep['oom_batches'] or 'none'}", "",
          "| label | batch | DeepCache | compile | status | cold s | warm s | s/video-min | speedup | proc peak MiB | torch reserved MiB | frames | dur s | PSNR vs baseline (mean/min) |",
          "|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        p = r["psnr_vs_baseline"]; ps = f"{p['psnr_mean']}/{p['psnr_min']}" if p else "-"
        md.append(f"| {r['label']} | {r['batch']} | {r['deepcache']} | {r['compile']} | {r['status']}{' OOM' if r['oom'] else ''} | {r['cold_wall_s'] or '-'} | {r['warm_wall_s'] or '-'} | "
                  f"{r['s_per_video_min'] or '-'} | {r['speedup_vs_baseline_warm'] or '-'} | {r['peak_process_mib'] or '-'} | {r['torch_peak_reserved_mib'] or '-'} | "
                  f"{r['output_frames'] or '-'} | {r['duration_s'] or '-'} | {ps} |")
    pathlib.Path(a.md).write_text("\n".join(md) + "\n"); print("\n".join(md)); print(f"-> {a.json}\n-> {a.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
