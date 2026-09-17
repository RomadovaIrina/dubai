#!/usr/bin/env python3
"""Merge several benchmark_05_gpu_coefficient.py result files (e.g. two repeat=1 passes) into one dataset report.

Aggregation mirrors benchmark_05: per-video median GPU-min/video-min over successful runs, dataset mean/median/min/max
of the per-video medians, target 5.0 GPU-min/video-min. Adds per-run routing / LatentSync facts read from the runner
manifests (<output>.manifest.json) when present. Writes JSON + Markdown; never touches the canonical 0.5 report.

    python scripts/optim/merge_05_runs.py --runs reports/pilot/optim/0.5_optimized_pass1.json reports/pilot/optim/0.5_optimized_pass2.json \
        --baseline-sanity /tmp/dabai_pilot_05/sanity_04.manifest.json --json reports/pilot/optim/0.5_optimized.json --md reports/pilot/optim/0.5_optimized.md
"""
from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import time


def manifest_facts(output: str) -> dict | None:
    p = pathlib.Path(output); mp = p.with_name(p.stem + ".manifest.json")
    if not mp.exists():
        return None
    m = json.load(open(mp)); ls = m.get("lipsync", {}) or {}
    return {"video_backend": m.get("video_backend"), "ls_fraction": ls.get("ls_fraction"), "latentsync_seconds_of_video": ls.get("latentsync_seconds_of_video"),
            "pass_through_seconds_of_video": ls.get("pass_through_seconds_of_video"), "frame_classes": ls.get("frame_classes"), "skip_reasons": ls.get("skip_reasons"),
            "latentsync_inference_total_s": ls.get("latentsync_inference_total_s"), "latentsync_calls": ls.get("latentsync_calls"),
            "by_action": ls.get("by_action"), "runner": ls.get("runner"), "stage_seconds": m.get("stage_seconds"), "validation_pass": (m.get("validation") or {}).get("pass"),
            "output_frame_check": ls.get("output_frame_check"), "codeformer": m.get("codeformer")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="benchmark_05 json files to merge (rows are concatenated, run index renumbered)")
    ap.add_argument("--target-ratio", type=float, default=5.0)
    ap.add_argument("--title", default="Pilot 0.5 — optimized video backend (RetinaFace router + accelerated LatentSync), CodeFormer OFF")
    ap.add_argument("--json", required=True)
    ap.add_argument("--md", required=True)
    a = ap.parse_args()
    reps = [json.load(open(f)) for f in a.runs]
    rows = []
    for k, rep in enumerate(reps, 1):
        for r in rep["rows"]:
            r = dict(r); r["pass"] = k; r["manifest"] = manifest_facts(r["output"]); rows.append(r)
    videos = sorted({r["video"] for r in rows})
    good = [r for r in rows if r["returncode"] == 0 and r["output_validation"].get("pass")]
    per_video = []
    for v in videos:
        rr = [r for r in good if r["video"] == v]; vals = [r["gpu_min_per_video_min"] for r in rr]; walls = [r["wall_s_per_video_min"] for r in rr]
        per_video.append({"video": v, "successful_runs": len(rr), "total_runs": sum(1 for r in rows if r["video"] == v),
                          "source_duration_s": next(r["source"]["duration_s"] for r in rows if r["video"] == v),
                          "median_gpu_min_per_video_min": round(statistics.median(vals), 4) if vals else None,
                          "median_wall_s_per_video_min": round(statistics.median(walls), 3) if walls else None,
                          "min_gpu_ratio": min(vals) if vals else None, "max_gpu_ratio": max(vals) if vals else None,
                          "ls_fraction": next((r["manifest"]["ls_fraction"] for r in rr if r.get("manifest")), None),
                          "latentsync_seconds_of_video": next((r["manifest"]["latentsync_seconds_of_video"] for r in rr if r.get("manifest")), None),
                          "peak_mib": max(r["gpu_peak_process_group_mib"] for r in rr) if rr else None})
    vals = [x["median_gpu_min_per_video_min"] for x in per_video if x["median_gpu_min_per_video_min"] is not None]
    gpu = reps[-1].get("gpu", {})
    rep = {"task": "0.5-optimized", "title": a.title, "gpu": gpu, "on_target_gpu": gpu.get("capability") == "sm_120",
           "command_template": reps[-1]["rows"][0]["command"] if reps[-1]["rows"] else None, "passes": len(reps),
           "total_runs": len(rows), "successful_runs": len(good), "all_valid": len(good) == len(rows),
           "codeformer_disabled_confirmed": all(r.get("manifest", {}) and "DISABLED" in str(r["manifest"].get("codeformer")) for r in rows if r.get("manifest")),
           "per_video": per_video, "target_gpu_min_per_video_min": a.target_ratio,
           "dataset": {"mean_gpu_min_per_video_min": round(statistics.mean(vals), 4) if vals else None, "median_gpu_min_per_video_min": round(statistics.median(vals), 4) if vals else None,
                       "min_gpu_min_per_video_min": min(vals) if vals else None, "max_gpu_min_per_video_min": max(vals) if vals else None,
                       "within_target_by_dataset_median": (statistics.median(vals) <= a.target_ratio) if vals else None,
                       "within_target_by_dataset_mean": (statistics.mean(vals) <= a.target_ratio) if vals else None,
                       "duration_weighted_gpu_min_per_video_min": round(sum(r["gpu_occupied_s"] for r in good) / sum(r["source"]["duration_s"] for r in good), 4) if good else None},
           "rows": rows, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    pathlib.Path(a.json).write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=str))
    md = [f"# {a.title}", "", f"GPU: `{gpu.get('name')}` / `{gpu.get('capability')}` · runs {len(good)}/{len(rows)} valid · target **{a.target_ratio} GPU-min/video-min**  ",
          f"Dataset median **{rep['dataset']['median_gpu_min_per_video_min']}**, mean **{rep['dataset']['mean_gpu_min_per_video_min']}**, "
          f"min {rep['dataset']['min_gpu_min_per_video_min']}, max {rep['dataset']['max_gpu_min_per_video_min']}, duration-weighted {rep['dataset']['duration_weighted_gpu_min_per_video_min']}  ",
          f"Target reached (dataset median <= {a.target_ratio}): **{rep['dataset']['within_target_by_dataset_median']}**", "",
          "| video | pass | src s | wall s | GPU occupied s | GPU-min/video-min | wall s/video-min | peak MiB | LS fraction | LS video s | valid |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        m = r.get("manifest") or {}
        md.append(f"| {r['video']} | {r['pass']} | {r['source']['duration_s']:.2f} | {r['wall_s']} | {r['gpu_occupied_s']} | {r['gpu_min_per_video_min']} | {r['wall_s_per_video_min']} | "
                  f"{r['gpu_peak_process_group_mib']} | {m.get('ls_fraction', '-')} | {m.get('latentsync_seconds_of_video', '-')} | {r['output_validation'].get('pass')} |")
    md += ["", "| video | median GPU-min/video-min | median wall s/video-min | min | max | runs |", "|---|---:|---:|---:|---:|---:|"]
    for x in per_video:
        md.append(f"| {x['video']} | {x['median_gpu_min_per_video_min']} | {x['median_wall_s_per_video_min']} | {x['min_gpu_ratio']} | {x['max_gpu_ratio']} | {x['successful_runs']}/{x['total_runs']} |")
    pathlib.Path(a.md).write_text("\n".join(md) + "\n"); print("\n".join(md)); print(f"-> {a.json}\n-> {a.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
