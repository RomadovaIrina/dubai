#!/usr/bin/env python3
"""Pilot 0.5 — GPU-minutes per source-video minute for the CLEAN real pipeline.

Formal meaning:
  - run the REAL clean E2E pipeline;
  - CodeFormer is explicitly disabled;
  - measure how long this job's process group owns CUDA memory;
  - normalize GPU-occupied seconds by source-video seconds.

This script is a measurement harness, not a replacement pipeline.

Example:
  source scripts/env.sh
  python scripts/pilot/benchmark_05_gpu_coefficient.py test_videos \
    --command-template 'python -m dabai.pipeline --input {input} --output {output} --no-codeformer' \
    --repeat 2 \
    --confirm-clean-pipeline \
    --confirm-codeformer-disabled

Generated media goes to /tmp by default. Compact JSON/MD evidence goes to reports/pilot/.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shlex
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT, ROOT, gpu_info, write_json

EXT = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
DEFAULT_WORK = pathlib.Path("/tmp/dabai_pilot_05")


def ffprobe(path: pathlib.Path) -> dict:
    p = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        capture_output=True, text=True, timeout=60, check=True,
    )
    d = json.loads(p.stdout)
    fmt = d.get("format", {})
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in d.get("streams", []) if s.get("codec_type") == "audio"), {})
    dur = float(fmt.get("duration") or v.get("duration") or 0)
    rate = v.get("avg_frame_rate") or v.get("r_frame_rate") or "0/1"
    try:
        n, den = rate.split("/")
        fps = float(n) / float(den) if float(den) else 0.0
    except Exception:
        fps = 0.0
    return {
        "duration_s": dur,
        "video": {
            "codec": v.get("codec_name"),
            "width": v.get("width"),
            "height": v.get("height"),
            "fps": fps,
            "frames": v.get("nb_frames"),
        },
        "audio": {
            "codec": a.get("codec_name"),
            "sample_rate": a.get("sample_rate"),
            "channels": a.get("channels"),
        },
    }


def gpu_processes() -> dict[int, int]:
    """pid -> used MiB for active CUDA compute processes."""
    p = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True,
    )
    out: dict[int, int] = {}
    if p.returncode != 0:
        return out
    for line in p.stdout.splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            pid = int(parts[0])
            mib = int(float(parts[1]))
            out[pid] = mib
        except Exception:
            continue
    return out


def pgrp(pid: int) -> int | None:
    try:
        return os.getpgid(pid)
    except Exception:
        return None


class GpuOccupancy(threading.Thread):
    """Sample CUDA ownership for one process group.

    The benchmark command is launched in a fresh process group. This catches children even if
    their direct parent exits, provided they stay in the job's process group.
    """

    def __init__(self, pgid: int, interval: float):
        super().__init__(daemon=True)
        self.pgid = pgid
        self.interval = interval
        self.stop_evt = threading.Event()
        self.samples = 0
        self.active_samples = 0
        self.mem_samples_mib: list[int] = []
        self.first_active_after_s: float | None = None
        self._t0 = time.perf_counter()

    def run(self) -> None:
        while not self.stop_evt.is_set():
            procs = gpu_processes()
            owned = [mib for pid, mib in procs.items() if pgrp(pid) == self.pgid]
            self.samples += 1
            if owned:
                if self.first_active_after_s is None:
                    self.first_active_after_s = time.perf_counter() - self._t0
                self.active_samples += 1
                self.mem_samples_mib.append(sum(owned))
            self.stop_evt.wait(self.interval)

    def stop(self) -> None:
        self.stop_evt.set()
        self.join(timeout=5)

    @property
    def gpu_seconds(self) -> float:
        return self.active_samples * self.interval

    @property
    def peak_mib(self) -> int:
        return max(self.mem_samples_mib, default=0)

    @property
    def mean_active_mib(self) -> float:
        return statistics.mean(self.mem_samples_mib) if self.mem_samples_mib else 0.0


def validate_output(src_meta: dict, out: pathlib.Path) -> dict:
    if not out.exists() or out.stat().st_size == 0:
        return {"pass": False, "reason": "output missing/empty"}
    try:
        meta = ffprobe(out)
    except Exception as e:
        return {"pass": False, "reason": f"ffprobe failed: {type(e).__name__}: {e}"}
    if not meta["video"]["codec"]:
        return {"pass": False, "reason": "no video stream", "output_meta": meta}

    src_dur = float(src_meta["duration_s"])
    out_dur = float(meta["duration_s"])
    fps = float(src_meta["video"].get("fps") or 25.0)
    tol = max(0.15, 2.0 / fps) if fps else 0.15
    delta = out_dur - src_dur

    # For a dubbed output the timeline should still be the source timeline.
    passed = abs(delta) <= tol
    return {
        "pass": passed,
        "reason": None if passed else f"duration delta {delta:+.3f}s > tolerance {tol:.3f}s",
        "duration_delta_s": round(delta, 4),
        "tolerance_s": round(tol, 4),
        "output_meta": meta,
    }


def run_once(video: pathlib.Path, template: str, idx: int, interval: float, work: pathlib.Path) -> dict:
    work.mkdir(parents=True, exist_ok=True)
    src_meta = ffprobe(video)
    if src_meta["duration_s"] <= 0:
        raise RuntimeError(f"invalid source duration for {video}")

    out = work / f"{video.stem}_r{idx}.mp4"
    if out.exists():
        out.unlink()

    cmd = template.format(
        input=shlex.quote(str(video.resolve())),
        output=shlex.quote(str(out.resolve())),
        stem=shlex.quote(video.stem),
    )

    t0 = time.perf_counter()
    p = subprocess.Popen(
        cmd,
        shell=True,
        cwd=str(ROOT),
        start_new_session=True,  # process group id == shell pid
    )
    mon = GpuOccupancy(p.pid, interval)
    mon.start()
    rc = p.wait()
    mon.stop()
    wall = time.perf_counter() - t0

    val = validate_output(src_meta, out) if rc == 0 else {"pass": False, "reason": f"returncode={rc}"}
    dur = float(src_meta["duration_s"])
    gpu_s = mon.gpu_seconds

    return {
        "video": video.name,
        "source": src_meta,
        "run": idx,
        "returncode": rc,
        "wall_s": round(wall, 3),
        "gpu_occupied_s": round(gpu_s, 3),
        "gpu_min_per_video_min": round(gpu_s / dur, 4),
        "wall_s_per_video_min": round(wall / dur * 60.0, 3),
        "gpu_occupancy_fraction_of_wall": round(gpu_s / wall, 4) if wall else None,
        "gpu_first_active_after_s": round(mon.first_active_after_s, 3) if mon.first_active_after_s is not None else None,
        "gpu_peak_process_group_mib": mon.peak_mib,
        "gpu_mean_active_process_group_mib": round(mon.mean_active_mib, 1),
        "samples": mon.samples,
        "active_samples": mon.active_samples,
        "output": str(out),
        "output_validation": val,
        "command": cmd,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def video_list(path: pathlib.Path) -> list[pathlib.Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        return []
    return sorted(x for x in path.iterdir() if x.is_file() and x.suffix.lower() in EXT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("videos", help="video file or directory")
    ap.add_argument(
        "--command-template", required=True,
        help="real clean-pipeline command; must contain {input} and {output}",
    )
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--interval", type=float, default=0.2)
    ap.add_argument("--work-dir", default=str(DEFAULT_WORK))
    ap.add_argument("--target-ratio", type=float, default=5.0, help="target GPU-min/video-min")
    ap.add_argument(
        "--confirm-clean-pipeline", action="store_true",
        help="formal closure flag: command is the real clean E2E pipeline, not a component benchmark",
    )
    ap.add_argument(
        "--confirm-codeformer-disabled", action="store_true",
        help="formal closure flag: CodeFormer is disabled for the measured command",
    )
    ap.add_argument("--json", default=str(PILOT / "0.5_gpu_coefficient.json"))
    ap.add_argument("--md", default=str(PILOT / "0.5_gpu_coefficient.md"))
    a = ap.parse_args()

    if "{input}" not in a.command_template or "{output}" not in a.command_template:
        ap.error("--command-template must contain {input} and {output}")
    if a.repeat < 1:
        ap.error("--repeat must be >= 1")
    if a.interval <= 0:
        ap.error("--interval must be > 0")

    vids = video_list(pathlib.Path(a.videos).resolve())
    if not vids:
        raise SystemExit("no videos found")

    rows: list[dict] = []
    work = pathlib.Path(a.work_dir)
    for v in vids:
        for i in range(1, a.repeat + 1):
            print(f"\n===== 0.5 {v.name} run {i}/{a.repeat} =====", flush=True)
            r = run_once(v, a.command_template, i, a.interval, work)
            rows.append(r)
            print(json.dumps({
                "video": r["video"],
                "run": r["run"],
                "rc": r["returncode"],
                "gpu_min_per_video_min": r["gpu_min_per_video_min"],
                "wall_s_per_video_min": r["wall_s_per_video_min"],
                "output_valid": r["output_validation"]["pass"],
                "peak_mib": r["gpu_peak_process_group_mib"],
            }, ensure_ascii=False), flush=True)

    good = [r for r in rows if r["returncode"] == 0 and r["output_validation"].get("pass")]
    gpu = gpu_info()
    on_target_gpu = gpu.get("capability") == "sm_120"

    per_video = []
    for v in vids:
        rr = [r for r in good if r["video"] == v.name]
        vals = [r["gpu_min_per_video_min"] for r in rr]
        walls = [r["wall_s_per_video_min"] for r in rr]
        per_video.append({
            "video": v.name,
            "successful_runs": len(rr),
            "required_runs": a.repeat,
            "median_gpu_min_per_video_min": round(statistics.median(vals), 4) if vals else None,
            "median_wall_s_per_video_min": round(statistics.median(walls), 3) if walls else None,
            "min_gpu_ratio": min(vals) if vals else None,
            "max_gpu_ratio": max(vals) if vals else None,
        })

    dataset_vals = [
        x["median_gpu_min_per_video_min"] for x in per_video
        if x["median_gpu_min_per_video_min"] is not None
    ]
    all_videos_complete = (
        len(per_video) == len(vids)
        and all(x["successful_runs"] == a.repeat for x in per_video)
    )
    formal_inputs = a.confirm_clean_pipeline and a.confirm_codeformer_disabled
    closed = bool(dataset_vals) and all_videos_complete and on_target_gpu and formal_inputs

    dataset_mean = round(statistics.mean(dataset_vals), 4) if dataset_vals else None
    dataset_median = round(statistics.median(dataset_vals), 4) if dataset_vals else None
    within_target = dataset_median <= a.target_ratio if dataset_median is not None else None

    if closed:
        verdict = (
            f"CLOSED — {'PASS_TARGET' if within_target else 'FAIL_TARGET'}; "
            f"median {dataset_median} GPU-min/video-min vs target {a.target_ratio}"
        )
    else:
        missing = []
        if not all_videos_complete:
            missing.append("not all videos/runs produced valid outputs")
        if not on_target_gpu:
            missing.append("not sm_120 target GPU")
        if not a.confirm_clean_pipeline:
            missing.append("clean E2E pipeline not confirmed")
        if not a.confirm_codeformer_disabled:
            missing.append("CodeFormer-disabled mode not confirmed")
        verdict = "OPEN — " + "; ".join(missing)

    rep = {
        "task": "0.5",
        "closed": closed,
        "verdict": verdict,
        "gpu": gpu,
        "method": (
            "job process-group CUDA occupancy sampled via nvidia-smi; "
            "GPU-min/video-min = GPU-owned seconds / source-video seconds"
        ),
        "clean_pipeline_confirmed": a.confirm_clean_pipeline,
        "codeformer_disabled_confirmed": a.confirm_codeformer_disabled,
        "target_gpu_min_per_video_min": a.target_ratio,
        "repeat": a.repeat,
        "sampling_interval_s": a.interval,
        "videos": len(vids),
        "successful_runs": len(good),
        "total_runs": len(rows),
        "per_video": per_video,
        "dataset": {
            "mean_gpu_min_per_video_min": dataset_mean,
            "median_gpu_min_per_video_min": dataset_median,
            "min_gpu_min_per_video_min": min(dataset_vals) if dataset_vals else None,
            "max_gpu_min_per_video_min": max(dataset_vals) if dataset_vals else None,
            "within_target_by_dataset_median": within_target,
        },
        "rows": rows,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(pathlib.Path(a.json), rep)

    md = [
        "# Pilot 0.5 — clean-pipeline GPU coefficient",
        "",
        f"**Verdict:** `{verdict}`  ",
        f"Target GPU: `{gpu.get('name')}` / `{gpu.get('capability')}`  ",
        f"CodeFormer disabled confirmed: **{a.confirm_codeformer_disabled}**  ",
        f"Clean E2E confirmed: **{a.confirm_clean_pipeline}**  ",
        f"Dataset median: **{dataset_median} GPU-min/video-min**  ",
        f"Dataset mean: **{dataset_mean} GPU-min/video-min**  ",
        f"Target: **{a.target_ratio} GPU-min/video-min**",
        "",
        "| video | successful runs | median GPU-min/video-min | median wall s/video-min | min | max |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for x in per_video:
        md.append(
            f"| {x['video']} | {x['successful_runs']}/{a.repeat} | "
            f"{x['median_gpu_min_per_video_min']} | {x['median_wall_s_per_video_min']} | "
            f"{x['min_gpu_ratio']} | {x['max_gpu_ratio']} |"
        )
    pathlib.Path(a.md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))
    print(f"\n-> {a.json}\n-> {a.md}")
    return 0 if closed else 1


if __name__ == "__main__":
    raise SystemExit(main())
