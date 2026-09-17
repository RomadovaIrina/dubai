#!/usr/bin/env python3
"""LatentSync execution benchmark for the optimization track (baseline vs. accelerated wrapper).

Fixed model config in every mode: stage2_512, 20 steps, guidance 1.5, fp16, seed 1247, DeepCache (unless --no-deepcache
for the A/B). Only the execution path changes:
  --mode baseline           upstream LipsyncPipeline.__call__ (sequential windows)
  --mode accel              scripts/optim/latentsync_accel.py (batched windows / SDPA backend / compile backend)

Per run: wall seconds, s per source-video-minute, torch peak allocated/reserved, nvidia-smi process peak (measure_vram
sampler, rows go to reports/pilot/optim/vram_optim.jsonl, never to the canonical reports/pilot/vram.jsonl), output
frames/duration, and PSNR against --ref (e.g. the baseline output) so batched output can be checked against sequential.
A CUDA OOM is recorded as a result (oom=true), not as a crash. Runtime media lives in --out-dir and is deleted unless
--keep-output.

    source scripts/env.sh
    python scripts/optim/bench_latentsync.py --video test_videos/04.mp4 --mode baseline --repeat 2 \
        --json reports/pilot/optim/latentsync_baseline.json --keep-output
    python scripts/optim/bench_latentsync.py --video test_videos/04.mp4 --mode accel --batch 4 \
        --ref /tmp/dabai_optim/baseline_r2.mp4 --json reports/pilot/optim/sweep_b4.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pilot"))
from pilot_common import PILOT, THIRD_PARTY, gpu_info, sh, write_json  # noqa: E402
from measure_vram import measure_gpu_memory                            # noqa: E402
from benchmark_06_codeformer import probe                              # noqa: E402

OPTIM = PILOT / "optim"
OPTIM_JSONL = OPTIM / "vram_optim.jsonl"
LS = THIRD_PARTY / "latentsync"


def count_frames(p: pathlib.Path) -> int:
    v = sh(f'ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "{p}"', timeout=600)
    return int(v.strip() or 0)


def extract_audio(video: pathlib.Path, wav: pathlib.Path) -> None:
    if not wav.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-nostdin", "-i", str(video), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(wav)], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True)
    ap.add_argument("--audio", default=None, help="16 kHz wav driving the lipsync; default: the video's own audio")
    ap.add_argument("--mode", choices=["baseline", "accel"], default="baseline")
    ap.add_argument("--batch", type=int, default=1, help="accel: temporal windows per UNet call")
    ap.add_argument("--no-deepcache", action="store_true")
    ap.add_argument("--compile", choices=["none", "inductor", "tensorrt"], default="none")
    ap.add_argument("--compile-mode", default=None, help="torch.compile mode for inductor (e.g. max-autotune-no-cudagraphs)")
    ap.add_argument("--sdpa", choices=["auto", "flash", "efficient", "math"], default="auto")
    ap.add_argument("--repeat", type=int, default=2, help="run 1 = cold (first CUDA kernels / compile), rest = warm")
    ap.add_argument("--out-dir", default="/tmp/dabai_optim")
    ap.add_argument("--label", default=None)
    ap.add_argument("--ref", default=None, help="reference mp4 for frame PSNR (e.g. baseline output)")
    ap.add_argument("--keep-output", action="store_true")
    ap.add_argument("--json", required=True)
    a = ap.parse_args()

    import torch
    from accelerate.utils import set_seed
    from omegaconf import OmegaConf
    from resident_vram_probe import load_latentsync

    src = pathlib.Path(a.video).resolve(); out_dir = pathlib.Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    OPTIM.mkdir(parents=True, exist_ok=True)
    label = a.label or (f"{a.mode}" + (f"_b{a.batch}" if a.mode == "accel" else "") + ("_nodc" if a.no_deepcache else "") +
                        (f"_{a.compile}" if a.compile != "none" else "") + (f"_{a.sdpa}" if a.sdpa != "auto" else ""))
    audio = pathlib.Path(a.audio).resolve() if a.audio else out_dir / f"{src.stem}_audio16k.wav"
    extract_audio(src, audio)
    info = probe(src)
    cfg = OmegaConf.load(LS / "configs/unet/stage2_512.yaml")
    config = {"unet_config": "configs/unet/stage2_512.yaml", "resolution": int(cfg.data.resolution), "num_frames": int(cfg.data.num_frames),
              "inference_steps": 20, "guidance_scale": 1.5, "deepcache": not a.no_deepcache, "dtype": "float16", "seed": 1247,
              "mode": a.mode, "window_batch_size": a.batch if a.mode == "accel" else 1, "compile_backend": a.compile,
              "compile_mode": a.compile_mode, "sdpa_backend": a.sdpa}
    print(f"== bench {label}: {src.name} {info['duration_s']:.2f}s  config={config}", flush=True)

    with measure_gpu_memory(f"optim/{label}/model-load", note="pipeline load only", jsonl=OPTIM_JSONL, quiet=True) as ml:
        pipe = load_latentsync()
    model_load = {"seconds": ml.result["seconds"], "peak_device_mib": ml.result["peak_device_mib"], "torch_peak_allocated_mib": ml.result.get("torch_peak_allocated_mib")}
    print(f"   model load {model_load}", flush=True)

    acc = None; helper = None; compile_setup_s = None
    if a.mode == "accel":
        from latentsync_accel import AccelLatentSync
        copts = {"mode": a.compile_mode} if (a.compile == "inductor" and a.compile_mode) else None
        try:
            acc = AccelLatentSync(pipe, window_batch_size=a.batch, deepcache=not a.no_deepcache, compile_backend=a.compile, sdpa_backend=a.sdpa, compile_options=copts)
        except Exception as e:
            rep = {"label": label, "config": config, "status": "SETUP_FAIL", "error": f"{type(e).__name__}: {str(e)[:1500]}", "gpu": gpu_info()}
            write_json(pathlib.Path(a.json), rep); print(json.dumps(rep, indent=1)); return 0
        compile_setup_s = acc.compile_s
    elif not a.no_deepcache:
        from DeepCache import DeepCacheSDHelper
        helper = DeepCacheSDHelper(pipe=pipe); helper.set_params(cache_interval=3, cache_branch_id=0); helper.enable()

    runs = []; oom = False
    for i in range(1, a.repeat + 1):
        kind = "cold" if i == 1 else "warm"
        out = out_dir / f"{label}_r{i}.mp4"
        if out.exists():
            out.unlink()
        temp = out_dir / f"ls_temp_{label}_r{i}"
        row = {"run": i, "kind": kind, "output": str(out)}
        try:
            with measure_gpu_memory(f"optim/{label}/{src.stem}/inference-{kind}-r{i}", note="in-process, model loaded", jsonl=OPTIM_JSONL, quiet=True) as m:
                if acc is not None:
                    stats = acc(str(src), str(audio), str(out), str(temp))
                    row["stages"] = stats["seconds"]; row["windows"] = {k: stats[k] for k in ("windows", "full_windows", "partial_windows", "groups", "unet_calls")}
                else:
                    set_seed(1247); old = os.getcwd(); os.chdir(LS)
                    try:
                        pipe(video_path=str(src), audio_path=str(audio), video_out_path=str(out), num_frames=cfg.data.num_frames,
                             num_inference_steps=20, guidance_scale=1.5, weight_dtype=torch.float16, width=cfg.data.resolution,
                             height=cfg.data.resolution, mask_image_path=cfg.data.mask_image_path, temp_dir=str(temp))
                    finally:
                        os.chdir(old)
            sec = float(m.result["seconds"])
            row.update(ok=True, oom=False, wall_s=round(sec, 3), s_per_video_min=round(sec / info["duration_s"] * 60, 3),
                       peak_device_mib=m.result["peak_device_mib"], peak_process_mib=m.result.get("peak_process_mib"),
                       torch_peak_allocated_mib=m.result.get("torch_peak_allocated_mib"), torch_peak_reserved_mib=m.result.get("torch_peak_reserved_mib"))
            if out.exists() and out.stat().st_size > 0:
                op = probe(out); row["output_probe"] = op; row["output_frames"] = count_frames(out)
                row["output_valid"] = bool(op.get("duration_s")) and row["output_frames"] > 0
                if a.ref:
                    from latentsync_accel import compare_videos
                    row["psnr_vs_ref"] = compare_videos(pathlib.Path(a.ref), out)
            else:
                row["output_valid"] = False
        except torch.cuda.OutOfMemoryError as e:
            oom = True
            row.update(ok=False, oom=True, error=f"OutOfMemoryError: {str(e)[:300]}", output_valid=False,
                       torch_peak_allocated_mib=round(torch.cuda.max_memory_allocated() / 2**20, 1), torch_peak_reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20, 1))
            print(f"   OOM at {label} run {i}", flush=True)
        except Exception as e:
            row.update(ok=False, oom=False, error=f"{type(e).__name__}: {str(e)[:1500]}", output_valid=False)
            print(f"   FAIL at {label} run {i}: {row['error'][:300]}", flush=True)
        shutil.rmtree(temp, ignore_errors=True)
        print(f"   run {i} {kind}: " + json.dumps({k: row.get(k) for k in ("ok", "oom", "wall_s", "s_per_video_min", "peak_process_mib", "torch_peak_allocated_mib", "output_frames", "output_valid", "psnr_vs_ref")}), flush=True)
        runs.append(row)
        if not a.keep_output and out.exists() and not (a.ref is None and i == a.repeat):
            out.unlink()
        if oom or not row["ok"]:
            break
    if not a.keep_output:
        for r in runs:
            p = pathlib.Path(r["output"])
            if p.exists():
                p.unlink()

    warm = [r for r in runs if r["kind"] == "warm" and r.get("ok")]
    cold = next((r for r in runs if r["kind"] == "cold" and r.get("ok")), None)
    steady = min(warm, key=lambda r: r["wall_s"]) if warm else None
    rep = {"label": label, "video": str(src), "audio": str(audio), "source": info, "config": config, "gpu": gpu_info(),
           "status": "OOM" if oom else ("PASS" if runs and all(r.get("ok") for r in runs) else "FAIL"),
           "model_load": model_load, "compile_setup_s": compile_setup_s, "runs": runs,
           "cold": {k: cold[k] for k in ("wall_s", "s_per_video_min", "peak_process_mib", "torch_peak_allocated_mib")} if cold else None,
           "steady_state": {k: steady[k] for k in ("wall_s", "s_per_video_min", "peak_process_mib", "torch_peak_allocated_mib", "torch_peak_reserved_mib")} if steady else None,
           "output_frames": next((r.get("output_frames") for r in runs if r.get("output_frames")), None),
           "psnr_vs_ref": next((r.get("psnr_vs_ref") for r in reversed(runs) if r.get("psnr_vs_ref")), None),
           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(pathlib.Path(a.json), rep)
    print(json.dumps({k: rep[k] for k in ("status", "model_load", "cold", "steady_state", "output_frames", "psnr_vs_ref")}, indent=1))
    print(f"-> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
