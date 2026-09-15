#!/usr/bin/env python3
"""Pilot 0.6 — LatentSync seconds per video-minute (companion to benchmark_06_codeformer.py).

Loads the LatentSync 1.6 pipeline ONCE (same loader as resident_vram_probe.py) and runs the
production pilot config (stage2_512, 20 steps, guidance 1.5, DeepCache, fp16, seed 1247) N times
on a real video + its own audio. Run 1 is the in-process cold run (first CUDA kernels/cuDNN
autotune, insightface init), runs 2..N are steady-state. Model load is timed separately and is
NOT inside the inference wall time. Each run is sampled with the nvidia-smi process-tree sampler.

    source scripts/env.sh
    python scripts/pilot/benchmark_06_latentsync.py test_videos/04.mp4 --audio /tmp/x.wav --repeat 2 \
        --out-dir /tmp/dabai_pilot_06

Writes reports/pilot/0.6_latentsync.json; feed its steady s/video-min into
benchmark_06_codeformer.py --latentsync-s-per-min.
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT, THIRD_PARTY, gpu_info, write_json
from measure_vram import measure_gpu_memory
from benchmark_06_codeformer import probe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--audio", required=True, help="audio track to drive lipsync (extracted beforehand, not timed)")
    ap.add_argument("--repeat", type=int, default=2, help="run 1 = in-process cold, rest = steady-state")
    ap.add_argument("--out-dir", default="/tmp/dabai_pilot_06")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=1247)
    ap.add_argument("--json", default=str(PILOT / "0.6_latentsync.json"))
    a = ap.parse_args()

    import torch
    from omegaconf import OmegaConf
    from DeepCache import DeepCacheSDHelper
    from accelerate.utils import set_seed
    from resident_vram_probe import load_latentsync, snap

    src = pathlib.Path(a.video).resolve(); out_dir = pathlib.Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    info = probe(src)
    ls = THIRD_PARTY / "latentsync"; cfg = OmegaConf.load(ls / "configs/unet/stage2_512.yaml")
    config = {"unet_config": "configs/unet/stage2_512.yaml", "resolution": int(cfg.data.resolution),
              "num_frames": int(cfg.data.num_frames), "inference_steps": a.steps, "guidance_scale": a.guidance,
              "deepcache": True, "dtype": "float16", "seed": a.seed}

    with measure_gpu_memory(f"pilot0.6/latentsync/{src.stem}/model-load", note="pipeline load only") as ml:
        pipe = load_latentsync()
    helper = DeepCacheSDHelper(pipe=pipe); helper.set_params(cache_interval=3, cache_branch_id=0); helper.enable()

    runs = []
    for i in range(1, a.repeat + 1):
        kind = "cold" if i == 1 else "warm"
        out = out_dir / f"latentsync_{src.stem}_r{i}.mp4"
        set_seed(a.seed)
        old = os.getcwd(); os.chdir(ls)
        try:
            with measure_gpu_memory(f"pilot0.6/latentsync/{src.stem}/inference-{kind}-r{i}",
                                    note="in-process, model already loaded") as m:
                pipe(video_path=str(src), audio_path=str(a.audio), video_out_path=str(out),
                     num_frames=cfg.data.num_frames, num_inference_steps=a.steps, guidance_scale=a.guidance,
                     weight_dtype=torch.float16, width=cfg.data.resolution, height=cfg.data.resolution,
                     mask_image_path=cfg.data.mask_image_path, temp_dir=str(out_dir / f"ls_temp_r{i}"))
        finally:
            os.chdir(old)
        o = probe(out) if out.exists() else {}
        sec = float(m.result["seconds"])
        runs.append({"run": i, "kind": kind, "wall_s": round(sec, 3), "model_load_included": False,
                     "s_per_video_min": round(sec / info["duration_s"] * 60, 3),
                     "peak_device_mib": m.result["peak_device_mib"], "peak_process_mib": m.result.get("peak_process_mib"),
                     "torch_peak_reserved_mib": m.result.get("torch_peak_reserved_mib"),
                     "output": str(out), "output_probe": o, "ok": m.result["ok"]})
        print(runs[-1], flush=True)

    warm = [r for r in runs if r["kind"] == "warm" and r["ok"]]
    steady = min(warm, key=lambda r: r["wall_s"]) if warm else None
    rep = {"task": "0.6", "component": "latentsync", "gpu": gpu_info(), "video": str(src), "audio": str(a.audio),
           "source": info, "config": config,
           "model_load": {"seconds": ml.result["seconds"], "peak_device_mib": ml.result["peak_device_mib"],
                          "device_used_after_load_mib": snap()["nvidia_smi_used_mib"]},
           "runs": runs,
           "steady_state": {"wall_s": steady["wall_s"], "s_per_video_min": steady["s_per_video_min"],
                            "peak_device_mib": steady["peak_device_mib"]} if steady else None,
           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(pathlib.Path(a.json), rep)
    print(json.dumps({k: rep[k] for k in ("model_load", "steady_state")}, indent=1))
    print(f"-> {a.json}")
    return 0 if all(r["ok"] for r in runs) else 1


if __name__ == "__main__":
    raise SystemExit(main())
