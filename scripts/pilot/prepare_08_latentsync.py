#!/usr/bin/env python3
"""Pilot 0.8 — produce identical LatentSync outputs for the real-content videos (preparation step).

For every input video: ffprobe metadata, own audio track (extracted beforehand by the caller or here, never timed),
then LatentSync 1.6 with the fixed pilot config (stage2_512, 20 steps, guidance 1.5, DeepCache, fp16, seed 1247).
The pipeline is loaded once (same loader as resident_vram_probe.py / benchmark_06_latentsync.py). Existing outputs
can be passed with --reuse NAME:MP4 (e.g. the 0.6 output for 04) and are symlinked instead of recomputed.

    python scripts/pilot/prepare_08_latentsync.py test_videos/01.MP4 ... --out-dir /tmp/dabai_pilot_08 \
        --reuse 04:/tmp/dabai_pilot_06/latentsync_04.mp4

Writes <out-dir>/latentsync_<name>.mp4 and reports/pilot/0.8_latentsync_prep.json. Media never goes into the repo.
"""
from __future__ import annotations
import argparse, json, os, pathlib, subprocess, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT, THIRD_PARTY, gpu_info, write_json, sh
from benchmark_06_codeformer import probe


def extract_audio(video: pathlib.Path, wav: pathlib.Path) -> None:
    if wav.exists():
        return
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-acodec", "pcm_s16le",
                    "-ar", "16000", "-ac", "1", str(wav)], check=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out-dir", default="/tmp/dabai_pilot_08")
    ap.add_argument("--reuse", nargs="*", default=[], metavar="NAME:MP4")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--seed", type=int, default=1247)
    ap.add_argument("--json", default=str(PILOT / "0.8_latentsync_prep.json"))
    a = ap.parse_args()
    out_dir = pathlib.Path(a.out_dir); (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    reuse = {x.split(":", 1)[0]: pathlib.Path(x.split(":", 1)[1]) for x in a.reuse}

    import torch
    from omegaconf import OmegaConf
    from DeepCache import DeepCacheSDHelper
    from accelerate.utils import set_seed
    from resident_vram_probe import load_latentsync
    ls = THIRD_PARTY / "latentsync"; cfg = OmegaConf.load(ls / "configs/unet/stage2_512.yaml")
    config = {"unet_config": "configs/unet/stage2_512.yaml", "resolution": int(cfg.data.resolution),
              "num_frames": int(cfg.data.num_frames), "inference_steps": a.steps, "guidance_scale": a.guidance,
              "deepcache": True, "dtype": "float16", "seed": a.seed,
              "commit": sh(f"git -C {ls} rev-parse HEAD")}
    pipe = None; helper = None
    rows = []
    for v in a.videos:
        src = pathlib.Path(v).resolve(); name = src.stem
        out = out_dir / f"latentsync_{name}.mp4"
        row = {"name": name, "input": str(src), "input_meta": probe(src), "output": str(out)}
        if name in reuse and reuse[name].exists():
            if out.is_symlink() or out.exists():
                out.unlink()
            out.symlink_to(reuse[name].resolve())
            row.update(reused=str(reuse[name]), latentsync_wall_s=None)
        else:
            wav = out_dir / "audio" / f"{name}_audio.wav"; extract_audio(src, wav)
            if pipe is None:
                t0 = time.perf_counter(); pipe = load_latentsync()
                helper = DeepCacheSDHelper(pipe=pipe); helper.set_params(cache_interval=3, cache_branch_id=0); helper.enable()
                config["model_load_s"] = round(time.perf_counter() - t0, 2)
            set_seed(a.seed)
            old = os.getcwd(); os.chdir(ls)
            t0 = time.perf_counter()
            try:
                pipe(video_path=str(src), audio_path=str(wav), video_out_path=str(out),
                     num_frames=cfg.data.num_frames, num_inference_steps=a.steps, guidance_scale=a.guidance,
                     weight_dtype=torch.float16, width=cfg.data.resolution, height=cfg.data.resolution,
                     mask_image_path=cfg.data.mask_image_path, temp_dir=str(out_dir / f"ls_temp_{name}"))
            except Exception as e:  # e.g. upstream "Face not detected": record it, keep going with the other videos
                row.update(error=f"{type(e).__name__}: {e}", latentsync_wall_s=round(time.perf_counter() - t0, 2))
                print(f"FAIL {name}: {row['error']}", flush=True)
            finally:
                os.chdir(old)
            row.update(audio=str(wav), peak_torch_reserved_mib=round(torch.cuda.max_memory_reserved() / 2**20, 1))
            if "error" not in row:
                row["latentsync_wall_s"] = round(time.perf_counter() - t0, 2)
        row["output_meta"] = probe(out) if out.exists() else None
        row["ok"] = bool(row["output_meta"]) and row["output_meta"]["frames"] > 0
        if row.get("latentsync_wall_s") and row["input_meta"]["duration_s"]:
            row["s_per_video_min"] = round(row["latentsync_wall_s"] / row["input_meta"]["duration_s"] * 60, 2)
        rows.append(row); print(json.dumps(row), flush=True)
    rep = {"task": "0.8", "step": "latentsync-preparation", "gpu": gpu_info(), "config": config, "rows": rows,
           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(pathlib.Path(a.json), rep); print(f"-> {a.json}")
    return 0 if all(r["ok"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
