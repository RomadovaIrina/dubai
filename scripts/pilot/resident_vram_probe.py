#!/usr/bin/env python3
"""Pilot 0.4 — real simultaneous model-residency probe.

Loads the GPU models into ONE process and keeps every successfully loaded model alive.
This answers the actual 0.4 question more directly than summing isolated nvidia-smi peaks.
It does not run full inference; combine this steady-residency measurement with the isolated
inference peaks from measure_vram.py/run_vram_suite.py to reserve transient headroom.

Examples:
  source scripts/env.sh
  python scripts/pilot/resident_vram_probe.py --profile audio
  python scripts/pilot/resident_vram_probe.py --profile video
  python scripts/pilot/resident_vram_probe.py --profile all --target-gib 32
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import MODELS, PILOT, ROOT, THIRD_PARTY, write_json

MIB = 1024 * 1024


def device_used_mib() -> int:
    p = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    try:
        return int(p.stdout.strip().splitlines()[0])
    except Exception:
        return 0


def snap() -> dict:
    import torch
    free, total = torch.cuda.mem_get_info()
    return {
        "nvidia_smi_used_mib": device_used_mib(),
        "torch_allocated_mib": round(torch.cuda.memory_allocated() / MIB, 1),
        "torch_reserved_mib": round(torch.cuda.memory_reserved() / MIB, 1),
        "cuda_free_mib": round(free / MIB, 1),
        "cuda_total_mib": round(total / MIB, 1),
    }


def load_whisper():
    import torch  # map CUDA libs first
    from faster_whisper import WhisperModel
    return WhisperModel(str(MODELS / "whisper-large-v3"), device="cuda",
                        compute_type="int8_float16")


def load_retinaface():
    from facexlib.detection import init_detection_model
    return init_detection_model("retinaface_resnet50", half=True, device="cuda",
                                model_rootpath=str(MODELS / "facexlib"))


def load_pyannote():
    import torch
    from pyannote.audio import Pipeline
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        raise RuntimeError("BLOCKED: HF_TOKEN is not set")
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=tok)
    if pipe is None:
        raise RuntimeError("BLOCKED: gated pyannote model access not accepted")
    pipe.to(torch.device("cuda"))
    return pipe


def load_chatterbox():
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    m = ChatterboxMultilingualTTS.from_local(MODELS / "chatterbox", "cuda", t3_model="v3")
    for mod in (m.t3, m.s3gen, m.ve):
        mod.to(torch.float16)
    return m


def load_latentsync():
    import torch
    from omegaconf import OmegaConf
    from diffusers import AutoencoderKL, DDIMScheduler

    ls = THIRD_PARTY / "latentsync"
    sys.path.insert(0, str(ls))
    from latentsync.models.unet import UNet3DConditionModel
    from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
    from latentsync.whisper.audio2feature import Audio2Feature

    cfg = OmegaConf.load(ls / "configs/unet/stage2_512.yaml")
    dtype = torch.float16
    old = pathlib.Path.cwd()
    try:
        os.chdir(ls)
        scheduler = DDIMScheduler.from_pretrained("configs")
        audio_encoder = Audio2Feature(
            model_path=str(ls / "checkpoints/whisper/tiny.pt"), device="cuda",
            num_frames=cfg.data.num_frames, audio_feat_length=cfg.data.audio_feat_length,
        )
        vae = AutoencoderKL.from_pretrained(str(MODELS / "sd-vae-ft-mse"), torch_dtype=dtype)
        vae.config.scaling_factor = 0.18215
        vae.config.shift_factor = 0
        unet, _ = UNet3DConditionModel.from_pretrained(
            OmegaConf.to_container(cfg.model), str(ls / "checkpoints/latentsync_unet.pt"),
            device="cpu")
        unet = unet.to(dtype=dtype)
        pipe = LipsyncPipeline(vae=vae, audio_encoder=audio_encoder,
                               unet=unet, scheduler=scheduler).to("cuda")
        return pipe
    finally:
        os.chdir(old)


def load_codeformer():
    import torch
    cf = THIRD_PARTY / "CodeFormer"
    sys.path.insert(0, str(cf))
    from basicsr.utils.registry import ARCH_REGISTRY
    # Import registers CodeFormer architecture.
    import basicsr.archs.codeformer_arch  # noqa: F401
    from facelib.utils.face_restoration_helper import FaceRestoreHelper

    device = torch.device("cuda")
    net = ARCH_REGISTRY.get("CodeFormer")(
        dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
        connect_list=["32", "64", "128", "256"]).to(device)
    ckpt = torch.load(cf / "weights/CodeFormer/codeformer.pth", map_location="cpu")
    net.load_state_dict(ckpt["params_ema"])
    net.eval()
    helper = FaceRestoreHelper(
        1, face_size=512, crop_ratio=(1, 1), det_model="retinaface_resnet50",
        save_ext="png", use_parse=True, device=device)
    return {"net": net, "helper": helper}


LOADERS = {
    "whisper": load_whisper,
    "retinaface": load_retinaface,
    "pyannote": load_pyannote,
    "chatterbox": load_chatterbox,
    "latentsync": load_latentsync,
    "codeformer": load_codeformer,
}
PROFILES = {
    "audio": ["whisper", "retinaface", "pyannote", "chatterbox"],
    "video": ["latentsync", "codeformer"],
    "all": ["whisper", "retinaface", "pyannote", "chatterbox", "latentsync", "codeformer"],
}


def infer_transient_peak_mib(rows_path: pathlib.Path) -> dict[str, float]:
    if not rows_path.exists():
        return {}
    aliases = {
        "whisper": "whisper", "faster-whisper": "whisper",
        "retinaface": "retinaface", "pyannote": "pyannote",
        "chatterbox": "chatterbox", "latentsync": "latentsync",
        "codeformer": "codeformer",
    }
    best: dict[str, float] = {}
    for line in rows_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        label = str(r.get("label", "")).lower()
        for needle, key in aliases.items():
            if needle in label:
                v = float(r.get("peak_device_delta_mib") or 0)
                best[key] = max(best.get(key, 0), v)
    return best


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=PROFILES, default="all")
    ap.add_argument("--models", nargs="*", choices=LOADERS,
                    help="override profile with an explicit ordered model list")
    ap.add_argument("--target-gib", type=float, default=32.0)
    ap.add_argument("--headroom-gib", type=float, default=3.0,
                    help="VRAM that must remain free for transient buffers")
    ap.add_argument("--json", default=str(PILOT / "0.4_residency.json"))
    a = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("FAIL: CUDA unavailable", file=sys.stderr)
        return 1
    torch.cuda.empty_cache(); gc.collect()

    order = a.models or PROFILES[a.profile]
    base = snap()
    kept: dict[str, object] = {}
    rows = []
    blocked = []
    oom = None
    print(f"GPU: {torch.cuda.get_device_name(0)} total={base['cuda_total_mib']/1024:.2f} GiB")
    print(f"profile={a.profile} models={order} target={a.target_gib:.1f} GiB headroom={a.headroom_gib:.1f} GiB")

    prev_used = base["nvidia_smi_used_mib"]
    for name in order:
        print(f"\n== load {name} ==")
        t0 = time.perf_counter()
        try:
            obj = LOADERS[name]()
            torch.cuda.synchronize()
            s = snap()
            row = {
                "model": name, "status": "PASS", "seconds": round(time.perf_counter()-t0, 3),
                "incremental_device_mib": max(0, s["nvidia_smi_used_mib"] - prev_used),
                **s,
            }
            rows.append(row)
            kept[name] = obj
            prev_used = s["nvidia_smi_used_mib"]
            print(row)
        except torch.cuda.OutOfMemoryError as e:
            oom = name
            rows.append({"model": name, "status": "OOM", "error": str(e)[:500]})
            print(f"OOM while loading {name}: {e}")
            break
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            status = "BLOCKED" if "BLOCKED:" in msg else "FAIL"
            rows.append({"model": name, "status": status, "error": msg[:700]})
            print(f"{status} {name}: {msg}")
            if status == "BLOCKED":
                blocked.append(name)
                continue
            break

    final = snap()
    steady_delta = max(0, final["nvidia_smi_used_mib"] - base["nvidia_smi_used_mib"])
    target_mib = a.target_gib * 1024
    headroom_mib = a.headroom_gib * 1024

    isolated_peaks = infer_transient_peak_mib(PILOT / "vram.jsonl")
    incremental = {r["model"]: float(r.get("incremental_device_mib") or 0)
                   for r in rows if r.get("status") == "PASS"}
    transient_extra = {
        k: max(0.0, isolated_peaks.get(k, 0.0) - incremental.get(k, 0.0))
        for k in incremental
    }
    max_transient_extra = max(transient_extra.values(), default=0.0)
    conservative_required = steady_delta + max_transient_extra + headroom_mib

    if oom:
        verdict = "UNLOAD_OR_REDUCE_RESIDENCY"
        closed = True
        reason = f"simultaneous residency OOM at {oom} on a GPU with {base['cuda_total_mib']/1024:.1f} GiB"
    elif steady_delta + headroom_mib > target_mib:
        verdict = "UNLOAD_BETWEEN_CONTOURS"
        closed = True
        reason = "steady simultaneous residency alone exceeds the 32 GiB target after headroom"
    elif isolated_peaks and conservative_required <= target_mib:
        verdict = "RESIDENT_SAFE"
        closed = True
        reason = "measured steady residency + largest observed transient + headroom fits target"
    elif blocked:
        verdict = "INCOMPLETE_BLOCKED_MODELS"
        closed = False
        reason = f"blocked models were not included: {', '.join(blocked)}"
    else:
        verdict = "LIKELY_RESIDENT_CONFIRM_TRANSIENTS"
        closed = False
        reason = "steady residency fits target, but isolated inference transient data is incomplete"

    rep = {
        "task": "0.4", "profile": a.profile, "models": order,
        "gpu": torch.cuda.get_device_name(0), "capability": torch.cuda.get_device_capability(0),
        "actual_gpu_total_mib": base["cuda_total_mib"],
        "target_mib": target_mib, "headroom_mib": headroom_mib,
        "baseline": base, "rows": rows, "final": final,
        "steady_resident_delta_mib": steady_delta,
        "isolated_peak_delta_mib": isolated_peaks,
        "estimated_transient_extra_mib": transient_extra,
        "max_transient_extra_mib": max_transient_extra,
        "conservative_required_mib": conservative_required,
        "verdict": verdict, "closed": closed, "reason": reason,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(pathlib.Path(a.json), rep)

    md = ["# Pilot 0.4 — VRAM residency", "",
          f"**Verdict:** `{verdict}`  ", f"**Closed:** {'YES' if closed else 'NO'}  ",
          f"**Reason:** {reason}", "",
          f"Actual GPU: **{rep['actual_gpu_total_mib']/1024:.2f} GiB**  ",
          f"Target: **{a.target_gib:.2f} GiB**, reserved headroom: **{a.headroom_gib:.2f} GiB**  ",
          f"Steady simultaneous residency delta: **{steady_delta/1024:.2f} GiB**  ",
          f"Conservative required (steady + max transient + headroom): **{conservative_required/1024:.2f} GiB**", "",
          "| model | status | incremental load MiB | device used MiB | load s |",
          "|---|---|---:|---:|---:|"]
    for r in rows:
        md.append(f"| {r['model']} | {r['status']} | {r.get('incremental_device_mib','-')} | {r.get('nvidia_smi_used_mib','-')} | {r.get('seconds','-')} |")
    (PILOT / "0.4_residency.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    print("\n" + "\n".join(md))
    print(f"\n-> {a.json}\n-> reports/pilot/0.4_residency.md")
    # Keep references alive until files are written; process exit frees them.
    return 0 if closed else 2


if __name__ == "__main__":
    raise SystemExit(main())
