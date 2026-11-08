#!/usr/bin/env python3
"""r16 T: the weights the e2e face path needs, checked at the start of a job (audit 08.11: a missing RealESRGAN_x2plus.pth made every
LatentSync face_comp fail and the job quietly fell back to the source picture; a missing codeformer.pth crashed the beard route).
    weights_check.py --face auto|keysync|latentsync --nc 0|1
stdout: technical flags (face_latentsync_unavailable, codeformer_missing_clean_face); stderr: one line per missing file.
exit 0 = fine; 2 = the requested face path cannot run (KeySync weights missing for auto / keysync; LatentSync weights missing for latentsync)."""
import argparse, os, pathlib, sys
ap = argparse.ArgumentParser(); ap.add_argument("--face", default="auto"); ap.add_argument("--nc", default="1"); a = ap.parse_args()
R = pathlib.Path(__file__).resolve().parents[3]; M = R / "models"
KS = pathlib.Path(os.environ.get("KS_ROOT", "/workspace/dub/keysync")); KM = M / "keysync"
lora = pathlib.Path(os.environ.get("KS_LORA", str(KS / "student/lora_pick.pt"))); cap = pathlib.Path(os.environ.get("KS_TRT_CAP", str(KS / "trt/cap")))
ks = [KS / "code/pretrained_models/checkpoints" / f for f in ("keyframe_dub.pt", "interpolation_dub.pt", "WavLM-Base+.pt")] + \
     [lora, lora.parent / "lora_meta.json", cap / "kf.pt", cap / "interp.pt", M / "mediapipe/face_landmarker.task", KS / "venv/bin/python"]
import glob   # r16 T (Q audit O2): sync_perframe.py (probe + route) needs SyncNet + S3FD; ks_fast needs HuBERT + the SVD VAE from the HF cache
AUX = R / "third_party/latentsync/checkpoints/auxiliary"; HF = pathlib.Path(os.environ.get("HF_HOME", str(M / "hf"))) / "hub"
hf = lambda repo, f: next(iter(sorted(glob.glob(str(HF / f"models--{repo.replace('/', '--')}/snapshots/*/{f}")))), str(HF / f"models--{repo.replace('/', '--')}/snapshots/MISSING/{f}"))
ks += [AUX / "syncnet_v2.model", AUX / "sfd_face.pth", pathlib.Path(hf("facebook/hubert-base-ls960", "pytorch_model.bin")),
       pathlib.Path(hf("stabilityai/stable-video-diffusion-img2vid", "vae/diffusion_pytorch_model.fp16.safetensors"))]
ls = [M / "latentsync/latentsync_unet.pt", M / "realesrgan/RealESRGAN_x2plus.pth", M / "mediapipe/face_landmarker.task"]
cf = R / "third_party/CodeFormer/weights/CodeFormer/codeformer.pth"
miss = lambda L: [str(p) for p in L if not (p.exists() and (p.is_dir() or p.stat().st_size > 0))]   # (exists() follows symlinks: a dangling volume link = missing)
mk, ml = miss(ks), miss(ls); flags = []; rc = 0
for f in mk: print(f"WEIGHTS MISSING (KeySync): {f}", file=sys.stderr)
for f in ml: print(f"WEIGHTS MISSING (LatentSync): {f}", file=sys.stderr)
if a.face in ("auto", "keysync") and mk: rc = 2
if a.face == "latentsync" and ml: rc = 2
if a.face == "auto" and ml and not mk:
    flags.append("face_latentsync_unavailable"); print("WEIGHTS: LatentSync unavailable -> a bearded speaker is lip-synced with KeySync (flagged)", file=sys.stderr)
if a.face in ("auto", "latentsync") and a.nc == "1" and not cf.exists():
    print(f"WEIGHTS: {cf} missing -> a LatentSync face would run without CodeFormer (the clean face; e2e flags codeformer_missing_clean_face if it does)", file=sys.stderr)
print(" ".join(flags)); sys.exit(rc)
