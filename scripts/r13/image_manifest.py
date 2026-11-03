#!/usr/bin/env python3
"""r13 spec 5.1 hand-off: the model / weights manifest of the CURRENT clean pipeline, measured on a working server (sizes, HF revisions,
sha256 of the small single files) + the licence of each. Writes models_manifest.json and MODELS.md.
    python scripts/r13/image_manifest.py --out reports/experiments/r13-spec/IMAGE"""
import argparse, hashlib, json, os, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[2]; W = pathlib.Path("/workspace/dub")
# name, local path, source, licence, used by, status
M = [
    ("faster-whisper large-v3", "models/whisper-large-v3", "hf:Systran/faster-whisper-large-v3", "MIT", "ASR (6.1), TTS QA, subtitle checks", "required"),
    ("Qwen2.5-14B-Instruct GGUF Q5_K_M", "models/qwen2.5-14b-instruct-gguf", "hf:Qwen/Qwen2.5-14B-Instruct-GGUF (*q5_k_m*)", "Apache-2.0", "translation / duration-aware rewrites / judge (6.4)", "required"),
    ("Qwen2.5-7B-Instruct GGUF Q8_0", "models/qwen2.5-7b-instruct-gguf", "hf:Qwen/Qwen2.5-7B-Instruct-GGUF", "Apache-2.0", "NOT used any more (14B on the GPU since 29.10)", "drop from the image (DUB_SKIP_QWEN7B=1)"),
    ("Chatterbox Multilingual v3", "models/chatterbox", "hf:ResembleAI/chatterbox (ve.pt, s3gen.pt, conds.pt, t3_mtl23ls_v3.safetensors, tokenizers)", "MIT", "TTS (7.2), VC (7.4)", "required"),
    ("LatentSync 1.6", "models/latentsync", "hf:ByteDance/LatentSync-1.6 (latentsync_unet.pt, whisper/tiny.pt, auxiliary/syncnet_v2.model, auxiliary/sfd_face.pth)", "OpenRAIL++ (user-approved)", "lip-sync (8.1), SyncNet metric (12.1)", "required"),
    ("sd-vae-ft-mse", "models/sd-vae-ft-mse", "hf:stabilityai/sd-vae-ft-mse", "MIT", "LatentSync VAE", "required"),
    ("wav2vec2-base-960h", "models/wav2vec2-base-960h", "hf:facebook/wav2vec2-base-960h", "Apache-2.0", "TTS QA (CTC), subtitle word timing (9.2), tempo audit", "required"),
    ("MediaPipe Face Landmarker", "models/mediapipe", "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task (sha256 pinned)", "Apache-2.0", "faces, head pose (5.4), crops", "required"),
    ("Spleeter 2stems", "models/spleeter", "https://github.com/deezer/spleeter/releases/download/v1.4.0/2stems.tar.gz (-> torch npz)", "MIT", "the Spleeter background bed", "required"),
    ("Real-ESRGAN x2plus", "models/realesrgan", "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth (sha256 pinned)", "BSD-3-Clause", "composite layer SR", "required"),
    ("CodeFormer + ParseNet", "third_party/CodeFormer/weights", "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/{codeformer,parsing_parsenet}.pth", "S-Lab License 1.0 = NON-COMMERCIAL", "mouth detail (8.3), DUB_ALLOW_NC=1", "LICENCE DEVIATION, user-accepted 29.10"),
    ("htdemucs (demucs 4.0.1)", "/workspace/dub/quarantine_nc/torch", "pip demucs==4.0.1 (code MIT) + its pretrained htdemucs checkpoint (dl.fbaipublicfiles.com)", "weights: not cleared for commercial use (treated as NC)", "the htdemucs (main) background bed, DUB_ALLOW_NC_BED=1", "LICENCE DEVIATION, user-accepted"),
    ("pyannote speaker-diarization-3.1", "models/hf (HF_HOME cache)", "hf:pyannote/speaker-diarization-3.1 + segmentation-3.0 + wespeaker-voxceleb-resnet34-LM (GATED: HF token + accepted terms)", "MIT", "diarization (6.3)", "required: pre-fetch into HF_HOME at build time (5.1: no download at start)"),
    ("Silero VAD", "pip silero-vad (weights inside the package)", "pip silero-vad", "MIT", "VAD (5.2, 6.2 guard), speech gate", "required"),
    ("KeySync", "/workspace/dub/keysync (own venv)", "git https://github.com/antonibigata/keysync @5e2503b + ks_sm120.patch; weights ~13 GB (scripts/keysync/ on experiment/r13-keysync)", "Stability AI Community License (< $1M revenue, registration)", "the alternative lip-sync generator (KeySync agent)", "OPTIONAL; LICENCE DEVIATION, user-accepted"),
]


def size(p: pathlib.Path) -> int:
    if p.is_file(): return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file() and ".cache" not in f.parts) if p.exists() else 0


def rev(p: pathlib.Path):
    f = next(iter(p.glob(".cache/huggingface/download/**/*.metadata")), None) if p.is_dir() else None
    return f.read_text().splitlines()[0] if f else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name, path, src, lic, use, status in M:
        p = pathlib.Path(path) if path.startswith("/") else ROOT / path
        if path.startswith("models/hf"): p = ROOT / "models" / "hf"
        files = []
        if p.is_dir() and p.exists() and size(p) < 2e8:   # small dirs: per-file sha256 (the big ones are pinned by HF revision)
            files = [{"file": str(f.relative_to(p)), "bytes": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest()} for f in sorted(p.rglob("*")) if f.is_file() and ".cache" not in f.parts and f.stat().st_size > 1e5]
        rows.append({"name": name, "path": path, "source": src, "hf_revision": rev(p) if p.exists() else None, "bytes": size(p), "gb": round(size(p) / 1e9, 2),
                     "licence": lic, "used_by": use, "status": status, "present_on_reference_server": p.exists(), "files_sha256": files})
    (out / "models_manifest.json").write_text(json.dumps(rows, indent=1))
    L = ["# Models and weights of the current clean pipeline (measured on r13-t, 03.11)", "",
         "| Model | Size | Licence | Source (HF revision) | Used by | Status |", "|---|---:|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['name']} | {r['gb']} GB | {r['licence']} | {r['source']}{' @' + r['hf_revision'][:10] if r['hf_revision'] else ''} | {r['used_by']} | {r['status']} |")
    req = sum(r["bytes"] for r in rows if r["status"].startswith(("required", "LICENCE")) and "OPTIONAL" not in r["status"])
    L += ["", f"Total of the required weights present here: **{req / 1e9:.1f} GB** (without the dropped Qwen 7B and the optional KeySync).",
          "Per-file sha256 of the small directories: `models_manifest.json`."]
    (out / "MODELS.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
