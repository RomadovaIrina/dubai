#!/usr/bin/env python3
"""DabAI RunPod Serverless worker — minimal handler for Pilot 0.10 (cold-start readiness).

READY (Pilot 0.10 definition): the worker is running, CUDA is usable, every production weight is locally available on
the network volume and a real dubbing job could start WITHOUT downloading anything. READY does not load the models into
VRAM and does not run inference — 0.10 measures start-to-ready, not the first completed job.

Actions (job["input"]["action"]):
  ready   full readiness validation -> {"ready": bool, "gpu": ..., "sm": ..., "models_root": ..., "missing": [...], ...}
  info    image / versions / environment summary (no GPU work)

CLI (also used at image build time and by the local smoke test):
  handler.py --check          readiness JSON on stdout, exit 0 iff ready
  handler.py --info           info JSON on stdout
  handler.py --import-smoke   import every production module (build-time proof, no GPU / volume needed)
  handler.py --mock-volume D  create a dummy volume tree under D (1-byte files) for entrypoint/handler smoke tests
  handler.py [runpod flags]   start the worker (runpod.serverless.start); e.g. --rp_serve_api, --test_input '{...}'

Secrets are never read or reported. Only the whitelisted environment variables below appear in any output.
"""
from __future__ import annotations

import argparse
import glob
import importlib
import importlib.metadata as md
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import time

MODELS_ROOT = pathlib.Path(os.environ.get("MODELS_ROOT", "/runpod-volume/models"))
DUB_ROOT = pathlib.Path(os.environ.get("DUB_ROOT", "/opt/dabai"))
LS_ROOT = DUB_ROOT / "third_party" / "latentsync"
LS_CKPT = LS_ROOT / "checkpoints"
BUILD_INFO = pathlib.Path(__file__).with_name("BUILD_INFO.json")
# Pilot 0.10 target topology is RTX 5090 / sm_120. A worker on another GPU is a topology violation for the benchmark,
# so it is reported as NOT ready unless DABAI_REQUIRE_SM=any (set that on non-Blackwell test endpoints only).
REQUIRE_SM = os.environ.get("DABAI_REQUIRE_SM", "sm_120").strip().lower()

# Mandatory production weights, relative to MODELS_ROOT. Source of truth: MODEL_VOLUME_MANIFEST.json (2026-09-20) and
# the loaders in scripts/pilot/run_clean_pipeline_05.py, resident_vram_probe.py, latentsync/utils/face_detector.py.
MANDATORY: dict[str, list[str]] = {
    "whisper": ["whisper-large-v3/model.bin", "whisper-large-v3/config.json", "whisper-large-v3/tokenizer.json",
                "whisper-large-v3/vocabulary.json", "whisper-large-v3/preprocessor_config.json"],
    "chatterbox": ["chatterbox/ve.pt", "chatterbox/s3gen.pt", "chatterbox/conds.pt", "chatterbox/t3_mtl23ls_v3.safetensors",
                   "chatterbox/grapheme_mtl_merged_expanded_v1.json"],
    "latentsync": ["latentsync/latentsync_unet.pt", "latentsync/whisper/tiny.pt"],
    "sd-vae-ft-mse": ["sd-vae-ft-mse/config.json", "sd-vae-ft-mse/diffusion_pytorch_model.safetensors"],
    "retinaface": ["facexlib/detection_Resnet50_Final.pth"],
    "insightface": ["insightface/models/buffalo_l/det_10g.onnx", "insightface/models/buffalo_l/2d106det.onnx"],
    "pkuseg": ["pkuseg/spacy_ontonotes.zip", "pkuseg/spacy_ontonotes/features.msgpack", "pkuseg/spacy_ontonotes/weights.npz"],
}
QWEN_DIR, QWEN_GLOB = "qwen2.5-7b-instruct-gguf", "qwen2.5-7b-instruct-q8_0*.gguf"
QWEN_SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")
# HF-cache-layout artifacts: (cache root relative to MODELS_ROOT, repo, files that the loader resolves offline)
HF_CACHE_ITEMS: list[tuple[str, str, list[str]]] = [
    ("hf/hub", "pyannote/speaker-diarization-3.1", ["config.yaml"]),
    ("hf/hub", "pyannote/segmentation-3.0", ["pytorch_model.bin", "config.yaml"]),
    ("hf/hub", "pyannote/wespeaker-voxceleb-resnet34-LM", ["pytorch_model.bin", "config.yaml"]),
    ("chatterbox", "ResembleAI/chatterbox", ["Cangjie5_TC.json"]),   # MTLTokenizer -> hf_hub_download(cache_dir=ckpt_dir)
]
REQUIRED_ENV = {
    "HF_HOME": str(MODELS_ROOT / "hf"),
    "HF_HUB_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "PYANNOTE_CACHE": str(MODELS_ROOT / "hf" / "hub"),   # pyannote.audio 3.4.0 CACHE_DIR (it does not read HF_HOME)
    "PKUSEG_HOME": str(MODELS_ROOT / "pkuseg"),          # spacy_pkuseg: $PKUSEG_HOME/spacy_ontonotes.zip must exist
}
SYMLINKS = {
    DUB_ROOT / "models": MODELS_ROOT,
    LS_CKPT / "latentsync_unet.pt": MODELS_ROOT / "latentsync" / "latentsync_unet.pt",
    LS_CKPT / "whisper": MODELS_ROOT / "latentsync" / "whisper",
    LS_CKPT / "auxiliary": MODELS_ROOT / "insightface",
}
REPORTED_ENV = ["MODELS_ROOT", "DUB_ROOT", "DUB_ENV", "HF_HOME", "HF_HUB_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "PYANNOTE_CACHE",
                "PKUSEG_HOME", "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "DUB_THREADS", "OMP_NUM_THREADS", "DABAI_REQUIRE_SM",
                "RUNPOD_ENDPOINT_ID", "RUNPOD_POD_ID", "RUNPOD_DC_ID", "RUNPOD_GPU_COUNT"]
IMPORT_SMOKE = [  # module, distribution for the version (None = module.__version__)
    ("torch", None), ("torchvision", None), ("torchaudio", None), ("numpy", None), ("cv2", "opencv-python"),
    ("transformers", None), ("huggingface_hub", None), ("diffusers", None), ("accelerate", None), ("safetensors", None),
    ("silero_vad", "silero-vad"), ("faster_whisper", "faster-whisper"), ("ctranslate2", None),
    ("pyannote.audio", "pyannote.audio"), ("facexlib.detection", "facexlib"), ("insightface.app", "insightface"),
    ("onnxruntime", "onnxruntime-gpu"), ("llama_cpp", "llama-cpp-python"), ("chatterbox.mtl_tts", "chatterbox-tts"),
    ("spacy_pkuseg", "spacy-pkuseg"), ("pykakasi", None), ("perth", "resemble-perth"), ("decord", None), ("DeepCache", "deepcache"),
    ("kornia", None), ("omegaconf", None), ("imageio", None), ("soundfile", None), ("librosa", None), ("matplotlib", None),
    ("runpod", None),
    # DabAI production code + LatentSync inference path (sys.path is extended below)
    ("pilot_common", None), ("resident_vram_probe", None), ("latentsync_accel", None), ("retinaface_router", None),
    ("face_aware_latentsync", None), ("face_aware_latentsync_accel", None), ("run_clean_pipeline_05", None),
    ("latentsync.pipelines.lipsync_pipeline", None), ("latentsync.utils.image_processor", None),
    ("latentsync.whisper.audio2feature", None), ("latentsync.models.unet", None),
]


# ------------------------------------------------------------------------------------------------------------------
def _file_ok(p: pathlib.Path) -> str | None:
    """None if p is an existing non-empty regular file (symlinks followed), else the reason."""
    if not p.exists():
        return "missing" if not p.is_symlink() else "broken symlink"
    if not p.is_file():
        return "not a file"
    if p.stat().st_size == 0:
        return "zero bytes"
    return None


def check_volume() -> tuple[list[str], list[str], dict]:
    missing, warnings, info = [], [], {"models_root": str(MODELS_ROOT)}
    if not MODELS_ROOT.is_dir():
        return [f"models_root:{MODELS_ROOT} is not a directory (network volume not mounted?)"], warnings, info
    man = MODELS_ROOT / "MODEL_VOLUME_MANIFEST.json"
    if man.is_file():
        try:
            m = json.loads(man.read_text())
            info["manifest"] = {"volume": m.get("volume"), "updated_at": m.get("updated_at") or m.get("generated_at"),
                                "total_size_bytes": m.get("total_size_bytes"), "qwen_quant": m.get("qwen_quant")}
        except Exception as e:  # noqa: BLE001
            warnings.append(f"manifest unreadable: {type(e).__name__}")
    else:
        warnings.append("MODEL_VOLUME_MANIFEST.json missing on the volume")
    return missing, warnings, info


def check_weights() -> tuple[list[str], dict]:
    missing, detail = [], {}
    for comp, rels in MANDATORY.items():
        bad = [f"{r} ({why})" for r in rels if (why := _file_ok(MODELS_ROOT / r))]
        detail[comp] = "ok" if not bad else "missing"
        missing += [f"weight:{b}" for b in bad]
    # Qwen Q8_0: every shard of the -of-N set must be present (llama.cpp opens the first shard and follows the rest)
    shards = sorted(glob.glob(str(MODELS_ROOT / QWEN_DIR / QWEN_GLOB)))
    if not shards:
        missing.append(f"weight:{QWEN_DIR}/{QWEN_GLOB} (no Q8_0 shard found)"); detail["qwen_q8_0"] = "missing"
    else:
        expected = None
        for s in shards:
            m = QWEN_SHARD_RE.search(s)
            if m:
                expected = int(m.group(2))
        if expected:
            have = {int(QWEN_SHARD_RE.search(s).group(1)) for s in shards if QWEN_SHARD_RE.search(s)}
            absent = sorted(set(range(1, expected + 1)) - have)
            for i in absent:
                missing.append(f"weight:{QWEN_DIR}/qwen2.5-7b-instruct-q8_0-{i:05d}-of-{expected:05d}.gguf (missing)")
        for s in shards:
            if why := _file_ok(pathlib.Path(s)):
                missing.append(f"weight:{os.path.relpath(s, MODELS_ROOT)} ({why})")
        detail["qwen_q8_0"] = {"shards": [os.path.basename(s) for s in shards], "expected": expected,
                               "bytes": sum(pathlib.Path(s).stat().st_size for s in shards if pathlib.Path(s).exists())}
    return missing, detail


def check_hf_cache() -> tuple[list[str], dict]:
    """Resolve <cache>/models--org--name/refs/main -> snapshots/<commit>/<file>, exactly like hf_hub_download offline."""
    missing, detail = [], {}
    for cache_rel, repo, files in HF_CACHE_ITEMS:
        repo_dir = MODELS_ROOT / cache_rel / f"models--{repo.replace('/', '--')}"
        ref = repo_dir / "refs" / "main"
        if not ref.is_file():
            missing.append(f"hf:{repo} (no refs/main under {repo_dir})"); detail[repo] = "missing"; continue
        commit = ref.read_text().strip()
        snap = repo_dir / "snapshots" / commit
        bad = [f"{f} ({why})" for f in files if (why := _file_ok(snap / f))]
        missing += [f"hf:{repo}/{b}" for b in bad]
        detail[repo] = {"commit": commit[:12], "status": "ok" if not bad else "missing"}
    return missing, detail


def check_env() -> tuple[list[str], dict]:
    missing, seen = [], {}
    for k, want in REQUIRED_ENV.items():
        got = os.environ.get(k)
        seen[k] = got
        ok = (got or "").strip().lower() in ("1", "true", "yes", "on") if k == "HF_HUB_OFFLINE" else (got == want)
        if not ok:
            missing.append(f"env:{k}={want!r} expected, got {got!r}")
    return missing, seen


def check_symlinks() -> tuple[list[str], dict]:
    missing, detail = [], {}
    for lnk, target in SYMLINKS.items():
        if not lnk.is_symlink():
            missing.append(f"symlink:{lnk} -> {target} ({'exists but is not a symlink' if lnk.exists() else 'missing'})")
            detail[str(lnk)] = "missing"; continue
        if not lnk.exists():
            missing.append(f"symlink:{lnk} (broken, points to {os.readlink(lnk)})"); detail[str(lnk)] = "broken"; continue
        if lnk.resolve() != target.resolve():
            missing.append(f"symlink:{lnk} points to {os.readlink(lnk)}, expected {target}"); detail[str(lnk)] = "wrong target"
        else:
            detail[str(lnk)] = "ok"
    return missing, detail


def check_cuda() -> tuple[list[str], dict]:
    missing, info = [], {}
    t0 = time.perf_counter()
    try:
        import torch  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        return [f"torch:import failed: {type(e).__name__}: {e}"], info
    info.update(torch=torch.__version__, torch_cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                torch_import_s=round(time.perf_counter() - t0, 3))
    if not torch.cuda.is_available():
        missing.append("cuda:torch.cuda.is_available() is False")
        return missing, info
    try:
        p = torch.cuda.get_device_properties(0)
        sm = f"sm_{p.major}{p.minor}"
        t1 = time.perf_counter()
        x = torch.ones(64, 64, device="cuda"); y = (x @ x).sum().item(); torch.cuda.synchronize()
        info.update(gpu=p.name, sm=sm, capability=[p.major, p.minor], vram_gib=round(p.total_memory / 2**30, 2),
                    device_count=torch.cuda.device_count(), arch_list=torch.cuda.get_arch_list(),
                    cuda_context_s=round(time.perf_counter() - t1, 3), matmul_ok=(y == 64.0 * 64 * 64))
        if not info["matmul_ok"]:
            missing.append("cuda:sanity matmul returned a wrong value")
        if REQUIRE_SM not in ("", "any") and sm != REQUIRE_SM:
            missing.append(f"gpu:{sm} ({p.name}) but the 0.10 target topology requires {REQUIRE_SM} (RTX 5090); "
                           f"set DABAI_REQUIRE_SM=any only for non-benchmark endpoints")
    except Exception as e:  # noqa: BLE001
        missing.append(f"cuda:device init failed: {type(e).__name__}: {e}")
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,compute_cap,memory.total", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        info["nvidia_smi"] = q.stdout.strip() or q.stderr.strip()[:200]
    except Exception as e:  # noqa: BLE001
        info["nvidia_smi"] = f"unavailable: {type(e).__name__}"
    return missing, info


def image_info() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    if BUILD_INFO.is_file():
        try:
            out["build"] = json.loads(BUILD_INFO.read_text())
        except Exception:  # noqa: BLE001
            out["build"] = "unreadable"
    vers = {}
    for dist in ("torch", "torchvision", "torchaudio", "faster-whisper", "ctranslate2", "pyannote.audio", "huggingface_hub",
                 "diffusers", "transformers", "llama-cpp-python", "chatterbox-tts", "facexlib", "insightface", "onnxruntime-gpu",
                 "silero-vad", "spacy-pkuseg", "deepcache", "runpod"):
        try:
            vers[dist] = md.version(dist)
        except md.PackageNotFoundError:
            vers[dist] = None
    out["versions"] = vers
    try:
        out["ffmpeg"] = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10).stdout.splitlines()[0]
    except Exception as e:  # noqa: BLE001
        out["ffmpeg"] = f"unavailable: {type(e).__name__}"
    ls_commit = LS_ROOT / "COMMIT"
    out["latentsync_commit"] = ls_commit.read_text().strip() if ls_commit.is_file() else None
    out["env"] = {k: os.environ.get(k) for k in REPORTED_ENV}
    return out


def readiness(include_cuda: bool = True) -> dict:
    t0 = time.perf_counter()
    missing: list[str] = []
    warnings: list[str] = []
    rep: dict = {"models_root": str(MODELS_ROOT), "dub_root": str(DUB_ROOT), "require_sm": REQUIRE_SM}
    m, w, vol = check_volume(); missing += m; warnings += w; rep["volume"] = vol
    if not m:  # volume present -> inspect it
        m, rep["weights"] = check_weights(); missing += m
        m, rep["hf_cache"] = check_hf_cache(); missing += m
    m, rep["env"] = check_env(); missing += m
    m, rep["symlinks"] = check_symlinks(); missing += m
    if include_cuda:
        m, cuda = check_cuda(); missing += m; rep["cuda"] = cuda
        rep["gpu"] = cuda.get("gpu"); rep["sm"] = cuda.get("sm")
    rep["missing"] = missing
    rep["warnings"] = warnings
    rep["ready"] = not missing
    rep["check_seconds"] = round(time.perf_counter() - t0, 3)
    rep["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return rep


# ------------------------------------------------------------------------------------------------------------------
def handler(job: dict) -> dict:
    inp = (job or {}).get("input") or {}
    action = str(inp.get("action", "ready")).lower()
    if action in ("ready", "check", "readiness"):
        return readiness()
    if action == "info":
        return image_info()
    return {"error": f"unknown action {action!r}", "actions": ["ready", "info"]}


def register_fitness_checks() -> bool:
    """Custom RunPod fitness checks (SDK >= 1.9): a worker with missing weights or no CUDA never receives a job."""
    try:
        import runpod  # noqa: PLC0415
        reg = runpod.serverless.register_fitness_check
    except Exception:  # noqa: BLE001
        return False

    @reg
    def dabai_volume_and_weights():
        r = readiness(include_cuda=False)
        if not r["ready"]:
            raise RuntimeError("volume/weights/env not ready: " + "; ".join(r["missing"][:8]))

    @reg
    def dabai_cuda():
        if os.environ.get("DABAI_FITNESS_CUDA", "1") == "0":
            return
        m, _ = check_cuda()
        if m:
            raise RuntimeError("; ".join(m))
    return True


def _extend_sys_path() -> None:
    for p in (DUB_ROOT / "scripts" / "pilot", DUB_ROOT / "scripts" / "optim", LS_ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def import_smoke() -> int:
    """Build-time proof that the image imports the whole production stack (CPU only, no volume, no network)."""
    _extend_sys_path()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    failures = []
    t_all = time.perf_counter()
    for mod, dist in IMPORT_SMOKE:
        t0 = time.perf_counter()
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", None)
            if ver is None and dist:
                try:
                    ver = md.version(dist)
                except md.PackageNotFoundError:
                    ver = "?"
            print(f"  ok   {mod:42s} {ver or '':14s} {time.perf_counter() - t0:6.2f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            failures.append(mod); print(f"  FAIL {mod:42s} {type(e).__name__}: {e}", flush=True)
    try:
        import onnxruntime as ort  # noqa: PLC0415
        print("  onnxruntime providers compiled in:", ort.get_available_providers())
    except Exception:  # noqa: BLE001
        pass
    for forbidden in ("basicsr", "facelib"):   # CodeFormer runtime must NOT be needed by the clean pipeline
        if forbidden in sys.modules:
            failures.append(f"{forbidden} was imported (CodeFormer runtime leaked into the production path)")
    print(f"import smoke: {len(IMPORT_SMOKE) - len(failures)}/{len(IMPORT_SMOKE)} ok in {time.perf_counter() - t_all:.1f}s")
    if failures:
        print("FAILED:", failures); return 1
    return 0


def make_mock_volume(root: pathlib.Path) -> None:
    """Dummy volume with 1-byte stand-ins for every mandatory artifact (smoke tests only; NOT usable for inference)."""
    def put(rel: str, data: bytes = b"0") -> None:
        p = root / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
    for rels in MANDATORY.values():
        for r in rels:
            put(r)
    for i in (1, 2, 3):
        put(f"{QWEN_DIR}/qwen2.5-7b-instruct-q8_0-{i:05d}-of-00003.gguf", b"GGUF")
    for cache_rel, repo, files in HF_CACHE_ITEMS:
        d = root / cache_rel / f"models--{repo.replace('/', '--')}"
        commit = "0" * 40
        (d / "refs").mkdir(parents=True, exist_ok=True); (d / "refs" / "main").write_text(commit)
        (d / "blobs").mkdir(exist_ok=True); (d / "snapshots" / commit).mkdir(parents=True, exist_ok=True)
        for f in files:
            blob = d / "blobs" / f"mock-{f.replace('/', '_')}"; blob.write_bytes(b"0")
            link = d / "snapshots" / commit / f
            if not link.exists():
                link.symlink_to(os.path.relpath(blob, link.parent))
    put("MODEL_VOLUME_MANIFEST.json", json.dumps({"volume": "MOCK", "generated_at": "mock", "total_size_bytes": 0}).encode())
    print(f"mock volume written under {root}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, add_help=False)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--info", action="store_true")
    ap.add_argument("--import-smoke", action="store_true")
    ap.add_argument("--mock-volume", metavar="DIR")
    ap.add_argument("--no-cuda", action="store_true", help="with --check: skip the CUDA checks (path/env validation only)")
    ap.add_argument("-h", "--help", action="store_true")
    a, rest = ap.parse_known_args()
    if a.help:
        print(__doc__); return 0
    if a.mock_volume:
        make_mock_volume(pathlib.Path(a.mock_volume)); return 0
    if a.import_smoke:
        return import_smoke()
    if a.info:
        print(json.dumps(image_info(), indent=2)); return 0
    if a.check:
        r = readiness(include_cuda=not a.no_cuda)
        print(json.dumps(r, indent=2)); return 0 if r["ready"] else 1
    # --- RunPod Serverless worker ---
    sys.argv = [sys.argv[0]] + rest   # leave only runpod's own flags (--rp_serve_api, --test_input, ...)
    import runpod  # noqa: PLC0415
    register_fitness_checks()
    print(f"[handler] dabai worker starting; models_root={MODELS_ROOT} require_sm={REQUIRE_SM}", flush=True)
    runpod.serverless.start({"handler": handler})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
