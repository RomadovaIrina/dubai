"""Print versions of every component in the dabai stack. No model loading. Same content as test-pipeline.ipynb."""
import importlib, importlib.metadata as md, platform, subprocess, sys, os

def ver(mod, dist=None):
    try:
        m = importlib.import_module(mod)
        v = getattr(m, "__version__", None)
        if v is None and dist:
            v = md.version(dist)
        return v or md.version(dist or mod)
    except md.PackageNotFoundError:
        m = importlib.import_module(mod)
        loc = getattr(m, "__file__", None) or (list(m.__path__)[0] if hasattr(m, "__path__") else "?")
        return f"vendored @ {loc}"
    except Exception as e:
        return f"MISSING ({type(e).__name__})"

def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        return f"ERR {e}"

print("== system ==")
print("python           ", sys.version.split()[0], sys.executable)
print("platform         ", platform.platform())
print("nvidia driver    ", sh("nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader"))
print("nvcc             ", sh("nvcc --version | grep release"))
print("ffmpeg           ", sh("ffmpeg -version | head -1"))
print("ffmpeg nvenc     ", sh("ffmpeg -hide_banner -encoders 2>/dev/null | grep -c nvenc"), "encoders listed (NVENC is non-functional on this host; use libx264)")
print("ffmpeg libx264   ", "yes" if "libx264" in sh("ffmpeg -hide_banner -buildconf") else "no")
print("cpu quota        ", sh("cat /sys/fs/cgroup/cpu.max"), "| os.cpu_count() =", os.cpu_count(), "(host value, do not trust)")

print("\n== torch baseline ==")
import torch
print("torch            ", torch.__version__)
print("torch.version.cuda", torch.version.cuda)
print("cudnn            ", torch.backends.cudnn.version())
print("cuda available   ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device           ", torch.cuda.get_device_name(0))
    print("capability       ", torch.cuda.get_device_capability(0))
    print("arch list        ", torch.cuda.get_arch_list())
print("torchvision      ", ver("torchvision"))
print("torchaudio       ", ver("torchaudio"))

print("\n== numeric / audio core ==")
for mod, dist in [("numpy", None), ("scipy", None), ("numba", None), ("librosa", None), ("soundfile", None), ("soxr", None), ("cv2", "opencv-python"), ("PIL", "Pillow")]:
    print(f"{mod:17s}", ver(mod, dist))

print("\n== HF stack ==")
for mod, dist in [("transformers", None), ("tokenizers", None), ("huggingface_hub", None), ("diffusers", None), ("safetensors", None), ("accelerate", None)]:
    print(f"{mod:17s}", ver(mod, dist))

print("\n== ASR / VAD / diarization ==")
for mod, dist in [("faster_whisper", "faster-whisper"), ("ctranslate2", None), ("silero_vad", "silero-vad"), ("pyannote.audio", "pyannote.audio"), ("lightning", None), ("speechbrain", None)]:
    print(f"{mod:17s}", ver(mod, dist))
try:
    import ctranslate2
    print("ct2 cuda types   ", ctranslate2.get_supported_compute_types("cuda"), "(int8 is disabled on sm_120 -> int8_float16 runs as float16)")
except Exception as e:
    print("ct2 cuda types    ERR", e)

print("\n== faces ==")
for mod, dist in [("facexlib", None), ("insightface", None), ("kornia", None), ("face_alignment", "face-alignment")]:
    print(f"{mod:17s}", ver(mod, dist))
try:
    import onnxruntime as ort
    print("onnxruntime      ", ort.__version__, ort.get_available_providers())
except Exception as e:
    print("onnxruntime       MISSING", type(e).__name__)

print("\n== LLM (CPU) ==")
print("llama_cpp        ", ver("llama_cpp", "llama-cpp-python"))

print("\n== TTS ==")
print("chatterbox       ", ver("chatterbox", "chatterbox-tts"))
try:
    d = md.distribution("chatterbox-tts")
    direct = d.read_text("direct_url.json")
    if direct:
        import json
        j = json.loads(direct); print("chatterbox source", j.get("url"), j.get("vcs_info", {}).get("commit_id", "")[:12])
except Exception:
    pass

print("\n== LatentSync / CodeFormer deps ==")
for mod, dist in [("decord", None), ("DeepCache", None), ("lpips", None), ("omegaconf", None), ("einops", None), ("imageio", None), ("scenedetect", None), ("basicsr", None), ("facelib", None)]:
    print(f"{mod:17s}", ver(mod, dist))
root = os.environ.get("DUB_ROOT", "/workspace/dub")
for repo in ("latentsync", "CodeFormer"):
    print(f"{repo:17s}", sh(f"git -C {root}/third_party/{repo} rev-parse --short HEAD 2>/dev/null") or "not cloned")
