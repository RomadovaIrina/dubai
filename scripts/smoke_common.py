import os, sys, time, pathlib
ROOT = pathlib.Path(os.environ.get("DUB_ROOT", "/workspace/dub"))
MODELS, REPORTS, TP = ROOT / "models", ROOT / "reports", ROOT / "third_party"
DEMO_WAV = TP / "latentsync/assets/demo1_audio.wav"
DEMO_MP4 = TP / "latentsync/assets/demo1_video.mp4"
THREADS = int(os.environ.get("DUB_THREADS", "16"))
REPORTS.mkdir(exist_ok=True)

class Timer:
    def __enter__(self): self.t = time.perf_counter(); return self
    def __exit__(self, *a): self.s = time.perf_counter() - self.t

def gpu_mem():
    try:
        import torch
        return f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GiB peak"
    except Exception:
        return "n/a"

def fail(msg):
    print("FAIL:", msg); sys.exit(1)
