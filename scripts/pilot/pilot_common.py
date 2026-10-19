"""Shared helpers for the pilot measurement scripts (0.3, 0.4, 0.11, env snapshot).

Kept separate from scripts/smoke_common.py on purpose: the smoke suite is the frozen
acceptance harness for the baseline, these are measurement tools that will keep changing.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import time

ROOT = pathlib.Path(os.environ.get("DUB_ROOT", "/workspace/dub"))
MODELS = ROOT / "models"
THIRD_PARTY = ROOT / "third_party"
PILOT = ROOT / "reports" / "pilot"
PILOT.mkdir(parents=True, exist_ok=True)
THREADS = int(os.environ.get("DUB_THREADS", "16"))


# r3 (2026-10-19, Agent C on server A's CPU): torch device of the audio models. DUB_DEVICE overrides; default cuda when visible, else cpu
# (CUDA_VISIBLE_DEVICES="" -> cpu). On the CPU Chatterbox runs fp32 only (fp16 autocast is a GPU speed path) and Whisper int8.
_DEV = None
def dev() -> str:
    global _DEV
    if _DEV is None:
        _DEV = os.environ.get("DUB_DEVICE") or ""
        if not _DEV:
            import torch
            _DEV = "cuda" if torch.cuda.is_available() else "cpu"
    return _DEV


def tts_dtypes(torch) -> tuple:
    return (torch.float16, torch.float32) if dev() == "cuda" else (torch.float32,)


def whisper_compute(gpu_type: str) -> str:
    return gpu_type if dev() == "cuda" else "int8"


def empty_cache() -> None:
    if dev() == "cuda":
        import torch
        torch.cuda.empty_cache()


def _cgroup_cpus() -> float | None:
    try:
        q, per = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        return None if q == "max" else int(q) / int(per)
    except Exception:
        return None


# llama.cpp threads (2026-10-13): greedy output is identical for any thread count (03 translations 16 / 13 / 12 / 10 / 8 threads), but more
# threads than the cgroup CPU quota spin-wait: 13.6 vCPU box, 16 threads 4.9 tok/s vs 12 threads 6.1 tok/s. Default = min(DUB_THREADS, quota - 1)
# (unchanged on a >= 17 vCPU box); DUB_QWEN_THREADS overrides.
_q = _cgroup_cpus()
QWEN_THREADS = int(os.environ.get("DUB_QWEN_THREADS") or (max(1, min(THREADS, int(_q) - 1)) if _q else THREADS))


def sh(cmd: str, timeout: int = 60) -> str:
    """Run a shell command, return stripped stdout ('' on any failure)."""
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip()
    except Exception:
        return ""


def cpu_quota() -> dict:
    """CPU quota from cgroup v2 *or* v1.

    scripts/check_versions.py only reads the v2 path (/sys/fs/cgroup/cpu.max); on a v1
    host that silently prints an empty quota even though a limit is in force. Read both.
    """
    v2 = pathlib.Path("/sys/fs/cgroup/cpu.max")
    if v2.exists():
        parts = v2.read_text().split()
        if len(parts) == 2 and parts[0] != "max":
            q, p = int(parts[0]), int(parts[1])
            return {"cgroup": "v2", "quota_us": q, "period_us": p, "vcpu": round(q / p, 2)}
        return {"cgroup": "v2", "quota_us": None, "period_us": None, "vcpu": None}
    qf = pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    pf = pathlib.Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if qf.exists() and pf.exists():
        q, p = int(qf.read_text()), int(pf.read_text())
        return {"cgroup": "v1", "quota_us": q, "period_us": p,
                "vcpu": round(q / p, 2) if q > 0 else None}
    return {"cgroup": "unknown", "quota_us": None, "period_us": None, "vcpu": None}


def mem_limit_bytes() -> int | None:
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        f = pathlib.Path(p)
        if f.exists():
            v = f.read_text().strip()
            if v.isdigit():
                n = int(v)
                # v1 reports a sentinel close to 2**63 when unlimited
                return None if n > 2**62 else n
    return None


def git_info(repo: pathlib.Path) -> dict:
    if not (repo / ".git").exists():
        return {"path": str(repo), "commit": None, "dirty": None}
    commit = sh(f"git -C {repo} rev-parse HEAD")
    branch = sh(f"git -C {repo} rev-parse --abbrev-ref HEAD")
    dirty = bool(sh(f"git -C {repo} status --porcelain"))
    return {"path": str(repo), "commit": commit, "branch": branch, "dirty": dirty}


def gpu_info() -> dict:
    info = {"nvidia_smi": sh("nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version"
                            " --format=csv,noheader")}
    try:
        import torch
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            free, total = torch.cuda.mem_get_info()
            info.update(
                name=p.name,
                capability=f"sm_{p.major}{p.minor}",
                total_vram_bytes=p.total_memory,
                total_vram_gib=round(p.total_memory / 2**30, 2),
                free_vram_gib=round(free / 2**30, 2),
                multi_processor_count=p.multi_processor_count,
                arch_list=torch.cuda.get_arch_list(),
            )
        else:
            info["error"] = "torch.cuda.is_available() is False"
    except Exception as e:  # torch missing or broken
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def append_jsonl(path: pathlib.Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def write_json(path: pathlib.Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str))


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.s = time.perf_counter() - self.t0
