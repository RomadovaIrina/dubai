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
