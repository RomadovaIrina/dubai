#!/usr/bin/env python
"""Freeze everything a measurement run depends on, so a chart taken today can be
traced back to the exact environment that produced it.

    source scripts/env.sh
    python scripts/pilot/env_snapshot.py                 # write a snapshot, print its id
    python scripts/pilot/env_snapshot.py --list          # list snapshots taken so far
    python scripts/pilot/env_snapshot.py --show <id>     # print one as markdown

Writes to reports/pilot/env/:
    <id>.json            machine-readable snapshot
    <id>.md              human-readable summary
    <id>.pip-freeze.txt  full resolved dependency list

The id is <UTC timestamp>-<8 hex>, where the hex is a fingerprint over the facts that
actually change results (GPU, driver, CUDA, torch, python, repo commit, third-party
commits, pip freeze). Two runs with the same fingerprint ran on the same stack; quote
the id next to every number and graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from pilot_common import (PILOT, ROOT, THIRD_PARTY, cpu_quota, git_info, gpu_info,
                          mem_limit_bytes, sh, write_json)

ENV_DIR = PILOT / "env"


def collect() -> tuple[dict, str]:
    torch_info: dict = {}
    try:
        import torch
        torch_info = {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cuda_available": torch.cuda.is_available(),
        }
        for mod, dist in (("torchvision", "torchvision"), ("torchaudio", "torchaudio")):
            try:
                torch_info[dist] = __import__(mod).__version__
            except Exception:
                torch_info[dist] = None
    except Exception as e:
        torch_info = {"error": f"{type(e).__name__}: {e}"}

    freeze = sh(f"{sys.executable} -m pip freeze", timeout=180)

    snap = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "captured_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "kernel": platform.release(),
            "libc": " ".join(platform.libc_ver()),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "venv": sh("echo $VIRTUAL_ENV") or None,
        },
        "gpu": gpu_info(),
        "cuda_system": {
            "nvcc": sh("nvcc --version | tail -2 | head -1"),
            "cuda_home": sh("echo $CUDA_HOME") or "/usr/local/cuda",
            "cuda_symlink": sh("readlink -f /usr/local/cuda"),
        },
        "torch": torch_info,
        "ffmpeg": sh("ffmpeg -version | head -1"),
        "limits": {"cpu": cpu_quota(), "memory_limit_bytes": mem_limit_bytes(),
                   "os_cpu_count": __import__("os").cpu_count()},
        "git": {
            "dub": git_info(ROOT),
            "latentsync": git_info(THIRD_PARTY / "latentsync"),
            "CodeFormer": git_info(THIRD_PARTY / "CodeFormer"),
        },
        "pip_freeze_lines": len(freeze.splitlines()),
    }

    fingerprint_src = json.dumps(
        {
            "gpu": snap["gpu"].get("name"),
            "cap": snap["gpu"].get("capability"),
            "driver": snap["gpu"].get("nvidia_smi"),
            "cuda": snap["cuda_system"]["nvcc"],
            "torch": snap["torch"],
            "python": snap["python"]["version"],
            "git": snap["git"],
            "freeze": freeze,
        },
        sort_keys=True,
    )
    fp = hashlib.sha256(fingerprint_src.encode()).hexdigest()[:8]
    snap["fingerprint"] = fp
    snap["id"] = f"{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}-{fp}"
    return snap, freeze


def to_md(s: dict) -> str:
    g, t, l = s["gpu"], s["torch"], s["limits"]
    cpu = l["cpu"]
    L = [
        f"# env snapshot `{s['id']}`",
        "",
        f"captured {s['captured_at']} · fingerprint `{s['fingerprint']}`",
        "",
        "| | |",
        "|---|---|",
        f"| GPU | {g.get('name')} · {g.get('capability')} · {g.get('total_vram_gib')} GiB |",
        f"| driver / nvidia-smi | {g.get('nvidia_smi')} |",
        f"| torch arch list | {', '.join(g.get('arch_list') or [])} |",
        f"| torch | {t.get('torch')} (cuda {t.get('torch_cuda')}, cudnn {t.get('cudnn')}) |",
        f"| torchvision / torchaudio | {t.get('torchvision')} / {t.get('torchaudio')} |",
        f"| system CUDA | {s['cuda_system']['nvcc']} |",
        f"| python | {s['python']['version']} · {s['python']['executable']} |",
        f"| ffmpeg | {s['ffmpeg']} |",
        f"| CPU quota | {cpu.get('vcpu')} vCPU (cgroup {cpu.get('cgroup')}) · "
        f"os.cpu_count()={l['os_cpu_count']} |",
        f"| memory limit | {round(l['memory_limit_bytes']/2**30, 1) if l['memory_limit_bytes'] else 'unlimited'} GiB |",
        f"| kernel | {s['host']['kernel']} |",
        "",
        "| repo | commit | dirty |",
        "|---|---|---|",
    ]
    for name, gi in s["git"].items():
        L.append(f"| {name} | `{(gi.get('commit') or '-')[:12]}` | {gi.get('dirty')} |")
    L += ["", f"pip freeze: {s['pip_freeze_lines']} packages → `{s['id']}.pip-freeze.txt`", ""]
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true", help="list existing snapshots")
    ap.add_argument("--show", metavar="ID", help="print an existing snapshot as markdown")
    a = ap.parse_args()
    ENV_DIR.mkdir(parents=True, exist_ok=True)

    if a.list:
        rows = sorted(ENV_DIR.glob("*.json"))
        if not rows:
            print("no snapshots yet")
            return 0
        for p in rows:
            s = json.loads(p.read_text())
            print(f"{s['id']}  {s['gpu'].get('name')}  {s['gpu'].get('capability')}  "
                  f"torch {s['torch'].get('torch')}  dub@{(s['git']['dub'].get('commit') or '')[:8]}")
        return 0

    if a.show:
        p = ENV_DIR / f"{a.show}.json"
        if not p.exists():
            print(f"no such snapshot: {a.show}", file=sys.stderr)
            return 1
        print(to_md(json.loads(p.read_text())))
        return 0

    snap, freeze = collect()
    write_json(ENV_DIR / f"{snap['id']}.json", snap)
    (ENV_DIR / f"{snap['id']}.md").write_text(to_md(snap))
    (ENV_DIR / f"{snap['id']}.pip-freeze.txt").write_text(freeze + "\n")
    print(to_md(snap))
    print(f"-> reports/pilot/env/{snap['id']}.{{json,md,pip-freeze.txt}}")
    print(f"SNAPSHOT_ID={snap['id']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
