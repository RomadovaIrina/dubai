#!/usr/bin/env python3
"""Pilot 0.1 — formal Blackwell/sm_120 validation.

Checks the immutable baseline from the pilot spec:
  * system CUDA toolkit 12.8
  * PyTorch >= 2.7 with cu128 build
  * GPU compute capability sm_120
  * real CUDA matmul on the current device
  * optionally runs the existing component smoke suite

Writes compact JSON + Markdown evidence into reports/pilot/.
On non-Blackwell dev GPUs the script exits 2 and reports NOT_TARGET_GPU rather
than pretending that 0.1 is closed there.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT, ROOT, gpu_info, sh, write_json


def run(cmd: list[str], cwd: pathlib.Path | None = None) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return p.returncode, p.stdout


def torch_probe() -> dict:
    import torch
    out = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
    }
    if not torch.cuda.is_available():
        return out
    props = torch.cuda.get_device_properties(0)
    out.update(
        device=props.name,
        capability=[props.major, props.minor],
        vram_gib=round(props.total_memory / 2**30, 3),
    )
    # A real kernel launch catches "no kernel image" failures that metadata cannot.
    try:
        a = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
        b = torch.randn((4096, 4096), device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        c = a @ b
        torch.cuda.synchronize()
        out["matmul"] = {
            "ok": bool(torch.isfinite(c).all().item()),
            "seconds": round(time.perf_counter() - t0, 4),
        }
        del a, b, c
        torch.cuda.empty_cache()
    except Exception as e:
        out["matmul"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    return out


def parse_nvcc() -> dict:
    txt = sh("nvcc --version", timeout=30)
    m = re.search(r"release\s+([0-9]+\.[0-9]+)", txt)
    return {"raw": txt, "release": m.group(1) if m else None,
            "which": sh("which nvcc")}


def parse_smoke(text: str) -> dict:
    # run_smoke.sh prints PASS/FAIL/SKIP component lines; preserve unknown lines too.
    statuses: dict[str, str] = {}
    for line in text.splitlines():
        m = re.search(r"\b(PASS|FAIL|SKIP)\s+([A-Za-z0-9_.+/-]+)", line)
        if m:
            statuses[m.group(2)] = m.group(1)
    return statuses


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-smoke", action="store_true",
                    help="run scripts/run_smoke.sh; slow but gives formal model-start evidence")
    ap.add_argument("--allow-pyannote-skip", action="store_true", default=True)
    ap.add_argument("--json", default=str(PILOT / "0.1_blackwell.json"))
    a = ap.parse_args()

    rep: dict = {
        "task": "0.1",
        "title": "Blackwell image",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu": gpu_info(),
        "nvcc": parse_nvcc(),
    }
    try:
        rep["torch_probe"] = torch_probe()
    except Exception as e:
        rep["torch_probe"] = {"error": f"{type(e).__name__}: {e}"}

    tp = rep["torch_probe"]
    target_gpu = tp.get("capability") == [12, 0]
    torch_ok = str(tp.get("torch", "")).startswith(("2.7", "2.8", "2.9", "2.10", "2.11")) and "+cu128" in str(tp.get("torch", ""))
    cuda_ok = rep["nvcc"].get("release") == "12.8"
    matmul_ok = bool(tp.get("matmul", {}).get("ok"))

    rep["baseline_checks"] = {
        "sm_120": target_gpu,
        "torch_2_7_plus_cu128": torch_ok,
        "system_cuda_12_8": cuda_ok,
        "real_cuda_kernel": matmul_ok,
    }

    smoke_rc = None
    smoke_output = None
    smoke_statuses: dict[str, str] = {}
    if a.run_smoke:
        smoke_rc, smoke_output = run(["bash", str(ROOT / "scripts" / "run_smoke.sh")], ROOT)
        (PILOT / "0.1_smoke.log").write_text(smoke_output, encoding="utf-8")
        smoke_statuses = parse_smoke(smoke_output)
        rep["smoke"] = {"returncode": smoke_rc, "statuses": smoke_statuses,
                        "log": "reports/pilot/0.1_smoke.log"}

    if not target_gpu:
        verdict = "NOT_TARGET_GPU"
        closed = False
        reason = "0.1 is defined specifically for Blackwell sm_120; this machine is a dev/portability host."
    elif not (torch_ok and cuda_ok and matmul_ok):
        verdict = "FAIL"
        closed = False
        reason = "Immutable Blackwell baseline check failed."
    elif a.run_smoke:
        fail = [k for k, v in smoke_statuses.items() if v == "FAIL"]
        skip = [k for k, v in smoke_statuses.items() if v == "SKIP"]
        if fail:
            verdict, closed = "FAIL", False
            reason = f"component smoke failures: {', '.join(fail)}"
        elif skip and not (a.allow_pyannote_skip and all("pyannote" in x.lower() for x in skip)):
            verdict, closed = "INCOMPLETE", False
            reason = f"component smoke skipped: {', '.join(skip)}"
        else:
            verdict, closed = "PASS", True
            reason = "sm_120 baseline and real model smoke suite passed" + ("; pyannote gated skip accepted" if skip else "")
    else:
        verdict, closed = "BASELINE_PASS_SMOKE_NOT_RERUN", False
        reason = "hardware/torch baseline passes; rerun with --run-smoke for formal closure evidence"

    rep.update(verdict=verdict, closed=closed, reason=reason)
    write_json(pathlib.Path(a.json), rep)

    md = [
        "# Pilot 0.1 — Blackwell image", "",
        f"**Verdict:** `{verdict}`  ",
        f"**Closed:** {'YES' if closed else 'NO'}  ",
        f"**Reason:** {reason}", "",
        "| Check | Result |", "|---|---|",
        f"| GPU | {tp.get('device', rep['gpu'].get('name', '-'))} |",
        f"| capability | {tp.get('capability', '-')} |",
        f"| torch | {tp.get('torch', '-')} |",
        f"| torch CUDA | {tp.get('torch_cuda', '-')} |",
        f"| nvcc | {rep['nvcc'].get('release', '-')} |",
        f"| real CUDA matmul | {'PASS' if matmul_ok else 'FAIL'} |",
    ]
    if a.run_smoke:
        md += ["", "## Component smoke", "", "| component | status |", "|---|---|"]
        md += [f"| {k} | {v} |" for k, v in sorted(smoke_statuses.items())]
    (PILOT / "0.1_blackwell.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\n-> {a.json}\n-> reports/pilot/0.1_blackwell.md")
    return 0 if closed else (2 if verdict == "NOT_TARGET_GPU" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
