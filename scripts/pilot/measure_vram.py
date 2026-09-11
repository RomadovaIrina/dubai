#!/usr/bin/env python
"""VRAM + wall-time measurement for pilot stages.

Why not just torch.cuda.max_memory_allocated(): half of this stack allocates GPU memory
*outside* the torch caching allocator — onnxruntime-gpu (insightface/buffalo_l in
LatentSync), cuDNN/cuBLAS workspaces, the CUDA context itself (~300-500 MiB). Torch
stats miss all of it. So every measurement also samples nvidia-smi, both device-wide and
attributed to our own process tree, and reports the two side by side.

Use it three ways:

    from measure_vram import measure_gpu_memory

    @measure_gpu_memory("LatentSync")            # decorator
    def run_latentsync(...): ...

    with measure_gpu_memory("CodeFormer"):       # context manager
        ...

    # external process (LatentSync runs as its own `python -m scripts.inference`)
    python scripts/pilot/measure_vram.py --label LatentSync -- \
        python -m scripts.inference --unet_config_path ...

    python scripts/pilot/measure_vram.py --report        # markdown table of everything so far
    python scripts/pilot/measure_vram.py --self-test     # sanity-check the sampler

Every run appends one row to reports/pilot/vram.jsonl, tagged with the env snapshot id
when one is available, so numbers stay traceable to the stack that produced them.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import pathlib
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT, append_jsonl, sh

JSONL = PILOT / "vram.jsonl"
MIB = 1024 * 1024


def _latest_snapshot_id() -> str | None:
    d = PILOT / "env"
    snaps = sorted(d.glob("*.json")) if d.exists() else []
    return snaps[-1].stem if snaps else None


def _descendants(pid: int) -> set[int]:
    """pid and every descendant, read straight from /proc (no psutil dependency)."""
    out, frontier = {pid}, [pid]
    while frontier:
        p = frontier.pop()
        try:
            kids = pathlib.Path(f"/proc/{p}/task").glob("*/children")
            for f in kids:
                for c in f.read_text().split():
                    c = int(c)
                    if c not in out:
                        out.add(c)
                        frontier.append(c)
        except Exception:
            pass
    return out


class GpuSampler(threading.Thread):
    """Polls nvidia-smi; tracks device-wide used MiB and MiB attributed to a pid tree."""

    def __init__(self, root_pid: int, interval: float = 0.2):
        super().__init__(daemon=True)
        self.root_pid, self.interval = root_pid, interval
        self._stop_evt = threading.Event()
        self.peak_device_mib = 0
        self.peak_proc_mib = 0
        self.baseline_device_mib = self._device_used()
        self.samples = 0
        self.attribution_seen = False

    @staticmethod
    def _device_used() -> int:
        v = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", timeout=15)
        try:
            return int(v.splitlines()[0])
        except Exception:
            return 0

    def _proc_used(self) -> int:
        raw = sh("nvidia-smi --query-compute-apps=pid,used_memory "
                 "--format=csv,noheader,nounits", timeout=15)
        if not raw:
            return 0
        mine = _descendants(self.root_pid)
        total = 0
        for line in raw.splitlines():
            try:
                pid_s, mem_s = (x.strip() for x in line.split(",")[:2])
                if int(pid_s) in mine:
                    total += int(mem_s)
                    self.attribution_seen = True
            except Exception:
                continue
        return total

    def run(self):
        while not self._stop_evt.is_set():
            self.peak_device_mib = max(self.peak_device_mib, self._device_used())
            self.peak_proc_mib = max(self.peak_proc_mib, self._proc_used())
            self.samples += 1
            self._stop_evt.wait(self.interval)

    def stop(self):
        self._stop_evt.set()
        self.join(timeout=5)


def _host_rss_peak_mib() -> float:
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except Exception:
        return 0.0


class measure_gpu_memory:
    """Context manager *and* decorator. Records peak VRAM and wall time for a labelled block."""

    def __init__(self, label: str, *, note: str = "", interval: float = 0.2,
                 jsonl: pathlib.Path | None = None, quiet: bool = False,
                 extra: dict | None = None):
        self.label, self.note, self.interval = label, note, interval
        self.jsonl = jsonl or JSONL
        self.quiet, self.extra = quiet, dict(extra or {})
        self.result: dict = {}

    def __enter__(self):
        self._torch = None
        try:
            import torch
            if torch.cuda.is_available():
                self._torch = torch
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                self._alloc0 = torch.cuda.memory_allocated()
        except Exception:
            pass
        self.sampler = GpuSampler(os.getpid(), self.interval)
        self.sampler.start()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._torch is not None:
            try:
                self._torch.cuda.synchronize()
            except Exception:
                pass
        dur = time.perf_counter() - self._t0
        self.sampler.stop()
        row = {
            "label": self.label,
            "note": self.note,
            "ok": exc_type is None,
            "error": f"{exc_type.__name__}: {exc}" if exc_type else None,
            "seconds": round(dur, 3),
            "peak_device_mib": self.sampler.peak_device_mib,
            "baseline_device_mib": self.sampler.baseline_device_mib,
            "peak_device_delta_mib": max(0, self.sampler.peak_device_mib
                                         - self.sampler.baseline_device_mib),
            "peak_process_mib": self.sampler.peak_proc_mib or None,
            "process_attribution": self.sampler.attribution_seen,
            "samples": self.sampler.samples,
            "host_rss_peak_mib": round(_host_rss_peak_mib(), 1),
            "mode": "in-process",
            "env_snapshot": _latest_snapshot_id(),
            "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **self.extra,
        }
        if self._torch is not None:
            row.update(
                torch_peak_allocated_mib=round(self._torch.cuda.max_memory_allocated() / MIB, 1),
                torch_peak_reserved_mib=round(self._torch.cuda.max_memory_reserved() / MIB, 1),
                torch_alloc_at_entry_mib=round(self._alloc0 / MIB, 1),
            )
        self.result = row
        append_jsonl(self.jsonl, row)
        if not self.quiet:
            print(_fmt_row(row), flush=True)
        return False  # never swallow exceptions

    def __call__(self, fn):
        @functools.wraps(fn)
        def wrapper(*a, **kw):
            with measure_gpu_memory(self.label, note=self.note, interval=self.interval,
                                    jsonl=self.jsonl, quiet=self.quiet, extra=self.extra) as m:
                out = fn(*a, **kw)
            wrapper.last_measurement = m.result
            return out
        wrapper.last_measurement = None
        return wrapper


def _fmt_row(r: dict) -> str:
    t = (f"torch {r['torch_peak_allocated_mib']:.0f}/{r['torch_peak_reserved_mib']:.0f} MiB"
         if r.get("torch_peak_allocated_mib") is not None else "torch n/a")
    proc = f"{r['peak_process_mib']} MiB" if r.get("peak_process_mib") else "n/a"
    return (f"[vram] {r['label']:<22} {r['seconds']:>8.2f}s  "
            f"proc peak {proc:>10}  device peak {r['peak_device_mib']} MiB "
            f"(Δ{r['peak_device_delta_mib']})  {t}"
            + ("" if r["ok"] else f"  FAILED: {r['error']}"))


def measure_command(label: str, cmd: list[str], note: str = "", interval: float = 0.2,
                    cwd: str | None = None) -> dict:
    """Measure an external process (its whole pid tree)."""
    t0 = time.perf_counter()
    proc = subprocess.Popen(cmd, cwd=cwd)
    sampler = GpuSampler(proc.pid, interval)
    sampler.start()
    rc = proc.wait()
    dur = time.perf_counter() - t0
    sampler.stop()
    row = {
        "label": label, "note": note, "ok": rc == 0,
        "error": None if rc == 0 else f"exit code {rc}",
        "seconds": round(dur, 3),
        "peak_device_mib": sampler.peak_device_mib,
        "baseline_device_mib": sampler.baseline_device_mib,
        "peak_device_delta_mib": max(0, sampler.peak_device_mib - sampler.baseline_device_mib),
        "peak_process_mib": sampler.peak_proc_mib or None,
        "process_attribution": sampler.attribution_seen,
        "samples": sampler.samples,
        "mode": "subprocess", "cmd": " ".join(cmd), "cwd": cwd,
        "env_snapshot": _latest_snapshot_id(),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    append_jsonl(JSONL, row)
    print(_fmt_row(row), flush=True)
    return row


def report() -> str:
    if not JSONL.exists():
        return "no measurements yet"
    rows = [json.loads(l) for l in JSONL.read_text().splitlines() if l.strip()]
    L = ["# VRAM measurements", "",
         "| label | ok | time | peak proc | peak device (Δ) | torch alloc/reserved | mode | env |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        ta = r.get("torch_peak_allocated_mib")
        tr = r.get("torch_peak_reserved_mib")
        L.append(
            f"| {r['label']} | {'PASS' if r['ok'] else 'FAIL'} | {r['seconds']:.2f}s "
            f"| {r.get('peak_process_mib') or '-'} MiB "
            f"| {r['peak_device_mib']} MiB (Δ{r['peak_device_delta_mib']}) "
            f"| {f'{ta:.0f}/{tr:.0f} MiB' if ta is not None else '-'} "
            f"| {r.get('mode','-')} | `{(r.get('env_snapshot') or '-')[:24]}` |")
    return "\n".join(L) + "\n"


def _self_test() -> int:
    import torch
    if not torch.cuda.is_available():
        print("no CUDA device"); return 1

    @measure_gpu_memory("self-test/alloc-2GiB", note="sanity check of the sampler")
    def burn():
        x = torch.empty(int(2 * 2**30 // 2), dtype=torch.float16, device="cuda")  # 2 GiB
        torch.cuda.synchronize()
        time.sleep(1.5)
        del x

    burn()
    r = burn.last_measurement
    ok = r["torch_peak_allocated_mib"] > 1900 and (r["peak_process_mib"] or 0) > 1900
    print("self-test:", "PASS" if ok else "SUSPECT — check nvidia-smi attribution")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="command")
    ap.add_argument("--note", default="")
    ap.add_argument("--cwd", default=None)
    ap.add_argument("--interval", type=float, default=0.2)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER,
                    help="-- followed by the command to measure")
    a = ap.parse_args()

    if a.report:
        out = report()
        print(out)
        (PILOT / "vram.md").write_text(out)
        print(f"-> reports/pilot/vram.md")
        return 0
    if a.self_test:
        return _self_test()
    cmd = [c for c in a.cmd if c != "--"]
    if not cmd:
        ap.print_help()
        return 2
    r = measure_command(a.label, cmd, a.note, a.interval, a.cwd)
    return 0 if r["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
