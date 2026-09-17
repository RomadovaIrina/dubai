#!/usr/bin/env python3
"""Pilot 0.10 — true cold-start start-to-READY benchmark.

Formal topology from the pilot:
  - target deployment topology;
  - multi-stage image;
  - model weights on network NVMe;
  - target: <60 seconds start-to-ready.

This harness does NOT pretend that restarting Python, dropping caches, or timing an already
running pod is a cold start. It controls the target worker through user-supplied commands.

`--ready-command` must exit 0 only when the worker is ACTUALLY able to accept a real job with
required weights accessible. Merely having SSH/HTTP up is not enough.

Typical shape:
  python scripts/pilot/benchmark_10_cold_start.py \
    --start-command './infra/start_node.sh' \
    --ready-command './infra/worker_ready_with_models.sh' \
    --stop-command './infra/stop_node.sh' \
    --ensure-stopped-command './infra/ensure_stopped.sh' \
    --repeat 3 \
    --confirm-cold-node \
    --confirm-multistage-image \
    --confirm-network-nvme \
    --topology-note 'RunPod target worker; multi-stage image; weights on network NVMe'

Provider-specific start/stop scripts are intentionally outside this benchmark.
"""
from __future__ import annotations

import argparse
import pathlib
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT, write_json


def call(cmd: str, timeout: float | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )


def tail(s: str | None, n: int = 4000) -> str:
    return (s or "")[-n:]


def one(a, idx: int) -> dict:
    print(f"\n===== cold start {idx}/{a.repeat} =====", flush=True)

    preflight = None
    if a.ensure_stopped_command:
        try:
            q = call(a.ensure_stopped_command, timeout=a.stop_timeout)
            preflight = {
                "returncode": q.returncode,
                "output": tail(q.stdout, 2000),
            }
            if q.returncode != 0:
                return {
                    "run": idx,
                    "status": "PREFLIGHT_STOP_FAIL",
                    "seconds": None,
                    "preflight": preflight,
                }
        except Exception as e:
            return {
                "run": idx,
                "status": "PREFLIGHT_STOP_FAIL",
                "seconds": None,
                "preflight": {"error": f"{type(e).__name__}: {e}"},
            }

    # Formal timer begins immediately BEFORE the provider start request.
    t0 = time.perf_counter()
    try:
        s = call(a.start_command, timeout=a.start_timeout)
    except subprocess.TimeoutExpired as e:
        return {
            "run": idx,
            "status": "START_COMMAND_TIMEOUT",
            "seconds": round(time.perf_counter() - t0, 3),
            "start_output": tail(getattr(e, "stdout", "")),
            "preflight": preflight,
        }
    except Exception as e:
        return {
            "run": idx,
            "status": "START_EXCEPTION",
            "seconds": round(time.perf_counter() - t0, 3),
            "start_output": f"{type(e).__name__}: {e}",
            "preflight": preflight,
        }

    start_return_s = time.perf_counter() - t0
    if s.returncode != 0:
        row = {
            "run": idx,
            "status": "START_FAIL",
            "seconds": round(start_return_s, 3),
            "start_command_return_s": round(start_return_s, 3),
            "start_output": tail(s.stdout),
            "preflight": preflight,
        }
        _stop(a, row)
        return row

    deadline = time.monotonic() + a.timeout
    attempts = 0
    last = ""
    ready_s = None
    status = "TIMEOUT"

    while time.monotonic() < deadline:
        attempts += 1
        try:
            r = call(a.ready_command, timeout=a.ready_timeout)
            last = tail(r.stdout, 3000)
            if r.returncode == 0:
                ready_s = time.perf_counter() - t0
                status = "PASS"
                break
        except subprocess.TimeoutExpired:
            last = "ready-command timed out"
        except Exception as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(a.poll_interval)

    row = {
        "run": idx,
        "status": status,
        "seconds": round(ready_s if ready_s is not None else time.perf_counter() - t0, 3),
        "start_command_return_s": round(start_return_s, 3),
        "poll_attempts": attempts,
        "start_output": tail(s.stdout, 2500),
        "ready_output": last,
        "preflight": preflight,
    }
    _stop(a, row)

    if idx < a.repeat and a.cooldown:
        time.sleep(a.cooldown)
    return row


def _stop(a, row: dict) -> None:
    if not a.stop_command:
        return
    try:
        t = time.perf_counter()
        q = call(a.stop_command, timeout=a.stop_timeout)
        row["stop"] = {
            "returncode": q.returncode,
            "seconds": round(time.perf_counter() - t, 3),
            "output": tail(q.stdout, 2000),
        }
    except Exception as e:
        row["stop"] = {"error": f"{type(e).__name__}: {e}"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start-command", required=True)
    ap.add_argument("--ready-command", required=True)
    ap.add_argument("--stop-command", required=True)
    ap.add_argument(
        "--ensure-stopped-command",
        help="optional preflight executed before timing each run; should ensure worker is actually stopped/cold",
    )
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--poll-interval", type=float, default=1.0)
    ap.add_argument("--start-timeout", type=float, default=60.0)
    ap.add_argument("--ready-timeout", type=float, default=10.0)
    ap.add_argument("--stop-timeout", type=float, default=120.0)
    ap.add_argument("--cooldown", type=float, default=5.0)
    ap.add_argument("--target-seconds", type=float, default=60.0)
    ap.add_argument("--confirm-cold-node", action="store_true")
    ap.add_argument("--confirm-multistage-image", action="store_true")
    ap.add_argument("--confirm-network-nvme", action="store_true")
    ap.add_argument(
        "--topology-note",
        default="UNCONFIRMED — target must be multi-stage image + network NVMe",
    )
    ap.add_argument("--json", default=str(PILOT / "0.10_cold_start.json"))
    ap.add_argument("--md", default=str(PILOT / "0.10_cold_start.md"))
    a = ap.parse_args()

    if a.repeat < 1:
        ap.error("--repeat must be >=1")
    if a.poll_interval <= 0:
        ap.error("--poll-interval must be >0")

    rows = [one(a, i) for i in range(1, a.repeat + 1)]
    good = [r["seconds"] for r in rows if r.get("status") == "PASS" and r.get("seconds") is not None]

    mean_s = round(statistics.mean(good), 3) if good else None
    med_s = round(statistics.median(good), 3) if good else None
    max_s = round(max(good), 3) if good else None
    min_s = round(min(good), 3) if good else None
    all_pass = len(good) == a.repeat
    within_target_all = all_pass and all(x < a.target_seconds for x in good)

    topology_confirmed = (
        a.confirm_cold_node
        and a.confirm_multistage_image
        and a.confirm_network_nvme
    )
    enough_repeats = a.repeat >= 3
    closed = all_pass and topology_confirmed and enough_repeats

    if closed:
        verdict = (
            f"CLOSED — {'PASS_TARGET' if within_target_all else 'FAIL_TARGET'}; "
            f"median {med_s}s, max {max_s}s, target <{a.target_seconds}s"
        )
    else:
        missing = []
        if not all_pass:
            missing.append(f"successful cold starts {len(good)}/{a.repeat}")
        if not enough_repeats:
            missing.append("need >=3 runs for formal closure")
        if not a.confirm_cold_node:
            missing.append("true cold node/start not confirmed")
        if not a.confirm_multistage_image:
            missing.append("multi-stage image not confirmed")
        if not a.confirm_network_nvme:
            missing.append("network NVMe weights not confirmed")
        verdict = "OPEN/BLOCKED — " + "; ".join(missing)

    rep = {
        "task": "0.10",
        "closed": closed,
        "verdict": verdict,
        "target_seconds": a.target_seconds,
        "topology_note": a.topology_note,
        "topology": {
            "cold_node_confirmed": a.confirm_cold_node,
            "multistage_image_confirmed": a.confirm_multistage_image,
            "network_nvme_confirmed": a.confirm_network_nvme,
        },
        "repeat": a.repeat,
        "rows": rows,
        "successful_runs": len(good),
        "mean_seconds": mean_s,
        "median_seconds": med_s,
        "min_seconds": min_s,
        "max_seconds": max_s,
        "within_target_all": within_target_all,
        "timing_definition": (
            "t0 immediately before provider start-command; READY only when ready-command exits 0 "
            "after worker + required model weights are usable"
        ),
        "anti_fake_note": (
            "Restarting Python, clearing CUDA/cache, or timing an already-running pod is not accepted "
            "as a formal cold start."
        ),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    write_json(pathlib.Path(a.json), rep)

    md = [
        "# Pilot 0.10 — cold start",
        "",
        f"**Verdict:** `{verdict}`  ",
        f"Topology: {a.topology_note}  ",
        f"Target: **<{a.target_seconds:.1f}s**  ",
        f"Median: **{med_s}s**  ",
        f"Mean: **{mean_s}s**  ",
        f"Max: **{max_s}s**  ",
        f"All runs below target: **{within_target_all}**",
        "",
        "| run | status | start command return s | READY s | polls |",
        "|---:|---|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['run']} | {r['status']} | {r.get('start_command_return_s', '-')} | "
            f"{r.get('seconds', '-')} | {r.get('poll_attempts', '-')} |"
        )
    pathlib.Path(a.md).write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md))
    print(f"\n-> {a.json}\n-> {a.md}")
    return 0 if closed else 1


if __name__ == "__main__":
    raise SystemExit(main())
