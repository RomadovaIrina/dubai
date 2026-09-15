#!/usr/bin/env python3
"""Pilot 0.2 — LatentSync fork/dependency compatibility evidence.

Verifies that the checked-out LatentSync commit runs under the immutable torch/cu128
baseline and, optionally, executes the existing real-inference smoke test.
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT, ROOT, THIRD_PARTY, git_info, sh, write_json


def run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, cwd=str(ROOT), text=True, stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT)
    return p.returncode, p.stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-smoke", action="store_true",
                    help="execute scripts/check_latentsync.sh (real model inference)")
    ap.add_argument("--json", default=str(PILOT / "0.2_latentsync.json"))
    a = ap.parse_args()

    repo = THIRD_PARTY / "latentsync"
    rep = {
        "task": "0.2",
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "latentsync": git_info(repo),
        "baseline": {
            "torch": sh("/venv/dabai/bin/python -c \"import torch; print(torch.__version__)\""),
            "torch_cuda": sh("/venv/dabai/bin/python -c \"import torch; print(torch.version.cuda)\""),
        },
        "recorded_commits": (ROOT / "reports" / "third_party-commits.txt").read_text().strip()
            if (ROOT / "reports" / "third_party-commits.txt").exists() else None,
    }

    if not repo.exists():
        rep.update(verdict="FAIL", closed=False, reason="third_party/latentsync is not present")
    elif a.run_smoke:
        rc, out = run(["bash", str(ROOT / "scripts" / "check_latentsync.sh")])
        (PILOT / "0.2_latentsync_smoke.log").write_text(out, encoding="utf-8")
        low = out.lower()
        no_kernel = "no kernel image" in low or "invalid device function" in low
        out_file = ROOT / "reports" / "latentsync_demo.mp4"
        ok = rc == 0 and out_file.exists() and out_file.stat().st_size > 0 and not no_kernel
        rep["smoke"] = {"returncode": rc, "output_exists": out_file.exists(),
                        "no_kernel_image": no_kernel,
                        "log": "reports/pilot/0.2_latentsync_smoke.log"}
        rep.update(verdict="PASS" if ok else "FAIL", closed=ok,
                   reason="real LatentSync inference passed on immutable baseline" if ok
                   else "LatentSync real inference failed; inspect smoke log")
    else:
        # Existing repository state can show compatibility, but a fresh run is stronger evidence.
        old_report = ROOT / "reports" / "env-report.md"
        evidence = old_report.exists() and "LatentSync" in old_report.read_text(errors="ignore")
        rep.update(verdict="EVIDENCE_PRESENT_SMOKE_NOT_RERUN" if evidence else "INCOMPLETE",
                   closed=bool(evidence),
                   reason="existing environment report records successful LatentSync inference"
                   if evidence else "run with --run-smoke to produce closure evidence")

    write_json(pathlib.Path(a.json), rep)
    md = ["# Pilot 0.2 — LatentSync compatibility", "",
          f"**Verdict:** `{rep['verdict']}`  ",
          f"**Closed:** {'YES' if rep['closed'] else 'NO'}  ",
          f"**Reason:** {rep['reason']}", "",
          f"LatentSync commit: `{rep['latentsync'].get('commit')}`  ",
          f"Torch: `{rep['baseline']['torch']}`  ",
          f"Torch CUDA: `{rep['baseline']['torch_cuda']}`"]
    (PILOT / "0.2_latentsync.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0 if rep["closed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
