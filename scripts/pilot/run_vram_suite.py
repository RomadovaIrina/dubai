#!/usr/bin/env python3
"""Pilot 0.4 — isolated inference VRAM suite.

Runs the already-working smoke commands under the nvidia-smi sampler from measure_vram.py.
The resulting peaks complement resident_vram_probe.py:
  * run_vram_suite.py -> real per-component inference peaks
  * resident_vram_probe.py -> simultaneous steady residency
Together they are enough to make the 32 GiB RESIDENT vs UNLOAD decision.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT, ROOT, write_json
from measure_vram import JSONL, measure_command, report


COMMANDS = {
    "whisper": (["/venv/dabai/bin/python", str(ROOT / "scripts/check_whisper.py")], str(ROOT)),
    "retinaface": (["/venv/dabai/bin/python", str(ROOT / "scripts/check_retinaface.py")], str(ROOT)),
    "pyannote": (["/venv/dabai/bin/python", str(ROOT / "scripts/check_pyannote.py")], str(ROOT)),
    "chatterbox": (["/venv/dabai/bin/python", str(ROOT / "scripts/check_chatterbox.py")], str(ROOT)),
    "latentsync": (["bash", str(ROOT / "scripts/check_latentsync.sh")], str(ROOT)),
    "codeformer": (["bash", str(ROOT / "scripts/check_codeformer.sh")], str(ROOT)),
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("components", nargs="*", choices=list(COMMANDS),
                    help="default: all; CodeFormer will reuse/create the LatentSync smoke output")
    ap.add_argument("--clear", action="store_true",
                    help="clear reports/pilot/vram.jsonl before this suite")
    ap.add_argument("--continue-on-error", action="store_true", default=True)
    a = ap.parse_args()

    components = a.components or list(COMMANDS)
    if a.clear and JSONL.exists():
        JSONL.unlink()

    rows = []
    for name in components:
        cmd, cwd = COMMANDS[name]
        print(f"\n===== 0.4 isolated VRAM: {name} =====")
        r = measure_command(f"pilot0.4/{name}", cmd, note="real smoke inference", cwd=cwd)
        rows.append(r)
        if not r["ok"] and not a.continue_on_error:
            break

    rep = {
        "task": "0.4",
        "kind": "isolated-inference-peaks",
        "rows": rows,
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "Use together with resident_vram_probe.py for the final 32 GiB decision.",
    }
    write_json(PILOT / "0.4_isolated_peaks.json", rep)
    md = report()
    (PILOT / "vram.md").write_text(md, encoding="utf-8")
    print("\n" + md)
    print("-> reports/pilot/0.4_isolated_peaks.json")
    return 0 if all(r["ok"] for r in rows if "pyannote" not in r.get("label", "")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
