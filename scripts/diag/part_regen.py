#!/usr/bin/env python3
"""Diagnostic: regenerate EVERY TTS part of a reproduced run with the production text / seed / reference, in three numeric modes:
  A_prod_fp16.wav (sanity: must equal the runner's raw output), A_fp16fix.wav (fp16 + int64 speech-token ids), A_fp32.wav.
Writes into <chunks>/<id>/<tag>/ next to the lineage files.   python scripts/diag/part_regen.py --id 01
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, soundfile as sf, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tts_experiments as tx


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--id", required=True); ap.add_argument("--repro", default="/workspace/dub/tts_diag/repro"); ap.add_argument("--chunks", default="/workspace/dub/tts_diag/chunks")
    ap.add_argument("--modes", default="prod_fp16,fp16fix,fp32"); a = ap.parse_args()
    man = json.loads((pathlib.Path(a.repro) / a.id / f"{a.id}.manifest.json").read_text()); ref = man["tts"]["references"][man["units"][0]["speaker"]]["file"]
    rows = json.loads((pathlib.Path(a.chunks) / a.id / "chunks.json").read_text()); modes = a.modes.split(",")
    for mode in modes:
        dt = "fp32" if mode == "fp32" else "fp16"; m = tx.load(dt); dtype = torch.float16 if dt == "fp16" else torch.float32
        with torch.autocast("cuda", dtype=dtype, enabled=dt == "fp16"): m.prepare_conditionals(ref)
        tx.install_token_fix(m, mode == "fp16fix")
        for r in rows:
            f = pathlib.Path(a.chunks) / a.id / r["tag"] / f"A_{mode}.wav"
            if f.exists(): continue
            torch.manual_seed(r["seed"])
            with torch.autocast("cuda", dtype=dtype, enabled=dt == "fp16"): wav = m.generate(r["translated_text"], language_id="en")
            sf.write(str(f), wav.detach().float().cpu().numpy()[0], m.sr, subtype="PCM_16")
        print("mode done", mode, flush=True); del m; torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
