#!/usr/bin/env python3
"""Diagnostic: does text form matter for the current fragment parts?  On the fp16-fixed path, per part (3 seeds each):
  asis    the part text exactly as the pipeline sends it (Chatterbox punc_norm then capitalises it and appends '.' when no ender is present)
  comma   a non-final part of a unit gets a trailing ',' instead (punc_norm adds nothing after ','), capitalised first letter removed -> the fragment is
          not declared a finished sentence; final parts unchanged.
Scores CTC CER / Whisper WER / duration.   python scripts/diag/part_punct.py --id 03
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, soundfile as sf, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import tts_experiments as tx, intel_metrics as im


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--id", required=True); a = ap.parse_args()
    man = json.loads(pathlib.Path(f"/workspace/dub/tts_diag/repro/{a.id}/{a.id}.manifest.json").read_text()); ref = man["tts"]["references"][man["units"][0]["speaker"]]["file"]
    rows = json.loads(pathlib.Path(f"/workspace/dub/tts_diag/chunks/{a.id}/chunks.json").read_text()); out = pathlib.Path(f"/workspace/dub/tts_diag/punct/{a.id}"); out.mkdir(parents=True, exist_ok=True)
    m = tx.load("fp16")
    with torch.autocast("cuda", dtype=torch.float16): m.prepare_conditionals(ref)
    tx.install_token_fix(m, True); res = []
    for r in rows:
        nonfinal = r["part_id"] < r["n_parts"] - 1; t = r["translated_text"].strip()
        variants = {"asis": t, "comma": (t.rstrip(".!? ") + ",") if nonfinal else t}
        for name, txt in variants.items():
            for k in range(3):
                f = out / f"{r['tag']}_{name}_s{k}.wav"
                if not f.exists():
                    torch.manual_seed(r["seed"] + k)
                    with torch.autocast("cuda", dtype=torch.float16): wav = m.generate(txt, language_id="en")
                    sf.write(str(f), wav.detach().float().cpu().numpy()[0], m.sr, subtype="PCM_16")
    del m; torch.cuda.empty_cache(); asr = im.Asr()
    for r in rows:
        exp = r["translated_text"]
        for name in ("asis", "comma"):
            for k in range(3):
                y, sr = sf.read(str(out / f"{r['tag']}_{name}_s{k}.wav"), dtype="float32"); y16 = asr.to16(y, sr); w = asr.whisper(y16); c = asr.ctc(y16)
                res.append({"tag": r["tag"], "variant": name, "seed": k, "nonfinal": r["part_id"] < r["n_parts"] - 1, "words": len(im.norm(exp)), "dur": round(len(y) / sr, 3), "w_wer": im.score(exp, w["text"])["wer"], "c_cer": im.score(exp, c["text"])["cer"],
                            "c_final_missing": im.score(exp, c["text"])["final_word_missing"], "whisper": w["text"]})
    (out / "scores.json").write_text(json.dumps(res, indent=1, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
