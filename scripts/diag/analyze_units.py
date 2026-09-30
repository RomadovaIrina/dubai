#!/usr/bin/env python3
"""Diagnostic: unit-level (natural sentence, with context) independent-ASR scoring of each lineage stage. Whisper is unreliable on 1-2 s
fragments (it invents 'Thank you for watching'), so the stage comparison is repeated on whole units:
  A_u raw parts concatenated | B_u trimmed parts (+0.12 s gaps) | C_u after atempo (+0.12 s gaps) | D_u reproduced final track | E_u real first_res audio
Expected text = the unit's full translation.   python scripts/diag/analyze_units.py --id 01 --chunks /workspace/dub/tts_diag/chunks
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, soundfile as sf
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import intel_metrics as im

SR = 24000


def rd(p):
    return sf.read(str(p), dtype="float32")[0]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--id", required=True); ap.add_argument("--chunks", required=True); ap.add_argument("--repro", default="/workspace/dub/tts_diag/repro")
    ap.add_argument("--first-res-audio", default="/workspace/dub/tts_diag/first_res_audio"); a = ap.parse_args()
    d = pathlib.Path(a.chunks) / a.id; rows = json.loads((d / "chunks.json").read_text()); asr = im.Asr()
    final = rd(pathlib.Path(a.repro) / a.id / f"work_{a.id}" / "dubbed_24k.wav"); fr = rd(pathlib.Path(a.first_res_audio) / f"{a.id}_final_24k.wav")
    units = {}
    for r in rows: units.setdefault(r["unit_id"], []).append(r)
    gap = np.zeros(int(0.12 * SR), dtype=np.float32); out = []
    for uid, parts in units.items():
        exp = parts[0]["unit_translation"]; res = {"video": a.id, "unit_id": uid, "expected": exp, "source": parts[0]["unit_source_text"], "n_parts": len(parts), "unit_mode": parts[0]["unit_mode"],
                                                  "atempo_max": max(p["atempo"] for p in parts), "t0": min(p["placed_start"] for p in parts), "t1": max(p["placed_end"] for p in parts), "stages": {}}
        cat = lambda fn, g: np.concatenate(sum([[rd(d / p["tag"] / fn), g] for p in parts], [])[:-1])
        aud = {"A": cat("A_raw_tts.wav", np.zeros(int(0.25 * SR), dtype=np.float32)), "B": cat("B_after_trim.wav", gap), "C": cat("C_after_atempo.wav", gap),
               "D": final[max(0, int((res["t0"] - 0.05) * SR)):int((res["t1"] + 0.05) * SR)], "E": fr[max(0, int((res["t0"] - 0.05) * SR)):int((res["t1"] + 0.05) * SR)]}
        for st, y in aud.items():
            y16 = asr.to16(y); w = asr.whisper(y16, pad_s=0.5); c = asr.ctc(y16)
            res["stages"][st] = {"dur_s": round(len(y) / SR, 2), "whisper": w["text"], **{f"w_{k}": v for k, v in im.score(exp, w["text"]).items()}, "ctc": c["text"], **{f"c_{k}": v for k, v in im.score(exp, c["text"]).items()},
                                 "w_avg_logprob": w["avg_logprob"]}
        out.append(res); s = res["stages"]; print(f"{a.id} u{uid:02d} t={res['t0']:.1f} parts={len(parts)} A w{s['A']['w_wer']:.2f}/c{s['A']['c_cer']:.2f}  B w{s['B']['w_wer']:.2f}  C w{s['C']['w_wer']:.2f}  E w{s['E']['w_wer']:.2f}/c{s['E']['c_cer']:.2f} | {exp[:60]}", flush=True)
    (d / "units_analysis.json").write_text(json.dumps(out, indent=1, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
