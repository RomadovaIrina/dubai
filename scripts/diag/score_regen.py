#!/usr/bin/env python3
"""Diagnostic: score the regenerated parts (part_regen.py) per chunk with the independent ASRs, and repeat the post-processing sweeps on the
CLEAN audio (A_fp16fix): production VAD trim (exactly run_clean_pipeline_05.align_units_burst), H2 trim variants, H3 atempo curve.
Writes <chunks>/<id>/regen_scores.json.   python scripts/diag/score_regen.py --id 01
"""
from __future__ import annotations
import argparse, json, pathlib, sys, tempfile, subprocess
import numpy as np, soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1] / "scripts" / "quality")); sys.path.insert(0, str(HERE.parents[1] / "scripts" / "pilot"))
import intel_metrics as im
import e1_burst_align as e1
from analyze_chunks import atempo, evaluate

SR = 24000


def prod_trim(y: np.ndarray, tmp: pathlib.Path) -> tuple[np.ndarray, float, float]:
    """Exactly the pipeline: 16 k copy -> Silero (min_silence 50 ms, min_speech 60 ms, pad 40 ms) -> span = first.start-0.04 .. last.end+0.06."""
    import librosa
    y16 = librosa.resample(y, orig_sr=SR, target_sr=16000); sf.write(str(tmp / "t16.wav"), y16, 16000, subtype="PCM_16")
    raw = e1.vad_intervals(tmp / "t16.wav") or [(0.0, len(y) / SR)]
    s_, e_ = max(0.0, raw[0][0] - 0.04), min(len(y) / SR, raw[-1][1] + 0.06)
    return y[int(s_ * SR):int(e_ * SR)], s_, e_


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--id", required=True); ap.add_argument("--chunks", default="/workspace/dub/tts_diag/chunks"); a = ap.parse_args()
    d = pathlib.Path(a.chunks) / a.id; rows = json.loads((d / "chunks.json").read_text()); asr = im.Asr(); tmp = pathlib.Path(tempfile.mkdtemp()); out = []
    for r in rows:
        cd = d / r["tag"]; exp = r["translated_text"]; res = {"video": a.id, "tag": r["tag"], "atempo": r["atempo"], "text": exp}
        for mode in ("prod_fp16", "fp16fix", "fp32"):
            f = cd / f"A_{mode}.wav"
            if f.exists():
                y = sf.read(str(f), dtype="float32")[0]; res[mode] = evaluate(asr, y, exp, with_sound=False); res[mode]["held"] = im.held_vowel_tail(y)
        if "fp16fix" in res:
            A = sf.read(str(cd / "A_fp16fix.wav"), dtype="float32")[0]; B, s_, e_ = prod_trim(A, tmp); res["fix_trim"] = {"left_ms": round(s_ * 1000), "right_ms": round((len(A) / SR - e_) * 1000), "trimmed_s": round(len(B) / SR, 3)}
            res["fix_B"] = evaluate(asr, B, exp)
            res["fix_h2"] = {}
            for name, (l, rr) in {"none": (None, None), "cur": (0, 0), "L50_R100": (0.05, 0.10), "L50_R200": (0.05, 0.20), "L50_R300": (0.05, 0.30)}.items():
                seg = A if name == "none" else A[max(0, int((s_ - l) * SR)):min(len(A), int((e_ + rr) * SR))]; res["fix_h2"][name] = evaluate(asr, seg, exp)
            res["fix_h3"] = {str(t): evaluate(asr, atempo(B, t, tmp), exp) for t in (1.0, 1.15, 1.3, 1.45, 1.6)}
        out.append(res); p, f_ = res.get("prod_fp16"), res.get("fp16fix")
        print(f"{a.id} {r['tag']:7s} prod CER {p['c_cer'] if p else '-'} fix CER {f_['c_cer'] if f_ else '-'} | {exp[:45]}", flush=True)
    (d / "regen_scores.json").write_text(json.dumps(out, indent=1, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
