#!/usr/bin/env python3
"""Diagnostic: independent-ASR intelligibility of every lineage stage (A raw, B trim, C atempo, D final track, E first_res) of each TTS part,
plus controlled sweeps on the SAME raw audio:  H2 trim variants, H3 atempo curve.   Writes <chunks>/<id>/analysis.json

    python scripts/diag/analyze_chunks.py --id 01 --chunks /workspace/dub/tts_diag/chunks
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys, tempfile
import numpy as np, soundfile as sf
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import intel_metrics as im

SR = 24000


def atempo(y: np.ndarray, tempo: float, tmp: pathlib.Path) -> np.ndarray:
    if abs(tempo - 1.0) < 1e-3: return y
    sf.write(str(tmp / "_i.wav"), y, SR, subtype="PCM_16")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-nostdin", "-i", str(tmp / "_i.wav"), "-af", f"atempo={tempo:.6f}", "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", str(tmp / "_o.wav")], check=True)
    return sf.read(str(tmp / "_o.wav"), dtype="float32")[0]


def evaluate(asr: im.Asr, y: np.ndarray, expected: str, with_sound=False) -> dict:
    y16 = asr.to16(y); w = asr.whisper(y16); c = asr.ctc(y16)
    r = {"whisper": w["text"], "w_avg_logprob": w["avg_logprob"], "w_words": w["words"], **{f"w_{k}": v for k, v in im.score(expected, w["text"]).items()},
         "ctc": c["text"], **{f"c_{k}": v for k, v in im.score(expected, c["text"]).items()}, "ctc_conf": c["conf_mean"], "ctc_max_same_symbol_s": c["max_same_symbol_s"], "ctc_max_symbol": c["max_symbol"],
         "dur_s": round(len(y) / SR, 3)}
    if with_sound:
        r.update(im.unexplained_sound(y16, c)); r.update(im.voiced_stable_run(y16))
    return r


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--id", required=True); ap.add_argument("--chunks", required=True); ap.add_argument("--no-sweeps", action="store_true"); a = ap.parse_args()
    d = pathlib.Path(a.chunks) / a.id; rows = json.loads((d / "chunks.json").read_text()); asr = im.Asr(); tmp = pathlib.Path(tempfile.mkdtemp()); out = []
    for n, r in enumerate(rows):
        cd = d / r["tag"]; exp = r["translated_text"]; res = dict(r); res["stages"] = {}
        for st, fn in (("A", "A_raw_tts.wav"), ("B", "B_after_trim.wav"), ("C", "C_after_atempo.wav"), ("D", "D_final_track.wav"), ("E", "E_first_res.wav")):
            if (cd / fn).exists():
                y = sf.read(str(cd / fn), dtype="float32")[0]; res["stages"][st] = evaluate(asr, y, exp, with_sound=(st == "A"))
        A = sf.read(str(cd / "A_raw_tts.wav"), dtype="float32")[0]; B = sf.read(str(cd / "B_after_trim.wav"), dtype="float32")[0]
        s_ = r["trim_left_ms"] / 1000; e_raw = len(A) / SR - r["trim_right_ms"] / 1000
        res["trim_cut"] = {"head_dbfs": im.tail_energy(A[:int(s_ * SR)]), "tail_dbfs": im.tail_energy(A[int(e_raw * SR):]), "tail_ms": r["trim_right_ms"], "speech_dbfs": im.tail_energy(B)}
        if not a.no_sweeps:
            res["h2_trim"] = {}
            for name, (l, rr) in {"none": (None, None), "cur": (0, 0), "L50_R100": (0.05, 0.10), "L50_R200": (0.05, 0.20), "L50_R300": (0.05, 0.30)}.items():
                seg = A if name == "none" else A[max(0, int((s_ - l) * SR)):min(len(A), int((e_raw + rr) * SR))]
                res["h2_trim"][name] = evaluate(asr, seg, exp)
            res["h3_tempo"] = {}
            for t in (1.0, 1.15, 1.3, 1.45, 1.6):
                res["h3_tempo"][str(t)] = evaluate(asr, atempo(B, t, tmp), exp)
        out.append(res)
        A_ = res["stages"]["A"]; print(f"{a.id} {r['tag']:8s} A wer={A_['w_wer']:.2f} cer={A_['c_cer']:.2f} | B {res['stages']['B']['w_wer']:.2f} | C {res['stages']['C']['w_wer']:.2f} | E {res['stages'].get('E',{}).get('w_wer','-')} | {exp[:50]}", flush=True)
    (d / "analysis.json").write_text(json.dumps(out, indent=1, ensure_ascii=False)); print("done", d / "analysis.json"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
