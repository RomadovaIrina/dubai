#!/usr/bin/env python3
"""Evaluate a dubbing run (new or old) on the AUDIO that is actually in the output: per-unit independent ASR (Whisper WER, CTC CER vs the unit's dubbed text),
stray interjections, final-word-missing, TTS QA / retry statistics from the manifest, atempo distribution.  Works for any burst-mode manifest.

    python scripts/diag/eval_run.py --manifest <x>.manifest.json --audio <dubbed_24k.wav | final.mp4 audio> --out report.json [--label old]
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys, tempfile
import numpy as np, soundfile as sf
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot"))
import tts_qa as qa_mod

SR = 24000


def load_audio(p: str) -> np.ndarray:
    if p.lower().endswith((".mp4", ".mov", ".m4a")):
        t = pathlib.Path(tempfile.mkdtemp()) / "a.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", p, "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(t)], check=True); p = str(t)
    y, sr = sf.read(p, dtype="float32"); assert sr == SR, sr
    return y if y.ndim == 1 else y.mean(1)


def unit_text(u: dict) -> str:
    return " ".join(u["parts"]) if u.get("parts") else u["translation"]


def unit_window(u: dict) -> tuple[float, float]:
    pls = [p for p in u["alignment"].get("placements", []) if p.get("placed")]
    if pls: return min(p["placed"][0] for p in pls), max(p["placed"][1] for p in pls)
    return u["slot"]["start"], u["slot"]["end"]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--manifest", required=True); ap.add_argument("--audio", required=True); ap.add_argument("--out", required=True); ap.add_argument("--label", default="run"); a = ap.parse_args()
    man = json.loads(pathlib.Path(a.manifest).read_text()); y = load_audio(a.audio); qa = qa_mod.TtsQA(); rows = []
    for u in man["units"]:
        t0, t1 = unit_window(u); seg = y[max(0, int((t0 - 0.05) * SR)):int((t1 + 0.05) * SR)]; exp = unit_text(u)
        if len(seg) < 2400: continue
        r = qa.assess(exp, seg)
        rows.append({"unit": u["id"], "t0": round(t0, 2), "t1": round(t1, 2), "expected": exp, "whisper": r["whisper"], "ctc": r["ctc"], "wer": round(qa_mod.wer(exp, r["whisper"]), 3), "cer_ctc": r["ctc_cer"], "cer_whisper": r["w_cer"],
                     "interjections": r["interjections"], "final_word_missing": r["final_word_missing"], "words": r["words"]})
    at = []; parts = 0
    for u in man["units"]:
        for p in u["alignment"].get("placements", []):
            if p.get("placed"): at.append(p.get("atempo", 1.0)); parts += 1
    at = np.array(at) if at else np.array([1.0])
    tts = [tp for u in man["units"] for tp in u.get("tts_parts", [])]
    att = [len(tp.get("attempts", [])) for tp in tts if tp.get("attempts")]
    retried = [tp for tp in tts if len(tp.get("attempts", [])) > 1]
    succ = [tp for tp in retried if not tp.get("qa_bad_final", False)]
    summ = {"label": a.label, "units": len(rows), "wer_whisper_mean": round(float(np.mean([r["wer"] for r in rows])), 3), "cer_ctc_mean": round(float(np.mean([r["cer_ctc"] for r in rows])), 3),
            "cer_whisper_mean": round(float(np.mean([r["cer_whisper"] for r in rows])), 3), "units_bad(cer_ctc>=0.3)": sum(r["cer_ctc"] >= 0.3 for r in rows),
            "interjection_units": sum(bool(r["interjections"]) for r in rows), "final_word_missing_units": sum(r["final_word_missing"] for r in rows),
            "tts_parts": len(tts) or parts, "placed_groups": parts, "parts_min_words": min((len(qa_mod.norm(tp["text"])) for tp in tts), default=None), "parts_le2_words": sum(len(qa_mod.norm(tp["text"])) <= 2 for tp in tts),
            "retried_parts": len(retried), "retry_success": len(succ), "retry_success_rate": round(len(succ) / len(retried), 3) if retried else None,
            "atempo_max": round(float(at.max()), 3), "atempo_gt_1.15": int((at > 1.15 + 1e-3).sum()), "atempo_gt_1.2": int((at > 1.2 + 1e-3).sum()), "atempo_gt_1.3": int((at > 1.3 + 1e-3).sum()), "atempo_median": round(float(np.median(at)), 3)}
    pathlib.Path(a.out).write_text(json.dumps({"summary": summ, "units": rows}, indent=1, ensure_ascii=False))
    print(json.dumps(summ, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
