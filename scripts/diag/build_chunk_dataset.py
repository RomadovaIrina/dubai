#!/usr/bin/env python3
"""Diagnostic: per TTS part of a reproduced run (scripts/diag/repro_audio.py) write the audio lineage files and a metadata table.

  A_raw_tts.wav        raw Chatterbox output (as saved by the runner)
  B_after_trim.wav     A cut to its Silero speech span  (runner: t["speech_span"])
  C_after_atempo.wav   B after the atempo filter that place_groups / place_slot_fallback applied (B itself when no tempo was applied)
  D_final_track.wav    fragment of the reproduced dubbed track (dubbed_24k.wav) at the placed interval (+-0.15 s context)
  E_first_res.wav      same interval cut from the audio of the real first_res video (final AAC) - proves the lineage

    python scripts/diag/build_chunk_dataset.py --id 01 --repro /workspace/dub/tts_diag/repro --first-res-audio /workspace/dub/tts_diag/first_res_audio --out /workspace/dub/tts_diag/chunks
"""
from __future__ import annotations
import argparse, csv, json, pathlib, subprocess
import numpy as np, soundfile as sf

SR = 24000; CTX = 0.15; GAP = 0.06


def atempo(y: np.ndarray, tempo: float, tmp: pathlib.Path) -> np.ndarray:
    sf.write(str(tmp / "_i.wav"), y, SR, subtype="PCM_16")
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-nostdin", "-i", str(tmp / "_i.wav"), "-af", f"atempo={tempo:.6f}", "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", str(tmp / "_o.wav")], check=True)
    return sf.read(str(tmp / "_o.wav"), dtype="float32")[0]


def cut(y: np.ndarray, a: float, b: float) -> np.ndarray:
    return y[max(0, int(round(a * SR))):max(0, int(round(b * SR)))]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--id", required=True); ap.add_argument("--repro", required=True); ap.add_argument("--first-res-audio", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rd = pathlib.Path(a.repro) / a.id; man = json.loads((rd / f"{a.id}.manifest.json").read_text()); units = man["units"]
    final, sr = sf.read(str(rd / f"work_{a.id}" / "dubbed_24k.wav"), dtype="float32"); assert sr == SR
    fr, sr2 = sf.read(str(pathlib.Path(a.first_res_audio) / f"{a.id}_final_24k.wav"), dtype="float32"); assert sr2 == SR
    out = pathlib.Path(a.out) / a.id; out.mkdir(parents=True, exist_ok=True); tmp = out / "_tmp"; tmp.mkdir(exist_ok=True)
    rows = []
    for u in units:
        al = u["alignment"]; pls = al["placements"]; fallback = pls[0].get("mode") == "slot_fallback"; tempo_u = pls[0]["atempo"] if fallback else None
        b_chunks = []
        for k, t in enumerate(u["tts_parts"]):
            y, sr_ = sf.read(t["file"], dtype="float32"); assert sr_ == SR
            s_, e_ = t["speech_span"]; B = y[int(s_ * SR):int(e_ * SR)]; b_chunks.append(B)
        cum = 0.0
        for k, t in enumerate(u["tts_parts"]):
            tag = f"u{u['id']:02d}_p{k}"; d = out / tag; d.mkdir(exist_ok=True)
            A, _ = sf.read(t["file"], dtype="float32"); s_, e_ = t["speech_span"]; B = b_chunks[k]
            if fallback:
                tempo = tempo_u; C = atempo(B, tempo, tmp) if tempo > 1.0 else B
                p0 = al["placements"][0]["placed"][0] + cum / tempo; p1 = p0 + len(B) / SR / tempo; cum += len(B) / SR + GAP
                placed = [round(p0, 3), round(p1, 3)]; pl = pls[0]; win = pl["window"]; burst = pl["burst"]; ratio = pl["ratio"]; spill = 0.0; cutend = 0.0; mode = "slot_fallback"
            else:
                pl = pls[k] if k < len(pls) else {}; tempo = pl.get("atempo", 1.0); placed = pl.get("placed"); win = pl.get("window"); burst = pl.get("burst"); ratio = pl.get("ratio")
                spill = pl.get("spilled_past_slot_s", 0.0); cutend = pl.get("cut_at_slot_end_s", 0.0); mode = "burst"
                tf = rd / "atempo" / f"u{u['id']:02d}_{k}.wav"
                C = sf.read(str(tf), dtype="float32")[0] if tf.exists() else B
            sf.write(str(d / "A_raw_tts.wav"), A, SR, subtype="PCM_16"); sf.write(str(d / "B_after_trim.wav"), B, SR, subtype="PCM_16"); sf.write(str(d / "C_after_atempo.wav"), C, SR, subtype="PCM_16")
            if placed:
                sf.write(str(d / "D_final_track.wav"), cut(final, placed[0] - CTX, placed[1] + CTX), SR, subtype="PCM_16")
                sf.write(str(d / "E_first_res.wav"), cut(fr, placed[0] - CTX, placed[1] + CTX), SR, subtype="PCM_16")
            src_text = pls[k]["source_words"] if (not fallback and k < len(pls) and "source_words" in pls[k]) else (pls[0].get("source_words", "").split(" | ")[k] if fallback and k < len(pls[0].get("source_words", "").split(" | ")) else u["text"])
            rows.append({"video": a.id, "tag": tag, "unit_id": u["id"], "part_id": k, "n_parts": len(u["tts_parts"]), "unit_mode": mode, "speaker": u["speaker"], "source_text": src_text,
                         "unit_source_text": u["text"], "unit_translation": u["translation"], "translated_text": t["text"], "split_method": u["split"]["method"],
                         "seed": 1247 + 100 * u["id"] + k, "raw_duration": round(len(A) / SR, 3), "trimmed_duration": round(len(B) / SR, 3), "trim_left_ms": round(s_ * 1000),
                         "trim_right_ms": round((len(A) / SR - e_) * 1000), "burst_window_s": burst, "window_s": win, "fit_ratio": ratio, "atempo": tempo, "spill": spill, "cut_at_slot_end_s": cutend,
                         "fallback_used": fallback, "placed_start": placed[0] if placed else None, "placed_end": placed[1] if placed else None,
                         "slot": [u["slot"]["start"], u["slot"]["end"]], "dtype": u["tts"]["dtype"]})
    (out / "chunks.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    with open(out / "chunks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); [w.writerow(r) for r in rows]
    for p in tmp.iterdir(): p.unlink()
    tmp.rmdir(); print(f"{a.id}: {len(rows)} parts in {len(units)} units -> {out}"); return 0


if __name__ == "__main__":
    raise SystemExit(main())
