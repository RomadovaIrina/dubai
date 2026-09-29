#!/usr/bin/env python3
"""Phase C — WhisperX forced alignment vs faster-whisper word timestamps on the synthesized TTS chunks (B2B requirement A/B).

For every TTS part of a burst-aware run (<burst-dir>/work_<id>/burst_alignment.json, tts_u*_p*.wav with known text):
  FW   faster-whisper large-v3, word_timestamps=True (project model, same call as the pipeline ASR) -> words with start/end
  WX   whisperx.align() with the wav2vec2 English alignment model (WAV2VEC2_ASR_BASE_960H, torchaudio pipeline) forced on the
       KNOWN text of the part -> per-word start/end (char-level CTC alignment)
  VAD  Silero VAD speech span of the chunk = what the placement uses today to trim the chunk (e1_burst_align.vad_intervals)
Reports: per chunk the speech span (first word start, last word end) from FW / WX / VAD, the differences in ms, word count
agreement, mean |Δ| of matched word boundaries, and the consequence for placement: how much the chunk START offset on the timeline
would move if the chunk were trimmed by WX / FW word boundaries instead of VAD (in ms and in 25 fps frames). Optionally writes an
alternative dubbed track trimmed by WX boundaries (--write-wx-track) for a full candidate run.
    python scripts/quality/phaseC_whisperx_ab.py --id 04 --burst-dir /tmp/dabai_quality/candidates/burst_s --json /tmp/dabai_quality/manifests/04_phaseC.json
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pilot"))
FPS = 25


def main() -> int:
    import soundfile as sf, torch
    from e1_burst_align import SR16, TTS_SR, ff, vad_intervals
    from pilot_common import MODELS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True); ap.add_argument("--burst-dir", required=True); ap.add_argument("--json", default=None); ap.add_argument("--lang", default="en")
    ap.add_argument("--write-wx-track", action="store_true", help="also build <burst-dir>_wx/work_<id>/ with chunks trimmed by WhisperX word boundaries and re-placed")
    a = ap.parse_args()
    work = pathlib.Path(a.burst_dir) / f"work_{a.id}"; rep = json.loads((work / "burst_alignment.json").read_text())
    from faster_whisper import WhisperModel
    import whisperx
    t0 = time.perf_counter(); fw = WhisperModel(str(MODELS / "whisper-large-v3"), device="cuda", compute_type="int8_float16"); t_fw_load = time.perf_counter() - t0
    t0 = time.perf_counter(); align_model, meta = whisperx.load_align_model(language_code=a.lang, device="cuda"); t_wx_load = time.perf_counter() - t0
    rows = []; t_fw = t_wx = t_vad = 0.0
    for u in rep["units"]:
        for k, t in enumerate(u["tts"]):
            wav = pathlib.Path(t["file"]); y, sr = sf.read(str(wav), dtype="float32")
            if y.ndim > 1: y = y.mean(axis=1)
            dur = len(y) / sr
            w16 = work / f"_ab16_u{u['id']:02d}_p{k}.wav"; ff(["-i", str(wav), "-ar", str(SR16), "-ac", "1", "-c:a", "pcm_s16le", str(w16)])
            s = time.perf_counter(); vad = vad_intervals(w16); t_vad += time.perf_counter() - s
            s = time.perf_counter(); segs, _ = fw.transcribe(str(w16), word_timestamps=True, beam_size=5, language=a.lang, vad_filter=False)
            fw_words = [(w.word.strip(), float(w.start), float(w.end)) for sg in segs for w in (sg.words or [])]; t_fw += time.perf_counter() - s
            s = time.perf_counter(); audio = whisperx.load_audio(str(w16))
            res = whisperx.align([{"text": t["text"], "start": 0.0, "end": dur}], align_model, meta, audio, "cuda", return_char_alignments=False)
            wx_words = [(w["word"], float(w["start"]), float(w["end"])) for sg in res["segments"] for w in sg.get("words", []) if "start" in w]; t_wx += time.perf_counter() - s
            w16.unlink()
            def span(ws): return (round(ws[0][1], 3), round(ws[-1][2], 3)) if ws else None
            vs = (round(max(0.0, vad[0][0] - 0.04), 3), round(min(dur, vad[-1][1] + 0.06), 3)) if vad else None
            fs, xs = span(fw_words), span(wx_words)
            # matched word boundaries (same count and order -> pairwise)
            pair = None
            if fw_words and wx_words and len(fw_words) == len(wx_words):
                d = [abs(f[1] - x[1]) for f, x in zip(fw_words, wx_words)] + [abs(f[2] - x[2]) for f, x in zip(fw_words, wx_words)]
                pair = {"mean_ms": round(1000 * float(np.mean(d)), 1), "p95_ms": round(1000 * float(np.percentile(d, 95)), 1), "max_ms": round(1000 * max(d), 1)}
            rows.append({"unit": u["id"], "part": k, "text": t["text"], "dur_s": round(dur, 3), "n_words_text": len(t["text"].split()), "n_words_fw": len(fw_words), "n_words_wx": len(wx_words),
                         "span_vad": vs, "span_fw": fs, "span_wx": xs,
                         "start_delta_ms": {"fw_vs_vad": round(1000 * (fs[0] - vs[0]), 1) if fs and vs else None, "wx_vs_vad": round(1000 * (xs[0] - vs[0]), 1) if xs and vs else None, "wx_vs_fw": round(1000 * (xs[0] - fs[0]), 1) if xs and fs else None},
                         "end_delta_ms": {"fw_vs_vad": round(1000 * (fs[1] - vs[1]), 1) if fs and vs else None, "wx_vs_vad": round(1000 * (xs[1] - vs[1]), 1) if xs and vs else None, "wx_vs_fw": round(1000 * (xs[1] - fs[1]), 1) if xs and fs else None},
                         "word_boundaries_wx_vs_fw": pair, "fw_words": fw_words, "wx_words": wx_words})
    def agg(key, sub):
        v = [abs(r[key][sub]) for r in rows if r[key].get(sub) is not None]; return {"mean_ms": round(float(np.mean(v)), 1), "p95_ms": round(float(np.percentile(v, 95)), 1), "n": len(v)} if v else None
    pairs = [r["word_boundaries_wx_vs_fw"] for r in rows if r["word_boundaries_wx_vs_fw"]]
    summary = {"chunks": len(rows), "words_text": sum(r["n_words_text"] for r in rows), "fw_word_count_matches_text": sum(r["n_words_fw"] == r["n_words_text"] for r in rows),
               "wx_word_count_matches_text": sum(r["n_words_wx"] == r["n_words_text"] for r in rows),
               "chunk_start_abs_delta": {"fw_vs_vad": agg("start_delta_ms", "fw_vs_vad"), "wx_vs_vad": agg("start_delta_ms", "wx_vs_vad"), "wx_vs_fw": agg("start_delta_ms", "wx_vs_fw")},
               "chunk_end_abs_delta": {"fw_vs_vad": agg("end_delta_ms", "fw_vs_vad"), "wx_vs_vad": agg("end_delta_ms", "wx_vs_vad"), "wx_vs_fw": agg("end_delta_ms", "wx_vs_fw")},
               "word_boundary_wx_vs_fw": {"chunks_compared": len(pairs), "mean_ms": round(float(np.mean([p["mean_ms"] for p in pairs])), 1) if pairs else None, "p95_ms": round(float(np.mean([p["p95_ms"] for p in pairs])), 1) if pairs else None},
               "start_moves_ge_1_frame": {"wx_vs_vad": sum(1 for r in rows if r["start_delta_ms"]["wx_vs_vad"] is not None and abs(r["start_delta_ms"]["wx_vs_vad"]) >= 1000 / FPS),
                                          "fw_vs_vad": sum(1 for r in rows if r["start_delta_ms"]["fw_vs_vad"] is not None and abs(r["start_delta_ms"]["fw_vs_vad"]) >= 1000 / FPS)},
               "seconds": {"fw_load": round(t_fw_load, 2), "wx_load": round(t_wx_load, 2), "fw_per_chunk": round(t_fw / max(len(rows), 1), 3), "wx_per_chunk": round(t_wx / max(len(rows), 1), 3), "vad_per_chunk": round(t_vad / max(len(rows), 1), 3)}}
    print(json.dumps(summary, indent=1))
    if a.write_wx_track:   # chunks trimmed by WhisperX first-word start / last-word end (same margins as the VAD trim), re-placed with the run's options
        from e1_burst_align import PlaceOpts, build_dubbed_track, place_groups, placement_summary
        out = pathlib.Path(str(a.burst_dir) + "_wx") / f"work_{a.id}"; out.mkdir(parents=True, exist_ok=True)
        opts = PlaceOpts(rep["atempo_cap"], rep["hard_cap"], rep["extend"] == "on", 0.06, rep["fill_slowdown"], spill=rep.get("spill") == "on"); units_out = []
        by = {(r["unit"], r["part"]): r for r in rows}
        for u in rep["units"]:
            chunks = []
            for k, t in enumerate(u["tts"]):
                y, sr = sf.read(t["file"], dtype="float32"); dur = len(y) / sr; r = by[(u["id"], k)]; sp = r["span_wx"] or r["span_vad"]
                s_, e_ = max(0.0, sp[0] - 0.04), min(dur, sp[1] + 0.06); chunks.append(y[int(s_ * sr):int(e_ * sr)])
            bursts = [tuple(p["burst"]) for p in u["placements"]]; slot = tuple(u["slot"])
            nxt = rep["units"][rep["units"].index(u) + 1]["placements"][0]["burst"][0] if rep.get("spill") == "on" and rep["units"].index(u) + 1 < len(rep["units"]) else (rep["units"][rep["units"].index(u) + 1]["slot"][0] if rep["units"].index(u) + 1 < len(rep["units"]) else slot[1] + 3600)
            track, placements = place_groups(chunks, bursts, slot, nxt, opts, out, f"u{u['id']:02d}", sr=TTS_SR, labels=[[k] for k in range(len(chunks))])
            f = out / f"aligned_u{u['id']:02d}.wav"; sf.write(str(f), track, TTS_SR, subtype="PCM_16")
            units_out.append({"id": u["id"], "slot": list(slot), "placements": placements, "max_atempo": max(p.get("atempo", 1.0) for p in placements), "file": str(f), "tts": u["tts"]})
        man_dur = json.loads((pathlib.Path("/tmp/dabai_quality/baseline") / f"{a.id}_baseline.manifest.json").read_text())["source"]["duration_s"]
        tr = build_dubbed_track([(r["file"], r["slot"][0]) for r in units_out], man_dur, out)
        rep2 = {**rep, "mode": "burst_aware_wx_trim", "units": units_out, "summary": {**placement_summary(units_out), **tr}}
        (out / "burst_alignment.json").write_text(json.dumps(rep2, indent=1, ensure_ascii=False)); (out / "e1_alignment.json").write_text(json.dumps(rep2, indent=1, ensure_ascii=False))
        summary["wx_track"] = str(out); print("wx-trimmed track ->", out, json.dumps(rep2["summary"]))
    if a.json:
        pathlib.Path(a.json).parent.mkdir(parents=True, exist_ok=True); pathlib.Path(a.json).write_text(json.dumps({"id": a.id, "summary": summary, "chunks": rows}, indent=1, ensure_ascii=False)); print(f"-> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
