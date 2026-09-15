#!/usr/bin/env python
"""0.11 — stage timing for VAD-split + concat + subtitle burn, TIMELINE-PRESERVING.

    S = T_vad_split + T_concat + T_subtitle_burn

Production semantics (fixed 2026-09-15; the previous version dropped non-speech intervals and compressed the
timeline — those results are LEGACY_INVALID_TIMELINE and live in stages_04.json / stages.jsonl only):

  original timeline -> ordered chunks covering [0, source_duration] with no gaps:
      SPEECH        interval from Silero VAD  (would go to the dubbing/lipsync slot)
      PASS_THROUGH  every interval between/around speech (copied as-is)
  T_vad_split       Silero inference + building that timeline + actually cutting EVERY chunk (speech and pass-through)
  T_concat          assembling ALL chunks back, in the original order, into one full-length clip
  T_subtitle_burn   hard-burning subtitles whose cues are on the ORIGINAL timeline into the full-length assembled clip

Deliberately NOT in S (reported as excluded_from_S): demux (source -> 16 kHz mono wav) and ASR (--asr, real subtitle
text with real timestamps). No scaling, no fps change: the source is measured as-is. No dubbing/LatentSync here.

Split mode: reencode (frame-accurate, the contractual result) or copy (keyframe-snapped, informational only).
Runtime media goes to /tmp/dabai_pilot_11/<name>/ (never into the repo); the repo gets 0.11_stages_<name>.json.

    source scripts/env.sh
    python scripts/pilot/measure_stages.py test_videos/04.mp4 --repeat 2 --asr
    python scripts/pilot/measure_stages.py test_videos/04.mp4 --split-mode copy
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT, ROOT, THREADS, append_jsonl, sh, write_json
from measure_vram import measure_gpu_memory, _latest_snapshot_id

STAGES_JSONL = PILOT / "0.11_stages.jsonl"          # new methodology only; stages.jsonl is legacy
WORK_ROOT = pathlib.Path("/tmp/dabai_pilot_11")
IN_S = ("vad_split", "concat", "subtitle_burn")


def ff(args: list[str], cwd: str | None = None) -> None:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-nostdin", *args],
                       capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(args[:10])}…\n{r.stderr[-700:]}")


def probe(path: pathlib.Path) -> dict:
    """Record a file exactly as it is — no assumptions, no normalisation."""
    raw = sh(f'ffprobe -v error -print_format json -show_format -show_streams "{path}"', timeout=120)
    d = json.loads(raw) if raw else {}
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in d.get("streams", []) if s.get("codec_type") == "audio"), {})

    def _fps(rate):
        try:
            num, den = rate.split("/")
            return round(int(num) / int(den), 4) if int(den) else None
        except Exception:
            return None
    r_rate, avg_rate = v.get("r_frame_rate", "0/0"), v.get("avg_frame_rate", "0/0")
    fps, avg = _fps(r_rate), _fps(avg_rate)
    return {
        "duration_s": float(d.get("format", {}).get("duration", 0) or 0),
        "size_bytes": int(d.get("format", {}).get("size", 0) or 0),
        "container": d.get("format", {}).get("format_name"),
        "video": {"codec": v.get("codec_name"), "profile": v.get("profile"), "width": v.get("width"), "height": v.get("height"),
                  "r_frame_rate": r_rate, "avg_frame_rate": avg_rate, "fps": fps, "avg_fps": avg,
                  "vfr": (fps is not None and avg is not None and abs(fps - avg) > 0.01),
                  "pix_fmt": v.get("pix_fmt"), "nb_frames": v.get("nb_frames"), "duration_s": float(v.get("duration") or 0),
                  "bit_rate": v.get("bit_rate")},
        "audio": {"codec": a.get("codec_name"), "sample_rate": a.get("sample_rate"), "channels": a.get("channels"),
                  "channel_layout": a.get("channel_layout"), "bit_rate": a.get("bit_rate"), "duration_s": float(a.get("duration") or 0)},
        "has_video": bool(v), "has_audio": bool(a),
    }


def srt_ts(t: float) -> str:
    ms = int(round(t * 1000)); h, ms = divmod(ms, 3600000); m, ms = divmod(ms, 60000); s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: pathlib.Path, cues) -> None:
    path.write_text("\n".join(f"{i}\n{srt_ts(a)} --> {srt_ts(b)}\n{t}\n" for i, (a, b, t) in enumerate(cues, 1)), encoding="utf-8")


def build_timeline(speech: list[dict], duration: float, min_gap: float = 0.0) -> list[dict]:
    """Ordered chunks covering [0, duration] with no gaps: SPEECH intervals from VAD, PASS_THROUGH everywhere else."""
    chunks, t = [], 0.0
    for s in speech:
        a, b = max(0.0, float(s["start"])), min(duration, float(s["end"]))
        if b <= a:
            continue
        if a - t > min_gap:
            chunks.append({"start_s": t, "end_s": a, "type": "PASS_THROUGH"})
        chunks.append({"start_s": max(a, t), "end_s": b, "type": "SPEECH"})
        t = b
    if duration - t > 1e-6:
        chunks.append({"start_s": t, "end_s": duration, "type": "PASS_THROUGH"})
    if not chunks:
        chunks.append({"start_s": 0.0, "end_s": duration, "type": "PASS_THROUGH"})
    for i, c in enumerate(chunks):
        c["index"] = i; c["duration_s"] = round(c["end_s"] - c["start_s"], 4)
        c["start_s"] = round(c["start_s"], 4); c["end_s"] = round(c["end_s"], 4)
    return chunks


class Stage:
    """Times a block and records its VRAM footprint through the 0.4 machinery."""

    def __init__(self, name: str, results: dict):
        self.name, self.results = name, results

    def __enter__(self):
        self._m = measure_gpu_memory(f"stage/{self.name}", quiet=True, jsonl=PILOT / "vram.jsonl")
        self._m.__enter__(); self._t0 = time.perf_counter(); return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self._t0; self._m.__exit__(*exc)
        self.results[self.name] = {"seconds": round(dt, 4), "peak_process_mib": self._m.result.get("peak_process_mib")}
        return False


def run_once(src: pathlib.Path, source: dict, args) -> dict:
    work = WORK_ROOT / src.stem
    if work.exists():
        shutil.rmtree(work)
    (work / "chunks").mkdir(parents=True)
    res: dict = {}
    wav = work / "audio16k.wav"
    duration = source["duration_s"]

    # ---- demux: preparatory, excluded from S --------------------------------
    with Stage("demux", res):
        ff(["-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)])

    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
    torch.set_num_threads(THREADS)
    model = load_silero_vad()            # model load is startup, not pipeline work
    audio = read_audio(str(wav), sampling_rate=16000)

    # ---- T_vad_split: VAD + full timeline + actual cutting of EVERY chunk ----
    chunk_ext = "mov"   # pcm audio keeps chunk boundaries sample-accurate; the final burn encodes aac
    with Stage("vad_split", res):
        speech = get_speech_timestamps(audio, model, sampling_rate=16000, return_seconds=True)
        chunks = build_timeline(speech, duration)        # no speech -> one PASS_THROUGH chunk, never a failure
        for c in chunks:
            out = work / "chunks" / f"{c['index']:05d}_{c['type']}.{chunk_ext}"
            cut = ["-ss", f"{c['start_s']:.4f}", "-to", f"{c['end_s']:.4f}", "-i", str(src)]
            if args.split_mode == "copy":
                ff([*cut, "-c", "copy", "-avoid_negative_ts", "make_zero", out.with_suffix(".mp4")]); c["file"] = out.with_suffix(".mp4").name
            else:   # frame-accurate: decode + encode, no scaling / fps change
                ff([*cut, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "pcm_s16le", str(out)]); c["file"] = out.name
    speech_s = round(sum(c["duration_s"] for c in chunks if c["type"] == "SPEECH"), 3)
    pt = [c for c in chunks if c["type"] == "PASS_THROUGH"]
    res["vad_split"].update(split_mode=args.split_mode, speech_intervals=sum(1 for c in chunks if c["type"] == "SPEECH"),
                            speech_seconds=speech_s, speech_pct=round(speech_s / duration * 100, 2) if duration else None,
                            pass_through_intervals=len(pt), pass_through_seconds=round(sum(c["duration_s"] for c in pt), 3),
                            chunks_total=len(chunks), timeline_covered_s=round(sum(c["duration_s"] for c in chunks), 4))

    # ---- optional ASR: real subtitle text on the ORIGINAL timeline, timed separately ----
    if args.asr:
        from faster_whisper import WhisperModel
        m = WhisperModel(str(ROOT / "models" / "whisper-large-v3"), device="cuda", compute_type="int8_float16")
        with Stage("asr", res):
            segs, _ = m.transcribe(str(wav), word_timestamps=False)
            cues = [(float(s.start), float(s.end), s.text.strip()) for s in segs if s.text.strip()]
        res["asr"]["cues"] = len(cues); res["asr"]["source"] = "faster-whisper large-v3 segment timestamps (original timeline)"
        del m; torch.cuda.empty_cache()
    else:
        cues = [(c["start_s"], c["end_s"], f"[speech {i + 1}]") for i, c in enumerate(x for x in chunks if x["type"] == "SPEECH")]
    if not cues:
        cues = [(0.0, min(2.0, duration), "[no speech detected]")]

    # ---- T_concat: ALL chunks, original order ---------------------------------
    (work / "chunks" / "concat.txt").write_text("".join(f"file '{c['file']}'\n" for c in chunks))
    assembled = work / ("assembled.mp4" if args.split_mode == "copy" else "assembled.mov")
    with Stage("concat", res):
        ff(["-f", "concat", "-safe", "0", "-i", "concat.txt", "-c", "copy", str(assembled)], cwd=str(work / "chunks"))
    ap_ = probe(assembled)
    res["concat"].update(out_seconds=round(ap_["duration_s"], 4), out_resolution=f"{ap_['video']['width']}x{ap_['video']['height']}",
                         out_fps=ap_["video"]["fps"], has_video=ap_["has_video"], has_audio=ap_["has_audio"], chunks=len(chunks))

    # ---- T_subtitle_burn: hard-burn into the FULL-LENGTH assembled clip -----
    write_srt(work / "subs.srt", cues)
    burned = work / "final_burned.mp4"
    with Stage("subtitle_burn", res):
        ff(["-i", str(assembled), "-vf", "subtitles=subs.srt", "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k", str(burned)], cwd=str(work))
    bp = probe(burned)
    res["subtitle_burn"].update(cues=len(cues), first_cue=[round(cues[0][0], 3), round(cues[0][1], 3)], last_cue=[round(cues[-1][0], 3), round(cues[-1][1], 3)],
                                out_resolution=f"{bp['video']['width']}x{bp['video']['height']}", out_fps=bp["video"]["fps"],
                                out_seconds=round(bp["duration_s"], 4), has_video=bp["has_video"], has_audio=bp["has_audio"])

    # ---- validation ----------------------------------------------------------
    fps = source["video"]["avg_fps"] or source["video"]["fps"] or 25.0
    tol = max(0.10, 2.0 / fps)
    d_asm = round(ap_["duration_s"] - duration, 4); d_fin = round(bp["duration_s"] - duration, 4)
    cover_ok = abs(res["vad_split"]["timeline_covered_s"] - duration) < 1e-3 and all(
        abs(chunks[i]["end_s"] - chunks[i + 1]["start_s"]) < 1e-6 for i in range(len(chunks) - 1)) and chunks[0]["start_s"] == 0.0
    checks = {
        "timeline_covered_no_gaps": cover_ok,
        "concat_includes_speech_and_pass_through": all(c["file"] for c in chunks) and len(chunks) == res["concat"]["chunks"],
        "assembled_duration_within_tol": abs(d_asm) <= tol,
        "final_duration_within_tol": abs(d_fin) <= tol,
        "assembled_has_video_audio": ap_["has_video"] and ap_["has_audio"],
        "final_has_video_audio": bp["has_video"] and bp["has_audio"],
        "resolution_preserved": (bp["video"]["width"], bp["video"]["height"]) == (source["video"]["width"], source["video"]["height"]),
        "subtitles_on_original_timeline": all(0.0 <= a <= b <= duration + tol for a, b, _ in cues),
        "timeline_not_compressed": ap_["duration_s"] >= duration - tol,
    }
    res["validation"] = {"tolerance_s": round(tol, 4), "tolerance_rule": "max(0.10 s, 2 source frames)", "assembled_delta_s": d_asm,
                         "final_delta_s": d_fin, "checks": checks, "pass": all(checks.values())}
    res["timeline"] = chunks
    res["cues"] = [[round(a, 3), round(b, 3), t] for a, b, t in cues]
    res["_S_seconds"] = round(sum(res[k]["seconds"] for k in IN_S), 4)
    res["_final_probe"] = bp; res["_assembled_probe"] = ap_
    res["work_dir"] = str(work)
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--split-mode", choices=["reencode", "copy"], default="reencode",
                    help="reencode = frame-accurate cuts (contractual); copy = keyframe-snapped (informational)")
    ap.add_argument("--asr", action="store_true", help="real transcription for subtitle text (timed separately, not in S)")
    ap.add_argument("--tag", default="", help="suffix for the output json (e.g. 'copy' for the informational copy-mode run)")
    ap.add_argument("--keep-work", action="store_true")
    a = ap.parse_args()
    if not a.input:
        ap.print_help(); return 2
    src = pathlib.Path(a.input).resolve()
    if not src.exists():
        print(f"no such file: {src}", file=sys.stderr); return 1
    source = probe(src)
    if source["duration_s"] <= 0:
        print(f"FAIL: ffprobe reports no duration for {src}", file=sys.stderr); return 1

    v, au = source["video"], source["audio"]
    print("=" * 74); print("0.11  timeline-preserving S  (source as-is: no upscale, no fps change)"); print("=" * 74)
    print(f"  file        {src}\n  duration    {source['duration_s']:.3f} s")
    print(f"  video       {v['width']}x{v['height']}  r={v['fps']} avg={v['avg_fps']} {'VFR' if v['vfr'] else 'CFR'}  {v['codec']} {v['pix_fmt']}")
    print(f"  audio       {au['codec']}  {au['sample_rate']} Hz  {au['channels']}ch")
    print(f"  split mode  {a.split_mode}    asr={a.asr}    repeat={a.repeat}\n")

    runs = []
    for i in range(a.repeat):
        if a.repeat > 1:
            print(f"--- run {i + 1}/{a.repeat}")
        r = run_once(src, source, a); runs.append(r)
        for k in ("demux", "vad_split", "asr", "concat", "subtitle_burn"):
            if k in r:
                print(f"    {k:<16} {r[k]['seconds']:8.3f}s{'' if k in IN_S else '   <- excluded from S'}")
        print(f"    {'S':<16} {r['_S_seconds']:8.3f}s   validation {'PASS' if r['validation']['pass'] else 'FAIL'} "
              f"(assembled Δ {r['validation']['assembled_delta_s']:+.3f}s, final Δ {r['validation']['final_delta_s']:+.3f}s)\n")
        if not a.keep_work and i < a.repeat - 1:
            shutil.rmtree(r["work_dir"], ignore_errors=True)

    med = {k: round(statistics.median([r[k]["seconds"] for r in runs]), 4) for k in ("demux", *IN_S)}
    if all("asr" in r for r in runs):
        med["asr"] = round(statistics.median([r["asr"]["seconds"] for r in runs]), 4)
    med["S"] = round(sum(med[k] for k in IN_S), 4)
    last = runs[-1]; dur = source["duration_s"]; spm = med["S"] / dur * 60
    vs = last["vad_split"]
    row = {
        "task": "0.11", "methodology": "timeline-preserving (SPEECH + PASS_THROUGH chunks, full-length concat, cues on original timeline)",
        "input": src.name, "source": source, "split_mode": a.split_mode, "asr": a.asr, "repeat": a.repeat,
        "median": med, "S_seconds": med["S"], "S_per_video_min": round(spm, 3),
        "speech_intervals": vs["speech_intervals"], "speech_seconds": vs["speech_seconds"], "speech_pct": vs["speech_pct"],
        "pass_through_intervals": vs["pass_through_intervals"], "pass_through_seconds": vs["pass_through_seconds"],
        "chunks_total": vs["chunks_total"],
        "assembled_seconds": last["concat"]["out_seconds"], "final_seconds": last["subtitle_burn"]["out_seconds"],
        "assembled_delta_s": last["validation"]["assembled_delta_s"], "final_delta_s": last["validation"]["final_delta_s"],
        "final_resolution": last["subtitle_burn"]["out_resolution"], "final_fps": last["subtitle_burn"]["out_fps"],
        "validation": last["validation"], "validation_all_runs_pass": all(r["validation"]["pass"] for r in runs),
        "excluded_from_S": {"demux_s": med["demux"], "asr_s": med.get("asr")},
        "timeline": last["timeline"], "cues": last["cues"],
        "runs": [{k: v for k, v in r.items() if k not in ("timeline", "cues", "_final_probe", "_assembled_probe")} for r in runs],
        "final_probe": last["_final_probe"], "env_snapshot": _latest_snapshot_id(),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    append_jsonl(STAGES_JSONL, row)
    out = PILOT / f"0.11_stages_{src.stem}{('_' + a.tag) if a.tag else ''}.json"
    write_json(out, row)

    print("=" * 74); print(f"  S = T_vad_split + T_concat + T_subtitle_burn" + (f"   (median of {a.repeat})" if a.repeat > 1 else "")); print("=" * 74)
    for k in IN_S:
        print(f"  T_{k:<15} {med[k]:8.3f}s   {med[k] / med['S'] * 100:5.1f}% of S")
    print(f"  {'S':<17} {med['S']:8.3f}s   100.0%\n")
    print(f"  speech / pass-through    {vs['speech_intervals']} intervals {vs['speech_seconds']:.2f}s ({vs['speech_pct']:.1f}%) / "
          f"{vs['pass_through_intervals']} intervals {vs['pass_through_seconds']:.2f}s")
    print(f"  source / assembled / final  {dur:.3f} / {row['assembled_seconds']:.3f} / {row['final_seconds']:.3f} s  "
          f"(Δ {row['assembled_delta_s']:+.3f} / {row['final_delta_s']:+.3f}, tol {last['validation']['tolerance_s']:.3f})")
    print(f"  final resolution / fps   {row['final_resolution']} @ {row['final_fps']}")
    print(f"  S / duration * 60        {spm:.2f} s per minute of source")
    print(f"  excluded from S          demux {med['demux']:.3f}s" + (f", asr {med['asr']:.3f}s" if "asr" in med else ""))
    print(f"  validation               {'PASS' if row['validation_all_runs_pass'] else 'FAIL'}  {last['validation']['checks']}")
    print(f"\n-> {out}   env={row['env_snapshot']}")
    if not a.keep_work:
        shutil.rmtree(last["work_dir"], ignore_errors=True)
    return 0 if row["validation_all_runs_pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
