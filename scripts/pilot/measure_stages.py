#!/usr/bin/env python
"""0.11 — stage timing for VAD-split + concat + subtitle burn.

    S = T_vad_split + T_concat + T_subtitle_burn

Stage definitions follow the 0.11 spec exactly:

  T_vad_split       silero-vad on the source audio, PLUS the actual cutting of the source
                    into per-speech-segment files.
  T_concat          final assembly of those segments back into a single clip.
  T_subtitle_burn   hard-burning subtitles into the assembled clip (libass).

Deliberately NOT part of S (reported separately so they can never quietly inflate it):
  demux             source -> 16 kHz mono wav, the input VAD needs. Preparatory I/O.
  asr               faster-whisper transcription used to fill real subtitle text (--asr).
No upscaling or preparatory re-encoding is performed at all: the source is measured as-is,
at its own resolution and fps.

Recorded per run: source duration/resolution/fps/codecs/audio params, segment count,
total segment duration, output resolution/fps, S, and S/duration*60 — seconds of
sequential processing per minute of source video.

    source scripts/env.sh
    python scripts/pilot/measure_stages.py test_videos/04.mp4 --asr
    python scripts/pilot/measure_stages.py test_videos/04.mp4 --split-mode copy
    python scripts/pilot/measure_stages.py --report
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

STAGES_JSONL = PILOT / "stages.jsonl"
WORK = PILOT / "stages_work"


def ff(args: list[str], cwd: str | None = None) -> None:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args],
                       capture_output=True, text=True, cwd=cwd)
    if r.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {' '.join(args[:8])}…\n{r.stderr[-700:]}")


def probe(path: pathlib.Path) -> dict:
    """Record the source exactly as it is — no assumptions, no normalisation."""
    raw = sh(f'ffprobe -v error -print_format json -show_format -show_streams "{path}"',
             timeout=60)
    d = json.loads(raw) if raw else {}
    v = next((s for s in d.get("streams", []) if s.get("codec_type") == "video"), {})
    a = next((s for s in d.get("streams", []) if s.get("codec_type") == "audio"), {})
    rate = v.get("r_frame_rate", "0/0")
    try:
        num, den = rate.split("/")
        fps = round(int(num) / int(den), 4) if int(den) else None
    except Exception:
        fps = None
    return {
        "duration_s": float(d.get("format", {}).get("duration", 0) or 0),
        "size_bytes": int(d.get("format", {}).get("size", 0) or 0),
        "container": d.get("format", {}).get("format_name"),
        "video": {"codec": v.get("codec_name"), "profile": v.get("profile"),
                  "width": v.get("width"), "height": v.get("height"),
                  "r_frame_rate": rate, "fps": fps,
                  "pix_fmt": v.get("pix_fmt"), "nb_frames": v.get("nb_frames"),
                  "bit_rate": v.get("bit_rate")},
        "audio": {"codec": a.get("codec_name"), "sample_rate": a.get("sample_rate"),
                  "channels": a.get("channels"), "channel_layout": a.get("channel_layout"),
                  "bit_rate": a.get("bit_rate")},
    }


def srt_ts(t: float) -> str:
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(path: pathlib.Path, cues) -> None:
    path.write_text("\n".join(f"{i}\n{srt_ts(a)} --> {srt_ts(b)}\n{t}\n"
                              for i, (a, b, t) in enumerate(cues, 1)), encoding="utf-8")


class Stage:
    """Times a block and records its VRAM footprint through the 0.4 machinery."""

    def __init__(self, name: str, results: dict):
        self.name, self.results = name, results

    def __enter__(self):
        self._m = measure_gpu_memory(f"stage/{self.name}", quiet=True,
                                     jsonl=PILOT / "vram.jsonl")
        self._m.__enter__()
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self._t0
        self._m.__exit__(*exc)
        self.results[self.name] = {
            "seconds": round(dt, 4),
            "peak_process_mib": self._m.result.get("peak_process_mib"),
        }
        return False


IN_S = ("vad_split", "concat", "subtitle_burn")


def run_once(src: pathlib.Path, args) -> dict:
    work = WORK / src.stem
    if work.exists():
        shutil.rmtree(work)
    (work / "segs").mkdir(parents=True)
    res: dict = {}
    wav = work / "audio16k.wav"

    # ---- demux: preparatory, excluded from S --------------------------------
    with Stage("demux", res):
        ff(["-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            str(wav)])

    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
    torch.set_num_threads(THREADS)
    model = load_silero_vad()            # model load is startup, not pipeline work
    audio = read_audio(str(wav), sampling_rate=16000)

    # ---- T_vad_split: VAD + actual cutting into segments ---------------------
    segments: list = []
    seg_paths: list[pathlib.Path] = []
    with Stage("vad_split", res):
        segments = get_speech_timestamps(audio, model, sampling_rate=16000,
                                         return_seconds=True)
        if not segments:
            raise RuntimeError("VAD found no speech in this file")
        for i, s in enumerate(segments):
            seg = work / "segs" / f"{i:05d}.mp4"
            cut = ["-ss", f"{s['start']:.3f}", "-to", f"{s['end']:.3f}", "-i", str(src)]
            if args.split_mode == "copy":
                ff([*cut, "-c", "copy", "-avoid_negative_ts", "make_zero", str(seg)])
            else:
                # accurate cuts; no scaling, no fps change — source geometry preserved
                ff([*cut, "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                    "-c:a", "aac", "-b:a", "192k", str(seg)])
            seg_paths.append(seg)
    res["vad_split"].update(
        segments=len(segments), split_mode=args.split_mode,
        speech_seconds=round(sum(s["end"] - s["start"] for s in segments), 3))

    # ---- optional ASR: real subtitle text, timed separately ------------------
    texts = [f"[speech {i + 1}]" for i in range(len(segments))]
    if args.asr:
        from faster_whisper import WhisperModel
        m = WhisperModel(str(ROOT / "models" / "whisper-large-v3"), device="cuda",
                         compute_type="int8_float16")
        with Stage("asr", res):
            segs, _ = m.transcribe(str(wav), word_timestamps=False)
            tr = [(s.start, s.end, s.text.strip()) for s in segs]
        res["asr"]["cues"] = len(tr)
        texts = []
        for s in segments:
            mid = (s["start"] + s["end"]) / 2
            texts.append(next((t for a, b, t in tr if a <= mid <= b), "") or "[speech]")
        del m
        torch.cuda.empty_cache()

    # ---- T_concat: final assembly of the segments into one clip --------------
    listfile = work / "concat.txt"
    listfile.write_text("\n".join(f"file '{p.name}'" for p in seg_paths) + "\n")
    shutil.move(str(listfile), str(work / "segs" / "concat.txt"))
    assembled = work / "assembled.mp4"
    with Stage("concat", res):
        ff(["-f", "concat", "-safe", "0", "-i", "concat.txt", "-c", "copy",
            str(assembled)], cwd=str(work / "segs"))
    ap_ = probe(assembled)
    res["concat"].update(out_seconds=round(ap_["duration_s"], 3),
                         out_resolution=f"{ap_['video']['width']}x{ap_['video']['height']}",
                         out_fps=ap_["video"]["fps"])

    # ---- T_subtitle_burn: hard-burn subtitles into the assembled clip --------
    cues, t = [], 0.0
    for s, txt in zip(segments, texts):
        d = s["end"] - s["start"]
        cues.append((t, t + d, txt))
        t += d
    write_srt(work / "subs.srt", cues)
    burned = work / "burned.mp4"
    with Stage("subtitle_burn", res):
        ff(["-i", str(assembled), "-vf", "subtitles=subs.srt",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-c:a", "copy", str(burned)], cwd=str(work))
    bp = probe(burned)
    res["subtitle_burn"].update(
        cues=len(cues),
        out_resolution=f"{bp['video']['width']}x{bp['video']['height']}",
        out_fps=bp["video"]["fps"], out_seconds=round(bp["duration_s"], 3))

    res["_S_seconds"] = round(sum(res[k]["seconds"] for k in IN_S), 4)
    res["_final_probe"] = bp
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", nargs="?")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--split-mode", choices=["reencode", "copy"], default="reencode",
                    help="reencode = frame-accurate cuts (default); copy = keyframe-snapped")
    ap.add_argument("--asr", action="store_true",
                    help="real transcription for subtitle text (timed separately, not in S)")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()

    if a.report:
        if not STAGES_JSONL.exists():
            print("no runs yet")
            return 0
        rows = [json.loads(l) for l in STAGES_JSONL.read_text().splitlines() if l.strip()]
        L = ["# 0.11 — S = T_vad_split + T_concat + T_subtitle_burn", "",
             "| input | source | dur | split | segs | T_vad_split | T_concat | "
             "T_subtitle_burn | **S** | s/min | env |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            m, s = r["median"], r["source"]
            L.append(
                f"| {r['input']} | {s['video']['width']}x{s['video']['height']}@"
                f"{s['video']['fps']} {s['video']['codec']} | {s['duration_s']:.1f}s "
                f"| {r['split_mode']} | {r['segments']} "
                f"| {m['vad_split']:.2f}s | {m['concat']:.2f}s | {m['subtitle_burn']:.2f}s "
                f"| **{m['S']:.2f}s** | {r['seconds_per_minute']:.1f} "
                f"| `{(r.get('env_snapshot') or '-')[:24]}` |")
        out = "\n".join(L) + "\n"
        print(out)
        (PILOT / "stages.md").write_text(out)
        print("-> reports/pilot/stages.md")
        return 0

    if not a.input:
        ap.print_help()
        return 2
    src = pathlib.Path(a.input).resolve()
    if not src.exists():
        print(f"no such file: {src}", file=sys.stderr)
        return 1
    source = probe(src)
    if source["duration_s"] <= 0:
        print(f"FAIL: ffprobe reports no duration for {src} (corrupt file?)", file=sys.stderr)
        return 1

    v, au = source["video"], source["audio"]
    print("=" * 74)
    print("0.11  source as-is (no upscale, no resolution/fps change)")
    print("=" * 74)
    print(f"  file        {src}")
    print(f"  duration    {source['duration_s']:.3f} s")
    print(f"  video       {v['width']}x{v['height']}  {v['fps']} fps "
          f"({v['r_frame_rate']})  {v['codec']} {v['profile']}  {v['pix_fmt']}"
          + (f"  {int(v['bit_rate'])//1000} kb/s" if v.get("bit_rate") else ""))
    print(f"  audio       {au['codec']}  {au['sample_rate']} Hz  {au['channels']}ch "
          f"({au['channel_layout']})" + (f"  {int(au['bit_rate'])//1000} kb/s"
                                          if au.get("bit_rate") else ""))
    print(f"  container   {source['container']}   {source['size_bytes']/1048576:.2f} MiB")
    print(f"\n  split mode  {a.split_mode}    asr={a.asr}    repeat={a.repeat}\n")

    runs = []
    for i in range(a.repeat):
        if a.repeat > 1:
            print(f"--- run {i + 1}/{a.repeat}")
        r = run_once(src, a)
        runs.append(r)
        for k in ("demux", "vad_split", "concat", "asr", "subtitle_burn"):
            if k in r:
                tag = "" if k in IN_S else "   <- not in S"
                print(f"    {k:<16} {r[k]['seconds']:8.3f}s{tag}")
        print(f"    {'S':<16} {r['_S_seconds']:8.3f}s")
        print()

    med = {k: round(statistics.median([r[k]["seconds"] for r in runs]), 4)
           for k in ("demux", *IN_S)}
    med["S"] = round(sum(med[k] for k in IN_S), 4)
    last = runs[-1]
    dur = source["duration_s"]
    spm = med["S"] / dur * 60
    row = {
        "input": src.name, "source": source, "split_mode": a.split_mode, "asr": a.asr,
        "repeat": a.repeat, "median": med,
        "segments": last["vad_split"]["segments"],
        "speech_seconds": last["vad_split"]["speech_seconds"],
        "assembled_seconds": last["concat"]["out_seconds"],
        "final_seconds": last["subtitle_burn"]["out_seconds"],
        "final_resolution": last["subtitle_burn"]["out_resolution"],
        "final_fps": last["subtitle_burn"]["out_fps"],
        "seconds_per_minute": round(spm, 3),
        "runs": runs, "env_snapshot": _latest_snapshot_id(),
        "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    append_jsonl(STAGES_JSONL, row)
    write_json(PILOT / f"stages_{src.stem}.json", row)

    print("=" * 74)
    print(f"  S = T_vad_split + T_concat + T_subtitle_burn"
          + (f"   (median of {a.repeat})" if a.repeat > 1 else ""))
    print("=" * 74)
    for k in IN_S:
        print(f"  T_{k:<15} {med[k]:8.3f}s   {med[k]/med['S']*100:5.1f}% of S")
    print(f"  {'S':<17} {med['S']:8.3f}s   100.0%")
    print()
    print(f"  segments                 {row['segments']}")
    print(f"  speech total             {row['speech_seconds']:.3f} s "
          f"of {dur:.3f} s source ({row['speech_seconds']/dur*100:.1f}%)")
    print(f"  assembled / final        {row['assembled_seconds']:.3f} s / "
          f"{row['final_seconds']:.3f} s")
    print(f"  final resolution / fps   {row['final_resolution']} @ {row['final_fps']}")
    print(f"  S / duration * 60        {spm:.2f} s of sequential work per minute of source")
    print(f"  excluded from S          demux {med['demux']:.3f}s"
          + (f", asr {last['asr']['seconds']:.3f}s" if "asr" in last else ""))
    print(f"\n-> reports/pilot/stages_{src.stem}.json   env={row['env_snapshot']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
