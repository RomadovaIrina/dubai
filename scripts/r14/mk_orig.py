#!/usr/bin/env python3
"""r14 student data: make an external clip (GRID / NASA / EC) look like a finished AUDIO stage whose "dub" is the clip's OWN audio, so the
production face generator (r7_e2e.sh -> r4_video.sh -> capture_gen.py) captures its LatentSync windows unchanged (LatentSync's own training
setup: the mouth is reconstructed for the real audio). Writes
  <inputs>/<id>.mp4                     the clip, normalised: 25 fps CFR, <= 720 px high, H.264 + AAC, [--ss, --t] cut
  <audio_root>/<id>/dubbed_16k.wav, dubbed_24k.wav, final_mix_aac.m4a, <id>_orig.manifest.json
The manifest carries what the generator reads: input, work_dir, source (ffprobe), speech_intervals (Silero VAD from faster-whisper), one
diarization speaker, dubbed_track, no units; "orig_audio": true marks it for the trainer.
    python scripts/r14/mk_orig.py --video n01.mp4 --id x_n01 --inputs DIR --audio-root DIR [--ss 30 --t 90] [--concat a.mpg b.mpg ...]"""
import argparse, json, pathlib, subprocess, tempfile


def run(c): subprocess.run(c, check=True, capture_output=True)


def probe(p):
    o = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(p)], capture_output=True, text=True, check=True).stdout)
    v = next(s for s in o["streams"] if s["codec_type"] == "video"); au = next((s for s in o["streams"] if s["codec_type"] == "audio"), {})
    fr = lambda s: (lambda n, d: float(n) / float(d) if float(d) else 0.0)(*s.split("/"))
    dur = float(o["format"]["duration"])
    return {"duration_s": dur, "size_bytes": int(o["format"]["size"]), "container": o["format"]["format_name"],
            "video": {"codec": v["codec_name"], "width": v["width"], "height": v["height"], "r_frame_rate": v["r_frame_rate"], "avg_frame_rate": v["avg_frame_rate"],
                      "fps": fr(v["r_frame_rate"]), "avg_fps": fr(v["avg_frame_rate"]), "vfr": False, "pix_fmt": v.get("pix_fmt"), "nb_frames": v.get("nb_frames"), "duration_s": dur},
            "audio": {"codec": au.get("codec_name"), "sample_rate": int(au.get("sample_rate", 0) or 0), "channels": au.get("channels")}}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--video"); ap.add_argument("--concat", nargs="*"); ap.add_argument("--id", required=True)
    ap.add_argument("--inputs", required=True); ap.add_argument("--audio-root", required=True); ap.add_argument("--ss", type=float); ap.add_argument("--t", type=float)
    ap.add_argument("--source", default=""); a = ap.parse_args()
    inp = pathlib.Path(a.inputs); inp.mkdir(parents=True, exist_ok=True); out = inp / f"{a.id}.mp4"
    AD = pathlib.Path(a.audio_root) / a.id; AD.mkdir(parents=True, exist_ok=True)
    norm = ["-vf", "fps=25,scale=-2:'min(720,ih)'", "-c:v", "libx264", "-crf", "16", "-preset", "fast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"]
    if a.concat:   # GRID: one speaker's 3-second clips as one video (one generator run instead of dozens)
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            for c in a.concat: f.write(f"file '{pathlib.Path(c).resolve()}'\n")
        run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", f.name] + norm + [str(out)])
    else:
        cut = (["-ss", str(a.ss)] if a.ss else []) + (["-t", str(a.t)] if a.t else [])
        run(["ffmpeg", "-v", "error", "-y"] + cut + ["-i", a.video] + norm + [str(out)])
    for sr, name in ((16000, "dubbed_16k.wav"), (24000, "dubbed_24k.wav")):
        run(["ffmpeg", "-v", "error", "-y", "-i", str(out), "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(AD / name)])
    run(["ffmpeg", "-v", "error", "-y", "-i", str(out), "-vn", "-c:a", "copy", str(AD / "final_mix_aac.m4a")])
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    y = decode_audio(str(AD / "dubbed_16k.wav"), sampling_rate=16000)
    ts = get_speech_timestamps(y, VadOptions(min_silence_duration_ms=300, speech_pad_ms=100))
    speech = [{"start": round(t["start"] / 16000, 2), "end": round(t["end"] / 16000, 2)} for t in ts]
    src = probe(out); D = src["duration_s"]
    man = {"task": "orig_audio_capture", "input": str(out), "output": None, "target_lang": None, "work_dir": str(AD), "source": src, "speech_intervals": speech,
           "diarization": {"model": "none (one speaker)", "turns": [{"start": s["start"], "end": s["end"], "speaker": "SPEAKER_00"} for s in speech]},
           "dubbed_track": {"wav24k": str(AD / "dubbed_24k.wav"), "wav16k": str(AD / "dubbed_16k.wav"), "aac": str(AD / "final_mix_aac.m4a"), "seconds": round(D, 3)},
           "units": [], "orig_audio": True, "data_source": a.source}
    (AD / f"{a.id}_orig.manifest.json").write_text(json.dumps(man, indent=1))
    print(f"{a.id}: {D:.1f} s, {src['video']['width']}x{src['video']['height']}, speech {sum(s['end'] - s['start'] for s in speech):.1f} s in {len(speech)} intervals")


if __name__ == "__main__":
    main()
