#!/usr/bin/env python3
"""A/B of the background bed (background_bed.py) on a finished run, WITHOUT re-running LatentSync: the video stream of the run's mp4 is
re-muxed with the mixed track (the bed never touches the frames, the lip-sync input or the speech gate).

Per video it reports
  dead air        seconds of the muxed track below -60 dBFS (50 ms frames): OLD (dub over silence) / NEW (dub + bed) / the ORIGINAL source
  bed leakage     source-voice residue in the bed: Silero VAD seconds on the bed, and those NOT covered by the dub voice (audible murmur);
                  Whisper (source language) words recognised on the bed alone, and on the bed inside the dub pauses
  intelligibility per-unit Whisper WER / CTC CER of the dubbed text measured on the dub-only track and on the NEW mixed track (eval_run.unit_eval)
  levels          bed gain, duck, bed / mix level in the dub pauses, original background level for reference
and writes <out>/<id>_bg.mp4 (A/B listening copy) + <out>/<id>_bg.json.

    python scripts/quality/background_ab.py --manifest RUN/03_final.manifest.json --mp4 RUN/03_final.mp4 --out OUT [--duck-db -6] [--offset-db 0]
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np, soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "diag"))


def ff(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-nostdin", *args], check=True)


def read_mono(path, sr=16000):
    t = pathlib.Path(str(path) + f".{sr}.wav"); ff(["-i", str(path), "-vn", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_f32le", str(t)])
    y, _ = sf.read(str(t), dtype="float32"); t.unlink(); return y


def dead_air_s(y, sr=16000, thr_db=-60.0, hop_s=0.05):
    h = int(hop_s * sr); n = len(y) // h
    if n == 0: return 0.0
    rms = np.sqrt(np.mean(y[: n * h].reshape(n, h) ** 2, axis=1) + 1e-12)
    return round(float(np.sum(20 * np.log10(rms) < thr_db) * hop_s), 2)


def vad(y, sr=16000):
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    torch.set_num_threads(8)
    return [(t["start"], t["end"]) for t in get_speech_timestamps(torch.from_numpy(y), load_silero_vad(), sampling_rate=sr, return_seconds=True)]


def overlap(a, b):
    return sum(max(0.0, min(e1, e2) - max(s1, s2)) for s1, e1 in a for s2, e2 in b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--mp4", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--duck-db", type=float, default=-6.0); ap.add_argument("--offset-db", type=float, default=0.0); ap.add_argument("--tag", default="bg")
    ap.add_argument("--no-asr", action="store_true"); ap.add_argument("--no-room-fill", action="store_true")
    a = ap.parse_args()
    from background_bed import mix_background, separate_background
    man = json.loads(pathlib.Path(a.manifest).read_text()); vid = pathlib.Path(man["input"]).stem
    out = pathlib.Path(a.out); work = out / f"bgwork_{vid}"; work.mkdir(parents=True, exist_ok=True)
    work_run = pathlib.Path(man["work_dir"]); dub24 = pathlib.Path(man["dubbed_track"]["wav24k"]); duration = man["source"]["duration_s"]
    src_speech = [(v["start"], v["end"]) for v in man["speech_intervals"]]
    dub16 = read_mono(dub24); dub_speech = vad(dub16)
    sep = separate_background(pathlib.Path(man["input"]), work)
    mix = mix_background(sep, dub24, src_speech, dub_speech, duration, work, bg_offset_db=a.offset_db, duck_db=a.duck_db, room_fill=not a.no_room_fill)
    mp4 = out / f"{vid}_{a.tag}.mp4"
    ff(["-i", a.mp4, "-i", mix["aac"], "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy", str(mp4)])
    old = read_mono(a.mp4); new = read_mono(mix["wav"]); src = read_mono(man["input"])
    bed = read_mono(sep["bg"]); bed_vad = vad(bed)
    pauses = []; t = 0.0
    for s, e in sorted(dub_speech):
        if s > t: pauses.append((t, s))
        t = max(t, e)
    if t < duration: pauses.append((t, duration))
    rep = {"id": vid, "manifest": a.manifest, "out_mp4": str(mp4), "separation": sep, "mix": mix,
           "dead_air_s": {"old_dub_only": dead_air_s(old), "new_dub_plus_bed": dead_air_s(new), "original_source": dead_air_s(src), "duration_s": duration},
           "bed_leakage": {"bed_vad_s": round(sum(e - s for s, e in bed_vad), 2), "bed_vad_in_dub_pauses_s": round(overlap(bed_vad, pauses), 2),
                           "bed_vad_in_source_speech_s": round(overlap(bed_vad, src_speech), 2), "source_speech_s": round(sum(e - s for s, e in src_speech), 2)}}
    if not a.no_asr:
        import tts_qa as qa_mod
        from faster_whisper import WhisperModel
        wm = WhisperModel(str(qa_mod.MODELS / "whisper-large-v3"), device="cuda", compute_type="float16")
        lang = man.get("source_lang") or "ru"
        def words(y):
            segs, _ = wm.transcribe(y, language=lang, beam_size=5, vad_filter=False, condition_on_previous_text=False, no_speech_threshold=0.6)
            return " ".join(s.text.strip() for s in segs if s.no_speech_prob < 0.6).strip()
        bed_scaled = bed * (10 ** (mix["bg_gain_db"] / 20))
        txt_all = words(bed_scaled)
        m = np.zeros(len(bed), bool)
        for s, e in pauses: m[int(s * 16000):int(e * 16000)] = True
        txt_pause = words(np.where(m, bed_scaled, 0.0).astype(np.float32))
        rep["bed_leakage"].update({"whisper_words_bed": len(txt_all.split()), "whisper_text_bed": txt_all[:600],
                                   "whisper_words_bed_in_dub_pauses": len(txt_pause.split()), "whisper_text_bed_in_dub_pauses": txt_pause[:600]})
        del wm
        import eval_run
        ev_dub = eval_run.evaluate(man, str(dub24)); ev_mix = eval_run.evaluate(man, mix["wav"])
        rep["intelligibility"] = {"dub_only": ev_dub["summary"], "mixed": ev_mix["summary"]}
        rep["intelligibility_units"] = [{"unit": a_["unit"], "expected": a_["expected"], "dub_whisper": a_["whisper"], "mix_whisper": b_["whisper"], "dub_wer": a_["wer"], "mix_wer": b_["wer"],
                                         "dub_cer_ctc": a_["cer_ctc"], "mix_cer_ctc": b_["cer_ctc"]} for a_, b_ in zip(ev_dub["units"], ev_mix["units"])]
    (out / f"{vid}_{a.tag}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    print(json.dumps({k: rep[k] for k in ("dead_air_s", "bed_leakage")}, ensure_ascii=False, indent=1))
    if "intelligibility" in rep: print(json.dumps(rep["intelligibility"], indent=1))
    print("bed gain", mix["bg_gain_db"], "dB; mix in pauses", mix["mix_db_in_dub_pauses"], "->", mp4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
