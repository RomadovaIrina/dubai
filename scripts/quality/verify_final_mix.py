#!/usr/bin/env python3
"""Round 2 (track A): rebuild the final mix of a canonical baseline work dir from its stems exactly the way run_clean_pipeline_05 does
(background_bed.mix_background: bg / vocals stems + dubbed_24k, source speech + Silero VAD of dubbed_16k, the manifest's mix parameters)
and compare it with the stored final_mix_44k.wav (decoded PCM md5) and the final mp4's audio stream md5 (stream-copied AAC packets).
No separation model is run (the stems are reused).   python scripts/quality/verify_final_mix.py --id 03 [--canonical DIR]
"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, subprocess, sys, tempfile
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot"))


def pcm_md5(p):
    return subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(p), "-map", "0:a", "-f", "md5", "-"], capture_output=True, text=True, check=True).stdout.strip()


def stream_md5(p):
    return subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(p), "-map", "0:a", "-c", "copy", "-f", "md5", "-"], capture_output=True, text=True, check=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--id", required=True); ap.add_argument("--canonical", default="/workspace/dub/canonical/artifacts/baseline")
    a = ap.parse_args(); base = pathlib.Path(a.canonical) / a.id; w = base / "work" / f"work_{a.id}"
    man = json.loads((base / "final" / f"{a.id}_final.manifest.json").read_text())
    ref_stream = (base / "final" / f"{a.id}_final.audio_stream.md5").read_text().strip() if (base / "final" / f"{a.id}_final.audio_stream.md5").exists() else None
    mixp = man["background"]["mix"]; speech = [(v["start"], v["end"]) for v in man["speech_intervals"]]
    import run_clean_pipeline_05 as rp
    from background_bed import mix_background
    dub_vad = rp.run_vad(w / "dubbed_16k.wav")
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        mix = mix_background({"bg": str(w / "bg_44k.wav"), "vocals": str(w / "vocals_44k.wav")}, w / "dubbed_24k.wav", speech,
                             [(v["start"], v["end"]) for v in dub_vad], man["source"]["duration_s"], td, bg_offset_db=mixp["bg_offset_db"],
                             duck_db=mixp["duck_db"], src_duck_db=mixp.get("src_duck_db", 0.0))
        res = {"id": a.id, "rebuilt_pcm": pcm_md5(td / "final_mix_44k.wav"), "stored_pcm": pcm_md5(w / "final_mix_44k.wav"),
               "rebuilt_aac_stream": stream_md5(td / "final_mix_aac.m4a"), "stored_aac_stream": stream_md5(w / "final_mix_aac.m4a"), "manifest_stream": ref_stream,
               "bg_gain_db": [mix["bg_gain_db"], mixp["bg_gain_db"]], "peak_norm": [mix["peak_norm"], mixp["peak_norm"]],
               "dub_vad_speech_s": round(sum(v["end"] - v["start"] for v in dub_vad), 3)}
        import numpy as np, soundfile as sf
        x, _ = sf.read(str(td / "final_mix_44k.wav"), dtype="int16"); y, _ = sf.read(str(w / "final_mix_44k.wav"), dtype="int16")
        dd = np.abs(x.astype(int) - y.astype(int)); res["pcm_max_abs_diff_lsb"] = int(dd.max()); res["pcm_samples_differing"] = int((dd > 0).sum())
    res["pcm_match"] = res["rebuilt_pcm"] == res["stored_pcm"]; res["stream_match_manifest"] = res["stored_aac_stream"] == ref_stream
    res["rebuilt_stream_match"] = res["rebuilt_aac_stream"] == ref_stream
    print(json.dumps(res))


if __name__ == "__main__":
    main()
