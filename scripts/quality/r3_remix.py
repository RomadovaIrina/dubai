#!/usr/bin/env python3
"""Round 3 D: remix the final mix of an audio run WITHOUT re-running TTS (user: "the background breathes / pumps").
The dub voice and its timing are untouched (lip renders stay valid); only the bed mix and the loudness stage change:
  * duck -2 dB, attack 0.4 s, release 0.8 s (was -4 dB / 0.25 / 0.6); residue duck -6 dB (was -15);
  * loudness: ONE static gain to the target (-16 LUFS) + a true-peak guard (limiter on a 4x oversampled signal at --tp), never ffmpeg loudnorm's
    dynamic mode (on 02 / 03 / 05 it acted as an AGC: bed gain jitter p95 7.8 dB on 03 vs 1.8 dB with the static gain).
Writes <out>/<id>/ = symlinks of the run's work dir + new final_mix_44k.wav / final_mix_aac.m4a + <id>_r4b.manifest.json, and checks it.
    python scripts/quality/r3_remix.py --src /workspace/dub/artifacts/r3d/audio-r4 --out /workspace/dub/artifacts/r3d/audio-r4b 03 04 ..."""
from __future__ import annotations
import argparse, json, pathlib, re, subprocess, sys
import soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE))


def measure(f):
    e = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(f), "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True).stderr
    t = e[e.rfind("Summary:"):]; I = re.search(r"I:\s+(-?[\d.]+) LUFS", t); P = re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", t)
    return {"lufs": float(I.group(1)) if I else None, "true_peak_dbtp": float(P.group(1)) if P and P.group(1) != "-inf" else None}


def static_loudness(wav: pathlib.Path, aac: pathlib.Path, target: float, tp: float) -> dict:
    before = measure(wav); tmp = wav.with_name("_static.wav"); gain = target - before["lufs"]; lim = 10 ** (tp / 20); hist = []
    for _ in range(3):   # the limiter can take a little loudness off: correct the static gain (still ONE constant gain per pass)
        af = f"volume={gain:.3f}dB,aresample=176400,alimiter=limit={lim:.5f}:attack=1:release=60:level=disabled:latency=1,aresample=44100"
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), "-af", af, "-c:a", "pcm_f32le", str(tmp)], check=True)
        m = measure(tmp); hist.append({"gain_db": round(gain, 3), **m})
        if m["lufs"] is None or abs(m["lufs"] - target) <= 0.3: break
        gain += target - m["lufs"]
    tmp.replace(wav)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), "-c:a", "aac", "-b:a", "192k", str(aac)], check=True)
    for _ in range(3):   # r4: the AAC encode can overshoot the limiter (v4repro 03: -0.9 dBTP) -> lower the whole wav by the overshoot and re-encode
        tp_aac = measure(aac)["true_peak_dbtp"]
        if tp_aac is None or tp_aac <= -1.1: break
        y, sr = sf.read(str(wav), dtype="float32"); sf.write(str(wav), y * 10 ** ((-1.2 - tp_aac) / 20), sr, subtype="FLOAT"); hist.append({"aac_tp_trim_db": round(-1.2 - tp_aac, 2)})
        subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), "-c:a", "aac", "-b:a", "192k", str(aac)], check=True)
    return {"mode": "static gain + oversampled limiter", "target_lufs": target, "limiter_dbfs": tp, "before": before, "passes": hist, "after_wav": measure(wav), "after_aac": measure(aac)}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--out", required=True); ap.add_argument("ids", nargs="+")
    ap.add_argument("--duck-db", type=float, default=-2.0); ap.add_argument("--attack", type=float, default=0.4); ap.add_argument("--release", type=float, default=0.8)
    ap.add_argument("--residue-db", type=float, default=-6.0); ap.add_argument("--src-duck-db", type=float, default=0.0); ap.add_argument("--target", type=float, default=-16.0); ap.add_argument("--tp", type=float, default=-1.5)
    a = ap.parse_args()
    import background_bed as bb, run_clean_pipeline_05 as rp, r3_bedswing  # noqa: F401
    for id_ in a.ids:
        W = (pathlib.Path(a.src) / id_).resolve(); O = pathlib.Path(a.out) / id_; O.mkdir(parents=True, exist_ok=True)
        for f in W.iterdir():
            if f.name.startswith("final_mix") or f.name.endswith(".manifest.json"): continue
            (O / f.name).unlink(missing_ok=True); (O / f.name).symlink_to(f)
        man = json.loads(next(W.glob(f"{id_}_*.manifest.json")).read_text())
        sep = {"bg": str(W / "bg_44k.wav"), "vocals": str(W / "vocals_44k.wav")}
        speech = [(s["start"], s["end"]) for s in man["speech_intervals"]]; dub_speech = [(s["start"], s["end"]) for s in rp.run_vad(W / "dubbed_16k.wav")]
        mix = bb.mix_background(sep, W / "dubbed_24k.wav", speech, dub_speech, man["source"]["duration_s"], O, duck_db=a.duck_db, attack_s=a.attack,
                                release_s=a.release, residue_duck_db=a.residue_db, src_duck_db=a.src_duck_db)
        loud = static_loudness(O / "final_mix_44k.wav", O / "final_mix_aac.m4a", a.target, a.tp)
        man["background"] = {**man.get("background", {}), "mix": mix, "loudness": loud, "remix": "r3 D r3_remix.py (bed duck -2 dB / 0.4 / 0.8 s, residue -6 dB, static loudness)"}
        man["dubbed_track"] = {**man["dubbed_track"], "aac": str(O / "final_mix_aac.m4a")}
        (O / f"{id_}_r4b.manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
        sw = subprocess.run([sys.executable, str(HERE / "r3_bedswing.py"), "--mix", str(O / "final_mix_44k.wav"), "--dub", str(W / "dubbed_24k.wav"), "--bg", sep["bg"]],
                            capture_output=True, text=True).stdout.strip()
        print(f"== {id_} r4b: {loud['after_aac']} bed {sw}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
