#!/usr/bin/env python3
"""Final-mix loudness normalisation (Agent C r3, 2026-10-19): one integrated-loudness target for every video + true-peak ceiling.
D's scorecard: audio-r2/r3 final mixes were -12.9 .. -20.4 LUFS with true peak up to -0.2 dBTP (spec: <= -1 dBTP).
Two-pass ffmpeg loudnorm (EBU R128, I = -16 LUFS online-video target, TP -1.5 dBTP so the AAC re-encode stays <= -1, linear when possible),
in place on <dir>/final_mix_44k.wav, re-encodes final_mix_aac.m4a, verifies both (ebur128) and records the result in the manifest `loudness`.
The dub voice stems (dubbed_16k / 24k) are NOT touched (LatentSync / speech gate input).

    python scripts/quality/loudness.py --dir /workspace/dub/c/artifacts/audio-r3b/03 [--target -16 --tp -1.5]
"""
from __future__ import annotations
import argparse, json, pathlib, re, subprocess


def measure(f: pathlib.Path) -> dict:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(f), "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True).stderr
    s = r[r.rfind("Summary:"):]
    g = lambda k: float(re.search(rf"{k}:\s+(-?[\d.]+|-inf)", s).group(1).replace("-inf", "-200"))
    return {"lufs": g("I"), "lra": g("LRA"), "true_peak_dbtp": g("Peak")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True); ap.add_argument("--target", type=float, default=-16.0); ap.add_argument("--tp", type=float, default=-1.5)
    ap.add_argument("--lra", type=float, default=11.0)
    a = ap.parse_args(); d = pathlib.Path(a.dir); wav = d / "final_mix_44k.wav"; aac = d / "final_mix_aac.m4a"; tmp = d / "_ln.wav"
    before = measure(wav); f0 = f"loudnorm=I={a.target}:TP={a.tp}:LRA={a.lra}"
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(wav), "-af", f0 + ":print_format=json", "-f", "null", "-"], capture_output=True, text=True).stderr
    m = json.loads(r[r.rfind("{"):r.rfind("}") + 1])
    f1 = (f0 + f":measured_I={m['input_i']}:measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
          f":offset={m['target_offset']}:linear=true:print_format=json")
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-y", "-i", str(wav), "-af", f1, "-ar", "44100", "-c:a", "pcm_f32le", str(tmp)], capture_output=True, text=True).stderr
    m2 = json.loads(r[r.rfind("{"):r.rfind("}") + 1]); tmp.replace(wav)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav), "-c:a", "aac", "-b:a", "192k", str(aac)], check=True)
    rec = {"target_lufs": a.target, "tp_ceiling_dbtp": a.tp, "mode": m2.get("normalization_type"), "before": before, "after_wav": measure(wav), "after_aac": measure(aac)}
    for mp in d.glob("*.manifest.json"):
        man = json.loads(mp.read_text()); man.setdefault("background", {})["loudness"] = rec; mp.write_text(json.dumps(man, indent=1, ensure_ascii=False))
    print(d.name, json.dumps({k: rec[k] for k in ("mode", "before", "after_wav", "after_aac")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
