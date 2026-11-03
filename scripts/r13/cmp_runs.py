#!/usr/bin/env python3
"""r13 0.5: is a speed lever output-preserving? Compares the audio-stage outputs of profiling runs (snapshots of run_prof.sh) against a reference
run: ASR text, translations, placed texts, the voice track (bit-identical? else SNR of the difference), the final mixes (SNR + loudness).
    python scripts/r13/cmp_runs.py --dir /workspace/dub/r13s/prof --ref base2 --runs base s3c qa1 --ids old03 m03 m18"""
import argparse, hashlib, json, pathlib
import numpy as np, soundfile as sf


def load(d: pathlib.Path):
    S = d / "snap"
    if not S.exists(): return None
    m = json.loads(next(S.glob("*_r4b.manifest.json")).read_text())
    out = {"asr": " ".join(s["text"] for s in m["asr"]["segments"]), "tr": [u.get("translation_fitted") or u.get("translation") for u in m["units"]],
           "pl": [p.get("text") for u in m["units"] for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed")]}
    for k, f in (("dub", "dubbed_16k.wav"), ("mix", "final_mix_44k.wav"), ("mix_htd", "final_mix_44k_htd.wav")):
        y, sr = sf.read(str(S / f), dtype="float32", always_2d=True); out[k] = y.mean(1)
    return out


def snr(a, b):
    n = min(len(a), len(b)); d = a[:n] - b[:n]
    return round(float(10 * np.log10(np.sum(a[:n] ** 2) / (np.sum(d ** 2) + 1e-20))), 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--ids", nargs="+", required=True); ap.add_argument("--json"); a = ap.parse_args(); D = pathlib.Path(a.dir); res = {}
    for i in a.ids:
        ref = load(D / f"{a.ref}_{i}")
        for t in a.runs:
            o = load(D / f"{t}_{i}")
            if o is None or ref is None: print(f"{i} {t}: no snapshot"); continue
            r = {"asr_same": o["asr"] == ref["asr"], "translations_differ": sum(x != y for x, y in zip(o["tr"], ref["tr"])) + abs(len(o["tr"]) - len(ref["tr"])),
                 "placed_texts_same": o["pl"] == ref["pl"], "dub_bit_identical": hashlib.md5(o["dub"].tobytes()).digest() == hashlib.md5(ref["dub"].tobytes()).digest(),
                 "dub_len_s": [round(len(o["dub"]) / 16000, 2), round(len(ref["dub"]) / 16000, 2)], "dub_diff_snr_db": snr(ref["dub"], o["dub"]),
                 "mix_diff_snr_db": snr(ref["mix"], o["mix"]), "mix_htd_diff_snr_db": snr(ref["mix_htd"], o["mix_htd"])}
            res[f"{i}/{t}"] = r; print(f"{i} {t:6s} vs {a.ref}: {r}")
    if a.json: pathlib.Path(a.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
