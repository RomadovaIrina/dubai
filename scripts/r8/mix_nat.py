#!/usr/bin/env python3
"""r8-sep: how natural the bed of a FINAL mix sounds where the dub is silent (dub gaps G of the run's own dub VAD, gap_metrics.gaps_of), 20-ms frames:
  cont_db   median mix level in G & S (the source spoke: separated bed) - median in G & P (true pauses: the room); 0 = the room carries on, < 0 = holes
  shape_db  RMS dB distance of the mean 64-mel log spectra G & S vs G & P after removing their mean offset (the bed under the removed voice sounds
            like another room: dull / hollow / warbly)
  air_db    energy > 11.5 kHz rel. the total in G (src_air_db: the source in its pauses; Spleeter 2stems cuts everything over 11 kHz)
    python scripts/r8/mix_nat.py --tags r8v4c sepH0 ... --out X.json 01 02 ..."""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "r7"))
import bed_metrics as bm
A = pathlib.Path("/workspace/dub/artifacts/r4a"); FR = 0.02; SR = 44100


def air(mono, m):
    import scipy.signal as ss
    f, _, Z = ss.stft(mono, SR, nperseg=2048, noverlap=2048 - int(FR * SR)); Z = np.abs(Z) ** 2; k = min(Z.shape[1], len(m)); mm = m[:k]
    return round(float(10 * np.log10(Z[f > 11500][:, :k][:, mm].sum() / (Z[:, :k][:, mm].sum() + 1e-20) + 1e-15)), 1) if mm.any() else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--out", required=True); ap.add_argument("ids", nargs="+"); a = ap.parse_args()
    import gap_metrics as gm
    R = {}; src_ck = {}
    for id_ in a.ids:
        src = bm.source44(id_); R[id_] = {}
        for t in a.tags:
            run = A / t / id_; man = json.loads(next(run.glob(f"{id_}_*.manifest.json")).read_text())
            y = sf.read(str(run / "final_mix_44k.wav"), dtype="float32")[0]; mono = y.mean(1); n = len(mono) // int(FR * SR)
            sp = [(s["start"], s["end"]) for s in man["speech_intervals"]]; S = bm.fmask(sp, n); P = ~bm.fmask(sp, n, 0.3)
            gaps, _ = gm.gaps_of(run); G = bm.fmask(gaps, n); GS, GP = G & S, G & P
            L = bm.lev(mono, n); _, mel, _ = bm.spec(mono, n)
            r = {"G_s": round(float(G.sum() * FR), 1), "GS_s": round(float(GS.sum() * FR), 1), "GP_s": round(float(GP.sum() * FR), 1)}
            if GS.sum() > 10 and GP.sum() > 10:
                r["cont_db"] = round(float(np.median(L[GS]) - np.median(L[GP])), 2)
                d = mel[:, GS].mean(1) - mel[:, GP].mean(1); r["shape_db"] = round(float(np.sqrt(np.mean((d - d.mean()) ** 2))), 2)
            r["air_db"] = air(mono, G); r["src_air_db"] = air(src.mean(1), P[:len(src) // int(FR * SR)])
            import clicks   # the dub is the same in every tag: a click of the final mix INSIDE a dub gap is the bed's (under the dub it is the dub's own)
            ck = [e["t"] for e in clicks.detect(y, SR) if G[min(n - 1, int(e["t"] / FR))]]; r["gap_clicks"] = ck
            if t != a.tags[0] and a.tags[0] in R[id_]:
                r["new_gap_clicks"] = [x for x in ck if not any(abs(x - q) < 0.02 for q in R[id_][a.tags[0]]["gap_clicks"])]
                sc = src_ck.setdefault(id_, [e["t"] for e in clicks.detect(src, SR)])   # the source's own transients (01 music: drum hits Spleeter dulled)
                r["new_gap_clicks_not_in_src"] = [x for x in r["new_gap_clicks"] if not any(abs(x - q) < 0.02 for q in sc)]
            R[id_][t] = r; print(id_, t, r, flush=True)
    pathlib.Path(a.out).write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
