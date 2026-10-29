#!/usr/bin/env python3
"""r8-sep listen list: the dub gaps (bed heard alone) where the PREV and NEW final mixes differ most (the dub is identical), with A / B timecodes of
the AB WATCH (B = t + offset). Per gap: level of A / B, the A-B spectral distance (64-mel, dB RMS), air > 11.5 kHz A / B, Silero voice A / B,
and what the ear should hear there.
    python scripts/r8/listen_sep.py --id 02 --prev r8v4c --new sepH5p --offset 111.9 [--top 10] > X.md"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "r7"))
import bed_metrics as bm
A = pathlib.Path("/workspace/dub/artifacts/r4a"); FR = 0.02; SR = 44100


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--id", required=True); ap.add_argument("--prev", default="r8v4c"); ap.add_argument("--new", required=True)
    ap.add_argument("--offset", type=float, required=True); ap.add_argument("--top", type=int, default=10); ap.add_argument("--ver", default="sep"); a = ap.parse_args()
    import gap_metrics as gm, gap_clean as gc, scipy.signal as ss
    run = A / a.new / a.id; man = json.loads(next(run.glob(f"{a.id}_*.manifest.json")).read_text())
    gaps, _ = gm.gaps_of(run); ys = {k: sf.read(str(A / t / a.id / "final_mix_44k.wav"), dtype="float32")[0].mean(1) for k, t in (("A", a.prev), ("B", a.new))}
    n = min(len(y) for y in ys.values()) // int(FR * SR); S = bm.fmask([(s["start"], s["end"]) for s in man["speech_intervals"]], n)
    L = {k: bm.lev(y, n) for k, y in ys.items()}; M = {k: bm.spec(y, n)[1] for k, y in ys.items()}
    V = {}
    for k, y in ys.items():
        p, ph = gc.bed_voice_prob(y, SR); V[k] = np.interp((np.arange(n) + .5) * FR, (np.arange(len(p)) + .5) * ph, p)
    H = {}
    for k, y in ys.items():
        f, _, Z = ss.stft(y, SR, nperseg=2048, noverlap=2048 - int(FR * SR)); Z = np.abs(Z) ** 2; H[k] = (f, Z)
    rows = []
    for g0, g1 in gaps:
        i0, i1 = int(g0 / FR), min(n, int(np.ceil(g1 / FR)))
        if i1 - i0 < 10: continue
        d = M["A"][:, i0:i1].mean(1) - M["B"][:, i0:i1].mean(1); dist = float(np.sqrt(np.mean(d ** 2)))
        air = {k: float(10 * np.log10(Z[f > 11500][:, i0:i1].sum() / (Z[:, i0:i1].sum() + 1e-20) + 1e-15)) for k, (f, Z) in H.items()}
        rows.append({"t": (g0, g1), "src": bool(S[i0:i1].mean() > 0.3), "dist": dist, "lA": float(np.median(L["A"][i0:i1])), "lB": float(np.median(L["B"][i0:i1])),
                     "airA": air["A"], "airB": air["B"], "vA": float((V["A"][i0:i1] >= .5).sum() * FR), "vB": float((V["B"][i0:i1] >= .5).sum() * FR)})
    rows.sort(key=lambda r: -r["dist"]); mm = lambda t: f"{int(t // 60)}:{t % 60:05.2f}"
    print(f"# {a.id} listen list: A = {a.prev} (Spleeter bed), B = {a.new} (htdemucs bed); B timecode = A + {a.offset} s in the AB WATCH\n")
    print("Dub gaps = the bed is heard alone. Sorted by how different A and B sound there (64-mel distance). 'src speaks' = the separator removed the voice there.\n")
    print("| A time | B time | len s | src speaks | A / B level dB | A-B dist dB | air >11.5k A / B dB | voice s A / B | listen for |\n|---|---|---|---|---|---|---|---|---|")
    for r in rows[:a.top]:
        g0, g1 = r["t"]; what = []
        if r["src"]: what.append("the room under the removed voice: B carries on like the pauses, A thinner / hollow" if r["lB"] >= r["lA"] - 1 else "where the voice was cut: B quieter here")
        if r["airB"] - r["airA"] > 10: what.append("top end (air / hiss) present in B, cut in A")
        if r["vA"] > r["vB"] + 0.04: what.append("A has a voice-like residue B does not")
        if r["vB"] > r["vA"] + 0.04: what.append("CHECK: B voice-like")
        if not what: what.append("bed texture / level")
        print(f"| {mm(g0)}-{mm(g1)} | {mm(g0 + a.offset)}-{mm(g1 + a.offset)} | {g1 - g0:.2f} | {'yes' if r['src'] else 'no'} | {r['lA']:.1f} / {r['lB']:.1f} | {r['dist']:.1f} | "
              f"{r['airA']:.0f} / {r['airB']:.0f} | {r['vA']:.2f} / {r['vB']:.2f} | {'; '.join(what)} |")


if __name__ == "__main__":
    main()
