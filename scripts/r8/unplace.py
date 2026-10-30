#!/usr/bin/env python3
"""r8: undo scripts/r7/lip_place.py on a published handoff (the r7 server artefacts are gone; the handoff keeps only the PLACED dub + manifest).
lip_place's blocks tile the track, cut in the middle of each dub pause, and man["lip_place"]["shifts"] keeps every block's loud core on the
pre-placement timeline + its shift, so the tiles are recomputed exactly and each one is moved back (cut points sit in pauses: the 20-ms fades
lip_place put there are in near-silence). Placements / track starts / pause times in the manifest are moved back by their lip_shift_s.
Output = a work dir on the r6/retake timeline (dubbed_24k.wav / dubbed_16k.wav / <id>_r4a.manifest.json + symlinked stems) that lip_place
can place again. --check renders the published shifts forward again and reports the difference to the published dub.
    python scripts/r8/unplace.py --placed handoff/.../audio-r7-v3a1/04 --stems r8a/stems/04 --out r8a/base/v3a1u/04 [--check]"""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np, soundfile as sf
FR = 0.02


def tiles(shifts, n):
    co = [(int(round(a / FR)), int(round(b / FR))) for a, b, _, _ in shifts]
    return [[(co[i - 1][1] + a) // 2 if i else 0, (b + co[i + 1][0]) // 2 if i + 1 < len(co) else n] for i, (a, b) in enumerate(co)]


def render(y, T, K, h):
    z = np.zeros_like(y)
    for q, ((b0, b1), k) in enumerate(zip(T, K)):
        i0, i1 = b0 * h, (min(len(y), b1 * h) if q + 1 < len(T) else len(y)); seg = y[i0:i1]
        j0 = i0 + k * h; c0 = max(0, -j0); c1 = max(0, j0 + len(seg) - len(z)); seg = seg[c0:len(seg) - c1]; j0 = max(0, j0); z[j0:j0 + len(seg)] += seg
    return z


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--placed", required=True); ap.add_argument("--stems", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    P, O = pathlib.Path(a.placed), pathlib.Path(a.out); O.mkdir(parents=True, exist_ok=True)
    mp = next(P.glob("*_r4b.manifest.json")); man = json.loads(mp.read_text()); lp = man["lip_place"]
    z, sr = sf.read(str(P / "dubbed_24k.wav"), dtype="float32"); h = int(FR * sr); n = len(z) // h + 1
    sh = lp["shifts"]; T = tiles(sh, n); K = [int(round(s[2] / FR)) for s in sh]
    y = np.zeros_like(z)
    for q, ((b0, b1), k) in enumerate(zip(T, K)):   # tile q's samples on the old timeline = the placed track at +k
        i0, i1 = b0 * h, (min(len(z), b1 * h) if q + 1 < len(T) else len(z)); j0, j1 = i0 + k * h, i1 + k * h
        c0, c1 = max(0, -j0), max(0, j1 - len(z)); y[i0 + c0:i1 - c1] = z[j0 + c0:j1 - c1]
    sf.write(str(O / "dubbed_24k.wav"), y, sr, subtype="FLOAT")
    import librosa
    sf.write(str(O / "dubbed_16k.wav"), librosa.resample(y, orig_sr=sr, target_sr=16000), 16000, subtype="PCM_16")
    for u in man["units"]:
        al = u.get("alignment") or {}; s_first = None
        for p in al.get("placements", []):
            if not p.get("placed") or "lip_shift_s" not in p: continue
            s0, s1 = p.pop("lip_shift_s"); s_first = s0 if s_first is None else s_first
            p["placed"] = [round(p["placed"][0] - s0, 3), round(p["placed"][1] - s1, 3)]
            if "track_start" in p: p["track_start"] = round(p["track_start"] - s0, 3)
            for qq in (p.get("spread") or {}).get("pauses", []):
                if "t" in qq: qq["t"] = round(qq["t"] - s0, 3)
        if "track_start" in al and s_first is not None: al["track_start"] = round(al["track_start"] - s_first, 3)
    man["r8_unplaced_from"] = {"lip_place": lp, "dub_gate": man.pop("dub_gate", None), "src": str(P)}; man.pop("lip_place")
    man["dubbed_track"] = {**man.get("dubbed_track", {}), "lip_placed": False}
    mid = mp.name.split("_")[0]; (O / f"{mid}_r4a.manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    for f in ("bg_44k.wav", "vocals_44k.wav"):
        (O / f).unlink(missing_ok=True); (O / f).symlink_to((pathlib.Path(a.stems) / f).resolve())
    rep = {"tiles": len(T), "moved": sum(1 for k in K if k)}
    if a.check:
        z2 = render(y, T, K, h); d = z2[:len(z)] - z; e = 20 * np.log10(np.sqrt((d ** 2).mean()) + 1e-12); ref = 20 * np.log10(np.sqrt((z ** 2).mean()) + 1e-12)
        fr = 20 * np.log10(np.sqrt((d[:len(d) // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-12)
        rep.update({"roundtrip_err_db_rel": round(e - ref, 1), "frames_err_gt_-60dBFS": int((fr > -60).sum()), "worst_err_frames_s": [round(float(t * FR), 2) for t in np.argsort(fr)[-5:]],
                    "worst_err_dbfs": round(float(fr.max()), 1)})
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
