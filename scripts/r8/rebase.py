#!/usr/bin/env python3
"""r8: rebuild the r7 retake output (= the input of lip_place) from the two published handoffs, so v3a1's takes can be placed again on this
server without re-running Qwen / Chatterbox (not deterministic across GPUs / builds: README_v1). Exactly what scripts/r7/retake.py wrote:
  the r6_audio_v1 dub (handoff audio-r6-v1, pre-r7 timeline, pre-gate); for every re-taken unit of v3a1 its old span is faded out (20-ms sin^2
  ramps, span +-0.03 s as retake.py) and v3a1's new pieces (cut from the published PLACED dub at their placement: the piece samples, its 10-ms edge
  fades, level gain and v3a1's tail hush included) are added back at their pre-placement position (placed - lip_shift_s).
Manifest = v3a1's with every placement moved back by its lip_shift_s (man["r8_rebase"] keeps the published lip_place record).
--check: places the result again with the PUBLISHED shifts (lip_place's tiling) + the r7 dub gate and compares it with the published dub.
    python scripts/r8/rebase.py --r6 in_r7/audio-r6-v1/04 --v3 in_r7/audio-r7-v3a1/04 --stems r8a/stems/04 --out r8a/base/v3a1r/04 [--check]"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "quality")); sys.path.insert(0, str(HERE.parent / "pilot"))
FR = 0.02


def ramp(n):
    return np.sin(np.linspace(0, np.pi / 2, n)) ** 2


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--r6", required=True); ap.add_argument("--v3", required=True); ap.add_argument("--stems", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--check", action="store_true"); a = ap.parse_args()
    import unplace
    R6, V3, O = pathlib.Path(a.r6), pathlib.Path(a.v3), pathlib.Path(a.out); O.mkdir(parents=True, exist_ok=True)
    m6 = json.loads(next(R6.glob("*_r4b.manifest.json")).read_text()); mp = next(V3.glob("*_r4b.manifest.json")); man = json.loads(mp.read_text())
    y, sr = sf.read(str(R6 / "dubbed_24k.wav"), dtype="float32"); zp, sr2 = sf.read(str(V3 / "dubbed_24k.wav"), dtype="float32"); assert sr == sr2
    y = y if y.ndim == 1 else y.mean(1); zp = zp if zp.ndim == 1 else zp.mean(1); z = y.copy(); zp0 = zp.copy()
    dg = man.get("dub_gate")
    if dg and dg.get("runs"):   # the published dub is gated: undo r3_remix.dub_gate's gain (same curve from its runs) so lip_place sees the pre-gate take
        hh = sr // 100; nn = len(zp) // hh; g = np.ones(nn)
        for r0, r1 in dg["runs"]: g[int(round(r0 * 100)):int(round(r1 * 100))] = 0.0
        gs = np.repeat(g, hh); gs = np.concatenate([gs, np.full(len(zp) - len(gs), gs[-1])]); kk = int(dg["ramp_s"] * sr)
        er = np.convolve((gs < 0.5).astype(float), np.ones(kk), "same") >= kk - 0.5; lin = np.clip(np.convolve(er.astype(float), np.ones(kk) / kk, "same"), 0, 1)
        gain = 1 - (1 - 10 ** (dg["floor_db"] / 20)) * np.sin(np.pi / 2 * lin) ** 2; zp = (zp / gain).astype(np.float32)
    u6 = {u["id"]: u for u in m6["units"]}; changed = []
    h = int(FR * sr); T = unplace.tiles(man["lip_place"]["shifts"], len(z) // h + 1); K = [int(round(s[2] / FR)) for s in man["lip_place"]["shifts"]]
    def unplaced(q, L):   # samples [q, q + L) of the pre-placement track, read through lip_place's tiles (a piece may span two blocks)
        out = np.zeros(L, np.float32)
        for (b0, b1), k in zip(T, K):
            x0, x1 = max(q, b0 * h), min(q + L, b1 * h)
            if x1 <= x0: continue
            j0, j1 = x0 + k * h, x1 + k * h; c0, c1 = max(0, -j0), max(0, j1 - len(zp)); out[x0 - q + c0:x1 - q - c1] = zp[j0 + c0:j1 - c1]
        return out
    for u in man["units"]:
        ps = [p for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed")]
        if not any(p.get("r7_retake") for p in ps): continue
        o = [p for p in u6[u["id"]]["alignment"]["placements"] if p.get("placed")]
        a0, a1 = min(p["placed"][0] for p in o), max(p["placed"][1] for p in o); i0, i1 = int((a0 - 0.03) * sr), int((a1 + 0.03) * sr)
        k = int(0.02 * sr); g = np.zeros(i1 - i0); g[:k] = 1 - ramp(k); g[-k:] = ramp(k); z[i0:i1] *= g
        for p in ps:
            s0 = p["lip_shift_s"][0]; L = int(round(p["group_s"] * sr)); t0 = round(p["placed"][0] - s0, 3); q = int(t0 * sr)
            seg = unplaced(q, min(L, len(z) - q)); z[q:q + len(seg)] += seg
        changed.append(u["id"])
    sf.write(str(O / "dubbed_24k.wav"), z, sr, subtype="FLOAT")
    import librosa
    sf.write(str(O / "dubbed_16k.wav"), librosa.resample(z, orig_sr=sr, target_sr=16000), 16000, subtype="PCM_16")
    lp = man.pop("lip_place")
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
    man["r8_rebase"] = {"lip_place": lp, "dub_gate": man.pop("dub_gate", None), "r6": str(R6), "v3": str(V3), "retaken": changed}
    man["dubbed_track"] = {**man.get("dubbed_track", {}), "lip_placed": False}
    (O / f"{mp.name.split('_')[0]}_r4a.manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False))
    for f in ("bg_44k.wav", "vocals_44k.wav"):
        (O / f).unlink(missing_ok=True); (O / f).symlink_to((pathlib.Path(a.stems) / f).resolve())
    rep = {"retaken": changed}
    if a.check:   # forward with the published shifts, lip_place's fades, then the r7 dub gate -> vs the published dub
        h = int(FR * sr); n = len(z) // h + 1; T = unplace.tiles(lp["shifts"], n); K = [int(round(s[2] / FR)) for s in lp["shifts"]]
        w = np.zeros_like(z); fd = int(0.02 * sr); rp_ = np.sin(np.linspace(0, np.pi / 2, fd))
        for q, ((b0, b1), k) in enumerate(zip(T, K)):
            i0, i1 = b0 * h, (min(len(z), b1 * h) if q + 1 < len(T) else len(z)); seg = z[i0:i1].copy()
            if len(seg) > 2 * fd:
                if q and K[q - 1] != k: seg[:fd] *= rp_
                if q + 1 < len(T) and K[q + 1] != k: seg[-fd:] *= rp_[::-1]
            j0 = i0 + k * h; c0 = max(0, -j0); c1 = max(0, j0 + len(seg) - len(w)); seg = seg[c0:len(seg) - c1]; j0 = max(0, j0); w[j0:j0 + len(seg)] += seg
        w = np.round(np.clip(w, -1, 1) * 32767) / 32767   # lip_place wrote PCM_16
        tmp = O / "_chk"; tmp.mkdir(exist_ok=True); sf.write(str(tmp / "p24.wav"), w.astype(np.float32), sr, subtype="FLOAT")
        import r3_remix
        r3_remix.dub_gate(tmp / "p24.wav", tmp / "g24.wav", tmp / "g16.wav")
        g, _ = sf.read(str(tmp / "g24.wav"), dtype="float32"); m_ = min(len(g), len(zp0)); d = g[:m_] - zp0[:m_]
        fr = 20 * np.log10(np.sqrt((d[:m_ // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-12)
        rep.update({"check_err_db_rel": round(float(20 * np.log10(np.sqrt((d ** 2).mean()) + 1e-12) - 20 * np.log10(np.sqrt((zp0 ** 2).mean()) + 1e-12)), 1),
                    "frames_err_gt_-50dBFS": int((fr > -50).sum()), "worst_dbfs": round(float(fr.max()), 1), "worst_at_s": [round(float(t * FR), 2) for t in np.argsort(fr)[-4:]]})
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
