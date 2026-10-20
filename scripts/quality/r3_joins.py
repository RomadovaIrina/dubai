#!/usr/bin/env python3
"""Round 3 D: prosody at the joins between the placed parts of ONE sentence (user: "choppy voice"). For every join k|k+1 inside a unit:
  gap_s       silence inserted between the two placed parts (placements[k].placed end -> [k+1] start)
  d_tempo     |atempo[k+1] - atempo[k]|
  f0_jump_st  |median F0 (last 200 ms voiced of part k) - median F0 (first 200 ms voiced of part k+1)| in semitones, on the PLACED dub track
  f0_raw_st   the same at the same word boundary inside the uncut whole-sentence generation (tts_u<id>_raw.wav): the natural reference
  e_jump_db   |RMS last 100 ms - RMS first 100 ms| (dB) on the placed track
    python scripts/quality/r3_joins.py MANIFEST [MANIFEST ...] [--json out.json]   (dubbed_24k.wav / raw clips next to the manifest)"""
import argparse, json, pathlib
import numpy as np, soundfile as sf


def f0_med(y, sr):
    import librosa
    if len(y) < int(0.06 * sr): return None
    f0 = librosa.yin(y.astype(np.float32), fmin=70, fmax=400, sr=sr, frame_length=1024, hop_length=128)
    rms = librosa.feature.rms(y=y.astype(np.float32), frame_length=1024, hop_length=128)[0][:len(f0)]
    v = f0[(rms > 0.25 * rms.max()) & (f0 > 75) & (f0 < 390)] if rms.size else []
    return float(np.median(v)) if len(v) >= 3 else None


def seg(y, sr, a, b): return y[max(0, int(a * sr)):max(0, int(b * sr))]


def db(x): return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-9) if len(x) else -120.0


def analyse(mp):
    mp = pathlib.Path(mp); m = json.loads(mp.read_text()); d = mp.parent
    y, sr = sf.read(str(d / "dubbed_24k.wav"), dtype="float32"); y = y if y.ndim == 1 else y.mean(1); rows = []
    for u in m["units"]:
        pl = (u.get("alignment") or {}).get("placements") or []; parts = u.get("tts_parts") or []
        raw = None
        if u.get("tts_raw_file"):
            rp = d / pathlib.Path(u["tts_raw_file"]).name
            if rp.exists(): raw, rsr = sf.read(str(rp), dtype="float32"); raw = raw if raw.ndim == 1 else raw.mean(1)
        for k in range(len(pl) - 1):
            e, s = pl[k]["placed"][1], pl[k + 1]["placed"][0]
            fa, fb = f0_med(seg(y, sr, e - 0.2, e), sr), f0_med(seg(y, sr, s, s + 0.2), sr)
            r = {"unit": u["id"], "k": k, "gap_s": round(s - e, 3), "d_tempo": round(abs(pl[k + 1].get("atempo", 1) - pl[k].get("atempo", 1)), 3),
                 "f0_jump_st": round(abs(12 * np.log2(fb / fa)), 2) if fa and fb else None,
                 "e_jump_db": round(abs(db(seg(y, sr, e - 0.1, e)) - db(seg(y, sr, s, s + 0.1))), 2)}
            if raw is not None and k + 1 < len(parts):
                ce, cs = parts[k]["clip_span"][1], parts[k + 1]["clip_span"][0]
                ra, rb = f0_med(seg(raw, rsr, ce - 0.2, ce), rsr), f0_med(seg(raw, rsr, cs, cs + 0.2), rsr)
                r["f0_raw_st"] = round(abs(12 * np.log2(rb / ra)), 2) if ra and rb else None
                r["e_raw_db"] = round(abs(db(seg(raw, rsr, ce - 0.1, ce)) - db(seg(raw, rsr, cs, cs + 0.1))), 2)   # r4: the same boundary in the uncut take
            rows.append(r)
    def med(key, f=lambda x: x): v = [f(r[key]) for r in rows if r.get(key) is not None]; return round(float(np.median(v)), 3) if v else None
    fj = [r["f0_jump_st"] for r in rows if r.get("f0_jump_st") is not None]
    exc = [r["f0_jump_st"] - r["f0_raw_st"] for r in rows if r.get("f0_jump_st") is not None and r.get("f0_raw_st") is not None]
    return {"manifest": str(mp), "joins": len(rows), "gap_median_s": med("gap_s"), "joins_gap_gt_0.3s": sum(1 for r in rows if r["gap_s"] > 0.3),
            "d_tempo_mean": round(float(np.mean([r["d_tempo"] for r in rows])), 3) if rows else None, "f0_jump_median_st": med("f0_jump_st"),
            "f0_jump_gt_3st": sum(1 for x in fj if x > 3), "f0_excess_over_raw_median_st": round(float(np.median(exc)), 2) if exc else None,
            "e_jump_median_db": med("e_jump_db"),
            "e_excess_over_raw_median_db": round(float(np.median([r["e_jump_db"] - r["e_raw_db"] for r in rows if r.get("e_raw_db") is not None])), 2) if any(r.get("e_raw_db") is not None for r in rows) else None,
            "rows": rows}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("manifests", nargs="+"); ap.add_argument("--json", default=None); a = ap.parse_args(); out = []
    for mp in a.manifests:
        r = analyse(mp); out.append(r); print(json.dumps({k: v for k, v in r.items() if k != "rows"}))
    if a.json: pathlib.Path(a.json).write_text(json.dumps(out, indent=1))
