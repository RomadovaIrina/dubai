#!/usr/bin/env python3
"""r8 v3 detectors (user on r8v1: "I'm saying ..... the phrase", words cut off abruptly), on a placed / final dub (dubbed_24k.wav + manifest):
  cut_word      an abrupt edge in the dub voice: within --edge-ms the level falls from >= ref + --hi-db (a word's body; ref = the voice's p95, 5-ms
                frames) to <= ref + --lo-db (digital silence / the gate floor), or rises that fast (an onset that starts mid-word). A natural word decay /
                attack takes >= 30 ms. Reported with the dub piece it belongs to.
  inserted_gap  a dub pause inside one sentence (between its first and last placed piece) longer than --gap-s, that is > --gap-s longer than the
                longest source pause it overlaps ("unless the source itself pauses that long there"); kind = 'split' when it falls on a boundary
                between two placed pieces (a cut made by retake / burst_align / lip_place), 'take' inside one piece (the TTS's own pause)
    python scripts/r8/cuts.py --run <dir> --id 02 --out X      -> X.json / X.md"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--id", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=""); ap.add_argument("--edge-ms", type=float, default=15.0); ap.add_argument("--hi-db", type=float, default=-25.0)
    ap.add_argument("--lo-db", type=float, default=-55.0); ap.add_argument("--gap-s", type=float, default=0.35); a = ap.parse_args()
    import run_clean_pipeline_05 as rp
    R = pathlib.Path(a.run); man = json.loads(next(R.glob(f"{a.id}_*.manifest.json")).read_text()); D = man["source"]["duration_s"]
    y, sr = sf.read(str(R / "dubbed_24k.wav"), dtype="float32"); y = y if y.ndim == 1 else y.mean(1)
    h = int(0.005 * sr); L = 20 * np.log10(np.sqrt((y[:len(y) // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-9); ref = float(np.percentile(L[L > -90], 95))
    pls = sorted([(u["id"], p) for u in man["units"] for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed")], key=lambda x: x[1]["placed"][0])
    piece_at = lambda t: next(((uid, p["text"]) for uid, p in pls if p["placed"][0] - 0.1 <= t <= p["placed"][1] + 0.1), (None, ""))
    k = max(1, int(round(a.edge_ms / 5))); hi, lo = ref + a.hi_db, ref + a.lo_db; ev = []
    i = k
    while i < len(L) - k:
        if L[i - 1] >= hi and L[i - 1:i + k].min() <= lo and (L[i:i + k] <= lo).any():   # fall
            j = i + int(np.argmax(L[i:i + k] <= lo)); t = j * h / sr; uid, txt = piece_at(t)
            ev.append({"kind": "cut_word", "edge": "end", "t": round(t, 3), "drop_db": round(float(L[i - 1] - L[j]), 1), "unit": uid, "piece": txt}); i = j + k; continue
        if L[i + k - 1] >= hi and L[i - 1] <= lo and L[i - 1:i + k].max() >= hi and (L[i - k:i] <= lo).all():   # rise
            t = i * h / sr; uid, txt = piece_at(t)
            ev.append({"kind": "cut_word", "edge": "start", "t": round(t, 3), "rise_db": round(float(L[i + k - 1] - L[i - 1]), 1), "unit": uid, "piece": txt}); i += k; continue
        i += 1
    # inserted gaps
    FR = 0.02; n = int(D / FR) + 1; Dm = np.zeros(n, bool); Sm = np.zeros(n, bool)
    for s in rp.run_vad(R / "dubbed_16k.wav"): Dm[int(s["start"] / FR):int(np.ceil(s["end"] / FR))] = True
    for s in man["speech_intervals"]: Sm[int(s["start"] / FR):int(np.ceil(s["end"] / FR))] = True
    d = np.diff(np.concatenate([[0], (~Dm).astype(np.int8), [0]])); sil = list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))
    ds = np.diff(np.concatenate([[0], (~Sm).astype(np.int8), [0]])); ssil = list(zip(np.flatnonzero(ds == 1), np.flatnonzero(ds == -1)))
    for u in man["units"]:
        ps = sorted([p for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed")], key=lambda p: p["placed"][0])
        if not ps: continue
        u0, u1 = ps[0]["placed"][0], ps[-1]["placed"][1]
        for s, e in sil:
            t0, t1 = s * FR, e * FR; Lg = t1 - t0
            if t0 <= u0 + 0.05 or t1 >= u1 - 0.05 or Lg <= a.gap_s: continue
            sp = max([(min(e, b) - max(s, a_)) * FR for a_, b in ssil if b > s and a_ < e] or [0.0])
            if Lg - sp <= a.gap_s: continue
            bnd = next((q for q in range(len(ps) - 1) if ps[q]["placed"][1] - 0.15 <= t0 + 0.1 and ps[q + 1]["placed"][0] + 0.15 >= t1 - 0.1), None)
            ev.append({"kind": "inserted_gap", "t": round(t0, 2), "t1": round(t1, 2), "len_s": round(Lg, 2), "src_pause_s": round(sp, 2), "unit": u["id"],
                       "where": "split" if bnd is not None else "take", "after": (ps[bnd]["text"].split() or [""])[-1] if bnd is not None else ""})
    ev.sort(key=lambda x: x["t"])
    summ = {"cut_word": sum(1 for x in ev if x["kind"] == "cut_word"), "cut_word_end": sum(1 for x in ev if x["kind"] == "cut_word" and x["edge"] == "end"),
            "inserted_gap": sum(1 for x in ev if x["kind"] == "inserted_gap"), "inserted_gap_split": sum(1 for x in ev if x["kind"] == "inserted_gap" and x["where"] == "split"),
            "inserted_gap_s": round(sum(x["len_s"] for x in ev if x["kind"] == "inserted_gap"), 2)}
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"id": a.id, "label": a.label, "summary": summ, "events": ev}, indent=1, ensure_ascii=False))
    mm = lambda t: f"{int(t // 60)}:{t % 60:05.2f}"
    md = [f"# {a.id} {a.label}: cut words / inserted mid-sentence gaps", "", "summary: " + json.dumps(summ), "", "| at | kind | detail |", "|---|---|---|"]
    md += [f"| {mm(x['t'])} | {x['kind']} | " + ", ".join(f"{k_}={v}" for k_, v in x.items() if k_ not in ("kind", "t")) + " |" for x in ev]
    out.with_suffix(".md").write_text("\n".join(md) + "\n"); print(a.id, a.label, json.dumps(summ))


if __name__ == "__main__":
    main()
