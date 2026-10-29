#!/usr/bin/env python3
"""r8 coverage check (management on final_r7 02 / 05: "he keeps DISAPPEARING"; face: "mouth about to vomit, says nothing"): every span where the
SOURCE speaks and the DUB is silent, on and off screen, with its cause. Dub speech = Silero on the final dub (dubbed_16k.wav of a remix dir,
the gated one); source speech = manifest speech_intervals. Spans <= --breath s are natural breaths and not listed.
Causes (first that applies):
  gate        the span overlaps a dub_gate run and the take had sound there (the gate muted it: check it is not a quiet word)
  take_pause  inside a placed piece: the take's own silence (Chatterbox pause / lead-in / tail)
  burst_pause the unit's burst spans a source pause >= 1 s and the English piece was placed at the burst onset, so the words after the pause
              are said too early (02 u3: "...a Russian banya" at 30.5 s, the speaker says it at 40.3 s)
  en_short    inside the unit's burst, after its English ended: the English is shorter than the Russian
  between     between two units' bursts (segmentation: the speech belongs to no unit's burst)
  no_unit     no unit and no ASR word there (ASR / hallucination-filter miss)
    python scripts/r8/coverage.py --run <remix dir with dubbed_16k.wav + <id>_r4b.manifest.json> --id 02 [--face npz] --out X"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "r7"))
FR = 0.02


def runs(m):
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]])); return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--id", required=True); ap.add_argument("--face"); ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=""); ap.add_argument("--breath", type=float, default=0.3); a = ap.parse_args()
    import run_clean_pipeline_05 as rp
    R = pathlib.Path(a.run); man = json.loads(next(R.glob(f"{a.id}_*.manifest.json")).read_text()); D = man["source"]["duration_s"]; n = int(D / FR) + 1
    Dm = np.zeros(n, bool); Sm = np.zeros(n, bool)
    for s in rp.run_vad(R / "dubbed_16k.wav"): Dm[int(s["start"] / FR):int(np.ceil(s["end"] / FR))] = True
    for s in man["speech_intervals"]: Sm[int(s["start"] / FR):int(np.ceil(s["end"] / FR))] = True
    if a.face and pathlib.Path(a.face).exists():
        import lip_place; fv, _ = lip_place.face_mask(a.face, n, D)
    else: fv = np.zeros(n, bool)
    gate = [(g0, g1) for g0, g1 in (man.get("dub_gate") or {}).get("runs", [])]
    words = [w for sg in man["asr"]["segments"] for w in sg.get("words") or []]
    pls = [(u["id"], p) for u in man["units"] for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed")]
    ev = []
    for s, e in runs(Sm & ~Dm):
        L = (e - s) * FR
        if L <= a.breath: continue
        t0, t1 = s * FR, e * FR; mid = (t0 + t1) / 2; cause = None; unit = None; det = {}
        if any(g0 < t1 and t0 < g1 for g0, g1 in gate): cause = "gate"
        pin = [(uid, p) for uid, p in pls if p["placed"][0] - 0.02 <= t0 and t1 <= p["placed"][1] + 0.02]
        if not cause and pin: cause, unit = "take_pause", pin[0][0]
        if not cause:
            bin_ = [(uid, p) for uid, p in pls if p["burst"][0] - 0.05 <= mid <= p["burst"][1] + 0.05]
            if bin_:
                uid, p = bin_[0]; unit = uid
                # a source pause >= 1 s inside the burst before this span, and the piece already ended before that pause
                sil = [(x * FR, y * FR) for x, y in runs(~Sm[int(p["burst"][0] / FR):int(t0 / FR)])]
                long_p = [(x + p["burst"][0], y + p["burst"][0]) for x, y in sil if y - x >= 1.0]
                cause = "burst_pause" if long_p and p["placed"][1] <= long_p[-1][0] + 0.3 else "en_short"
                det = {"en": p["text"], "piece_end": round(p["placed"][1], 2), "burst": [round(x, 2) for x in p["burst"]]}
            else:
                cause = "between" if any(t0 - 0.3 <= w["start"] <= t1 + 0.3 for w in words) else "no_unit"
        wd = " ".join(w["word"].strip() for w in words if t0 - 0.15 <= (w["start"] + w["end"]) / 2 <= t1 + 0.15)
        ev.append({"t0": round(t0, 2), "t1": round(t1, 2), "len_s": round(L, 2), "face": bool(fv[s:e].mean() > 0.5), "cause": cause, "unit": unit, "ru": wd, **det})
    tot = float((Sm & ~Dm).sum() * FR); listed = sum(x["len_s"] for x in ev)
    summ = {"src_speaks_dub_silent_s": round(tot, 2), "beyond_breaths_s": round(listed, 2), "on_face_s": round(sum(x["len_s"] for x in ev if x["face"]), 2),
            "off_face_s": round(sum(x["len_s"] for x in ev if not x["face"]), 2), "dub_speaks_src_silent_s": round(float((Dm & ~Sm).sum() * FR), 2), "spans": len(ev),
            "by_cause_s": {c: round(sum(x["len_s"] for x in ev if x["cause"] == c), 2) for c in ("gate", "take_pause", "burst_pause", "en_short", "between", "no_unit")}}
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"id": a.id, "label": a.label, "summary": summ, "spans": ev}, indent=1, ensure_ascii=False))
    mm = lambda t: f"{int(t // 60)}:{t % 60:05.2f}"
    md = [f"# {a.id} {a.label}: source speaks, dub silent (> {a.breath} s)", "", "summary: " + json.dumps(summ, ensure_ascii=False), "", "| at | s | face | cause | unit | RU said there | EN piece |", "|---|---|---|---|---|---|---|"]
    md += [f"| {mm(x['t0'])}–{mm(x['t1'])} | {x['len_s']} | {'yes' if x['face'] else 'no'} | {x['cause']} | {x['unit']} | {x['ru']} | {x.get('en', '')} |" for x in ev]
    out.with_suffix(".md").write_text("\n".join(md) + "\n"); print(a.id, json.dumps(summ))


if __name__ == "__main__":
    main()
