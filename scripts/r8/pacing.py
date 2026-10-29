#!/usr/bin/env python3
"""r8 pacing check (user on final_r7 04: "strange starts and pauses"). Per video, on a placed dub (dubbed_16k.wav + manifest) vs the source
speech (manifest speech_intervals = Silero on the source) and the source's ASR words, lists every spot of these classes:
  join     two sentences glued: dub gap between units < 0.3 s while the source pauses >= 0.5 s between them (an abrupt start)
  early    a sentence's dub starts > 0.5 s before the speaker starts that sentence (it talks over the previous source sentence)
  late     a sentence's dub starts > 0.6 s after the speaker starts it
  hole     the dub is silent >= 0.5 s inside its speech while the source speaks >= 60 % of that time (a pause the speaker does not make)
  excess   a dub pause > 0.8 s longer than the source pause it overlaps
  midpause a dub pause >= 0.35 s inside a sentence at a word boundary with no punctuation (mirrored = the source pauses there too)
Units' dub spans come from the placements (lip-shifted), refined by Silero on the dub. Writes <out>.json and <out>.md (a listen list).
    python scripts/r8/pacing.py --run <dir with dubbed_16k.wav + *_r4b.manifest.json> --id 04 [--face npz] --out X"""
from __future__ import annotations
import argparse, json, pathlib, re, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "r7"))
FR = 0.02


def runs(m):
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]])); return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def unit_spans(man, D):
    """per unit: (dub on, dub off, source on, source off, [(piece text, placed)])"""
    S = [(s["start"], s["end"]) for s in man["speech_intervals"]]; out = []
    for u in man["units"]:
        ps = [p for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed")]
        if not ps: continue
        b0, b1 = ps[0]["burst"][0], ps[-1]["burst"][1]
        sp = [(max(a, b0), min(b, b1)) for a, b in S if b > b0 and a < b1]
        s0, s1 = (sp[0][0], sp[-1][1]) if sp else (b0, b1)
        out.append({"id": u["id"], "text": u["text"], "en": " ".join(p["text"] for p in ps), "dub": [ps[0]["placed"][0], ps[-1]["placed"][1]], "src": [s0, s1],
                    "pieces": [(p["text"], p["placed"]) for p in ps], "bursts": [p["burst"] for p in ps]})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--id", required=True); ap.add_argument("--face"); ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=""); a = ap.parse_args()
    import run_clean_pipeline_05 as rp
    R = pathlib.Path(a.run); man = json.loads(next(R.glob("*.manifest.json")).read_text()); D = man["source"]["duration_s"]; n = int(D / FR) + 1
    dubv = [(s["start"], s["end"]) for s in rp.run_vad(R / "dubbed_16k.wav")]
    Dm = np.zeros(n, bool); Sm = np.zeros(n, bool)
    for s, e in dubv: Dm[int(s / FR):int(np.ceil(e / FR))] = True
    for s in man["speech_intervals"]: Sm[int(s["start"] / FR):int(np.ceil(s["end"] / FR))] = True
    if a.face and pathlib.Path(a.face).exists():
        import lip_place; fv, _ = lip_place.face_mask(a.face, n, D)
    else: fv = np.ones(n, bool)
    U = unit_spans(man, D)
    for u in U:   # refine the dub on / off with the dub VAD (placements include pre / post roll)
        iv = [(s, e) for s, e in dubv if e > u["dub"][0] - 0.05 and s < u["dub"][1] + 0.05]
        if iv: u["dub_v"] = [max(u["dub"][0], iv[0][0]), min(u["dub"][1], iv[-1][1])]
        else: u["dub_v"] = u["dub"]
    ev = []
    def add(kind, t0, t1, **kw): ev.append({"kind": kind, "t0": round(float(t0), 2), "t1": round(float(t1), 2), **kw})
    for k, u in enumerate(U):
        on, src_on = u["dub_v"][0], u["src"][0]
        if fv[int(src_on / FR)]:
            if on < src_on - 0.5: add("early", on, src_on, unit=u["id"], by_s=round(src_on - on, 2), en=u["en"][:60])
            if on > src_on + 0.6: add("late", src_on, on, unit=u["id"], by_s=round(on - src_on, 2), en=u["en"][:60])
        if k:
            p = U[k - 1]; dg = on - p["dub_v"][1]; sg = src_on - p["src"][1]
            if dg < 0.3 and sg >= 0.5: add("join", p["dub_v"][1], on, unit=u["id"], dub_gap_s=round(dg, 2), src_gap_s=round(sg, 2), en=u["en"][:60])
    lo, hi = int(dubv[0][0] / FR), int(dubv[-1][1] / FR)
    for s, e in runs(~Dm[lo:hi]):
        s += lo; e += lo; L = (e - s) * FR
        if L < 0.35: continue
        t0, t1 = s * FR, e * FR; spk = float(Sm[s:e].mean())
        inside = next((u for u in U if u["dub_v"][0] < t0 and t1 < u["dub_v"][1]), None)
        if L >= 0.5 and spk >= 0.6 and fv[s:e].mean() > 0.5: add("hole", t0, t1, len_s=round(L, 2), src_speaks=round(spk, 2), unit=inside["id"] if inside else None)
        # the source pause(s) this dub pause overlaps
        sil = runs(~Sm); ov = [((b_ - a_) * FR) for a_, b_ in sil if b_ > s and a_ < e]; src_p = max(ov, default=0.0)
        if L - src_p > 0.8: add("excess", t0, t1, len_s=round(L, 2), src_pause_s=round(src_p, 2), unit=inside["id"] if inside else None)
        if inside:
            # the word boundary: the piece text before the pause (pieces = the placements; a pause between pieces or inside one)
            prev_txt = " ".join(t for t, pl in inside["pieces"] if pl[1] <= t0 + 0.1) or ""
            last = prev_txt.split()[-1] if prev_txt.split() else ""
            punct = bool(re.search(r"[,.;:!?—-]$", last))
            if not punct: add("midpause", t0, t1, len_s=round(L, 2), unit=inside["id"], after=last, mirrored=bool(Sm[s:e].mean() < 0.4), src_speaks=round(spk, 2))
    ev.sort(key=lambda x: x["t0"])
    summ = {k: sum(1 for e in ev if e["kind"] == k) for k in ("join", "early", "late", "hole", "excess", "midpause")}
    summ["hole_s"] = round(sum(e["len_s"] for e in ev if e["kind"] == "hole"), 2)
    summ["early_s"] = round(sum(e["by_s"] for e in ev if e["kind"] == "early"), 2); summ["late_s"] = round(sum(e["by_s"] for e in ev if e["kind"] == "late"), 2)
    summ["src_speaks_dub_silent_face_s"] = round(float((Sm & ~Dm & fv).sum() * FR), 2); summ["dub_speaks_src_silent_s"] = round(float((Dm & ~Sm).sum() * FR), 2)
    res = {"id": a.id, "label": a.label, "summary": summ, "events": ev,
           "units": [{"id": u["id"], "dub": [round(x, 2) for x in u["dub_v"]], "src": [round(x, 2) for x in u["src"]], "en": u["en"], "ru": u["text"]} for u in U]}
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); out.with_suffix(".json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    mm = lambda t: f"{int(t // 60)}:{t % 60:05.2f}"
    md = [f"# {a.id} {a.label} pacing", "", "summary: " + ", ".join(f"{k} {v}" for k, v in summ.items()), "", "| at | kind | detail |", "|---|---|---|"]
    for e in ev: md.append(f"| {mm(e['t0'])}–{mm(e['t1'])} | {e['kind']} | " + ", ".join(f"{k}={v}" for k, v in e.items() if k not in ("kind", "t0", "t1")) + " |")
    md += ["", "| unit | dub | source | EN |", "|---|---|---|---|"] + [f"| {u['id']} | {u['dub_v'][0]:.2f}–{u['dub_v'][1]:.2f} | {u['src'][0]:.2f}–{u['src'][1]:.2f} | {u['en']} |" for u in U]
    out.with_suffix(".md").write_text("\n".join(md) + "\n"); print(json.dumps(summ))


if __name__ == "__main__":
    main()
