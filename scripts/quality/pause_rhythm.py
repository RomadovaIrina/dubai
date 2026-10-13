#!/usr/bin/env python3
"""Speech rhythm / pause analysis of one run, original vs dub ("паузы заметны", 2026-10-13 feedback): not coverage alone but how the pauses look.

  speech_s, pause count, pause distribution (median / p90 / > 0.5 s / > 1.0 s) for original VAD (manifest speech_intervals) and dub VAD
  (Silero on dubbed_16k.wav, same call as the pipeline), inside [first speech, last speech]
  intra-sentence dub pauses: gaps between consecutive placed parts of ONE unit (a unit = one translated sentence) -> preserved (the source
  pauses at the same place too, >= 50 % of the gap is source silence) or inserted (the source keeps speaking through most of the gap)
  source_without_dub / dub_without_source: seconds + every event > 0.3 s, each classified by cause:
     late_start (inside a unit before its first part), inner_gap (between parts of a unit: burst mapping / phrase split), short_fill (after the
     last part of a unit while the source still speaks: TTS shorter than the source, fit-up / fill), between_units (no unit there: segmentation /
     ASR), unit_missing; dub_without_source: spill (past the slot end), lead (before the first burst), inside_slot_pause (a part over a source pause)
  timeline PNG: original VAD | TTS parts (unit colour, atempo) | dub VAD | LatentSync segments; 30 s per row

    python scripts/quality/pause_rhythm.py --manifest RUN/final/03_final.manifest.json --out RUN/pauses --tag reviewed
"""
from __future__ import annotations
import argparse, json, pathlib, re, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot"))
R = 0.01   # 10 ms grid


def mask(ivs, n):
    m = np.zeros(n, bool)
    for s, e in ivs: m[max(0, int(round(s / R))):min(n, int(round(e / R)))] = True
    return m


def runs(m):
    out = []; i = 0; n = len(m)
    while i < n:
        if m[i]:
            j = i
            while j < n and m[j]: j += 1
            out.append((i * R, j * R)); i = j
        else: i += 1
    return out


def pauses(ivs):
    ivs = sorted(ivs); return [b[0] - a[1] for a, b in zip(ivs, ivs[1:]) if b[0] - a[1] > 0.0]


def pstats(ivs) -> dict:
    p = np.array(pauses(ivs)); sp = sum(e - s for s, e in ivs)
    if not len(p): return {"speech_s": round(sp, 2), "pauses": 0}
    return {"speech_s": round(sp, 2), "segments": len(ivs), "pauses": int(len(p)), "pause_total_s": round(float(p.sum()), 2), "median": round(float(np.median(p)), 3),
            "p90": round(float(np.percentile(p, 90)), 3), "max": round(float(p.max()), 2), "gt_0_3": int((p > 0.3).sum()), "gt_0_5": int((p > 0.5).sum()), "gt_1_0": int((p > 1.0).sum()),
            "hist_s": {f"{lo}-{hi}": int(((p >= lo) & (p < hi)).sum()) for lo, hi in ((0, 0.15), (0.15, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.2), (1.2, 2.0), (2.0, 99))}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--min-event-s", type=float, default=0.3)
    a = ap.parse_args()
    man = json.loads(pathlib.Path(a.manifest).read_text()); W = pathlib.Path(man["work_dir"]); dur = man["source"]["duration_s"]
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); stem = f"{pathlib.Path(man['input']).stem}_{a.tag}"
    import run_clean_pipeline_05 as rp
    from tts_burst import FIXED_PHRASES, LONG_PAUSE_FUNC
    orig = [(v["start"], v["end"]) for v in man["speech_intervals"]]
    dubv = [(v["start"], v["end"]) for v in rp.run_vad(W / "dubbed_16k.wav")]
    n = int(dur / R) + 1; mo, md = mask(orig, n), mask(dubv, n)
    units = man["units"]; parts = []
    for u in units:
        for k, pl in enumerate(u["alignment"]["placements"]):
            if pl.get("placed"): parts.append({"unit": u["id"], "k": k, "t0": pl["placed"][0], "t1": pl["placed"][1], "atempo": pl.get("atempo", 1.0), "text": pl.get("text", "")})
    slot = {u["id"]: (u["slot"]["start"], u["slot"]["end"]) for u in units}
    by_unit = {}
    for p in parts: by_unit.setdefault(p["unit"], []).append(p)
    marks_of = {u["id"]: (set(u["pause_marks"]) if u.get("pause_marks") is not None else None) for u in units}
    word_end_of = {}   # (unit, part k) -> index of the last word of that part in the unit's final text
    for u in units:
        acc = -1
        for k, t in enumerate(u.get("tts_parts", [])):
            acc += len(t["text"].split()); word_end_of[(u["id"], k)] = acc
    # intra-sentence pauses of the dub (between parts of one unit)
    intra = []
    for uid, ps in by_unit.items():
        ps.sort(key=lambda p: p["t0"])
        for p, q in zip(ps, ps[1:]):
            g = q["t0"] - p["t1"]
            if g > 0.15:
                i0, i1 = int(p["t1"] / R), int(q["t0"] / R); src_sil = float(1 - mo[i0:i1].mean()) if i1 > i0 else 1.0
                last = re.sub(r"[^\w']", "", (p["text"].split() or [""])[-1].lower()); nxt_w = re.sub(r"[^\w']", "", (q["text"].split() or [""])[0].lower())
                punct = bool(re.search(r"[,.;:!?…]['\"»)]*$", p["text"].strip())); mk = marks_of.get(uid)
                weak = (not punct and word_end_of.get((uid, p["k"])) not in mk) if mk is not None else \
                    (not punct and (last in LONG_PAUSE_FUNC or (last, nxt_w) in FIXED_PHRASES or bool(re.search(r"\d", last))))
                intra.append({"unit": uid, "t": round(p["t1"], 2), "gap_s": round(g, 2), "source_silent_frac": round(src_sil, 2), "kind": "preserved" if src_sil >= 0.5 else "inserted",
                              "weak_boundary": weak, "before": p["text"][-35:], "after": q["text"][:35]})
    # source speaking, dub silent -> classify
    def classify_sd(t0, t1):
        tm = 0.5 * (t0 + t1)
        for u in units:
            s, e = slot[u["id"]]
            if s - 0.05 <= tm <= e + 0.05:
                ps = sorted(by_unit.get(u["id"], []), key=lambda p: p["t0"])
                if not ps: return "unit_missing", u["id"]
                if tm < ps[0]["t0"]: return "late_start", u["id"]
                if tm > ps[-1]["t1"]: return "short_fill", u["id"]
                return "inner_gap", u["id"]
        return "between_units", None
    sd = []
    for t0, t1 in runs(mo & ~md):
        if t1 - t0 >= a.min_event_s:
            c, uid = classify_sd(t0, t1); sd.append({"t": round(t0, 2), "seconds": round(t1 - t0, 2), "cause": c, "unit": uid})
    ds = []
    for t0, t1 in runs(md & ~mo):
        if t1 - t0 >= a.min_event_s:
            tm = 0.5 * (t0 + t1); c = "inside_slot_pause"; uid = None
            for u in units:
                s, e = slot[u["id"]]
                if s <= tm <= e: uid = u["id"]; break
            if uid is None:
                prev = [u for u in units if slot[u["id"]][1] < tm]; nxt = [u for u in units if slot[u["id"]][0] > tm]
                c = "spill" if prev and (not nxt or tm - slot[prev[-1]["id"]][1] < slot[nxt[0]["id"]][0] - tm) else "lead"
                uid = prev[-1]["id"] if c == "spill" and prev else (nxt[0]["id"] if nxt else None)
            ds.append({"t": round(t0, 2), "seconds": round(t1 - t0, 2), "cause": c, "unit": uid})
    def tot(lst, key=None):
        d = {}
        for x in lst: d[x["cause"]] = round(d.get(x["cause"], 0) + x["seconds"], 2)
        return d
    inter = np.logical_or(mo, md).sum(); iou = float((mo & md).sum() / max(inter, 1))
    rep = {"duration_s": dur, "original": pstats(orig), "dub": pstats(dubv), "vad_iou": round(iou, 3),
           "source_without_dub_s": round(float((mo & ~md).sum() * R), 2), "dub_without_source_s": round(float((md & ~mo).sum() * R), 2),
           "source_without_dub_events": sd, "source_without_dub_by_cause_s": tot(sd), "dub_without_source_events": ds, "dub_without_source_by_cause_s": tot(ds),
           "intra_sentence_pauses": intra, "intra_inserted": sum(1 for x in intra if x["kind"] == "inserted"), "intra_preserved": sum(1 for x in intra if x["kind"] == "preserved"),
           "intra_inserted_gt_0_3": sum(1 for x in intra if x["kind"] == "inserted" and x["gap_s"] > 0.3),
           "weak_boundary_pauses_gt_0_3": sum(1 for x in intra if x["weak_boundary"] and x["gap_s"] > 0.3), "weak_boundary_pauses_gt_0_8": sum(1 for x in intra if x["weak_boundary"] and x["gap_s"] > 0.8),
           "weak_boundary_pause_s": round(sum(x["gap_s"] for x in intra if x["weak_boundary"] and x["gap_s"] > 0.3), 2),
           "atempo": {"max": max((p["atempo"] for p in parts), default=None), "gt_1_15": sum(p["atempo"] > 1.15 for p in parts), "lt_1": sum(p["atempo"] < 0.995 for p in parts)}}
    # timeline figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    seg = [s for s in man.get("lipsync", {}).get("segments", []) if s.get("action") != "SPLIT"]
    rows = int(np.ceil(dur / 30)); fig, axs = plt.subplots(rows, 1, figsize=(14, 1.9 * rows), squeeze=False)
    cmap = plt.get_cmap("tab10")
    for r in range(rows):
        ax = axs[r][0]; lo, hi = 30 * r, min(dur, 30 * (r + 1))
        for s, e in orig: ax.broken_barh([(s, e - s)], (3, 0.8), color="#4c72b0")
        for p in parts:
            ax.broken_barh([(p["t0"], p["t1"] - p["t0"])], (2, 0.8), color=cmap(p["unit"] % 10), alpha=0.85)
            if lo <= p["t0"] <= hi and abs(p["atempo"] - 1) > 0.01: ax.text(p["t0"], 2.85, f"x{p['atempo']:.2f}", fontsize=5)
        for s, e in dubv: ax.broken_barh([(s, e - s)], (1, 0.8), color="#55a868")
        for s in seg: ax.broken_barh([(s["start_s"], s["end_s"] - s["start_s"])], (0, 0.8), color="#c44e52" if s["action"] == "LATENT_SYNC" else "#dddddd")
        for x in sd:
            if lo <= x["t"] <= hi: ax.axvspan(x["t"], x["t"] + x["seconds"], color="orange", alpha=0.25)
        for x in intra:
            if x["kind"] == "inserted" and lo <= x["t"] <= hi: ax.plot([x["t"], x["t"] + x["gap_s"]], [2.4, 2.4], color="red", lw=2)
        ax.set_xlim(lo, lo + 30); ax.set_ylim(-0.1, 4); ax.set_yticks([0.4, 1.4, 2.4, 3.4]); ax.set_yticklabels(["LS segs", "dub VAD", "TTS parts", "orig VAD"], fontsize=7); ax.tick_params(labelsize=7)
    fig.suptitle(f"{stem}: orange = source speaks / dub silent, red line = inserted intra-sentence pause", fontsize=9); fig.tight_layout()
    png = out / f"{stem}_timeline.png"; fig.savefig(png, dpi=80); plt.close(fig); rep["timeline_png"] = str(png)
    (out / f"{stem}.pauses.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    o, d = rep["original"], rep["dub"]
    print(f"== {stem}: speech orig {o['speech_s']}s dub {d['speech_s']}s | pauses orig {o['pauses']} (med {o.get('median')} p90 {o.get('p90')} >0.5 {o.get('gt_0_5')} >1 {o.get('gt_1_0')}) "
          f"dub {d['pauses']} (med {d.get('median')} p90 {d.get('p90')} >0.5 {d.get('gt_0_5')} >1 {d.get('gt_1_0')}) | IoU {rep['vad_iou']}")
    print(f"   source w/o dub {rep['source_without_dub_s']}s {rep['source_without_dub_by_cause_s']} | dub w/o source {rep['dub_without_source_s']}s {rep['dub_without_source_by_cause_s']} | "
          f"intra-sentence pauses inserted {rep['intra_inserted']} (>0.3 s {rep['intra_inserted_gt_0_3']}) preserved {rep['intra_preserved']} | "
          f"weak-boundary pauses >0.3 s {rep['weak_boundary_pauses_gt_0_3']} (>0.8 s {rep['weak_boundary_pauses_gt_0_8']}, {rep['weak_boundary_pause_s']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
