#!/usr/bin/env python3
"""One comparison table over run roots (2026-10-14 candidates): time, rhythm (thought_metrics), mouth in dub silence (mouth_silence json),
and, when present, e2e_report rows (WER / CER / SyncNet). Usage: compare_runs.py OUT.md label=root [label=root ...]"""
import json, pathlib, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from thought_metrics import metrics

DUR = {"01": 101.667, "02": 110.867, "03": 90.333, "04": 53.9, "05": 52.367}


def row(root: pathlib.Path, vid: str) -> dict | None:
    mp = root / "final" / f"{vid}_final.manifest.json"
    if not mp.exists():
        return None
    man = json.loads(mp.read_text()); st = man["stage_seconds"]; tm = metrics(man); r = {"wall": st.get("total")}
    r["s_per_min"] = round(st["total"] / DUR[vid] * 60) if st.get("total") else None
    q = sum(st.get(k) or 0 for k in ("segmentation", "translation")); ts = man["tts"].get("stats", {})
    r["qwen_s"] = round(q + (ts.get("fitup_qwen_seconds") or 0) + (ts.get("fitdown_qwen_seconds") or 0), 1)
    r["tts_s"] = st.get("tts"); r["ls_s"] = st.get("latentsync"); r["cf_s"] = st.get("codeformer")
    r.update({k: tm[k] for k in ("inside_pauses_gt_0.3", "weak_pauses_gt_0.3", "between_pause_max_s", "atempo_max", "parts_sped_up", "parts_slowed",
                                 "shift_max_s", "source_without_dub_s", "dub_outside_source_s")})
    al = man.get("alignment", {}); r["how"] = al.get("used")
    r["fit_down"] = ts.get("fit_parts_rewritten"); r["splits"] = (ts.get("thought_splits") or 0) + (ts.get("thought_clause_splits") or 0)
    mj = root / "mouth" / f"{vid}.json"
    if mj.exists():
        m = json.loads(mj.read_text()); k = "dub_silent&src_speech"
        r["silent_frames"] = m[k]["output"]["frames"]; r["silent_open_out"] = m[k]["output"]["open_mean"]; r["silent_open_src"] = m[k]["master"]["open_mean"]
        r["silent_shape_out"] = m[k]["output"]["shape_motion"]; r["speech_shape_out"] = m["dub_speech"]["output"]["shape_motion"]
    for rep in root.glob("report_*.json"):
        for x in json.loads(rep.read_text()):
            if x["id"] == vid:
                it = x["intelligibility"]; r["wer"] = it.get("wer_whisper_mean"); r["cer"] = it.get("cer_ctc_mean"); r["bad"] = it.get("units_bad(cer_ctc>=0.3)")
                r["interj"] = it.get("interjection_units"); r["final_miss"] = it.get("final_word_missing_units")
                sy = x.get("syncnet"); r["syncnet"] = sy.get("confidence") if isinstance(sy, dict) else sy
                sd = x.get("syncnet_dub_only") or {}; r["syncnet_dub"] = sd.get("confidence"); r["av_off"] = sy.get("av_offset") if isinstance(sy, dict) else None
                r["hot_tails"] = (x.get("edges") or {}).get("inner_tails_hot")
    return r


def main() -> int:
    out = pathlib.Path(sys.argv[1]); runs = [a.split("=", 1) for a in sys.argv[2:]]
    cols = [("wall s", "wall"), ("s / video-min", "s_per_min"), ("Qwen s", "qwen_s"), ("TTS stage s", "tts_s"), ("LatentSync s", "ls_s"), ("CodeFormer s", "cf_s"),
            ("pauses inside a sentence > 0.3 s", "inside_pauses_gt_0.3"), ("…on a weak word", "weak_pauses_gt_0.3"), ("longest pause between sentences s", "between_pause_max_s"),
            ("parts slowed (< 1.0)", "parts_slowed"), ("parts sped up", "parts_sped_up"), ("atempo max", "atempo_max"), ("next sentence shifted max s", "shift_max_s"),
            ("Qwen concise rewrites", "fit_down"), ("punctuation splits", "splits"), ("source speech without dub s", "source_without_dub_s"),
            ("dub outside source speech s", "dub_outside_source_s"), ("frames: dub silent, speaker talks", "silent_frames"), ("mouth opening there: output / source", None),
            ("lip motion: dub silent / dub speech", None), ("WER", "wer"), ("CTC CER", "cer"), ("bad units", "bad"), ("interjections", "interj"), ("final word missing", "final_miss"),
            ("SyncNet (mix) / dub-only", None), ("AV offset", "av_off")]
    L = []
    for vid in ("04", "01", "03"):
        rows = [(lab, row(pathlib.Path(root), vid)) for lab, root in runs]; rows = [(l, r) for l, r in rows if r]
        if not rows: continue
        L += [f"### {vid}", "", "| metric | " + " | ".join(l for l, _ in rows) + " |", "|---|" + "---:|" * len(rows)]
        for name, key in cols:
            vals = []
            for _, r in rows:
                if name.startswith("mouth opening"): v = f"{r.get('silent_open_out')} / {r.get('silent_open_src')}"
                elif name.startswith("lip motion"): v = f"{r.get('silent_shape_out')} / {r.get('speech_shape_out')}"
                elif name.startswith("SyncNet"): v = f"{r.get('syncnet')} / {r.get('syncnet_dub')}"
                else: v = r.get(key, "–")
                vals.append("–" if v is None else str(v))
            L.append(f"| {name} | " + " | ".join(vals) + " |")
        L.append("")
    out.write_text("\n".join(L)); print("\n".join(L)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
