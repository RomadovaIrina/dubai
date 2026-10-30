#!/usr/bin/env python3
"""r9 E2E: the per-video table of the REPORT from the manifest.json files of scripts/r9/e2e.sh runs.
    python scripts/r9/table.py --out reports/experiments/r9-e2e/table runs/*/manifest.json   -> table.md + table.csv + table.json"""
from __future__ import annotations
import argparse, csv, json, pathlib

STAGES = ["precheck", "upright", "audio", "sep", "face_gen", "face_comp", "mux"]


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict): return None
        d = d.get(k)
    return d


def row(m):
    me = m.get("metrics") or {}; t = m.get("timing") or {}; fg = me.get("face_gate") or {}; fs = me.get("face_score") or {}
    cov = me.get("coverage") or {}; i = m["id"]; c = lambda n: (cov.get(f"{n}_{i}.json") or {}).get("summary") or {}
    am = me.get("audio_metrics") or {}; spm = t.get("s_per_video_min") or {}
    prod = sum(v for k, v in (t.get("stage_wall_s") or {}).items() if k in STAGES)
    fb = me.get("fallback") or {}; lsa = me.get("lipsync_by_action") or {}
    ls_s = g(lsa, "LATENT_SYNC", "seconds") or 0; dur = t.get("duration_s") or 0
    r = {"id": i, "status": m.get("status"), "flags": " ".join(m.get("flags") or []), "dur_s": round(dur, 1),
         "fmt": f"{g(m, 'source', 'width')}x{g(m, 'source', 'height')} {g(m, 'source', 'fps')}",
         "out_matches_src": all(o.get("matches_source") for o in (m.get("outputs") or {}).values()) if m.get("outputs") else False,
         **{f"s/vmin {k}": spm.get(k) for k in STAGES if k in spm},
         "s/vmin production": round(prod / (dur / 60), 1) if dur else None, "s/vmin metrics": spm.get("metrics"),
         "s/vmin total": t.get("total_s_per_video_min"), "vram_peak_mib": t.get("vram_peak_mib"),
         "lipsync_s": round(ls_s, 1), "lipsync_pct": round(100 * ls_s / dur, 1) if dur else None,
         "syncnet_speech": fg.get("sync_speech", g(fs, "sync_speech", "conf_speech")), "flashes": fg.get("flash_strong", fs.get("flash_strong")),
         "fallback_pct": fb.get("fallback_weight_pct"), "pressed_lips_s": fg.get("pressed_lips_s", (me.get("closed_mouth") or {}).get("closed_while_source_talks_dub_silent_s")),
         "mismatch_s": fg.get("mouth_audio_mismatch_s", g(fs, "mouth_audio", "mouth_audio_mismatch_s")),
         "frame_gate": fg.get("frame_gate_verdict"), "occl_ring_p95_max": max(((me.get("occlusion") or {}).get("per_second_max") or {"_": None}).values(), key=lambda v: v or 0), "occl_frames_over_2x": (me.get("occlusion") or {}).get("frames_over_2x_median"), "frame_gate_worst_pct": fg.get("frame_gate_worst_pct"),
         "src_speaks_dub_silent_s": c("cov").get("src_speaks_dub_silent_s"), "on_face_s": c("cov").get("on_face_s"),
         "cut_words": c("cuts").get("cut_word"), "dead_air": c("mix").get("dead"), "abrupt": c("mix").get("abrupt"),
         "faith": g(am, "faith", "mean"), "faith_lt4": g(am, "faith", "lt4"), "wer": g(am, "asr", "wer_final_mix"),
         "gap_words_spleeter": g(me, "gaps_spleeter", "gap_words"), "gap_words_demucs": g(me, "gaps_demucs", "gap_words"),
         "errors": "; ".join(m.get("errors") or [])}
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("manifests", nargs="+"); a = ap.parse_args()
    rows = [row(json.loads(pathlib.Path(p).read_text())) for p in a.manifests]
    order = ["m19", "m01", "m04", "m03", "m15", "m17", "m18", "m21", "m20"]; rows.sort(key=lambda r: order.index(r["id"]) if r["id"] in order else 99)
    keys = list(dict.fromkeys(k for r in rows for k in r)); out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as f: w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    out.with_suffix(".json").write_text(json.dumps(rows, indent=1))
    fmt = lambda v: "" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))
    tot = sum(r["dur_s"] for r in rows) / 60
    md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)] + ["| " + " | ".join(fmt(r.get(k)) for k in keys) + " |" for r in rows]
    out.with_suffix(".md").write_text("\n".join(md) + f"\n\n{len(rows)} videos, {tot:.2f} video-min.\n"); print("\n".join(md))


if __name__ == "__main__":
    main()
