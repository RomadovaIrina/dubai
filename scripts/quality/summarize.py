#!/usr/bin/env python3
"""Phase 5 summary: one row per (candidate, video) from manifests/*.quality.json (+ face_metrics.json) -> summary.csv + summary.md.
    python scripts/quality/summarize.py [--manifests /tmp/dabai_quality/manifests] [--out reports/quality/summary]
"""
from __future__ import annotations
import argparse, csv, json, pathlib
CHANGED = {"baseline": "FINAL_ML_CONFIG (none)", "e1": "TTS placed per original speech burst (split at TTS pauses/energy valleys, atempo cap 1.3, extend into following silence)",
           "e1b": "e1 + slow-down to 0.85x to fill bursts", "e3": "slot extension into following silence + atempo cap 1.3 (no burst split)",
           "e4_mouthonly": "assembly: LatentSync pixels only inside feathered lower-face mask", "e6_xf0": "crossfade 0", "e6_xf8": "crossfade 8",
           "e6_margin045": "gate margin 0.45 s", "e6_gap10": "gate merge gap 1.0 s", "e6_stride1": "router stride 1", "g20": "guidance 2.0", "g25": "guidance 2.5"}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--manifests", default="/tmp/dabai_quality/manifests"); ap.add_argument("--out", default="reports/quality/summary"); a = ap.parse_args()
    M = pathlib.Path(a.manifests); rows = []
    for qp in sorted(M.glob("*.quality.json")):
        q = json.loads(qp.read_text())
        if "run_error" in q: continue
        vid, tag = q["video"], q["tag"]; fmp = M / f"{vid}_{tag}.face_metrics.json"; fm = json.loads(fmp.read_text()) if fmp.exists() else None
        sn = q.get("syncnet", {}); so = sn.get("output") or {}; sg = sn.get("original") or {}
        ms = [s["mouth_sharp_ratio"] for s in fm["segments"]] if fm else []; us = [s["upper_sharp_ratio"] for s in fm["segments"]] if fm else []; fl = [s["flicker_ratio"] for s in fm["segments"]] if fm else []
        jumps = [t["jump_vs_neighbours_out"] for t in fm["transitions"] if t.get("jump_vs_neighbours_out")] if fm else []
        rows.append({"candidate": tag, "changed_parameters": CHANGED.get(tag, q.get("config", {})), "video": vid, "syncnet_confidence": so.get("confidence", sn.get("output", {}).get("status", "-") if sn else "-"),
                     "av_offset": so.get("av_offset_frames", "-"), "original_syncnet": sg.get("confidence", "-"), "duration_delta_s": q["output"]["duration_delta_s"],
                     "frames_delta": q["output"]["frames"] - q["routing"]["master_frames"], "peak_vram_mib": q["latentsync"]["peak_vram_nvsmi_mib"], "wall_s": q["latentsync"]["wall_total_s"],
                     "ls_inference_s": q["latentsync"]["inference_total_s"], "ls_segments": q["routing"]["latentsync_segments"], "transitions": q["routing"]["n_transitions"],
                     "ls_silent_fraction": q["audio"].get("ls_silent_fraction", "-"), "ls_silent_while_original_speaks": q["audio"].get("ls_frames_dub_silent_while_original_speaks", "-"),
                     "units_sped_up_gt_1_25": q["audio"]["sped_up_gt_1_25"], "atempo_max": max([u.get("atempo", 1.0) for u in q["audio"]["units_table"]] or [1.0]),
                     "mouth_sharp_ratio": f"{min(ms):.2f}-{max(ms):.2f}" if ms else "-", "upper_sharp_ratio": f"{min(us):.2f}-{max(us):.2f}" if us else "-", "flicker_ratio": f"{min(fl):.2f}-{max(fl):.2f}" if fl else "-",
                     "boundary_jump_max_vs_neighbours": max(jumps) if jumps else "-", "validation": "PASS" if all(q["output"]["validation"].values()) and (q["output"]["frame_check"] or {}).get("pass") else "FAIL",
                     "notes": ""})
    rows.sort(key=lambda r: (r["video"], r["candidate"] != "baseline", r["candidate"]))
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out) + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    cols = ["video", "candidate", "syncnet_confidence", "original_syncnet", "av_offset", "ls_silent_while_original_speaks", "ls_silent_fraction", "units_sped_up_gt_1_25", "atempo_max", "mouth_sharp_ratio", "upper_sharp_ratio", "flicker_ratio", "boundary_jump_max_vs_neighbours", "ls_segments", "transitions", "duration_delta_s", "frames_delta", "peak_vram_mib", "wall_s", "validation"]
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    pathlib.Path(str(out) + ".md").write_text("\n".join(md) + "\n"); print("\n".join(md)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
