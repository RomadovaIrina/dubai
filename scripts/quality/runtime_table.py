#!/usr/bin/env python3
"""Runtime of a run set against the DabAI norm (300 s of processing per minute of video, judged like pilot 0.5: dataset median / mean of the
per-video ratio). Per video from <run-root>/final/<id>_final.manifest.json (+ .nvsmi.log):
  wall_per_min      total runner wall / source minutes                          (= pilot 0.5 per-job view; the process owns the GPU all along)
  audio_per_min     ASR, diarization, Qwen segmentation / translation, TTS (incl. CPU Qwen fit-up / judge), alignment, background
  video_per_min     speech gate, router, LatentSync (+ verify), CodeFormer, assembly
  pipelined_per_min max(audio, video): steady-state throughput when the CPU audio stage of the next job overlaps the GPU video stage
  qwen_cpu_s        Qwen segmentation + translation + fit-up/judge seconds (the CPU part of the audio stage)
    python scripts/quality/runtime_table.py --run-root /workspace/dub/runs/cand5 --ids 01 02 03 04 05 [--md out.md]
"""
from __future__ import annotations
import argparse, json, pathlib, statistics

AUDIO = ("audio_extract", "vad", "asr", "translation_load", "segmentation", "diarization", "translation", "tts_load", "qa_load", "tts", "alignment", "dubbed_track", "background_sep", "background_mix")
VIDEO = ("speech_gate", "face_router_load", "latentsync_load", "codeformer_load", "ls_master_25fps", "ls_master_decode", "face_classify", "latentsync", "latentsync_verify", "codeformer", "final_assembly")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", required=True); ap.add_argument("--ids", nargs="+", default=["01", "02", "03", "04", "05"]); ap.add_argument("--md", default=None)
    ap.add_argument("--label", default=None)
    a = ap.parse_args(); root = pathlib.Path(a.run_root); rows = []
    for vid in a.ids:
        mp = root / "final" / f"{vid}_final.manifest.json"
        if not mp.exists(): continue
        m = json.loads(mp.read_text()); st = m["stage_seconds"]; mins = m["source"]["duration_s"] / 60
        au = sum(st.get(k, 0) for k in AUDIO); vi = sum(st.get(k, 0) for k in VIDEO); ts = m.get("tts", {}).get("stats", {})
        qwen = st.get("segmentation", 0) + st.get("translation", 0) + ts.get("fitup_qwen_seconds", 0) + ts.get("fitdown_qwen_seconds", 0)   # judge time is inside fitup_qwen_seconds (fuller_rewrite calls faithful_check)
        nv = mp.with_name(f"{vid}_final.nvsmi.log"); peak = None
        if nv.exists():
            vals = [int(l.split(",")[1]) for l in nv.read_text().splitlines() if "," in l and l.split(",")[1].strip().isdigit()]
            peak = max(vals) if vals else None
        rows.append({"id": vid, "src_s": round(m["source"]["duration_s"], 1), "wall_s": round(st["total"], 1), "audio_s": round(au, 1), "video_s": round(vi, 1), "qwen_cpu_s": round(qwen, 1),
                     "tts_gen_s": ts.get("gen_seconds"), "latentsync_s": st.get("latentsync"), "codeformer_s": st.get("codeformer"), "peak_vram_mib": peak,
                     "wall_per_min": round(st["total"] / mins, 1), "audio_per_min": round(au / mins, 1), "video_per_min": round(vi / mins, 1), "pipelined_per_min": round(max(au, vi) / mins, 1)})
    agg = {}
    for k in ("wall_per_min", "audio_per_min", "video_per_min", "pipelined_per_min"):
        v = [r[k] for r in rows]; agg[k] = {"median": round(statistics.median(v), 1), "mean": round(statistics.mean(v), 1)} if v else None
    out = {"label": a.label or root.name, "rows": rows, "dataset": agg}
    lines = [f"| video | src s | wall s | audio s (Qwen CPU) | video s (LatentSync / CodeFormer) | peak VRAM MiB | wall s/min | audio s/min | video s/min | pipelined s/min |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['src_s']} | {r['wall_s']} | {r['audio_s']} ({r['qwen_cpu_s']}) | {r['video_s']} ({r['latentsync_s']} / {r['codeformer_s']}) | {r['peak_vram_mib']} | "
                     f"{r['wall_per_min']} | {r['audio_per_min']} | {r['video_per_min']} | {r['pipelined_per_min']} |")
    for k in ("median", "mean"):
        lines.append(f"| **{k}** | | | | | | **{agg['wall_per_min'][k]}** | {agg['audio_per_min'][k]} | {agg['video_per_min'][k]} | **{agg['pipelined_per_min'][k]}** |")
    md = "\n".join(lines); print(md)
    if a.md: pathlib.Path(a.md).write_text(md + "\n"); pathlib.Path(a.md).with_suffix(".json").write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
