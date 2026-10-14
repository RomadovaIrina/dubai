#!/usr/bin/env python3
"""One table for a set of finished E2E runs (run_clean_pipeline_05.py outputs under <run-root>/final, work dirs under <run-root>/work):
wall + per-stage timings (audio stages split into Qwen segmentation / translation / TTS generation / ASR QA / CTC alignment / rest = Qwen fit
rewrites + judge + I/O), LatentSync / CodeFormer, SyncNet confidence + AV offset (output and original), speech coverage (compare_audio_candidates.side),
dub-only intelligibility (eval_run.evaluate: Whisper WER, CTC CER, bad units, interjections, final word missing), TTS QA / fit stats, atempo
distribution, edge cuts (edge_cuts.check) and the background bed when present.

    python scripts/quality/e2e_report.py --run-root /workspace/dub/runs/baseline --ids 01 02 03 04 05 --label baseline [--syncnet]
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
for d in ("pilot", "diag", "quality"):
    sys.path.insert(0, str(HERE.parent / d))
SYNC_CACHE = pathlib.Path("/workspace/dub/runs/syncnet_original_cache.json")


def one(root: pathlib.Path, vid: str, qa, syncnet: bool) -> dict:
    import eval_run
    from compare_audio_candidates import side
    from edge_cuts import check
    mp = root / "final" / f"{vid}_final.manifest.json"; man = json.loads(mp.read_text()); st = man["stage_seconds"]; work = pathlib.Path(man["work_dir"])
    ev = eval_run.evaluate(man, man["dubbed_track"]["wav24k"], vid, qa=qa)
    (root / f"{vid}_eval.json").write_text(json.dumps(ev, indent=1, ensure_ascii=False))
    ls_segs = (man.get("lipsync") or {}).get("segments") or []
    cov = side(man, pathlib.Path(man["dubbed_track"]["wav16k"]), ev, work / "audio16k.wav", ls_segs)
    edges = check(man)["summary"]; tts = man["tts"]; ts = tts.get("stats", {})
    at = np.array([p.get("atempo", 1.0) for u in man["units"] for p in u["alignment"].get("placements", []) if p.get("placed")] or [1.0])
    gen, qas, ali = ts.get("gen_seconds", 0.0), ts.get("qa_seconds", 0.0), ts.get("align_seconds", 0.0)
    r = {"id": vid, "valid": man.get("validation", {}).get("pass"), "wall_total_s": st.get("total"),
         "audio": {"asr": st.get("asr"), "diarization": st.get("diarization"), "segmentation_qwen": st.get("segmentation"), "translation_qwen": st.get("translation"),
                   "tts_stage": st.get("tts"), "tts_gen": gen, "tts_qa_asr": qas, "ctc_align": ali, "tts_rest_qwen_fit_io": round((st.get("tts") or 0) - gen - qas - ali, 2),
                   "fitup_qwen_s": ts.get("fitup_qwen_seconds"), "fitdown_qwen_s": ts.get("fitdown_qwen_seconds"), "judge_calls": ts.get("judge_calls"), "judge_s": ts.get("judge_seconds"),
                   "background_sep": st.get("background_sep"), "background_mix": st.get("background_mix"),
                   "audio_total": round(sum(st.get(k) or 0 for k in ("audio_extract", "vad", "asr", "translation_load", "segmentation", "diarization", "translation", "tts_load", "qa_load", "tts",
                                                                       "alignment", "dubbed_track", "background_sep", "background_mix")), 2)},
         "video": {"latentsync": st.get("latentsync"), "codeformer": st.get("codeformer"), "face_classify": st.get("face_classify"), "final_assembly": st.get("final_assembly"),
                   "ls_frames": (man.get("lipsync") or {}).get("by_action", {}).get("LATENT_SYNC", {}).get("frames")},
         "coverage": {k: cov.get(k) for k in ("orig_speech_s", "dub_speech_s", "orig_without_dub_s", "dub_outside_orig_s", "iou", "bursts", "bursts_covered_lt_50pct", "bursts_uncovered",
                                              "silent_while_speaking_frames_vad", "cut_groups", "cut_max_s", "lost_numbers", "repeated_words")} | {"ls_frames_dub_silent_while_orig": (cov.get("rms_metric") or {}).get("ls_frames_dub_silent_while_original_speaks")},
         "intelligibility": ev["summary"], "tts_stats": {k: ts.get(k) for k in ("parts", "attempts", "retried_parts", "retry_success", "final_bad", "fit_up_candidates", "fit_up_accepted",
                                                                                 "fit_up_rejected", "fit_parts_rewritten", "spans", "span_merges", "syntax_burst_merges")},
         "atempo": {"max": round(float(at.max()), 3), "mean": round(float(at.mean()), 3), "lt_1": int((at < 0.995).sum()), "gt_1_15": int((at > 1.151).sum()), "gt_1_20": int((at > 1.201).sum()), "groups": len(at)},
         "edges": edges, "background": man.get("background", {"enabled": False}).get("mix") if man.get("background", {}).get("enabled") else None}
    if syncnet:
        from syncnet_common import eval_syncnet
        cache = json.loads(SYNC_CACHE.read_text()) if SYNC_CACHE.exists() else {}
        if man.get("lipsync", {}).get("by_action", {}).get("LATENT_SYNC", {}).get("frames", 0):
            try:
                o = eval_syncnet(pathlib.Path(man["output"])); r["syncnet"] = {"confidence": o["confidence"], "av_offset": o["av_offset_frames"]}
            except Exception as e:
                r["syncnet"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
            if man.get("background", {}).get("enabled"):   # the bed is in the muxed track: also score the same frames with the dub voice only
                import subprocess, tempfile
                t = pathlib.Path(tempfile.mkdtemp()) / f"{vid}_dubonly.mp4"
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", man["output"], "-i", man["dubbed_track"]["aac"], "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", str(t)], check=True)
                try:
                    o = eval_syncnet(t); r["syncnet_dub_only"] = {"confidence": o["confidence"], "av_offset": o["av_offset_frames"]}
                except Exception as e:
                    r["syncnet_dub_only"] = {"error": f"{type(e).__name__}: {str(e)[:120]}"}
                finally:   # the temp mp4 used to stay in /tmp (with eval_run's temp wavs it filled the 50 GB disk, 2026-10-14)
                    t.unlink(missing_ok=True); t.parent.rmdir()
            if vid not in cache:
                try:
                    o = eval_syncnet(pathlib.Path(man["input"])); cache[vid] = {"confidence": o["confidence"], "av_offset": o["av_offset_frames"]}; SYNC_CACHE.write_text(json.dumps(cache, indent=1))
                except Exception as e:
                    cache[vid] = {"error": str(e)[:120]}
            r["syncnet_original"] = cache.get(vid)
        else:
            r["syncnet"] = "N/A (no face)"
    return r


def md(rows: list, label: str) -> str:
    def g(r, path, d="-"):
        x = r
        for k in path.split("."):
            x = x.get(k) if isinstance(x, dict) else None
            if x is None:
                return d
        return f"{x:.3f}" if isinstance(x, float) and abs(x) < 10 else (f"{x:.1f}" if isinstance(x, float) else str(x))
    cols = [("valid", "valid"), ("wall s", "wall_total_s"), ("audio s", "audio.audio_total"), ("segm Qwen", "audio.segmentation_qwen"), ("transl Qwen", "audio.translation_qwen"),
            ("TTS stage", "audio.tts_stage"), ("TTS gen", "audio.tts_gen"), ("QA ASR", "audio.tts_qa_asr"), ("CTC align", "audio.ctc_align"), ("fit Qwen+IO", "audio.tts_rest_qwen_fit_io"),
            ("LatentSync", "video.latentsync"), ("CodeFormer", "video.codeformer"), ("SyncNet", "syncnet.confidence"), ("AV off", "syncnet.av_offset"), ("SyncNet dub-only", "syncnet_dub_only.confidence"), ("orig SyncNet", "syncnet_original.confidence"),
            ("orig w/o dub s", "coverage.orig_without_dub_s"), ("IoU", "coverage.iou"), ("silent LS frames", "coverage.ls_frames_dub_silent_while_orig"),
            ("WER", "intelligibility.wer_whisper_mean"), ("CTC CER", "intelligibility.cer_ctc_mean"), ("bad units", "intelligibility.units_bad(cer_ctc>=0.3)"),
            ("interj", "intelligibility.interjection_units"), ("final miss", "intelligibility.final_word_missing_units"), ("QA retried/ok/bad", None), ("fit-up acc/cand", None),
            ("atempo max/mean", None), (">1.15", "atempo.gt_1_15"), ("cut groups", "coverage.cut_groups"), ("hot inner tails/heads", None)]
    L = [f"## {label}", "", "| metric | " + " | ".join(r["id"] for r in rows) + " |", "|---|" + "---:|" * len(rows)]
    for name, path in cols:
        vals = []
        for r in rows:
            if name == "QA retried/ok/bad": vals.append(f"{g(r, 'tts_stats.retried_parts')}/{g(r, 'tts_stats.retry_success')}/{g(r, 'tts_stats.final_bad')}")
            elif name == "fit-up acc/cand": vals.append(f"{g(r, 'tts_stats.fit_up_accepted')}/{g(r, 'tts_stats.fit_up_candidates')}")
            elif name == "atempo max/mean": vals.append(f"{g(r, 'atempo.max')}/{g(r, 'atempo.mean')}")
            elif name == "hot inner tails/heads": vals.append(f"{g(r, 'edges.inner_tails_hot')}/{g(r, 'edges.inner_heads_hot')} of {g(r, 'edges.inner_cuts')}")
            elif name == "SyncNet" and isinstance(r.get("syncnet"), str): vals.append("N/A")
            elif name == "bad units": vals.append(str(r["intelligibility"].get("units_bad(cer_ctc>=0.3)", "-")))
            else: vals.append(g(r, path))
        L.append(f"| {name} | " + " | ".join(vals) + " |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-root", required=True); ap.add_argument("--ids", nargs="+", required=True); ap.add_argument("--label", required=True); ap.add_argument("--syncnet", action="store_true")
    a = ap.parse_args()
    import tts_qa
    root = pathlib.Path(a.run_root); qa = tts_qa.TtsQA(); rows = []
    for vid in a.ids:
        rows.append(one(root, vid, qa, a.syncnet)); print(json.dumps({k: rows[-1].get(k) for k in ("id", "wall_total_s", "syncnet")}), flush=True)
    qa.close()
    (root / f"report_{a.label}.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False, default=str))
    (root / f"report_{a.label}.md").write_text(md(rows, a.label)); print(md(rows, a.label)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
