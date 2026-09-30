#!/usr/bin/env python3
"""Package one finished E2E run into its result folder and write the per-video reports + the summary row of the final table.

    python scripts/diag/package_run.py --id 01 --out result_videos/tts_intelligibility_v2/01 --work /workspace/dub/work_v2/work_01 \
        [--old-video first_res/01_e2e_final.mp4 --old-manifest <old>.manifest.json --old-eval <old eval_run.json>]

Expects <out>/<id>_final.mp4 and <out>/<id>_final.manifest.json (written by run_clean_pipeline_05.py). Writes into <out>:
  dubbed_24k.wav, dubbed_16k.wav (copied from the work dir), intelligibility_report.json (eval_run on the AUDIO OF THE FINAL MP4), tts_qa_retry_report.json,
  alignment_report.json, final_quality_report.json / .md, summary_row.json
"""
from __future__ import annotations
import argparse, json, pathlib, shutil, subprocess, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "pilot"))


def vad_iou(orig: list, wav16: pathlib.Path) -> float:
    import soundfile as sf
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad
    y, sr = sf.read(str(wav16), dtype="float32")
    ts = get_speech_timestamps(torch.from_numpy(y), load_silero_vad(), sampling_rate=16000, return_seconds=True)
    dub = [(t["start"], t["end"]) for t in ts]; o = [(v["start"], v["end"]) for v in orig]; n = int(max(max(e for _, e in dub + o), 1) * 100) + 2
    a = np.zeros(n, bool); b = np.zeros(n, bool)
    for s, e in o: a[int(s * 100):int(e * 100)] = True
    for s, e in dub: b[int(s * 100):int(e * 100)] = True
    return round(float((a & b).sum() / max((a | b).sum(), 1)), 3)


def syncnet(video: pathlib.Path) -> dict:
    try:
        from syncnet_common import eval_syncnet
        r = eval_syncnet(video); return {"confidence": r["confidence"], "av_offset_frames": r["av_offset_frames"]}
    except Exception as e:   # the metric must never hide a finished run
        return {"confidence": None, "av_offset_frames": None, "error": f"{type(e).__name__}: {str(e)[:200]}"}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--id", required=True); ap.add_argument("--out", required=True); ap.add_argument("--work", required=True)
    ap.add_argument("--old-video"); ap.add_argument("--old-manifest"); ap.add_argument("--old-eval"); ap.add_argument("--old-label", default="old"); a = ap.parse_args()
    out = pathlib.Path(a.out); video = out / f"{a.id}_final.mp4"; man = json.loads((out / f"{a.id}_final.manifest.json").read_text()); work = pathlib.Path(a.work)
    for f in ("dubbed_24k.wav", "dubbed_16k.wav"): shutil.copy(work / f, out / f)
    subprocess.run([sys.executable, str(HERE / "eval_run.py"), "--manifest", str(out / f"{a.id}_final.manifest.json"), "--audio", str(video), "--out", str(out / "intelligibility_report.json"), "--label", "new"], check=True, stdout=subprocess.DEVNULL)
    new = json.loads((out / "intelligibility_report.json").read_text())["summary"]
    old = json.loads(pathlib.Path(a.old_eval).read_text())["summary"] if a.old_eval else None
    # ---- TTS QA / retry report
    parts = [dict(unit=u["id"], part=k, text=tp["text"], attempts=tp.get("attempts", []), qa_bad_final=tp.get("qa_bad_final"), seed_used=tp.get("seed_used"), temperature_used=tp.get("temperature_used"))
             for u in man["units"] for k, tp in enumerate(u.get("tts_parts", []))]
    fit = [dict(unit=u["id"], **r) for u in man["units"] for r in u.get("fit", [])]
    retried = [p for p in parts if len(p["attempts"]) > 1]
    qa_rep = {"tts": man["tts"].get("stats"), "qa": man["tts"].get("qa"), "speech_token_guard": man["tts"].get("speech_token_guard"), "parts_total": len(parts), "retried_parts": len(retried),
              "retry_success": sum(not p["qa_bad_final"] for p in retried), "final_bad_parts": sum(bool(p["qa_bad_final"]) for p in parts), "retried_detail": retried, "fit_operations": fit,
              "translation_number_retries": [dict(unit=u["id"], **u["translation_number_retry"]) for u in man["units"] if u.get("translation_number_retry")],
              "split": man["translation"].get("burst_split")}
    (out / "tts_qa_retry_report.json").write_text(json.dumps(qa_rep, indent=1, ensure_ascii=False))
    # ---- alignment report
    at = np.array([p.get("atempo", 1.0) for u in man["units"] for p in u["alignment"].get("placements", []) if p.get("placed")] or [1.0])
    al = {"alignment": man["alignment"], "atempo": {"groups": int(len(at)), "max": float(at.max()), "median": float(np.median(at)), "gt_1.15": int((at > 1.1501).sum()), "gt_1.2": int((at > 1.2001).sum()), "gt_1.3": int((at > 1.3001).sum())},
          "lead_groups": [(u["id"], p["lead_s"]) for u in man["units"] for p in u["alignment"].get("placements", []) if p.get("lead_s", 0) > 0],
          "cut_groups": [(u["id"], p["cut_at_slot_end_s"], p.get("text", "")[:40]) for u in man["units"] for p in u["alignment"].get("placements", []) if p.get("cut_at_slot_end_s", 0) > 0.05],
          "vad_iou": vad_iou(man["speech_intervals"], out / "dubbed_16k.wav")}
    (out / "alignment_report.json").write_text(json.dumps(al, indent=1, ensure_ascii=False))
    # ---- final quality report
    sn = syncnet(video); sn_old = syncnet(pathlib.Path(a.old_video)) if a.old_video else None
    cf = man.get("lipsync", {}).get("codeformer"); val = man.get("validation", {})
    fq = {"validation": val, "duration_delta_s": val.get("duration_delta_s"), "output_frame_check": man.get("lipsync", {}).get("output_frame_check"), "lipsync": {k: man.get("lipsync", {}).get(k) for k in ("ls_fraction", "latentsync_seconds_of_video", "frame_classes", "by_action", "latentsync_inference_total_s")},
          "codeformer": cf, "syncnet": sn, "syncnet_old": sn_old, "stage_seconds": man.get("stage_seconds"), "gpu": man.get("gpu", {}).get("name")}
    (out / "final_quality_report.json").write_text(json.dumps(fq, indent=1, ensure_ascii=False))
    cf_s = cf.get("seconds") if isinstance(cf, dict) else None
    row = {"video": a.id, "wer_old": old and old["wer_whisper_mean"], "wer_new": new["wer_whisper_mean"], "cer_old": old and old["cer_ctc_mean"], "cer_new": new["cer_ctc_mean"],
           "bad_units_old": old and old["units_bad(cer_ctc>=0.3)"], "bad_units_new": new["units_bad(cer_ctc>=0.3)"], "halluc_units_old": old and old["interjection_units"], "halluc_units_new": new["interjection_units"],
           "missing_final_old": old and old["final_word_missing_units"], "missing_final_new": new["final_word_missing_units"], "tts_parts_old": old and old["tts_parts"], "tts_parts_new": qa_rep["parts_total"],
           "retried": qa_rep["retried_parts"], "retry_success": qa_rep["retry_success"], "retry_success_pct": round(100 * qa_rep["retry_success"] / qa_rep["retried_parts"], 1) if qa_rep["retried_parts"] else None,
           "final_bad_parts": qa_rep["final_bad_parts"], "max_atempo_old": old and old["atempo_max"], "max_atempo_new": al["atempo"]["max"], "parts_gt_1.15_old": old and old["atempo_gt_1.15"], "parts_gt_1.15_new": al["atempo"]["gt_1.15"],
           "parts_gt_1.2_new": al["atempo"]["gt_1.2"], "cut_groups": len(al["cut_groups"]), "av_offset_frames": sn["av_offset_frames"], "syncnet": sn["confidence"], "syncnet_old": sn_old and sn_old["confidence"],
           "av_offset_old": sn_old and sn_old["av_offset_frames"], "codeformer_s": cf_s, "vad_iou": al["vad_iou"], "validation": val.get("pass"), "duration_delta_s": val.get("duration_delta_s")}
    (out / "summary_row.json").write_text(json.dumps(row, indent=1, ensure_ascii=False))
    (out / "final_quality_report.md").write_text("# " + a.id + " final quality\n\n" + "\n".join(f"- **{k}**: {v}" for k, v in row.items()) + "\n")
    print(json.dumps(row, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
