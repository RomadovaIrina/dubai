#!/usr/bin/env python3
"""Pipeline refresh (2026-10-02) — informational facts from the five FINAL runner manifests, merged into the 0.8 refresh evidence,
plus one result page per test video (reports/pilot/refresh_videos/<id>.md).

Reads <final-dir>/<id>_final.manifest.json (+ <id>_final.nvsmi.log written by refresh_e2e.sh) and the benchmark_08 JSON
(SyncNet rows) and records per video: TTS QA retries, atempo max / distribution, LatentSync seconds, CodeFormer seconds, peak VRAM,
validation, stage wall times. The SyncNet methodology / numbers are NOT touched: this only appends a section.

    python scripts/pilot/refresh_08_collect.py --final-dir /tmp/pilot_refresh/final \
        --json reports/pilot/refresh_0.8_current_e2e.json --md reports/pilot/refresh_0.8_current_e2e.md --per-video-dir reports/pilot/refresh_videos
"""
from __future__ import annotations
import argparse, json, pathlib, statistics, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT, ROOT, write_json

BINS = [(0.0, 0.995, "<1 (slowed)"), (0.995, 1.005, "1.0"), (1.005, 1.05, "(1.0,1.05]"), (1.05, 1.10, "(1.05,1.10]"), (1.10, 1.15, "(1.10,1.15]"), (1.15, 1.20, "(1.15,1.20]"), (1.20, 9.0, ">1.20")]


def nvsmi_peak(p: pathlib.Path) -> int | None:
    if not p.exists():
        return None
    vals = []
    for line in p.read_text().splitlines():
        try:
            vals.append(int(line.split(",")[-1].strip()))
        except Exception:
            pass
    return max(vals) if vals else None


def facts(man: dict, nvsmi: pathlib.Path) -> dict:
    st = man.get("stage_seconds") or {}; ls = man.get("lipsync") or {}; cf = ls.get("codeformer"); cfd = cf if isinstance(cf, dict) else {}
    tts = man.get("tts") or {}; ts = tts.get("stats") or {}; al = man.get("alignment") or {}; summ = al.get("summary") or {}
    units = man.get("units") or []
    ats = [p.get("atempo", 1.0) for u in units for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed")]
    hist = {label: sum(1 for x in ats if lo < x <= hi) for lo, hi, label in BINS}
    parts = [tp for u in units for tp in u.get("tts_parts", [])]
    retried = [tp for tp in parts if len(tp.get("attempts", [])) > 1]
    bad_final = [tp for tp in parts if tp.get("qa_bad_final")]
    segs = ls.get("segments") or []
    seg_cf = [s["codeformer"] for s in segs if isinstance(s.get("codeformer"), dict)]
    dur = float(man["source"]["duration_s"]); ls_s = float(st.get("latentsync") or 0.0); cf_s = float(st.get("codeformer") or 0.0)
    val = man.get("validation") or {}
    return {"source_duration_s": dur, "source_lang": man.get("source_lang"), "units": len(units), "speakers": (man.get("diarization") or {}).get("speakers"),
            "translation": {"quant": (man.get("translation") or {}).get("quant"), "tokens": (man.get("translation") or {}).get("usage"), "burst_split": (man.get("translation") or {}).get("burst_split"),
                            "number_retries": sum(1 for u in units if u.get("translation_number_retry")), "fitted_units": sum(1 for u in units if u.get("translation_fitted"))},
            "tts": {"dtype": tts.get("dtype"), "guard_violations": (tts.get("speech_token_guard") or {}).get("violations"), "parts": ts.get("parts", len(parts)), "attempts": ts.get("attempts"),
                    "retried_parts": ts.get("retried_parts", len(retried)), "retry_success": ts.get("retry_success"), "retry_success_rate": ts.get("retry_success_rate"), "final_bad_parts": ts.get("final_bad", len(bad_final)),
                    "fit_parts_rewritten": ts.get("fit_parts_rewritten"), "fit_parts_merged": ts.get("fit_parts_merged"), "fit_rewrite_rejected": ts.get("fit_rewrite_rejected"), "fit_rounds_used": ts.get("fit_rounds_used"),
                    "gen_seconds": ts.get("gen_seconds"), "qa_seconds": ts.get("qa_seconds"), "qa": tts.get("qa")},
            "alignment": {"mode": man.get("alignment_mode"), "atempo_cap": al.get("atempo_cap"), "hard_cap": al.get("hard_cap"), "units_slot_fallback": al.get("units_slot_fallback"),
                          "groups": summ.get("groups", len(ats)), "atempo_max": summ.get("atempo_max", max(ats) if ats else None), "atempo_min": summ.get("atempo_min", min(ats) if ats else None),
                          "atempo_mean": round(statistics.mean(ats), 4) if ats else None, "atempo_median": round(statistics.median(ats), 4) if ats else None,
                          "gt_1_15": summ.get("atempo_gt_1_15"), "gt_1_25": summ.get("atempo_gt_1_25"), "gt_1_3": summ.get("atempo_gt_1_3"), "slowed_lt_1": summ.get("slowed_lt_1"),
                          "overflow_groups": summ.get("overflow_groups"), "cut_groups": summ.get("cut_groups"), "atempo_histogram": hist},
            "video": {"backend": man.get("video_backend"), "frame_classes": ls.get("frame_classes"), "by_action": ls.get("by_action"), "master_frames": ls.get("master_frames"),
                      "latentsync_seconds_of_video": ls.get("latentsync_seconds_of_video"), "ls_fraction": ls.get("ls_fraction"), "latentsync_segments": (ls.get("by_action") or {}).get("LATENT_SYNC", {}).get("segments", 0),
                      "ls_failed_segments": (ls.get("by_action") or {}).get("PASS_THROUGH_LS_FAILED", {}).get("segments", 0), "verify_splits": len(ls.get("verify_splits") or []),
                      "latentsync_stage_s": ls_s, "latentsync_verify_s": st.get("latentsync_verify"), "latentsync_s_per_video_min": round(ls_s / dur * 60, 2),
                      "codeformer": man.get("codeformer"), "codeformer_stage_s": cf_s, "codeformer_faces": cfd.get("totals", {}).get("faces_restored"), "codeformer_eligible_frames": cfd.get("eligible_frames"),
                      "codeformer_fallback_detections": cfd.get("totals", {}).get("fallback_detections"), "codeformer_s_per_video_min": round(cf_s / dur * 60, 2),
                      "codeformer_torch_peak_mib": max([c.get("peak_vram_mib") or 0 for c in seg_cf] or [0]) or None, "speech_gate": {k: v for k, v in (man.get("speech_gate") or {}).items() if k != "intervals"}},
            "timing": {"total_wall_s": st.get("total"), "stage_seconds": st, "wall_s_per_video_min": round(float(st.get("total") or 0) / dur * 60, 2)},
            "peak_vram_nvsmi_mib": nvsmi_peak(nvsmi),
            "validation": {"pass": val.get("pass"), "checks": val.get("checks"), "duration_delta_s": val.get("duration_delta_s"), "tolerance_s": val.get("tolerance_s"),
                           "frames_written": (ls.get("assembly") or {}).get("frames_written"), "output_frame_check": ls.get("output_frame_check"), "output_meta": {k: (man.get("output_meta") or {}).get(k) for k in ("duration_s", "size_bytes")},
                           "output_video": (man.get("output_meta") or {}).get("video")}, "gpu": {k: (man.get("gpu") or {}).get(k) for k in ("name", "capability")}, "at": man.get("at")}


def unit_rows(man: dict) -> list[dict]:
    rows = []
    for u in man.get("units") or []:
        al = u.get("alignment") or {}; pls = al.get("placements") or []; parts = u.get("tts_parts") or []
        rows.append({"id": u["id"], "speaker": u.get("speaker"), "start": u["start"], "end": u["end"], "text": u.get("text"), "translation": u.get("translation"), "translation_fitted": u.get("translation_fitted"),
                     "parts": len(parts), "tts_s": (u.get("tts") or {}).get("seconds"), "slot_s": (u.get("slot") or {}).get("seconds"), "ratio": al.get("ratio_tts_to_slot"), "atempo_max": al.get("atempo"),
                     "atempos": [p.get("atempo") for p in pls if p.get("placed")], "retried": sum(len(tp.get("attempts", [])) > 1 for tp in parts), "bad_final": sum(bool(tp.get("qa_bad_final")) for tp in parts),
                     "qa_scores": [tp.get("qa_score") for tp in parts], "split": (u.get("split") or {}).get("method"), "fit": [f.get("action") for f in u.get("fit", [])],
                     "cut_s": round(sum(p.get("cut_at_slot_end_s", 0) or 0 for p in pls), 3), "overflow_s": round(sum(p.get("overflow_into_next_burst_s", 0) or 0 for p in pls), 3)})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final-dir", default="/tmp/pilot_refresh/final"); ap.add_argument("--json", default=str(PILOT / "refresh_0.8_current_e2e.json"))
    ap.add_argument("--md", default=str(PILOT / "refresh_0.8_current_e2e.md")); ap.add_argument("--per-video-dir", default=str(PILOT / "refresh_videos"))
    ap.add_argument("--coverage-dir", default=None, help="optional: <id>.json files written by scripts/quality/dub_coverage.py (dubbed-speech coverage of the original speech)")
    a = ap.parse_args(); fd = pathlib.Path(a.final_dir); pv = pathlib.Path(a.per_video_dir); pv.mkdir(parents=True, exist_ok=True)
    bench = json.loads(pathlib.Path(a.json).read_text()) if pathlib.Path(a.json).exists() else {}
    sn = {r["key"]: r for r in bench.get("rows", [])}
    info = {}; md_rows = []
    for mp in sorted(fd.glob("*_final.manifest.json")):
        vid = mp.name.split("_final")[0]; man = json.loads(mp.read_text()); f = facts(man, fd / f"{vid}_final.nvsmi.log"); f["syncnet"] = {k: sn.get(vid, {}).get(k) for k in ("status", "confidence", "av_offset_frames", "original_confidence", "original_av_offset_frames", "confidence_delta", "reason")}
        cov = None
        if a.coverage_dir and (pathlib.Path(a.coverage_dir) / f"{vid}.json").exists():
            c = json.loads((pathlib.Path(a.coverage_dir) / f"{vid}.json").read_text())
            cov = {k: c.get(k) for k in ("orig_speech_s", "dub_speech_s", "overlap_s", "iou", "orig_without_dub_s", "dub_outside_orig_s", "orig_bursts", "bursts_covered_lt_50pct", "bursts_uncovered")}
            cov["per_burst"] = c.get("per_burst")
        f["coverage"] = cov
        info[vid] = f
        T, A, V, X = f["tts"], f["alignment"], f["video"], f["timing"]
        md_rows.append(f"| {vid} | {f['units']} | {T['parts']} | {T['attempts']} | {T['retried_parts']} ({T['retry_success']} ok) | {T['final_bad_parts']} | {T['fit_parts_rewritten']} / {T['fit_parts_merged']} | "
                       f"{A['atempo_max']} | {A['gt_1_15']} / {A['groups']} | {A['atempo_histogram']['1.0']} / {A['atempo_histogram']['(1.0,1.05]']} / {A['atempo_histogram']['(1.05,1.10]']} / {A['atempo_histogram']['(1.10,1.15]']} / {A['atempo_histogram']['(1.15,1.20]']} / {A['atempo_histogram']['>1.20']} | "
                       f"{A['units_slot_fallback']} | {A['cut_groups']} | {V['latentsync_stage_s']:.1f} ({V['latentsync_s_per_video_min']}) | {V['latentsync_seconds_of_video']} | {V['codeformer_stage_s']:.1f} ({V['codeformer_s_per_video_min']}) | {V['codeformer_faces']} | "
                       f"{f['peak_vram_nvsmi_mib']} | {X['total_wall_s']} | {'PASS' if f['validation']['pass'] else 'FAIL'} |")
        # ---- per-video result page
        s = man["source"]; v = f["validation"]; ov = v.get("output_video") or {}; snr = f["syncnet"]
        page = [f"# Test video {vid} — current pipeline result ({time.strftime('%Y-%m-%d')})", "",
                f"Input `{man['input']}`: {s['duration_s']:.3f} s, {s['video'].get('width')}x{s['video'].get('height')} @ {s['video'].get('avg_fps')} fps, language {f['source_lang']} -> {man.get('target_lang')}. "
                f"Output `{man['output']}` (+ manifest). Run at {f['at']} on {f['gpu'].get('name')}.", "",
                f"**Validation: {'PASS' if v['pass'] else 'FAIL'}** — duration delta {v['duration_delta_s']} s (tol {v['tolerance_s']}), frames written {v['frames_written']} / master {V['master_frames']}, "
                f"output {ov.get('width')}x{ov.get('height')} {ov.get('nb_frames')} frames, decodable/no-blank {(v.get('output_frame_check') or {}).get('pass')}; checks: {v['checks']}", "",
                f"**SyncNet (0.8):** {snr.get('status')} — confidence {snr.get('confidence')} (original {snr.get('original_confidence')}, Δ {snr.get('confidence_delta')}), AV offset {snr.get('av_offset_frames')} (original {snr.get('original_av_offset_frames')}){(' — ' + snr['reason']) if snr.get('reason') else ''}", "",
                "## Audio", "",
                f"- ASR units {f['units']}, speakers {f['speakers']}; translation {f['translation']['quant']} ({f['translation']['tokens']}), burst split {f['translation']['burst_split']}, number retries {f['translation']['number_retries']}, units re-translated for duration {f['translation']['fitted_units']}",
                f"- TTS {T['dtype']} (token guard violations {T['guard_violations']}): parts {T['parts']}, attempts {T['attempts']}, retried parts {T['retried_parts']} (success {T['retry_success']}, rate {T['retry_success_rate']}), still bad after retries {T['final_bad_parts']}; fit: rewritten {T['fit_parts_rewritten']}, merged {T['fit_parts_merged']}, rewrite rejected {T['fit_rewrite_rejected']}; gen {T['gen_seconds']} s, QA {T['qa_seconds']} s",
                f"- Alignment `{A['mode']}` (cap {A['atempo_cap']} / hard {A['hard_cap']}): groups {A['groups']}, atempo max {A['atempo_max']} mean {A['atempo_mean']} median {A['atempo_median']}, >1.15: {A['gt_1_15']}, >1.2: {A['atempo_histogram']['>1.20']}, slowed {A['slowed_lt_1']}, overflow {A['overflow_groups']}, cut {A['cut_groups']}, slot-fallback units {A['units_slot_fallback']}; histogram {A['atempo_histogram']}",
                (f"- Coverage (Silero VAD, dub_coverage.py): original speech {cov['orig_speech_s']} s, dubbed speech {cov['dub_speech_s']} s, overlap {cov['overlap_s']} s, IoU {cov['iou']}, original speech without dub {cov['orig_without_dub_s']} s, dub outside original speech {cov['dub_outside_orig_s']} s; "
                 f"bursts {cov['orig_bursts']}, < 50 % covered {cov['bursts_covered_lt_50pct']}, uncovered {cov['bursts_uncovered']}") if cov else "- Coverage: n/a", "",
                "## Video", "",
                f"- backend {V['backend']}, speech gate {V['speech_gate']}", f"- frame classes {V['frame_classes']}; actions {V['by_action']}; LatentSync on {V['latentsync_seconds_of_video']} s of video ({V['ls_fraction']}), segments {V['latentsync_segments']}, LS-failed {V['ls_failed_segments']}, verify splits {V['verify_splits']}",
                f"- LatentSync stage {V['latentsync_stage_s']:.2f} s ({V['latentsync_s_per_video_min']} s/video-min, verify {V['latentsync_verify_s']} s); CodeFormer {V['codeformer']}: {V['codeformer_stage_s']:.2f} s ({V['codeformer_s_per_video_min']} s/video-min), faces {V['codeformer_faces']} / eligible {V['codeformer_eligible_frames']}, fallback det {V['codeformer_fallback_detections']}, torch peak {V['codeformer_torch_peak_mib']} MiB",
                f"- peak VRAM (nvidia-smi, whole run) {f['peak_vram_nvsmi_mib']} MiB; total wall {X['total_wall_s']} s ({X['wall_s_per_video_min']} s/video-min); stages: {X['stage_seconds']}", "",
                "## Units (source -> dubbed text, placement)", "", "| id | spk | slot s | source | translation (fitted) | parts | tts s | atempo max | atempos | retried | bad | QA scores | cut s |", "|---:|---|---|---|---|---:|---:|---:|---|---:|---:|---|---:|"]
        for r in unit_rows(man):
            tr = r["translation_fitted"] or r["translation"] or ""; tr = tr + (" *(fitted)*" if r["translation_fitted"] else "")
            page.append(f"| {r['id']} | {r['speaker']} | {r['start']:.2f}-{r['end']:.2f} | {(r['text'] or '').replace('|', '/')} | {tr.replace('|', '/')} | {r['parts']} | {r['tts_s']} | {r['atempo_max']} | {' '.join(f'{x:.2f}' for x in r['atempos'] if x is not None)} | {r['retried']} | {r['bad_final']} | {' '.join(str(x) for x in r['qa_scores'])} | {r['cut_s']} |")
        (pv / f"{vid}.md").write_text("\n".join(page) + "\n"); write_json(pv / f"{vid}.json", {"facts": f, "units": unit_rows(man)})
    # ---- merge into the 0.8 evidence
    if bench:
        bench["informational_from_manifests"] = info; write_json(pathlib.Path(a.json), bench)
    section = ["", "## Informational (from the runner manifests; not part of the SyncNet methodology)", "",
               "| video | units | TTS parts | attempts | retried (ok) | bad final | fit rewritten / merged | atempo max | >1.15 / groups | atempo hist 1.0 / ≤1.05 / ≤1.10 / ≤1.15 / ≤1.20 / >1.20 | slot-fallback units | cut groups | LatentSync s (s/video-min) | LS video s | CodeFormer s (s/video-min) | CF faces | peak VRAM MiB (nvsmi) | total wall s | valid |",
               "|---|---:|---:|---:|---|---:|---|---:|---|---|---:|---:|---|---:|---|---:|---:|---:|---|"] + md_rows + ["", f"Per-video result pages: `{pv.relative_to(ROOT) if str(pv).startswith(str(ROOT)) else pv}/<id>.md`."]
    mdp = pathlib.Path(a.md)
    if mdp.exists():
        txt = mdp.read_text()
        if "## Informational (from the runner manifests" in txt:
            txt = txt.split("## Informational (from the runner manifests")[0].rstrip() + "\n"
        mdp.write_text(txt.rstrip() + "\n" + "\n".join(section) + "\n")
    print("\n".join(section)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
