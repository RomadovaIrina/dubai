#!/usr/bin/env python3
"""Quality manifest for one run_clean_pipeline_05.py output (Phase 1 diagnosis; nothing in the pipeline is touched).

Reads <run-dir>/<id>_<tag>.manifest.json (+ .mp4, .nvsmi.log, work_<id>/ media) and writes
  <out>/<id>_<tag>.quality.json   source/output/routing/latentsync/audio(+silence-driven LS frames)/syncnet
  <out>/<id>_<tag>.quality.md     human summary
  <frames>/<id>_<tag>/...png      contact sheets: inside LS segments, segment start/end, -8..+8 around every LS<->pass-through
                                  transition (row 1 = master/original frame, row 2 = output), SMALL_FACE samples
    python scripts/quality/quality_manifest.py --run-dir /tmp/dabai_quality/baseline --tag baseline --ids 04 03 [--syncnet]
"""
from __future__ import annotations
import argparse, csv, json, math, pathlib, subprocess, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot"))
FPS, SR = 25, 16000
Q = pathlib.Path("/tmp/dabai_quality")


def probe(p: pathlib.Path) -> dict:
    j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-count_frames", "-of", "json", str(p)],
                                  check=True, capture_output=True, text=True).stdout)
    v = next((s for s in j["streams"] if s["codec_type"] == "video"), {}); a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    num, den = (v.get("avg_frame_rate") or "0/1").split("/")
    return {"duration_s": round(float(j["format"]["duration"]), 3), "width": v.get("width"), "height": v.get("height"),
            "fps": round(int(num) / max(int(den), 1), 4), "frames": int(v.get("nb_read_frames") or 0), "audio": a is not None,
            "audio_sr": a and a.get("sample_rate"), "audio_ch": a and a.get("channels")}


def read_wav(p: pathlib.Path) -> np.ndarray:
    import soundfile as sf
    y, sr = sf.read(str(p), dtype="float32"); assert sr == SR, sr
    return y if y.ndim == 1 else y.mean(axis=1)


def frame_rms(y: np.ndarray, n_frames: int) -> np.ndarray:
    hop = SR // FPS; out = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = y[i * hop:(i + 1) * hop]
        out[i] = float(np.sqrt(np.mean(seg ** 2))) if len(seg) else 0.0
    return out


def grab_frames(video: pathlib.Path, wanted: set[int]) -> dict[int, np.ndarray]:
    import cv2
    cap = cv2.VideoCapture(str(video)); got = {}; i = 0; mx = max(wanted) if wanted else -1
    while i <= mx:
        ok, fr = cap.read()
        if not ok: break
        if i in wanted: got[i] = fr
        i += 1
    cap.release(); return got


def sheet(frames: list[tuple[str, np.ndarray | None]], path: pathlib.Path, cols: int, scale: float = 0.5, title: str = ""):
    import cv2
    tiles = []
    for lab, fr in frames:
        if fr is None: continue
        t = cv2.resize(fr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.rectangle(t, (0, 0), (t.shape[1], 22), (0, 0, 0), -1)
        cv2.putText(t, lab, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA); tiles.append(t)
    if not tiles: return None
    h, w = tiles[0].shape[:2]; rows = math.ceil(len(tiles) / cols)
    canvas = np.zeros((rows * h + 26, cols * w, 3), dtype=np.uint8)
    cv2.putText(canvas, title, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    for k, t in enumerate(tiles):
        r, c = divmod(k, cols); canvas[26 + r * h:26 + (r + 1) * h, c * w:(c + 1) * w] = t
    path.parent.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(path), canvas); return str(path)


def mouth_crop(fr: np.ndarray, bbox: dict | None) -> np.ndarray:
    """Lower-face crop (bbox from the router) so mouth detail is visible in contact sheets; whole frame if no bbox."""
    if not bbox: return fr
    x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]; w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) // 2, y1 + int(h * 0.72); s = int(max(w, h) * 0.9)
    H, W = fr.shape[:2]; a = max(0, cx - s // 2); b = max(0, cy - s // 2)
    return fr[b:min(H, b + s), a:min(W, a + s)]


def transitions(segs: list[dict]) -> list[dict]:
    out = []
    for a, b in zip(segs, segs[1:]):
        la, lb = a["action"] == "LATENT_SYNC", b["action"] == "LATENT_SYNC"
        if la != lb:
            out.append({"frame": b["start_frame"], "t": round(b["start_frame"] / FPS, 3), "from": a["action"] + "/" + a["classification"],
                        "to": b["action"] + "/" + b["classification"], "left_frames": a["frames"], "right_frames": b["frames"]})
    return out


def analyse(run_dir: pathlib.Path, work_root: pathlib.Path, vid: str, tag: str, out_dir: pathlib.Path, frames_dir: pathlib.Path,
            syncnet: bool, syncnet_cache: dict) -> dict:
    mp = run_dir / f"{vid}_{tag}.manifest.json"; man = json.loads(mp.read_text())
    out_mp4 = run_dir / f"{vid}_{tag}.mp4"; work = work_root / f"work_{vid}"; src = pathlib.Path(man["input"])
    ls = man.get("lipsync") or {}; segs = sorted(ls.get("segments", []), key=lambda s: s["start_frame"])
    q: dict = {"video": vid, "tag": tag, "manifest": str(mp), "output": str(out_mp4), "config": {**ls.get("runner", {}), "router": ls.get("router_params"),
               "router_stride": ls.get("router_stride"), "speech_gate": man.get("speech_gate", {}), "crossfade_frames": ls.get("crossfade_frames")}}
    if "error" in man or "blocker" in man:
        q["run_error"] = man.get("error") or man.get("blocker"); return q
    # 1/2 source + output
    q["source"] = {**probe(src), "vfr": man["source"]["video"].get("vfr")}
    q["output"] = {**probe(out_mp4), "frame_check": ls.get("output_frame_check"), "validation": man["validation"]["checks"],
                   "duration_delta_s": man["validation"]["duration_delta_s"]}
    # 3 routing
    ls_segs = [s for s in segs if s["action"] == "LATENT_SYNC"]
    trs = transitions(segs)
    q["routing"] = {"frame_classes": ls.get("frame_classes"), "skip_reasons": ls.get("skip_reasons"), "by_action": ls.get("by_action"),
                    "router_detections": ls.get("router_detections"), "master_frames": ls.get("master_frames"),
                    "latentsync_segments": len(ls_segs), "segment_durations_s": [round(s["frames"] / FPS, 2) for s in ls_segs],
                    "short_ls_segments_lt2s": sum(1 for s in ls_segs if s["frames"] < 50),
                    "transitions": trs, "n_transitions": len(trs),
                    "verify_splits": ls.get("verify_splits"), "ls_failed_segments": [s for s in segs if s["action"] == "PASS_THROUGH_LS_FAILED"],
                    "small_face_segments": [{k: s.get(k) for k in ("index", "start_s", "end_s", "frames")} for s in segs if s.get("classification") == "SMALL_FACE"],
                    "no_face_segments": [{k: s.get(k) for k in ("index", "start_s", "end_s", "frames")} for s in segs if s.get("classification") == "NO_FACE"]}
    # 4 latentsync
    nv = run_dir / f"{vid}_{tag}.nvsmi.log"; peak = None
    if nv.exists():
        vals = [int(x) for x in nv.read_text().split() if x.strip().isdigit()]; peak = max(vals) if vals else None
    calls = ls.get("runner_calls") or []
    q["latentsync"] = {"inference_total_s": ls.get("latentsync_inference_total_s"), "stage_latentsync_s": man["stage_seconds"].get("latentsync"),
                       "verify_s": ls.get("verify_seconds"), "calls": len(calls), "unet_calls": sum(c.get("unet_calls", 0) for c in calls),
                       "unet_denoise_s": round(sum(c["stages"].get("unet_denoise", 0) for c in calls), 1),
                       "detector_fallback_calls": sum(c.get("detector_fallback_calls", 0) for c in calls),
                       "safety_face_failures": len(q["routing"]["ls_failed_segments"]), "split_fallbacks": len(ls.get("verify_splits") or []),
                       "peak_vram_nvsmi_mib": peak, "torch_free_after_gib": (man.get("gpu") or {}).get("free_vram_gib"),
                       "wall_total_s": man["stage_seconds"].get("total"), "stage_seconds": man["stage_seconds"]}
    # 5 audio / timing
    units = man["units"]; rows = []
    for u in units:
        a = u["alignment"]; rows.append({"id": u["id"], "speaker": u["speaker"], "slot_start": u["slot"]["start"], "slot_end": u["slot"]["end"],
                                         "slot_s": u["slot"]["seconds"], "tts_s": a["tts_seconds"], "ratio": a["ratio_tts_to_slot"], "atempo": a["atempo"],
                                         "mode": a["mode"], "src_words": u.get("words"), "src_chars": len(u["text"]), "tgt_chars": len(u["translation"]),
                                         "text": u["text"], "translation": u["translation"]})
    ratios = [r["ratio"] for r in rows]
    q["audio"] = {"source_lang": man.get("source_lang"), "target_lang": man.get("target_lang"), "asr_language_p": man["asr"].get("language_probability"),
                  "speech_intervals_vad": len(man["speech_intervals"]), "asr_segments": len(man["asr"]["segments"]), "units": len(units),
                  "speakers": man["diarization"]["speakers"], "references": man["tts"]["references"],
                  "source_speech_s": man["timeline"]["speech_seconds"], "dubbed_speech_s_vad": (man.get("speech_gate") or {}).get("dubbed_speech_seconds"),
                  "ratio_min": min(ratios), "ratio_median": float(np.median(ratios)), "ratio_max": max(ratios),
                  "sped_up_gt_1_15": sum(r > 1.15 for r in ratios), "sped_up_gt_1_25": sum(r > 1.25 for r in ratios), "sped_up_gt_1_40": sum(r > 1.40 for r in ratios),
                  "underfilled_lt_0_6": sum(r < 0.6 for r in ratios), "underfilled_lt_0_5": sum(r < 0.5 for r in ratios),
                  "worst_speedup": sorted(rows, key=lambda r: -r["ratio"])[:3], "worst_underfill": sorted(rows, key=lambda r: r["ratio"])[:3], "units_table": rows}
    # silence-driven LatentSync frames: LS frames whose dubbed-audio frame RMS is below -50 dBFS vs frames where the ORIGINAL audio has speech
    n = ls.get("master_frames", 0); own = work / "audio16k.wav"
    dub = pathlib.Path((man.get("dubbed_track") or {}).get("wav16k") or (work / "dubbed_16k.wav"))   # candidates carry their own dubbed track
    if dub.exists() and own.exists() and n:
        rd = frame_rms(read_wav(dub), n); ro = frame_rms(read_wav(own), n)
        silent_d = rd < 10 ** (-50 / 20); loud_o = ro > 10 ** (-35 / 20)
        ls_mask = np.zeros(n, dtype=bool)
        for s in ls_segs: ls_mask[s["start_frame"]:s["end_frame"]] = True
        q["audio"]["ls_frames"] = int(ls_mask.sum()); q["audio"]["ls_frames_dub_silent"] = int((ls_mask & silent_d).sum())
        q["audio"]["ls_frames_dub_silent_while_original_speaks"] = int((ls_mask & silent_d & loud_o).sum())
        q["audio"]["ls_silent_fraction"] = round(float((ls_mask & silent_d).sum() / max(ls_mask.sum(), 1)), 3)
        per_seg = []
        for s in ls_segs:
            m = np.zeros(n, dtype=bool); m[s["start_frame"]:s["end_frame"]] = True
            per_seg.append({"index": s["index"], "start_s": s["start_s"], "end_s": s["end_s"], "frames": s["frames"],
                            "dub_silent_frames": int((m & silent_d).sum()), "dub_silent_fraction": round(float((m & silent_d).sum() / s["frames"]), 3)})
        q["audio"]["ls_segments_silence"] = per_seg
    # 6 syncnet
    if syncnet:
        from syncnet_common import eval_syncnet
        res = {}
        for label, p in (("output", out_mp4), ("original", src)):
            key = str(p)
            if key in syncnet_cache: res[label] = syncnet_cache[key]; continue
            try:
                r = eval_syncnet(p); res[label] = {"confidence": r["confidence"], "av_offset_frames": r["av_offset_frames"], "seconds": r["seconds"]}
            except Exception as e:
                res[label] = {"status": "N/A" if "Face not detected" in str(e) else "FAIL", "error": str(e)[-300:]}
            syncnet_cache[key] = res[label]
        q["syncnet"] = res
    # 7 contact sheets
    fd = frames_dir / f"{vid}_{tag}"; master = work / "master_25fps.mp4"; sheets = []
    bbox_by_frame = {}
    wanted: set[int] = set(); plan = []
    for s in ls_segs[:6]:
        a, b = s["start_frame"], s["end_frame"]; mids = [a + int((b - a) * f) for f in (0.15, 0.3, 0.5, 0.7, 0.85)]
        plan.append(("inside", s["index"], mids)); wanted.update(mids)
        edge = [a, a + 1, a + 2, a + 3, b - 4, b - 3, b - 2, b - 1]; plan.append(("edges", s["index"], edge)); wanted.update(edge)
    for k, tr in enumerate(trs[:12]):
        f = tr["frame"]; rng = [f + d for d in range(-8, 9, 2) if 0 <= f + d < n]; plan.append(("transition", k, rng)); wanted.update(rng)
    small = [s for s in segs if s.get("classification") == "SMALL_FACE"][:4]
    for s in small:
        a, b = s["start_frame"], s["end_frame"]; mids = [a + int((b - a) * f) for f in (0.1, 0.5, 0.9)]; plan.append(("small_face", s["index"], mids)); wanted.update(mids)
    if wanted and out_mp4.exists():
        fo = grab_frames(out_mp4, wanted); fm = grab_frames(master, wanted) if master.exists() else {}
        for kind, idx, fr_idx in plan:
            if kind == "transition":
                tr = trs[idx]; rows_ = [(f"{'M' if src_ else 'O'} f{f} {f/FPS:.2f}s", (fm if src_ else fo).get(f)) for src_ in (True, False) for f in fr_idx]
                p = sheet(rows_, fd / f"transition_{idx:02d}_f{tr['frame']}.png", cols=len(fr_idx), scale=0.35,
                          title=f"{vid} transition {tr['t']}s {tr['from']} -> {tr['to']} | row1 master, row2 output, -8..+8 frames")
            elif kind == "inside":
                p = sheet([(f"O f{f}", fo.get(f)) for f in fr_idx] + [(f"M f{f}", fm.get(f)) for f in fr_idx], fd / f"inside_seg{idx:03d}.png", cols=len(fr_idx), scale=0.4,
                          title=f"{vid} inside LS segment {idx}: row1 output, row2 master")
            elif kind == "edges":
                p = sheet([(f"O f{f}", fo.get(f)) for f in fr_idx] + [(f"M f{f}", fm.get(f)) for f in fr_idx], fd / f"edges_seg{idx:03d}.png", cols=len(fr_idx), scale=0.4,
                          title=f"{vid} LS segment {idx} first 4 / last 4 frames (crossfade zone): row1 output, row2 master")
            else:
                p = sheet([(f"M f{f}", fm.get(f)) for f in fr_idx], fd / f"small_face_seg{idx:03d}.png", cols=len(fr_idx), scale=0.4, title=f"{vid} SMALL_FACE segment {idx} (pass-through)")
            if p: sheets.append(p)
    q["frames"] = {"dir": str(fd), "sheets": sheets}
    return q


def md_report(q: dict) -> str:
    if "run_error" in q: return f"# {q['video']} {q['tag']}\n\nRUN ERROR: {q['run_error']}\n"
    s, o, r, l, a = q["source"], q["output"], q["routing"], q["latentsync"], q["audio"]
    L = [f"# {q['video']} — {q['tag']}", "",
         f"source {s['duration_s']}s {s['width']}x{s['height']} @{s['fps']} ({s['frames']} frames, vfr={s['vfr']}) -> output {o['duration_s']}s {o['width']}x{o['height']} @{o['fps']} ({o['frames']} frames), "
         f"audio={o['audio']} Δdur={o['duration_delta_s']}s, frame_check={o['frame_check'] and o['frame_check']['pass']}, validation={all(o['validation'].values())}", "",
         f"routing: {r['frame_classes']} skip={r['skip_reasons']} | LS segments {r['latentsync_segments']} durations {r['segment_durations_s']} | transitions {r['n_transitions']} | "
         f"verify splits {len(r['verify_splits'] or [])} | LS failed {len(r['ls_failed_segments'])} | router detections {r['router_detections']}",
         f"latentsync: inference {l['inference_total_s']}s (unet {l['unet_denoise_s']}s, {l['unet_calls']} unet calls), stage {l['stage_latentsync_s']}s, peak VRAM {l['peak_vram_nvsmi_mib']} MiB, wall total {l['wall_total_s']}s", "",
         f"audio: {a['source_lang']}->{a['target_lang']} units {a['units']} (asr segs {a['asr_segments']}, vad {a['speech_intervals_vad']}), speakers {a['speakers']} | tts/slot ratio min/med/max {a['ratio_min']:.2f}/{a['ratio_median']:.2f}/{a['ratio_max']:.2f} | "
         f"sped-up >1.15/1.25/1.40: {a['sped_up_gt_1_15']}/{a['sped_up_gt_1_25']}/{a['sped_up_gt_1_40']} | under-filled <0.6/<0.5: {a['underfilled_lt_0_6']}/{a['underfilled_lt_0_5']}"]
    if "ls_frames" in a:
        L.append(f"LS frames driven by SILENT dubbed audio: {a['ls_frames_dub_silent']}/{a['ls_frames']} ({a['ls_silent_fraction']:.0%}); of which original speaker audible: {a['ls_frames_dub_silent_while_original_speaks']}")
    if "syncnet" in q: L.append(f"syncnet: output {q['syncnet'].get('output')} | original {q['syncnet'].get('original')}")
    L += ["", "| unit | spk | slot s | tts s | ratio | mode | source | translation |", "|---|---|---:|---:|---:|---|---|---|"]
    for u in a["units_table"]:
        L.append(f"| u{u['id']:02d} | {u['speaker']} | {u['slot_s']:.2f} | {u['tts_s']:.2f} | {u['ratio']:.2f} | {u['mode'][:12]} | {u['text'][:60]} | {u['translation'][:70]} |")
    L += ["", "transitions: " + "; ".join(f"{t['t']}s {t['from'].split('/')[1]}->{t['to'].split('/')[1]}" for t in r["transitions"][:30]), "", f"frames: {q['frames']['dir']} ({len(q['frames']['sheets'])} sheets)"]
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=str(Q / "baseline")); ap.add_argument("--work-dir", default=str(Q / "work")); ap.add_argument("--tag", default="baseline")
    ap.add_argument("--ids", nargs="+", required=True); ap.add_argument("--out", default=str(Q / "manifests")); ap.add_argument("--frames", default=str(Q / "frames"))
    ap.add_argument("--syncnet", action="store_true"); ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); cache_p = out / "syncnet_cache.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    rows = []
    for vid in a.ids:
        q = analyse(pathlib.Path(a.run_dir), pathlib.Path(a.work_dir), vid, a.tag, out, pathlib.Path(a.frames), a.syncnet, cache)
        (out / f"{vid}_{a.tag}.quality.json").write_text(json.dumps(q, indent=1, ensure_ascii=False, default=str))
        (out / f"{vid}_{a.tag}.quality.md").write_text(md_report(q)); cache_p.write_text(json.dumps(cache, indent=1))
        print(md_report(q).split("\n\n")[1] if "run_error" not in q else q["run_error"])
        if "run_error" not in q:
            sn = q.get("syncnet", {}); rows.append({"candidate": a.tag, "video": vid, "syncnet_conf": (sn.get("output") or {}).get("confidence"),
                    "av_offset": (sn.get("output") or {}).get("av_offset_frames"), "orig_conf": (sn.get("original") or {}).get("confidence"),
                    "duration_delta_s": q["output"]["duration_delta_s"], "frames_out": q["output"]["frames"], "frames_master": q["routing"]["master_frames"],
                    "ls_segments": q["routing"]["latentsync_segments"], "transitions": q["routing"]["n_transitions"], "ls_failed": len(q["routing"]["ls_failed_segments"]),
                    "peak_vram_mib": q["latentsync"]["peak_vram_nvsmi_mib"], "ls_inference_s": q["latentsync"]["inference_total_s"], "wall_s": q["latentsync"]["wall_total_s"],
                    "ratio_median": round(q["audio"]["ratio_median"], 3), "ratio_max": q["audio"]["ratio_max"], "sped_up_gt_1_25": q["audio"]["sped_up_gt_1_25"],
                    "ls_silent_fraction": q["audio"].get("ls_silent_fraction")})
    if a.csv and rows:
        p = pathlib.Path(a.csv); new = not p.exists()
        with p.open("a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); new and w.writeheader(); w.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
