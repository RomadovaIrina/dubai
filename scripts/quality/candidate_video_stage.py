#!/usr/bin/env python3
"""Re-run ONLY the video stage of run_clean_pipeline_05.py (speech gate -> RetinaFace router -> LatentSync -> assembly) on a baseline
work dir with a different dubbed track and/or different video-backend parameters. Audio stages (ASR/translation/TTS) are reused
from the baseline run, so E1/E6 candidates are compared on identical speech content. Writes <out>/<id>_<tag>.mp4 + manifest in the
runner's schema (so quality_manifest.py / face_metrics.py work unchanged).
    python scripts/quality/candidate_video_stage.py --id 04 --tag e1 --dubbed-dir /tmp/dabai_quality/candidates/e1/work_04 \
        --out /tmp/dabai_quality/candidates [--crossfade-frames 4] [--gate-margin-s 0.24] [--gate-merge-gap-s 0.6] [--router-stride 3] [--speech-gate on|off]
"""
from __future__ import annotations
import argparse, json, os, pathlib, shutil, sys, time
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "optim"))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
Q = pathlib.Path("/tmp/dabai_quality")


class MouthOnlyRunner:
    """E4 wrapper around AccelRunner: LatentSync output is composited back into the ORIGINAL frames only inside a feathered lower-face
    mask built from LatentSync's own per-frame detection bbox (replayed detections), so eyes/nose/skin keep the source texture."""

    def __init__(self, inner, top: float, feather: float):
        self.inner, self.top, self.feather = inner, top, feather
        self.config = {**inner.config, "mouth_only_pasteback": True, "mask_top": top, "mask_feather": feather}; self.calls = inner.calls
        self.steps, self.guidance, self.seed, self.load_s = inner.steps, inner.guidance, inner.seed, inner.load_s; self.masked_frames = 0

    def run_frames(self, frames_rgb, audio_wav, detections=None):
        import cv2, numpy as np
        outs, dt = self.inner.run_frames(frames_rgb, audio_wav, detections)
        if detections is None: return outs, dt
        res = np.empty_like(outs); H, W = frames_rgb.shape[1:3]
        for i in range(min(len(outs), len(frames_rgb))):
            bbox = detections[i][0] if i < len(detections) else None
            if bbox is None: res[i] = outs[i]; continue
            x1, y1, x2, y2 = [int(v) for v in bbox]; w, h = max(x2 - x1, 1), max(y2 - y1, 1)
            m = np.zeros((H, W), np.float32); cy = y1 + int(self.top * h); cx = (x1 + x2) // 2
            # lower face: ellipse spanning the bbox width (+10 %) from mask_top to slightly below the chin
            axes = (int(w * 0.55), int((y2 + int(0.06 * h) - cy) * 0.5 + 1)); centre = (cx, (cy + y2 + int(0.06 * h)) // 2)
            cv2.ellipse(m, centre, axes, 0, 0, 360, 1.0, -1)
            k = max(3, int(self.feather * w) | 1); m = cv2.GaussianBlur(m, (k, k), 0)[..., None]
            res[i] = (outs[i].astype(np.float32) * m + frames_rgb[i].astype(np.float32) * (1 - m)).round().clip(0, 255).astype(np.uint8); self.masked_frames += 1
        if len(outs) > len(frames_rgb): res = np.concatenate([res, outs[len(frames_rgb):]])
        return res, dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--run-dir", default=str(Q / "baseline")); ap.add_argument("--work-dir", default=str(Q / "work"))
    ap.add_argument("--dubbed-dir", default=None, help="dir with dubbed_16k.wav + dubbed_aac.m4a (+ aligned_u*.wav); default = baseline work dir")
    ap.add_argument("--out", default=str(Q / "candidates")); ap.add_argument("--work-out", default=None)
    ap.add_argument("--speech-gate", choices=["on", "off"], default="on"); ap.add_argument("--gate-margin-s", type=float, default=0.24); ap.add_argument("--gate-merge-gap-s", type=float, default=0.6)
    ap.add_argument("--crossfade-frames", type=int, default=4); ap.add_argument("--router-stride", type=int, default=3); ap.add_argument("--window-batch-size", type=int, default=2)
    ap.add_argument("--deepcache-interval", type=int, default=5); ap.add_argument("--no-deepcache", action="store_true"); ap.add_argument("--steps", type=int, default=20); ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--face-min-w", type=int, default=50); ap.add_argument("--face-min-h", type=int, default=80); ap.add_argument("--retina-conf", type=float, default=0.8)
    ap.add_argument("--min-ls-frames", type=int, default=25); ap.add_argument("--max-verify-depth", type=int, default=3)
    ap.add_argument("--mouth-only-pasteback", choices=["on", "off"], default="off", help="E4: outside a feathered lower-face mask keep the master pixels (LatentSync upstream untouched)")
    ap.add_argument("--mask-top", type=float, default=0.52, help="E4: mask starts at this fraction of the detected face height (0 = top of bbox)")
    ap.add_argument("--mask-feather", type=float, default=0.08, help="E4: gaussian feather as a fraction of the face width")
    a = ap.parse_args()
    import run_clean_pipeline_05 as rp
    import face_aware_latentsync_accel as faa
    from measure_stages import probe
    base_work = pathlib.Path(a.work_dir) / f"work_{a.id}"; man = json.loads((pathlib.Path(a.run_dir) / f"{a.id}_baseline.manifest.json").read_text())
    src = pathlib.Path(man["input"]); duration = man["source"]["duration_s"]; out_dir = pathlib.Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{a.id}_{a.tag}.mp4"; work = pathlib.Path(a.work_out or (out_dir / f"work_{a.id}_{a.tag}")); work.mkdir(parents=True, exist_ok=True)
    dd = pathlib.Path(a.dubbed_dir) if a.dubbed_dir else base_work
    dub16, dub_aac = dd / "dubbed_16k.wav", dd / "dubbed_aac.m4a"
    master = base_work / "master_25fps.mp4"
    if master.exists() and not (work / "master_25fps.mp4").exists(): os.symlink(master, work / "master_25fps.mp4")   # make_master() reuses it (identical ffmpeg call otherwise)
    times = {}; t_all = time.perf_counter()
    man = {**man, "output": str(out), "work_dir": str(work), "candidate": {"tag": a.tag, "dubbed_dir": str(dd), "video_params": {k: v for k, v in vars(a).items() if k not in ("id", "tag", "run_dir", "work_dir", "dubbed_dir", "out", "work_out")}}}
    if a.dubbed_dir:   # units' alignment info comes from the E1 report when present
        e1 = dd / "e1_alignment.json"
        if e1.exists():
            rep = json.loads(e1.read_text()); by = {r["id"]: r for r in rep["units"]}
            for u in man["units"]:
                r = by.get(u["id"])
                if r: u["alignment"] = {**u["alignment"], "mode": f"e1_{rep['mode']}", "atempo": r["max_atempo"], "e1_placements": r["placements"], "file": r["file"]}
            man["e1_summary"] = rep["summary"]
        man["dubbed_track"] = {**man["dubbed_track"], "wav16k": str(dub16), "aac": str(dub_aac), "wav24k": str(dd / "dubbed_24k.wav")}
    speech = man["speech_intervals"]; gate = None
    if a.speech_gate == "on":
        with rp.timed(times, "speech_gate"):
            dub_speech = rp.run_vad(dub16); gate = faa.build_speech_gate([speech, dub_speech], duration, a.gate_margin_s, a.gate_merge_gap_s)
        gated = sum(e - s for s, e in gate)
        man["speech_gate"] = {"enabled": True, "margin_s": a.gate_margin_s, "merge_gap_s": a.gate_merge_gap_s, "crossfade_frames": a.crossfade_frames, "source_speech_seconds": round(sum(i["end"] - i["start"] for i in speech), 3),
                              "dubbed_speech_seconds": round(sum(i["end"] - i["start"] for i in dub_speech), 3), "gate_seconds": round(gated, 3), "gate_fraction": round(gated / duration, 4), "intervals": gate}
        print(f"  [gate] {len(gate)} intervals, {gated:.1f}s of {duration:.1f}s", flush=True)
    else: man["speech_gate"] = {"enabled": False}
    with rp.timed(times, "face_router_load"): router = faa.make_router("retinaface", a.face_min_w, a.face_min_h, a.retina_conf)
    with rp.timed(times, "latentsync_load"): runner = faa.AccelRunner(a.window_batch_size, not a.no_deepcache, "none", "auto", steps=a.steps, guidance=a.guidance, deepcache_interval=a.deepcache_interval)
    if a.mouth_only_pasteback == "on":
        runner = MouthOnlyRunner(runner, a.mask_top, a.mask_feather)
    rep = faa.process_video(src, work, dub16, dub_aac, out, router=router, runner=runner, min_ls_frames=a.min_ls_frames, max_verify_depth=a.max_verify_depth, times=times,
                            log=lambda m: print(m, flush=True), speech_gate=gate, crossfade_frames=a.crossfade_frames, router_stride=a.router_stride, in_memory=True)
    rep["backend"] = "optimized"; man["lipsync"] = rep; rp.free_cuda(runner)
    times["total"] = round(time.perf_counter() - t_all, 3); op = probe(out); fps = man["source"]["video"].get("avg_fps") or 25.0; tol = max(0.15, 2.0 / fps); delta = round(op["duration_s"] - duration, 4)
    checks = {"output_exists_nonempty": out.exists() and out.stat().st_size > 0, "output_has_video_audio": op["has_video"] and op["has_audio"], "duration_within_tolerance": abs(delta) <= tol,
              "resolution_preserved": (op["video"]["width"], op["video"]["height"]) == (man["source"]["video"]["width"], man["source"]["video"]["height"]),
              "all_units_have_tts_and_translation": True, "frame_count_preserved": rep["assembly"]["frames_written"] == rep["master_frames"], "output_frames_decodable_no_blank": bool(rep["output_frame_check"]["pass"])}
    man["output_meta"] = op; man["validation"] = {"tolerance_s": round(tol, 4), "duration_delta_s": delta, "checks": checks, "pass": all(checks.values())}
    man["stage_seconds"] = times; man["gpu"] = rp.gpu_info(); man["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rp.write_json(out.with_name(out.stem + ".manifest.json"), man)
    for p in work.glob("seg*_in.wav"): p.unlink(missing_ok=True)
    print(f"== {a.id} {a.tag}: validation {'PASS' if man['validation']['pass'] else 'FAIL'} {checks} total {times['total']}s -> {out}")
    return 0 if man["validation"]["pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
