#!/usr/bin/env python3
"""E5 — LatentSync inference-quality probe on ONE real segment of a baseline run (Phase 3 §1/§2). Nothing in the pipeline changes.

Takes the master frames + dubbed-track slice + LatentSync detections of a LATENT_SYNC segment from a baseline run and re-runs the
AccelRunner with one parameter group changed per variant (DeepCache interval / off, steps, guidance, window batch). Writes
<out>/<id>_seg<idx>_<variant>.mp4 (segment with its dubbed audio), sharpness/flicker metrics vs master, SyncNet, time, VRAM.
    python scripts/quality/ls_probe.py --id 04 --segment 1 --variants dc5 dc3 dcoff s25 s30 g10 g20 b1
"""
from __future__ import annotations
import argparse, json, math, pathlib, sys, time
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "optim"))
Q = pathlib.Path("/tmp/dabai_quality"); FPS, SR = 25, 16000
VARIANTS = {"dc5": dict(), "dc3": dict(deepcache_interval=3), "dcoff": dict(deepcache=False), "s25": dict(steps=25), "s30": dict(steps=30),
            "g10": dict(guidance=1.0), "g20": dict(guidance=2.0), "g25": dict(guidance=2.5), "g30": dict(guidance=3.0), "b1": dict(window_batch_size=1)}
BASE = dict(window_batch_size=2, deepcache=True, deepcache_interval=5, steps=20, guidance=1.5, sdpa_backend="auto", compile_backend="none", seed=1247)


def sharp(g):
    import cv2
    return float(cv2.Laplacian(cv2.cvtColor(g, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())


def main() -> int:
    import cv2, soundfile as sf, torch
    ap = argparse.ArgumentParser(); ap.add_argument("--id", required=True); ap.add_argument("--segment", type=int, required=True)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS)); ap.add_argument("--run-dir", default=str(Q / "baseline")); ap.add_argument("--work-dir", default=str(Q / "work"))
    ap.add_argument("--out", default=str(Q / "candidates" / "ls_probe")); ap.add_argument("--syncnet", action="store_true")
    a = ap.parse_args(); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    man = json.loads((pathlib.Path(a.run_dir) / f"{a.id}_baseline.manifest.json").read_text())
    seg = next(s for s in man["lipsync"]["segments"] if s["index"] == a.segment); assert seg["action"] == "LATENT_SYNC"
    work = pathlib.Path(a.work_dir) / f"work_{a.id}"
    import face_aware_latentsync as fa
    from face_aware_latentsync_accel import AccelRunner
    from retinaface_router import RetinaFaceRouter
    frames = fa.read_frames(work / "master_25fps.mp4")[seg["start_frame"]:seg["end_frame"]]
    seg_rgb = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames])
    audio, sr = sf.read(str(work / "dubbed_16k.wav"), dtype="float32"); a0 = int(round(seg["start_frame"] / FPS * SR)); a1 = a0 + int(round(seg["frames"] / FPS * SR))
    wav = out / f"{a.id}_seg{a.segment:03d}_dub.wav"; sf.write(str(wav), audio[a0:a1], SR, subtype="PCM_16")
    print(f"segment {a.segment}: {seg['start_s']}-{seg['end_s']}s {len(frames)} frames", flush=True)
    dets = fa.detect_frames(seg_rgb)
    router = RetinaFaceRouter(50, 80, 0.8); bbs = []
    for i in range(0, len(frames), 5):
        r = router.classify_frame(frames[i]); bbs.append((r["bbox"]["x1"], r["bbox"]["y1"], r["bbox"]["x2"], r["bbox"]["y2"]) if r["bbox"] else None)
    def mouth(fr, k):
        bb = bbs[min(k // 5, len(bbs) - 1)]
        if bb is None: return None
        x1, y1, x2, y2 = bb; w, h = x2 - x1, y2 - y1; H, W = fr.shape[:2]
        return fr[max(0, y1 + int(0.55 * h)):min(H, y2 + int(0.1 * h)), max(0, x1 + int(0.15 * w)):min(W, x2 - int(0.15 * w))]
    def upper(fr, k):
        bb = bbs[min(k // 5, len(bbs) - 1)]
        if bb is None: return None
        x1, y1, x2, y2 = bb; H, W = fr.shape[:2]; return fr[max(0, y1):min(H, y1 + (y2 - y1) // 2), max(0, x1):min(W, x2)]
    def metrics(outs):
        ms = mo = us = uo = fm = fo = 0.0; n = nf = 0; prev = None
        for k, (o, m) in enumerate(zip(outs, frames)):
            mm, om = mouth(m, k), mouth(o, k)
            if mm is None or om is None or mm.shape != om.shape: prev = None; continue
            ms += sharp(mm); mo += sharp(om); us += sharp(upper(m, k)); uo += sharp(upper(o, k)); n += 1
            if prev is not None and prev[0].shape == mm.shape:
                fm += float(np.mean(np.abs(mm.astype(np.float32) - prev[0].astype(np.float32)))); fo += float(np.mean(np.abs(om.astype(np.float32) - prev[1].astype(np.float32)))); nf += 1
            prev = (mm, om)
        return {"mouth_sharp_ratio": round(mo / max(ms, 1e-6), 3), "upper_sharp_ratio": round(uo / max(us, 1e-6), 3), "flicker_ratio": round(fo / max(fm, 1e-6), 3),
                "mouth_sharp_out": round(mo / max(n, 1), 1), "flicker_out": round(fo / max(nf, 1), 2), "frames_measured": n}
    results = []; pipe_runner = None; prev_helper = None; ref_frames = None
    for v in a.variants:
        cfg = {**BASE, **VARIANTS[v]}
        if prev_helper is not None: prev_helper.disable()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        if pipe_runner is None:
            pipe_runner = AccelRunner(cfg["window_batch_size"], cfg["deepcache"], cfg["compile_backend"], cfg["sdpa_backend"], steps=cfg["steps"], guidance=cfg["guidance"], seed=cfg["seed"], deepcache_interval=cfg["deepcache_interval"])
            pipe = pipe_runner.acc.pipe
        else:
            from latentsync_accel import AccelLatentSync
            pipe_runner.acc = AccelLatentSync(pipe, window_batch_size=cfg["window_batch_size"], deepcache=cfg["deepcache"], compile_backend=cfg["compile_backend"], sdpa_backend=cfg["sdpa_backend"],
                                              steps=cfg["steps"], guidance=cfg["guidance"], seed=cfg["seed"], deepcache_interval=cfg["deepcache_interval"])
        prev_helper = pipe_runner.acc.deepcache_helper
        t0 = time.perf_counter(); outs_rgb, dt = pipe_runner.run_frames(seg_rgb, wav, dets); wall = round(time.perf_counter() - t0, 1)
        outs = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in outs_rgb][:len(frames)]
        peak = round(torch.cuda.max_memory_reserved() / 2**20)
        mp4 = out / f"{a.id}_seg{a.segment:03d}_{v}.mp4"; tmp = out / f"{a.id}_seg{a.segment:03d}_{v}_v.mp4"
        import imageio, subprocess
        w = imageio.get_writer(str(tmp), fps=FPS, codec="libx264", macro_block_size=None, ffmpeg_params=["-crf", "13"], ffmpeg_log_level="error")
        for f in outs: w.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
        w.close(); subprocess.run(["ffmpeg", "-v", "error", "-y", "-nostdin", "-i", str(tmp), "-i", str(wav), "-c:v", "copy", "-c:a", "aac", "-b:a", "128k", str(mp4)], check=True); tmp.unlink()
        row = {"variant": v, "changed": VARIANTS[v] or "baseline", "config": cfg, "wall_s": wall, "unet_calls": pipe_runner.acc.last_stats["unet_calls"], "peak_reserved_mib": peak, "frames_out": len(outs), **metrics(outs), "file": str(mp4)}
        if ref_frames is None: ref_frames = outs
        else:
            ps = []
            for o, r_ in zip(outs, ref_frames):
                mse = float(np.mean((o.astype(np.float32) - r_.astype(np.float32)) ** 2)); ps.append(99.0 if mse == 0 else 10 * math.log10(255**2 / mse))
            row["psnr_vs_dc5_mean"] = round(float(np.mean(ps)), 2); row["psnr_vs_dc5_min"] = round(float(np.min(ps)), 2)
        if a.syncnet:
            from syncnet_common import eval_syncnet
            try: r_ = eval_syncnet(mp4); row["syncnet_conf"] = r_["confidence"]; row["av_offset"] = r_["av_offset_frames"]
            except Exception as e: row["syncnet_conf"] = None; row["syncnet_error"] = str(e)[-200:]
        results.append(row); print(json.dumps({k: row[k] for k in row if k not in ("config", "file")}), flush=True)
        (out / f"{a.id}_seg{a.segment:03d}_probe.json").write_text(json.dumps({"video": a.id, "segment": seg, "results": results}, indent=1))
    # master reference clip for human A/B
    mp4 = out / f"{a.id}_seg{a.segment:03d}_master_original_audio.mp4"
    if not mp4.exists():
        import subprocess
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-nostdin", "-ss", f"{seg['start_frame']/FPS:.3f}", "-i", str(work / "master_25fps.mp4"), "-ss", f"{seg['start_frame']/FPS:.3f}", "-i", str(work / "audio16k.wav"),
                        "-t", f"{seg['frames']/FPS:.3f}", "-c:v", "libx264", "-crf", "13", "-c:a", "aac", str(mp4)], check=True)
    print("done ->", out); return 0


if __name__ == "__main__":
    raise SystemExit(main())
