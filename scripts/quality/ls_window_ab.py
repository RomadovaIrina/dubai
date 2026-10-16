#!/usr/bin/env python3
"""Video-stage A/B on a time WINDOW of a finished baseline run (track A, 2026-10-16): same master frames, same dubbed track, same speech
gate intervals (taken from the baseline manifest, not recomputed), same router / verify / crossfade / mouth-only paste-back / CodeFormer-mouth
settings as cand5; only the lip-sync generator variant changes (scripts/quality/ls_stable.py). The baseline arm must be re-run on the same
window too (LatentSync windows restart at the cut).

Writes <out>/<id>_<tag>.mp4 (window clip, final mix muxed; L2 muxes the source audio), <id>_<tag>_dubonly.mp4 (same frames, driving audio
only, for SyncNet), and <id>_<tag>.manifest.json whose segments are in ABSOLUTE master frames with "window" = {f0, f1} so
lipsync_jitter.py measures it against the full master.

    python scripts/quality/ls_window_ab.py --manifest RUN/final/03_final.manifest.json --t0 40 --t1 58 --tag L3 --out DIR
        [--mode ls|wrapper|vae] [--audio dub|src] [--stride 8] [--ref-mode same|fixed --ref-select calm_any|calm_closed]
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys, time
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
for d in ("pilot", "optim", "quality"):
    sys.path.insert(0, str(HERE.parent / d))
FPS = 25; SR = 16000


class WindowPrep:
    """Stands in for face_aware_latentsync_accel.VideoPrep: the window's master frames plus the same deterministic detector results."""

    def __init__(self, frames, router, ls_det):
        import cv2
        self.master = None; self.frames = frames; self.times = {}
        t0 = time.perf_counter(); self.router_rows = [router.classify_frame(f) for f in frames]; self.times["prep_router_all_frames"] = round(time.perf_counter() - t0, 3)
        t0 = time.perf_counter(); self.ls_dets = [ls_det(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames]; self.times["prep_ls_detect_all_frames"] = round(time.perf_counter() - t0, 3)


def make_dumper(MouthOnlyRunner, frames):
    """MouthOnlyRunner that also keeps, per LS segment, the RAW LatentSync frames (LatentSync's own restore / paste of its 512 crop) and
    the frames after CodeFormer (right before the mouth-only composite), so each stage can be measured on its own."""
    import cv2, hashlib
    index = {hashlib.md5(f.tobytes()).hexdigest(): i for i, f in enumerate(frames)}

    class Dumper(MouthOnlyRunner):
        raw: dict = {}; pre: dict = {}
        def run_frames(self, frames_rgb, audio_wav, detections=None):
            outs, dt = super().run_frames(frames_rgb, audio_wav, detections)
            self._start = index.get(hashlib.md5(cv2.cvtColor(frames_rgb[0], cv2.COLOR_RGB2BGR).tobytes()).hexdigest())
            if self._start is not None: Dumper.raw[self._start] = [cv2.cvtColor(np.asarray(o), cv2.COLOR_RGB2BGR) for o in outs[:len(frames_rgb)]]
            return outs, dt
        def composite(self, outs, originals, boxes=None):
            if getattr(self, "_start", None) is not None: Dumper.pre[self._start] = [np.array(o, copy=True) for o in outs]
            return super().composite(outs, originals, boxes)
    return Dumper


def write_stage(frames, stage: dict, path, audio):
    """master frames with the stage's LS frames substituted (no crossfade), muxed with `audio`."""
    import imageio, cv2
    out = list(frames)
    for st, frs in stage.items():
        for k, f in enumerate(frs):
            if st + k < len(out): out[st + k] = f
    tmp = path.with_name(path.stem + "_v.mp4")
    w = imageio.get_writer(str(tmp), fps=FPS, codec="libx264", macro_block_size=None, ffmpeg_params=["-crf", "13"], ffmpeg_log_level="error")
    for f in out: w.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    w.close()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", str(tmp), "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", str(path)], check=True)
    tmp.unlink(missing_ok=True)


class PrecomputedRunner:
    """Lip-sync frames produced OUTSIDE this process for the whole window (e.g. TBDub in its own venv, same driving audio), served per LS
    segment so gate / verify / CodeFormer-mouth / mouth-only composite / crossfade stay identical to the LatentSync arms."""

    def __init__(self, video, frames_bgr, label, seconds=None):
        import cv2, hashlib
        gen = read_frames(video, 0, len(frames_bgr))
        assert len(gen) == len(frames_bgr), f"{video}: {len(gen)} frames, window has {len(frames_bgr)}"
        assert gen[0].shape == frames_bgr[0].shape, f"{video}: {gen[0].shape} vs {frames_bgr[0].shape}"
        self.gen = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in gen]
        self.index = {hashlib.md5(cv2.cvtColor(f, cv2.COLOR_BGR2RGB).tobytes()).hexdigest(): i for i, f in enumerate(frames_bgr)}
        self.config = {"backend": label, "precomputed_video": str(video), "external_seconds": seconds}
        self.calls: list = []; self.steps = self.guidance = self.seed = None; self.load_s = 0.0

    def run_frames(self, frames_rgb, audio_wav, detections=None):
        import hashlib
        st = self.index[hashlib.md5(np.ascontiguousarray(frames_rgb[0]).tobytes()).hexdigest()]
        out = np.stack(self.gen[st:st + len(frames_rgb)]); self.calls.append({"frames": len(out), "start": st, "seconds": 0.0})
        return out, 0.0


def read_frames(path, f0, f1):
    import cv2
    cap = cv2.VideoCapture(str(path)); cap.set(cv2.CAP_PROP_POS_FRAMES, f0); out = []
    for _ in range(f1 - f0):
        ok, f = cap.read()
        if not ok: break
        out.append(f)
    cap.release(); return out


def cut_audio(src, t0, dur, dst, codec_args):
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-ss", f"{t0:.3f}", "-t", f"{dur:.3f}", "-i", str(src)] + codec_args + [str(dst)], check=True)


def pick_reference(lm, ok, mode, nref=16, exclude=None):
    """16 consecutive master frames for the fixed reference: all detected, calm head (low landmark motion), pose near the median,
    calm_closed additionally prefers a closed mouth (inner lips 60/62). lm: (T, 106, 2) master landmarks (NaN = no face)."""
    T = len(lm); iod = np.linalg.norm(lm[:, 87:97].mean(1) - lm[:, 33:43].mean(1), axis=1)
    nose = lm[:, 86]; mid = 0.5 * (lm[:, 33:43].mean(1) + lm[:, 87:97].mean(1))
    yaw = (nose[:, 0] - mid[:, 0]) / iod; pitch = (nose[:, 1] - mid[:, 1]) / iod
    motion = np.zeros(T); motion[1:] = np.linalg.norm(np.diff(lm[:, list(range(33, 43)) + list(range(72, 97))], axis=0), axis=2).mean(1) / iod[1:]
    opening = np.linalg.norm(lm[:, 60] - lm[:, 62], axis=1) / iod
    ym, pm = np.nanmedian(yaw[ok]), np.nanmedian(pitch[ok]); best = None
    for s in range(0, T - nref):
        sl = slice(s, s + nref)
        if not ok[sl].all() or (exclude is not None and exclude[sl].any()): continue
        cost = 10 * np.mean(motion[sl]) + abs(np.mean(yaw[sl]) - ym) + abs(np.mean(pitch[sl]) - pm) + (np.mean(opening[sl]) if mode == "calm_closed" else 0.0)
        if best is None or cost < best[0]: best = (float(cost), s)
    return best


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--t0", type=float, required=True); ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["ls", "wrapper", "vae"], default="ls"); ap.add_argument("--audio", choices=["dub", "src"], default="dub")
    ap.add_argument("--stride", type=int, default=None); ap.add_argument("--ref-mode", choices=["same", "fixed"], default="same")
    ap.add_argument("--ref-select", choices=["calm_any", "calm_closed"], default="calm_any"); ap.add_argument("--ref-lm", default=None, help="master landmark npz (lipsync_jitter cache) for the reference choice")
    # cand5 video settings (defaults = the baseline)
    ap.add_argument("--window-batch-size", type=int, default=2); ap.add_argument("--deepcache-interval", type=int, default=5); ap.add_argument("--no-deepcache", action="store_true")
    ap.add_argument("--steps", type=int, default=20); ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--crossfade-frames", type=int, default=4); ap.add_argument("--router-stride", type=int, default=3); ap.add_argument("--min-ls-frames", type=int, default=25)
    ap.add_argument("--mask-top", type=float, default=0.52); ap.add_argument("--mask-feather", type=float, default=0.08); ap.add_argument("--mask-smooth", type=int, default=3)
    ap.add_argument("--ls-landmark-smooth", type=int, default=2); ap.add_argument("--codeformer", choices=["off", "optimized"], default="optimized")
    ap.add_argument("--precomputed", default=None, help="window-length video of an external lip-sync backend (same driving audio); replaces LatentSync")
    ap.add_argument("--precomputed-label", default="external"); ap.add_argument("--precomputed-seconds", type=float, default=None)
    ap.add_argument("--dump-stages", action="store_true", help="also write <tag>_raw.mp4 (raw LatentSync frames) and <tag>_precomp.mp4 (after CodeFormer, before the mouth composite)")
    ap.add_argument("--outbox", default=None, help="hardlink the viewable clips into this dir (e.g. /workspace/dub/outbox/p2-03win-L3)")
    a = ap.parse_args()
    import cv2, soundfile as sf, torch
    import face_aware_latentsync as fa
    import face_aware_latentsync_accel as faa
    from candidate_video_stage import MouthOnlyRunner
    from ls_stable import StableRunner
    from pilot_common import gpu_info, write_json
    man = json.loads(pathlib.Path(a.manifest).read_text()); work_b = pathlib.Path(man["work_dir"]); vid = pathlib.Path(man["input"]).stem
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); work = out / f"work_{vid}_{a.tag}"; work.mkdir(parents=True, exist_ok=True)
    f0, f1 = int(round(a.t0 * FPS)), int(round(a.t1 * FPS)); t0s, dur = f0 / FPS, (f1 - f0) / FPS
    times = {}; t_all = time.perf_counter()
    frames = read_frames(work_b / "master_25fps.mp4", f0, f1); f1 = f0 + len(frames)
    # audio: driving (16 kHz) + mux
    drv_src = work_b / ("dubbed_16k.wav" if a.audio == "dub" else "own_audio_16k.wav")
    y, sr = sf.read(str(drv_src), dtype="float32"); assert sr == SR
    a0 = int(round(t0s * SR)); seg = y[a0:a0 + int(round(dur * SR))]
    seg = np.concatenate([seg, np.zeros(int(round(dur * SR)) - len(seg), np.float32)]) if len(seg) < int(round(dur * SR)) else seg
    drv = work / "drive_16k.wav"; sf.write(str(drv), seg, SR, subtype="PCM_16")
    mux_src = (work_b / "final_mix_44k.wav") if a.audio == "dub" and (work_b / "final_mix_44k.wav").exists() else drv_src
    mux = work / "mux_aac.m4a"; cut_audio(mux_src, t0s, dur, mux, ["-c:a", "aac", "-b:a", "192k"])
    drv_aac = work / "drive_aac.m4a"; subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", str(drv), "-c:a", "aac", "-b:a", "192k", str(drv_aac)], check=True)
    # gate: the baseline's intervals, shifted into the window
    gate = [(round(max(0.0, s - t0s), 3), round(min(dur, e - t0s), 3)) for s, e in man["speech_gate"]["intervals"] if e > t0s and s < t0s + dur]
    router = faa.make_router("retinaface", 50, 80, 0.8)
    t = time.perf_counter(); prep = WindowPrep(frames, router, fa.face_detector()); times["prep"] = round(time.perf_counter() - t, 3)
    t = time.perf_counter()
    if a.precomputed:
        runner = PrecomputedRunner(pathlib.Path(a.precomputed), frames, a.precomputed_label, a.precomputed_seconds)
    else:
        runner = StableRunner(a.window_batch_size, not a.no_deepcache, a.steps, a.guidance, 1247, a.deepcache_interval, stride=a.stride, ref_mode=a.ref_mode, mode=a.mode)
    times["latentsync_load"] = round(time.perf_counter() - t, 3)
    ref_info = None
    if a.ref_mode == "fixed" and not a.precomputed:
        lm_npz = pathlib.Path(a.ref_lm) if a.ref_lm else None
        if lm_npz and lm_npz.exists():
            LM = np.load(lm_npz)["lm"]
        else:
            det = fa.face_detector(); allf = read_frames(work_b / "master_25fps.mp4", 0, man["lipsync"]["master_frames"])
            LM = np.stack([np.asarray(d[1], np.float32) if d[0] is not None else np.full((106, 2), np.nan, np.float32) for d in (det(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in allf)])
        ok = np.isfinite(LM[:, 0, 0]); excl = np.zeros(len(LM), bool); excl[f0:f1] = True            # reference from OUTSIDE the window (as in training)
        cost, rs = pick_reference(LM, ok, a.ref_select, 16, excl)
        rf = read_frames(work_b / "master_25fps.mp4", rs, rs + 16); det = fa.face_detector()
        runner.acc.set_reference(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in rf]), [det(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in rf])
        ref_info = {"select": a.ref_select, "start_frame": int(rs), "t": round(rs / FPS, 2), "cost": round(cost, 4)}
        print(f"  [ref] fixed reference frames {rs}-{rs + 16} ({rs / FPS:.2f}s), cost {cost:.4f}", flush=True)
    Mo = make_dumper(MouthOnlyRunner, frames) if a.dump_stages else MouthOnlyRunner
    runner = Mo(runner, a.mask_top, a.mask_feather, a.mask_smooth, defer=True)
    cf = None
    if a.codeformer == "optimized":
        from codeformer_accel import CodeFormerAccel
        cf = CodeFormerAccel(fidelity_weight=1.0, batch_size=8, landmark_source="insightface", autocast=True, parse_autocast=True)
    clip = out / f"{vid}_{a.tag}.mp4"
    torch.cuda.reset_peak_memory_stats()
    rep = faa.process_video(pathlib.Path(man["input"]), work, drv, mux, clip, router=router, runner=runner, min_ls_frames=a.min_ls_frames, max_verify_depth=3,
                            codeformer=a.codeformer, times=times, log=lambda m: print(m, flush=True), speech_gate=gate, crossfade_frames=a.crossfade_frames,
                            router_stride=a.router_stride, in_memory=True, cf_runner=cf, lm_smooth=a.ls_landmark_smooth, prep=prep)
    peak = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    if cf is not None: cf.close()
    dubonly = out / f"{vid}_{a.tag}_dubonly.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", str(clip), "-i", str(drv_aac), "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", str(dubonly)], check=True)
    stage_clips = {}
    if a.dump_stages:
        for name, st in (("raw", Mo.raw), ("precomp", Mo.pre)):
            pth = out / f"{vid}_{a.tag}_{name}.mp4"; write_stage(frames, st, pth, mux); stage_clips[name] = str(pth)
    if a.outbox:
        import os
        ob = pathlib.Path(a.outbox); ob.mkdir(parents=True, exist_ok=True)
        for pth in [clip] + [pathlib.Path(x) for x in stage_clips.values()]:
            tmpl = ob / (".tmp_" + pth.name)
            if tmpl.exists(): tmpl.unlink()
            os.link(pth, tmpl); os.replace(tmpl, ob / pth.name)
    for s in rep["segments"]:
        s["start_frame"] += f0; s["end_frame"] += f0; s["start_s"] = round(s["start_frame"] / FPS, 3); s["end_s"] = round(s["end_frame"] / FPS, 3)
    rep["master_frames"] = man["lipsync"]["master_frames"]
    times["total"] = round(time.perf_counter() - t_all, 3)
    wm = {k: man[k] for k in ("input", "work_dir", "source", "dubbed_track", "speech_intervals")}
    wm.update({"output": str(clip), "output_dubonly": str(dubonly), "stage_clips": stage_clips, "window": {"t0": round(t0s, 3), "t1": round(f1 / FPS, 3), "f0": f0, "f1": f1},
               "variant": {"tag": a.tag, "mode": a.mode, "audio": a.audio, "stride": a.stride, "ref_mode": a.ref_mode, "reference": ref_info,
                           "video_params": {k: v for k, v in vars(a).items() if k not in ("manifest", "out", "tag")}},
               "speech_gate": {"intervals_window": gate, "source": "baseline manifest"}, "lipsync": rep, "stage_seconds": times,
               "peak_vram_mib_torch": peak, "gpu": gpu_info(), "mux_audio": str(mux_src), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    write_json(out / f"{vid}_{a.tag}.manifest.json", wm)
    for p in work.glob("seg*_in.wav"): p.unlink(missing_ok=True)
    ok = rep["output_frame_check"]["pass"]
    print(f"== {vid} {a.tag}: frames {rep['assembly']['frames_written']} check {'PASS' if ok else 'FAIL'} LS {rep['latentsync_inference_total_s']}s total {times['total']}s peak {peak} MiB -> {clip}")
    return 0 if ok else 4


if __name__ == "__main__":
    raise SystemExit(main())
