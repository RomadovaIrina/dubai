#!/usr/bin/env python3
"""Optimized face-aware LatentSync path (routing + batched execution) built ON TOP of face_aware_latentsync.py.

Baseline (face_aware_latentsync.py) stays untouched and selectable. This module adds:
  --face-router retinaface   fast routing decision per master frame with facexlib RetinaFace (scripts/optim/retinaface_router.py):
                             VALID_FACE -> LatentSync, SMALL_FACE / NO_FACE -> pass-through (reason recorded per frame),
                             LatentSync's own FaceDetector still re-verifies every cut segment right before inference
                             (verify_segment); disagreement -> split / pass-through, never "Face not detected" crash.
  --window-batch-size N      independent temporal windows batched into one UNet call (scripts/optim/latentsync_accel.py)
  --sdpa-backend auto|flash  PyTorch SDPA backend for the UNet (flash = force the fused FLASH_ATTENTION kernel)
  --compile-backend none|inductor|tensorrt   optional UNet compilation
  CodeFormer gate            OFF. The gate exists only as routing metadata: `codeformer_eligible` marks the segments
                             that a later, explicitly approved CodeFormer stage could touch (LATENT_SYNC segments only);
                             NO_FACE / SMALL_FACE segments are never eligible. Pilot 0.7 found no strict-safe weight.

Audio/video semantics are those of the E2E runner: the master frames of a pass-through interval stay untouched, the
DUBBED audio track covers the whole timeline (it is muxed as the only audio stream). Frame order and count follow the
25 fps master exactly (face_aware_latentsync.assemble).

    source scripts/env.sh
    python scripts/pilot/face_aware_latentsync_accel.py test_videos/04.mp4 --audio /tmp/x/dubbed_16k.wav \
        --out-dir /tmp/dabai_optim/fa --window-batch-size 4 --face-router retinaface
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "optim"))
from pilot_common import THIRD_PARTY, gpu_info, sh, write_json  # noqa: E402
import face_aware_latentsync as fa                               # noqa: E402

LS = THIRD_PARTY / "latentsync"
if str(LS) not in sys.path:
    sys.path.insert(0, str(LS))


class AccelRunner:
    """Same call interface as face_aware_latentsync.LatentSyncRunner, executed through AccelLatentSync."""

    def __init__(self, window_batch_size: int = 1, deepcache: bool = True, compile_backend: str = "none", sdpa_backend: str = "auto",
                 steps: int = 20, guidance: float = 1.5, seed: int = 1247):
        from resident_vram_probe import load_latentsync
        from latentsync_accel import AccelLatentSync
        t0 = time.perf_counter(); pipe = load_latentsync()
        self.acc = AccelLatentSync(pipe, window_batch_size=window_batch_size, deepcache=deepcache, compile_backend=compile_backend,
                                   sdpa_backend=sdpa_backend, steps=steps, guidance=guidance, seed=seed)
        self.load_s = round(time.perf_counter() - t0, 2)
        self.steps, self.guidance, self.seed = steps, guidance, seed
        self.config = {"window_batch_size": window_batch_size, "deepcache": deepcache, "compile_backend": compile_backend,
                       "sdpa_backend": sdpa_backend, "steps": steps, "guidance": guidance, "seed": seed, "fp16": True, "stage2_512": True}
        self.calls: list[dict] = []

    def __call__(self, video: pathlib.Path, audio: pathlib.Path, out: pathlib.Path, temp: pathlib.Path) -> float:
        t0 = time.perf_counter()
        st = self.acc(str(video), str(audio), str(out), str(temp))
        dt = round(time.perf_counter() - t0, 2)
        self.calls.append({"frames": st["frames"], "windows": st["windows"], "groups": st["groups"], "unet_calls": st["unet_calls"], "seconds": dt, "stages": st["seconds"]})
        return dt


def make_router(kind: str, min_w: int, min_h: int, conf: float):
    if kind == "retinaface":
        from retinaface_router import RetinaFaceRouter
        return RetinaFaceRouter(min_w, min_h, conf)
    return None


def check_output_frames(path: pathlib.Path, expected: int) -> dict:
    """Decode every output frame: count must match, no undecodable frames, no blank (constant) frames."""
    import cv2
    import numpy as np
    cap = cv2.VideoCapture(str(path)); n = 0; blank = 0; bad = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        if fr is None or fr.size == 0:
            bad += 1; continue
        if not np.isfinite(fr.astype(np.float32)).all():
            bad += 1
        elif float(fr.std()) < 1e-3:
            blank += 1
    cap.release()
    return {"decoded_frames": n, "expected_frames": expected, "blank_frames": blank, "undecodable_frames": bad,
            "pass": n == expected and bad == 0 and blank == 0}


def process_video(src: pathlib.Path, work: pathlib.Path, audio16k: pathlib.Path, audio_mux: pathlib.Path, out: pathlib.Path, *,
                  router, runner: AccelRunner | None, min_ls_frames: int = 25, max_verify_depth: int = 3, codeformer: str = "off",
                  times: dict | None = None, log=print) -> dict:
    """Face-aware LatentSync on `src` driven by `audio16k` (the dubbed track); `audio_mux` (AAC of the same track) is muxed."""
    times = times if times is not None else {}
    if codeformer != "off":
        raise NotImplementedError("CodeFormer gate: only LATENT_SYNC segments would be eligible; CodeFormer is disabled for pilot 0.5")
    t0 = time.perf_counter(); master, _own = fa.make_master(src, work); times["ls_master_25fps"] = round(time.perf_counter() - t0, 3)
    t0 = time.perf_counter()
    if router is not None:
        frames = router.classify_video(master); router_name = "retinaface"
    else:
        frames = fa.classify_frames(master); router_name = "insightface(latentsync)"
    times["face_classify"] = round(time.perf_counter() - t0, 3)
    segs = fa.segment(frames, min_ls_frames)
    counts = {c: sum(1 for f in frames if f["cls"] == c) for c in ("VALID_FACE", "SMALL_FACE", "NO_FACE")}
    reasons = {}
    for f in frames:
        if f.get("reason"):
            reasons[f["reason"]] = reasons.get(f["reason"], 0) + 1
    log(f"  [lipsync/{router_name}] {len(frames)} frames@25fps {counts} skip_reasons={reasons} segments={len(segs)}")
    replacements, ls_runs, verify_info = {}, [], []
    queue = [s for s in segs if s["action"] == "LATENT_SYNC"]
    t0 = time.perf_counter()
    while queue:
        s = queue.pop(0)
        if runner is None:
            s["action"] = "PASS_THROUGH_NO_RUNNER"; continue
        try:
            v, au = fa.cut_segment(master, audio16k, s, work, f"ls{s['index']:03d}")          # audio = DUBBED track slice
            if s.get("verify_depth", 0) < max_verify_depth:
                ok = fa.verify_segment(v); s["verify_depth"] = s.get("verify_depth", 0) + 1       # LatentSync's own detector = safety check
                if not all(ok):
                    subs = fa.split_by_mask(s, ok, min_ls_frames, next_index=len(segs))
                    verify_info.append({"segment": s["index"], "rejected_frames": ok.count(False), "split_into": [x["index"] for x in subs]})
                    log(f"  [lipsync] seg {s['index']}: {ok.count(False)}/{len(ok)} frames rejected by LatentSync detector -> split {len(subs)}")
                    s["action"] = "SPLIT"; s["split_into"] = [x["index"] for x in subs]
                    for x in subs:
                        x["verify_depth"] = s["verify_depth"]; segs.append(x)
                        if x["action"] == "LATENT_SYNC":
                            queue.append(x)
                    continue
            o = work / f"seg{s['index']:03d}_ls_out.mp4"
            dt = runner(v, au, o, work / f"ls_temp_{s['index']}")
            frs = fa.read_frames(o); replacements[s["index"]] = frs
            s["latentsync"] = {"seconds": dt, "frames_in": s["frames"], "frames_out": len(frs)}
            ls_runs.append(dt); log(f"  [lipsync] seg {s['index']} {s['start_s']}-{s['end_s']}s LatentSync {dt}s -> {len(frs)} frames")
            for p in (v, au, o, v.with_name(v.stem + "_lsview.mp4")):
                if p.exists():
                    p.unlink()                                                                    # runtime media freed as we go
        except Exception as e:                                                                    # "Face not detected" etc. -> pass-through
            s["action"] = "PASS_THROUGH_LS_FAILED"; s["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            log(f"  [lipsync] seg {s['index']} LatentSync FAILED -> pass-through: {s['error']}")
    times["latentsync"] = round(time.perf_counter() - t0, 3)
    segs[:] = sorted([s for s in segs if s["action"] != "SPLIT"], key=lambda s: s["start_frame"])
    for s in segs:
        s["codeformer_eligible"] = s["action"] == "LATENT_SYNC"     # gate metadata only; CodeFormer is not run
    t0 = time.perf_counter(); asm = fa.assemble(master, audio_mux, segs, replacements, out, work); times["final_assembly"] = round(time.perf_counter() - t0, 3)
    frame_check = check_output_frames(out, len(frames))
    by_action: dict[str, dict] = {}
    for s in segs:
        d = by_action.setdefault(s["action"], {"segments": 0, "frames": 0, "seconds": 0.0})
        d["segments"] += 1; d["frames"] += s["frames"]; d["seconds"] = round(d["seconds"] + s["frames"] / fa.FPS, 3)
    lipsynced = sum(s["frames"] for s in segs if s["action"] == "LATENT_SYNC")
    return {"router": router_name, "router_params": router.params() if router is not None else fa.LS and {"latentsync_filter": "w>=50 h>=80 score>=0.5"},
            "master_fps": fa.FPS, "master_frames": len(frames), "frame_classes": counts, "skip_reasons": reasons, "by_action": by_action,
            "latentsync_seconds_of_video": round(lipsynced / fa.FPS, 3), "pass_through_seconds_of_video": round((len(frames) - lipsynced) / fa.FPS, 3),
            "ls_fraction": round(lipsynced / max(len(frames), 1), 4), "latentsync_inference_total_s": round(sum(ls_runs), 2),
            "latentsync_calls": len(ls_runs), "verify_splits": verify_info, "assembly": asm, "output_frame_check": frame_check,
            "codeformer": "OFF (gate metadata only: codeformer_eligible on LATENT_SYNC segments)",
            "codeformer_eligible_segments": [s["index"] for s in segs if s["codeformer_eligible"]],
            "runner": runner.config if runner is not None else None, "runner_calls": runner.calls if runner is not None else None,
            "segments": [{k: v for k, v in s.items() if k != "face_bbox_stats"} for s in segs]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--audio", default=None, help="16 kHz mono wav driving the lipsync (dubbed track); default: the video's own audio")
    ap.add_argument("--out-dir", default="/tmp/dabai_optim/fa")
    ap.add_argument("--face-router", choices=["retinaface", "insightface"], default="retinaface")
    ap.add_argument("--face-min-w", type=int, default=50)
    ap.add_argument("--face-min-h", type=int, default=80)
    ap.add_argument("--retina-conf", type=float, default=0.8)
    ap.add_argument("--window-batch-size", type=int, default=1)
    ap.add_argument("--compile-backend", choices=["none", "inductor", "tensorrt"], default="none")
    ap.add_argument("--sdpa-backend", choices=["auto", "flash", "efficient", "math"], default="auto")
    ap.add_argument("--no-deepcache", action="store_true")
    ap.add_argument("--min-ls-frames", type=int, default=25)
    ap.add_argument("--max-verify-depth", type=int, default=3)
    ap.add_argument("--no-latentsync", action="store_true", help="routing + assembly only")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    src = pathlib.Path(a.video).resolve(); out_dir = pathlib.Path(a.out_dir); work = out_dir / f"work_{src.stem}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    import subprocess
    if a.audio:
        audio16k = pathlib.Path(a.audio).resolve()
    else:
        audio16k = work / "own_audio_16k.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-nostdin", "-i", str(src), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(audio16k)], check=True)
    audio_mux = work / "audio_aac.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-nostdin", "-i", str(audio16k), "-c:a", "aac", "-b:a", "192k", str(audio_mux)], check=True)
    router = make_router(a.face_router, a.face_min_w, a.face_min_h, a.retina_conf)
    runner = None if a.no_latentsync else AccelRunner(a.window_batch_size, not a.no_deepcache, a.compile_backend, a.sdpa_backend)
    times = {}
    out = out_dir / f"facesync_{src.stem}.mp4"
    t0 = time.perf_counter()
    rep = process_video(src, work, audio16k, audio_mux, out, router=router, runner=runner, min_ls_frames=a.min_ls_frames,
                        max_verify_depth=a.max_verify_depth, times=times)
    rep.update(video=str(src), output=str(out), output_meta=fa.probe(out), stage_seconds=times, wall_s_total=round(time.perf_counter() - t0, 2),
               model_load_s=runner.load_s if runner else None, gpu=gpu_info(), at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if a.json:
        write_json(pathlib.Path(a.json), rep)
    print(json.dumps({k: rep[k] for k in ("frame_classes", "skip_reasons", "by_action", "latentsync_inference_total_s", "output_frame_check", "stage_seconds", "wall_s_total")}, indent=1))
    return 0 if rep["output_frame_check"]["pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
