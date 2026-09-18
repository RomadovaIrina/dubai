#!/usr/bin/env python3
"""Optimized face-aware LatentSync path (routing + batched execution) built ON TOP of face_aware_latentsync.py.

Baseline (face_aware_latentsync.py) stays untouched and selectable. This module adds:
  --face-router retinaface   fast routing decision per master frame with facexlib RetinaFace (scripts/optim/retinaface_router.py):
                             VALID_FACE -> LatentSync, SMALL_FACE / NO_FACE -> pass-through (reason recorded per frame),
                             LatentSync's own FaceDetector still verifies every segment right before inference
                             (detect_frames / verify_segment); disagreement -> split / pass-through, never "Face not detected" crash.
  --router-stride N          RetinaFace runs on every N-th frame of a gated run, the frames in between inherit the decision
                             (segments are >= --min-ls-frames anyway; LatentSync's detector still sees every frame).
  speech gate                frames outside VAD(original speech) U VAD(dubbed speech), widened by --gate-margin-s and merged
                             over gaps < --gate-merge-gap-s, are passed through: nobody speaks there in the source and the
                             dubbed track is silent there by construction (TTS never leaves its slot), so the original mouth
                             is already correct. Class NO_SPEECH, reason outside_speech_gate. Boundaries LATENT_SYNC <->
                             pass-through are blended over --crossfade-frames frames (whole-frame alpha blend == mouth-region
                             blend: outside the pasted-back mouth LatentSync frames equal the master frames).
  in-memory segments         master frames are decoded once; a LatentSync segment is the in-memory frame range + a slice of
                             the dubbed 16 kHz track. No per-segment ffmpeg cut / re-encode / re-read / write / mux round trips,
                             and LatentSync's FaceDetector runs ONCE per frame (verify pass), its results are replayed inside
                             LatentSync (latentsync_accel._ReplayDetector). --segment-files restores the old file-based path.
  --window-batch-size N      independent temporal windows batched into one UNet call (scripts/optim/latentsync_accel.py)
  --deepcache-interval N     DeepCache cache_interval (5 = default, A/B-validated on 04; 3 = previous frozen setting)
  --sdpa-backend auto|flash  PyTorch SDPA backend for the UNet (flash = force the fused FLASH_ATTENTION kernel)
  --compile-backend none|inductor|tensorrt   optional UNet compilation
  CodeFormer gate            OFF. The gate exists only as routing metadata: `codeformer_eligible` marks the segments
                             that a later, explicitly approved CodeFormer stage could touch (LATENT_SYNC segments only);
                             NO_FACE / SMALL_FACE / NO_SPEECH segments are never eligible. Pilot 0.7 found no strict-safe weight.

Audio/video semantics are those of the E2E runner: the master frames of a pass-through interval stay untouched, the
DUBBED audio track covers the whole timeline (it is muxed as the only audio stream). Frame order and count follow the
25 fps master exactly.

    source scripts/env.sh
    python scripts/pilot/face_aware_latentsync_accel.py test_videos/04.mp4 --audio /tmp/x/dubbed_16k.wav \
        --out-dir /tmp/dabai_optim/fa --window-batch-size 2 --face-router retinaface --speech-gate vad --router-stride 3
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import shutil
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "optim"))
from pilot_common import THIRD_PARTY, gpu_info, sh, write_json  # noqa: E402
import face_aware_latentsync as fa                               # noqa: E402

LS = THIRD_PARTY / "latentsync"
if str(LS) not in sys.path:
    sys.path.insert(0, str(LS))
FPS = fa.FPS
SR = 16000
GATE_DEFAULTS = {"margin_s": 0.24, "merge_gap_s": 0.6, "crossfade_frames": 4}


class AccelRunner:
    """Same call interface as face_aware_latentsync.LatentSyncRunner, executed through AccelLatentSync (+ run_frames)."""

    def __init__(self, window_batch_size: int = 1, deepcache: bool = True, compile_backend: str = "none", sdpa_backend: str = "auto",
                 steps: int = 20, guidance: float = 1.5, seed: int = 1247, deepcache_interval: int = 3):
        from resident_vram_probe import load_latentsync
        from latentsync_accel import AccelLatentSync
        t0 = time.perf_counter(); pipe = load_latentsync()
        self.acc = AccelLatentSync(pipe, window_batch_size=window_batch_size, deepcache=deepcache, compile_backend=compile_backend,
                                   sdpa_backend=sdpa_backend, steps=steps, guidance=guidance, seed=seed, deepcache_interval=deepcache_interval)
        self.load_s = round(time.perf_counter() - t0, 2)
        self.steps, self.guidance, self.seed = steps, guidance, seed
        self.config = {"window_batch_size": window_batch_size, "deepcache": deepcache, "deepcache_interval": deepcache_interval if deepcache else None,
                       "compile_backend": compile_backend, "sdpa_backend": sdpa_backend, "steps": steps, "guidance": guidance, "seed": seed,
                       "fp16": True, "stage2_512": True}
        self.calls: list[dict] = []

    def _record(self, st: dict, dt: float) -> None:
        self.calls.append({"frames": st["frames"], "windows": st["windows"], "groups": st["groups"], "unet_calls": st["unet_calls"], "seconds": dt,
                           "stages": st["seconds"], "detector": st.get("detector"), "detector_fallback_calls": st.get("detector_fallback_calls", 0)})

    def __call__(self, video: pathlib.Path, audio: pathlib.Path, out: pathlib.Path, temp: pathlib.Path) -> float:
        t0 = time.perf_counter()
        st = self.acc(str(video), str(audio), str(out), str(temp))
        dt = round(time.perf_counter() - t0, 2); self._record(st, dt)
        return dt

    def run_frames(self, frames_rgb: np.ndarray, audio_wav: pathlib.Path, detections: list | None = None) -> tuple[np.ndarray, float]:
        t0 = time.perf_counter()
        frames = self.acc.run_frames(frames_rgb, str(audio_wav), detections)
        dt = round(time.perf_counter() - t0, 2); self._record(self.acc.last_stats, dt)
        return frames, dt


def make_router(kind: str, min_w: int, min_h: int, conf: float):
    if kind == "retinaface":
        from retinaface_router import RetinaFaceRouter
        return RetinaFaceRouter(min_w, min_h, conf)
    return None


# --------------------------------------------------------------------------------------------------------------
# speech gate
# --------------------------------------------------------------------------------------------------------------
def silero_speech_intervals(wav16k: pathlib.Path) -> list[tuple[float, float]]:
    """Silero VAD speech intervals (seconds) of a 16 kHz mono wav; same call as the E2E runner's run_vad()."""
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
    torch.set_num_threads(int(os.environ.get("DUB_THREADS", "16")))
    model = load_silero_vad()
    audio = read_audio(str(wav16k), sampling_rate=SR)
    return [(round(float(t["start"]), 3), round(float(t["end"]), 3)) for t in get_speech_timestamps(audio, model, sampling_rate=SR, return_seconds=True)]


def build_speech_gate(interval_lists: list, duration: float, margin_s: float = GATE_DEFAULTS["margin_s"],
                      merge_gap_s: float = GATE_DEFAULTS["merge_gap_s"]) -> list[tuple[float, float]]:
    """Union of speech intervals (each item of interval_lists is a list of (start, end) or {start, end} dicts), widened by
    margin_s on both sides, merged where the gap is < merge_gap_s, clamped to [0, duration]."""
    ivs = []
    for lst in interval_lists:
        for it in lst or []:
            s, e = (it["start"], it["end"]) if isinstance(it, dict) else it
            s, e = max(0.0, float(s) - margin_s), min(float(duration), float(e) + margin_s)
            if e > s:
                ivs.append([s, e])
    ivs.sort()
    out: list[list[float]] = []
    for s, e in ivs:
        if out and s - out[-1][1] < merge_gap_s:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(round(s, 3), round(e, 3)) for s, e in out]


def gate_mask(gate: list[tuple[float, float]], n_frames: int) -> np.ndarray:
    """Per-master-frame boolean: frame i (time i/FPS) lies inside a gate interval."""
    m = np.zeros(n_frames, dtype=bool)
    for s, e in gate:
        a = max(0, int(math.floor(s * FPS))); b = min(n_frames, int(math.ceil(e * FPS)))
        if b > a:
            m[a:b] = True
    return m


def classify_master(master_frames: list, router, gate: np.ndarray | None, stride: int) -> tuple[list[dict], int]:
    """Router classification of in-memory BGR master frames. Frames outside the gate -> NO_SPEECH (no detection run).
    Inside a gated run every `stride`-th frame is detected, the others inherit the last decision."""
    stride = max(1, int(stride)); rows: list[dict] = []; detected = 0; last = None; since = stride
    for i, fr in enumerate(master_frames):
        if gate is not None and not gate[i]:
            rows.append({"frame": i, "cls": "NO_SPEECH", "reason": "outside_speech_gate", "bbox": None}); last = None; since = stride
            continue
        if last is None or since >= stride:
            r = router.classify_frame(fr); r["frame"] = i; detected += 1; last = r; since = 1
        else:
            r = dict(last); r["frame"] = i; r["inherited"] = True; since += 1
        rows.append(r)
    return rows, detected


def apply_gate_to_rows(rows: list[dict], gate: np.ndarray) -> None:
    for f in rows:
        if not gate[f["frame"]]:
            f["face_cls"] = f["cls"]; f["cls"] = "NO_SPEECH"; f["reason"] = "outside_speech_gate"


# --------------------------------------------------------------------------------------------------------------
# assembly with boundary crossfade
# --------------------------------------------------------------------------------------------------------------
def assemble_frames(master_frames: list, audio_mux: pathlib.Path, segs: list[dict], replacements: dict, out: pathlib.Path, work: pathlib.Path) -> dict:
    """Write master frames (BGR, in memory) in order, substituting `replacements[seg index]` (BGR frames) for LATENT_SYNC
    segments; alpha-blend the first/last seg['crossfade']['in'/'out'] replaced frames with the master; mux the dubbed track."""
    import cv2, imageio
    tmp = work / (out.stem + "_video.mp4")
    writer = imageio.get_writer(str(tmp), fps=FPS, codec="libx264", macro_block_size=None, ffmpeg_params=["-crf", "13"], ffmpeg_log_level="error")
    written = 0; padded: dict = {}; blended = 0; pos = 0
    for s in segs:
        rep = replacements.get(s["index"]); xf = s.get("crossfade") or {}
        for k in range(s["frames"]):
            if pos >= len(master_frames):
                break
            fr = master_frames[pos]; pos += 1
            if rep is not None and k < len(rep):
                a = 1.0
                if xf.get("in") and k < xf["in"]:
                    a = (k + 1) / (xf["in"] + 1)
                if xf.get("out") and k >= s["frames"] - xf["out"]:
                    a = min(a, (s["frames"] - k) / (xf["out"] + 1))
                if a < 1.0:
                    fr = cv2.addWeighted(rep[k], a, fr, 1.0 - a, 0.0); blended += 1
                else:
                    fr = rep[k]
            elif rep is not None:
                padded[s["index"]] = padded.get(s["index"], 0) + 1   # LatentSync returned fewer frames: keep original
            writer.append_data(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); written += 1
    writer.close()
    fa.run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", str(tmp), "-i", str(audio_mux), "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "copy", str(out)])
    tmp.unlink(missing_ok=True)
    return {"frames_written": written, "master_frames": len(master_frames), "padded_with_original": padded, "crossfaded_frames": blended}


def check_output_frames(path: pathlib.Path, expected: int) -> dict:
    """Decode every output frame: count must match, no undecodable frames, no blank (constant) frames."""
    import cv2
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


# --------------------------------------------------------------------------------------------------------------
def process_video(src: pathlib.Path, work: pathlib.Path, audio16k: pathlib.Path, audio_mux: pathlib.Path, out: pathlib.Path, *,
                  router, runner: AccelRunner | None, min_ls_frames: int = 25, max_verify_depth: int = 3, codeformer: str = "off",
                  times: dict | None = None, log=print, speech_gate: list | None = None,
                  crossfade_frames: int = GATE_DEFAULTS["crossfade_frames"], router_stride: int = 1, in_memory: bool = True) -> dict:
    """Face-aware LatentSync on `src` driven by `audio16k` (the dubbed track); `audio_mux` (AAC of the same track) is muxed.
    `speech_gate`: list of (start_s, end_s) intervals (build_speech_gate); frames outside them are passed through."""
    import cv2
    import soundfile as sf
    times = times if times is not None else {}
    if codeformer != "off":
        raise NotImplementedError("CodeFormer gate: only LATENT_SYNC segments would be eligible; CodeFormer is disabled for pilot 0.5")
    t0 = time.perf_counter(); master, _own = fa.make_master(src, work); times["ls_master_25fps"] = round(time.perf_counter() - t0, 3)
    t0 = time.perf_counter(); master_frames = fa.read_frames(master); times["ls_master_decode"] = round(time.perf_counter() - t0, 3)
    n = len(master_frames)
    gate = gate_mask(speech_gate, n) if speech_gate is not None else None

    t0 = time.perf_counter()
    if router is not None:
        frames, n_detected = classify_master(master_frames, router, gate, router_stride); router_name = "retinaface"
    else:
        frames = fa.classify_frames(master); router_name = "insightface(latentsync)"; n_detected = len(frames)
        if gate is not None:
            apply_gate_to_rows(frames, gate)
    times["face_classify"] = round(time.perf_counter() - t0, 3)
    segs = fa.segment(frames, min_ls_frames)
    classes = ("VALID_FACE", "SMALL_FACE", "NO_FACE", "NO_SPEECH")
    counts = {c: sum(1 for f in frames if f["cls"] == c) for c in classes}
    reasons: dict = {}
    for f in frames:
        if f.get("reason"):
            reasons[f["reason"]] = reasons.get(f["reason"], 0) + 1
    gate_info = None
    if gate is not None:
        gated_in = int(gate.sum())
        gate_info = {"enabled": True, "intervals": [list(g) for g in speech_gate], "gated_in_frames": gated_in, "gated_out_frames": n - gated_in,
                     "gated_in_seconds": round(gated_in / FPS, 3), "gated_out_seconds": round((n - gated_in) / FPS, 3)}
    gate_note = "" if gate_info is None else f" gate_out={gate_info['gated_out_seconds']}s"
    log(f"  [lipsync/{router_name}] {n} frames@25fps {counts} skip_reasons={reasons} segments={len(segs)} "
        f"router_detections={n_detected} (stride {router_stride}){gate_note}")

    audio = None
    if in_memory:
        audio, sr = sf.read(str(audio16k), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != SR:
            raise RuntimeError(f"driving audio must be {SR} Hz mono, got {sr}")
    replacements, ls_runs, verify_info = {}, [], []
    t_verify = 0.0
    queue = [s for s in segs if s["action"] == "LATENT_SYNC"]
    t0 = time.perf_counter()
    while queue:
        s = queue.pop(0)
        if runner is None:
            s["action"] = "PASS_THROUGH_NO_RUNNER"; continue
        try:
            if in_memory:
                seg_rgb = np.stack([cv2.cvtColor(master_frames[i], cv2.COLOR_BGR2RGB) for i in range(s["start_frame"], s["end_frame"])])
                a0 = int(round(s["start_frame"] / FPS * SR)); a1 = a0 + int(round(s["frames"] / FPS * SR))
                seg_audio = audio[a0:a1]
                if len(seg_audio) < a1 - a0:
                    seg_audio = np.concatenate([seg_audio, np.zeros((a1 - a0) - len(seg_audio), dtype=np.float32)])
                wav = work / f"seg{s['index']:03d}_in.wav"; sf.write(str(wav), seg_audio, SR, subtype="PCM_16")   # audio = DUBBED track slice
                dets = None
                if s.get("verify_depth", 0) < max_verify_depth:
                    tv = time.perf_counter(); dets = fa.detect_frames(seg_rgb); t_verify += time.perf_counter() - tv
                    ok = [d[0] is not None for d in dets]; s["verify_depth"] = s.get("verify_depth", 0) + 1   # LatentSync's own detector = safety check
                    if not all(ok):
                        subs = fa.split_by_mask(s, ok, min_ls_frames, next_index=len(segs))
                        for x in subs:
                            if x.get("note"):
                                x["note"] = "rejected by LatentSync FaceDetector on the master frames"
                        verify_info.append({"segment": s["index"], "rejected_frames": ok.count(False), "split_into": [x["index"] for x in subs]})
                        log(f"  [lipsync] seg {s['index']}: {ok.count(False)}/{len(ok)} frames rejected by LatentSync detector -> split {len(subs)}")
                        s["action"] = "SPLIT"; s["split_into"] = [x["index"] for x in subs]
                        for x in subs:
                            x["verify_depth"] = s["verify_depth"]; segs.append(x)
                            if x["action"] == "LATENT_SYNC":
                                queue.append(x)
                        wav.unlink(missing_ok=True); continue
                frs_rgb, dt = runner.run_frames(seg_rgb, wav, dets)
                frs = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frs_rgb]
                wav.unlink(missing_ok=True); del seg_rgb, frs_rgb
            else:
                v, au = fa.cut_segment(master, audio16k, s, work, f"ls{s['index']:03d}")          # audio = DUBBED track slice
                if s.get("verify_depth", 0) < max_verify_depth:
                    tv = time.perf_counter(); ok = fa.verify_segment(v); t_verify += time.perf_counter() - tv
                    s["verify_depth"] = s.get("verify_depth", 0) + 1                                  # LatentSync's own detector = safety check
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
                frs = fa.read_frames(o)
                for p in (v, au, o, v.with_name(v.stem + "_lsview.mp4")):
                    if p.exists():
                        p.unlink()                                                                    # runtime media freed as we go
            replacements[s["index"]] = frs
            s["latentsync"] = {"seconds": dt, "frames_in": s["frames"], "frames_out": len(frs)}
            ls_runs.append(dt); log(f"  [lipsync] seg {s['index']} {s['start_s']}-{s['end_s']}s LatentSync {dt}s -> {len(frs)} frames")
        except Exception as e:                                                                    # "Face not detected" etc. -> pass-through
            s["action"] = "PASS_THROUGH_LS_FAILED"; s["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            log(f"  [lipsync] seg {s['index']} LatentSync FAILED -> pass-through: {s['error']}")
    times["latentsync"] = round(time.perf_counter() - t0, 3)
    times["latentsync_verify"] = round(t_verify, 3)
    segs[:] = sorted([s for s in segs if s["action"] != "SPLIT"], key=lambda s: s["start_frame"])
    xf = max(0, int(crossfade_frames))
    for j, s in enumerate(segs):
        s["codeformer_eligible"] = s["action"] == "LATENT_SYNC"     # gate metadata only; CodeFormer is not run
        if s["action"] != "LATENT_SYNC" or xf == 0:
            continue
        k = min(xf, s["frames"] // 3)
        s["crossfade"] = {"in": k if j > 0 and segs[j - 1]["action"] != "LATENT_SYNC" else 0,
                          "out": k if j + 1 < len(segs) and segs[j + 1]["action"] != "LATENT_SYNC" else 0}
    t0 = time.perf_counter(); asm = assemble_frames(master_frames, audio_mux, segs, replacements, out, work); times["final_assembly"] = round(time.perf_counter() - t0, 3)
    del master_frames, replacements
    frame_check = check_output_frames(out, n)
    by_action: dict[str, dict] = {}
    for s in segs:
        d = by_action.setdefault(s["action"], {"segments": 0, "frames": 0, "seconds": 0.0})
        d["segments"] += 1; d["frames"] += s["frames"]; d["seconds"] = round(d["seconds"] + s["frames"] / FPS, 3)
    lipsynced = sum(s["frames"] for s in segs if s["action"] == "LATENT_SYNC")
    return {"router": router_name, "router_params": router.params() if router is not None else {"latentsync_filter": "w>=50 h>=80 score>=0.5"},
            "router_stride": int(router_stride), "router_detections": n_detected, "speech_gate": gate_info or {"enabled": False},
            "execution": "in_memory_segments (detector shared with LatentSync)" if in_memory else "segment_files (legacy)",
            "crossfade_frames": xf, "master_fps": FPS, "master_frames": n, "frame_classes": counts, "skip_reasons": reasons, "by_action": by_action,
            "latentsync_seconds_of_video": round(lipsynced / FPS, 3), "pass_through_seconds_of_video": round((n - lipsynced) / FPS, 3),
            "ls_fraction": round(lipsynced / max(n, 1), 4), "latentsync_inference_total_s": round(sum(ls_runs), 2), "verify_seconds": round(t_verify, 2),
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
    ap.add_argument("--router-stride", type=int, default=1)
    ap.add_argument("--speech-gate", choices=["off", "vad"], default="off", help="vad = Silero VAD(video's own audio) U VAD(--audio)")
    ap.add_argument("--gate-margin-s", type=float, default=GATE_DEFAULTS["margin_s"])
    ap.add_argument("--gate-merge-gap-s", type=float, default=GATE_DEFAULTS["merge_gap_s"])
    ap.add_argument("--crossfade-frames", type=int, default=GATE_DEFAULTS["crossfade_frames"])
    ap.add_argument("--window-batch-size", type=int, default=1)
    ap.add_argument("--deepcache-interval", type=int, default=5, help="5 = A/B-validated default (reports/pilot/optim/deepcache_interval_ab_04.md), 3 = previous setting")
    ap.add_argument("--compile-backend", choices=["none", "inductor", "tensorrt"], default="none")
    ap.add_argument("--sdpa-backend", choices=["auto", "flash", "efficient", "math"], default="auto")
    ap.add_argument("--no-deepcache", action="store_true")
    ap.add_argument("--min-ls-frames", type=int, default=25)
    ap.add_argument("--max-verify-depth", type=int, default=3)
    ap.add_argument("--segment-files", action="store_true", help="legacy path: per-segment mp4 cut / verify re-encode / LatentSync file I/O")
    ap.add_argument("--no-latentsync", action="store_true", help="routing + assembly only")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    src = pathlib.Path(a.video).resolve(); out_dir = pathlib.Path(a.out_dir); work = out_dir / f"work_{src.stem}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    import subprocess
    own16k = work / "own_audio_16k.wav"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-nostdin", "-i", str(src), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(own16k)], check=True)
    audio16k = pathlib.Path(a.audio).resolve() if a.audio else own16k
    audio_mux = work / "audio_aac.m4a"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-nostdin", "-i", str(audio16k), "-c:a", "aac", "-b:a", "192k", str(audio_mux)], check=True)
    gate = None
    if a.speech_gate == "vad":
        duration = float(sh(f'ffprobe -v error -show_entries format=duration -of csv=p=0 "{src}"') or 0)
        lists = [silero_speech_intervals(own16k)] + ([silero_speech_intervals(audio16k)] if audio16k != own16k else [])
        gate = build_speech_gate(lists, duration, a.gate_margin_s, a.gate_merge_gap_s)
        print(f"speech gate: {len(gate)} intervals, {round(sum(e - s for s, e in gate), 2)}s of {duration}s")
    router = make_router(a.face_router, a.face_min_w, a.face_min_h, a.retina_conf)
    runner = None if a.no_latentsync else AccelRunner(a.window_batch_size, not a.no_deepcache, a.compile_backend, a.sdpa_backend,
                                                      deepcache_interval=a.deepcache_interval)
    times: dict = {}
    out = out_dir / f"facesync_{src.stem}.mp4"
    t0 = time.perf_counter()
    rep = process_video(src, work, audio16k, audio_mux, out, router=router, runner=runner, min_ls_frames=a.min_ls_frames,
                        max_verify_depth=a.max_verify_depth, times=times, speech_gate=gate, crossfade_frames=a.crossfade_frames,
                        router_stride=a.router_stride, in_memory=not a.segment_files)
    rep.update(video=str(src), output=str(out), output_meta=fa.probe(out), stage_seconds=times, wall_s_total=round(time.perf_counter() - t0, 2),
               model_load_s=runner.load_s if runner else None, gpu=gpu_info(), at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if a.json:
        write_json(pathlib.Path(a.json), rep)
    print(json.dumps({k: rep[k] for k in ("frame_classes", "skip_reasons", "by_action", "speech_gate", "latentsync_inference_total_s",
                                           "output_frame_check", "stage_seconds", "wall_s_total")}, indent=1, default=str))
    return 0 if rep["output_frame_check"]["pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
