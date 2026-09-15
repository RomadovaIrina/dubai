#!/usr/bin/env python3
"""Pilot 0.8 routing fix — face-aware preprocessing AROUND LatentSync (LatentSync internals untouched).

Per input video:
  1. master = input converted to 25 fps CFR exactly the way LatentSync does it (ffmpeg -r 25 -crf 18), own audio extracted;
  2. every master frame is classified with LatentSync's own FaceDetector (+ raw insightface for the SMALL/NO split):
       VALID_FACE  bbox passes LatentSync's own filter (w>=50, h>=80 px, 0.2<w/h<1.5, score>=0.5); each cut segment is
                   re-verified with that detector and split at rejected frames before LatentSync runs (--max-verify-depth)
       SMALL_FACE  a face is detected but does not pass the filter (LatentSync would raise "Face not detected")
       NO_FACE     no face at all
  3. frames are grouped into segments; VALID_FACE segments >= --min-ls-frames go through LatentSync (segment video +
     matching own-audio slice), everything else is passed through untouched;
  4. output is re-assembled frame by frame (same frame count / order as the master), original audio track muxed back;
  5. optional --crop-experiment: SMALL_FACE segments are cropped around the face (padding), upscaled so the face is big
     enough for LatentSync, lip-synced, downscaled and pasted back -> separate *_smallface_crop.mp4 + ROI preview;
  6. SyncNet (upstream evaluator) only on segments with action=LATENT_SYNC (cut from the final output); N/A elsewhere;
  7. manifest JSON with start_s, end_s, classification, action, bbox stats per segment.

    source scripts/env.sh
    python scripts/pilot/face_aware_latentsync.py test_videos/05.mp4 --out-dir /tmp/dabai_pilot_08fa --crop-experiment
"""
from __future__ import annotations
import argparse, json, math, os, pathlib, shutil, statistics, subprocess, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT, THIRD_PARTY, gpu_info, write_json, sh
from benchmark_06_codeformer import probe

LS = THIRD_PARTY / "latentsync"
FPS = 25
# LatentSync filter is w>=50, h>=80, 0.2<w/h<1.5, score>=0.5 (latentsync/utils/face_detector.py). We use exactly
# these thresholds for classification; robustness against re-encode flips comes from verify_segment(): every cut
# segment is re-checked with LatentSync's FaceDetector on the exact frames LatentSync will read, and split at any
# frame that fails, before LatentSync is called.
VALID_W, VALID_H, VALID_SCORE = 50, 80, 0.5
_DET = None


def face_detector():
    global _DET
    if _DET is None:
        old = os.getcwd(); os.chdir(LS)
        try:
            from latentsync.utils.face_detector import FaceDetector
            _DET = FaceDetector(device="cuda")
        finally:
            os.chdir(old)
    return _DET


def run(cmd: list[str], **kw) -> None:
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, **kw)


def ffprobe_frames(p: pathlib.Path) -> int:
    v = sh(f'ffprobe -v error -select_streams v:0 -count_frames -show_entries stream=nb_read_frames -of csv=p=0 "{p}"', timeout=600)
    return int(v.strip() or 0)


def make_master(src: pathlib.Path, work: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path]:
    master = work / "master_25fps.mp4"; wav = work / "own_audio_16k.wav"
    if not master.exists():
        run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", str(src), "-r", str(FPS), "-crf", "18", "-an", str(master)])
    if not wav.exists():
        run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", str(src), "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(wav)])
    return master, wav


def classify_frames(master: pathlib.Path) -> list[dict]:
    import cv2
    det = face_detector()
    cap = cv2.VideoCapture(str(master)); rows = []; i = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        faces = det.app.get(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))   # LatentSync's read_video feeds RGB to the detector
        best = None
        for f in faces:
            x1, y1, x2, y2 = f.bbox.astype(int).tolist(); w, h = x2 - x1, y2 - y1
            if best is None or w * h > best["w"] * best["h"]:
                best = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "w": w, "h": h, "score": round(float(f.det_score), 3)}
        if best is None:
            cls = "NO_FACE"
        elif best["w"] >= VALID_W and best["h"] >= VALID_H and 0.2 < best["w"] / max(best["h"], 1) < 1.5 and best["score"] >= VALID_SCORE:
            cls = "VALID_FACE"
        else:
            cls = "SMALL_FACE"
        rows.append({"frame": i, "cls": cls, "bbox": best}); i += 1
    cap.release()
    return rows


def segment(frames: list[dict], min_ls_frames: int) -> list[dict]:
    segs = []
    for f in frames:
        if segs and segs[-1]["cls"] == f["cls"]:
            segs[-1]["end"] = f["frame"] + 1
        else:
            segs.append({"cls": f["cls"], "start": f["frame"], "end": f["frame"] + 1})
    out = []
    for s in segs:
        n = s["end"] - s["start"]
        bb = [frames[i]["bbox"] for i in range(s["start"], s["end"]) if frames[i]["bbox"]]
        stats = None
        if bb:
            stats = {"n_detected": len(bb), "w_median": statistics.median(b["w"] for b in bb), "h_median": statistics.median(b["h"] for b in bb),
                     "h_min": min(b["h"] for b in bb), "h_max": max(b["h"] for b in bb), "score_mean": round(statistics.mean(b["score"] for b in bb), 3),
                     "cx_median": statistics.median((b["x1"] + b["x2"]) / 2 for b in bb), "cy_median": statistics.median((b["y1"] + b["y2"]) / 2 for b in bb),
                     "x1_min": min(b["x1"] for b in bb), "y1_min": min(b["y1"] for b in bb), "x2_max": max(b["x2"] for b in bb), "y2_max": max(b["y2"] for b in bb)}
        if s["cls"] == "VALID_FACE":
            action = "LATENT_SYNC" if n >= min_ls_frames else "PASS_THROUGH_SHORT_VALID"
        else:
            action = "PASS_THROUGH"
        out.append({"index": len(out), "start_frame": s["start"], "end_frame": s["end"], "frames": n,
                    "start_s": round(s["start"] / FPS, 3), "end_s": round(s["end"] / FPS, 3),
                    "classification": s["cls"], "action": action, "face_bbox_stats": stats})
    return out


def cut_segment(master: pathlib.Path, wav: pathlib.Path, seg: dict, work: pathlib.Path, tag: str) -> tuple[pathlib.Path, pathlib.Path]:
    v = work / f"seg{seg['index']:03d}_{tag}_in.mp4"; a = work / f"seg{seg['index']:03d}_{tag}_in.wav"
    n = seg["frames"]; t0 = seg["start_frame"] / FPS
    run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", str(master), "-vf", f"select=between(n\\,{seg['start_frame']}\\,{seg['end_frame']-1}),setpts=N/{FPS}/TB",
         "-r", str(FPS), "-frames:v", str(n), "-crf", "18", "-an", str(v)])
    run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", str(wav), "-ss", f"{t0:.4f}", "-t", f"{n / FPS:.4f}", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", str(a)])
    got = ffprobe_frames(v)
    if got != n:
        raise RuntimeError(f"segment cut produced {got} frames, expected {n}")
    return v, a


def verify_segment(video: pathlib.Path) -> list[bool]:
    """Run LatentSync's FaceDetector (with its own filters) on the exact frames LatentSync will see: its read_video()
    re-encodes the input once more with `ffmpeg -r 25 -crf 18`, so we replicate that encode (x264 is deterministic) and
    detect on the result. True = LatentSync accepts the frame."""
    import cv2
    ls_view = video.with_name(video.stem + "_lsview.mp4")
    run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", str(video), "-r", str(FPS), "-crf", "18", str(ls_view)])
    det = face_detector(); cap = cv2.VideoCapture(str(ls_view)); ok = []
    while True:
        r, fr = cap.read()
        if not r:
            break
        bbox, _ = det(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); ok.append(bbox is not None)   # RGB, as in LatentSync
    cap.release(); return ok


def split_by_mask(seg: dict, ok: list[bool], min_frames: int, next_index: int) -> list[dict]:
    """Split a segment into runs of accepted / rejected frames (relative to the master timeline)."""
    out = []; i = 0
    while i < len(ok):
        j = i
        while j < len(ok) and ok[j] == ok[i]:
            j += 1
        n = j - i; good = ok[i]
        out.append({"index": next_index + len(out), "start_frame": seg["start_frame"] + i, "end_frame": seg["start_frame"] + j, "frames": n,
                    "start_s": round((seg["start_frame"] + i) / FPS, 3), "end_s": round((seg["start_frame"] + j) / FPS, 3),
                    "classification": "VALID_FACE" if good else "SMALL_FACE", "face_bbox_stats": seg.get("face_bbox_stats"),
                    "action": ("LATENT_SYNC" if n >= min_frames else "PASS_THROUGH_SHORT_VALID") if good else "PASS_THROUGH",
                    "split_from": seg["index"], "note": None if good else "rejected by LatentSync FaceDetector on the re-encoded segment"})
        i = j
    return out


class LatentSyncRunner:
    def __init__(self, steps=20, guidance=1.5, seed=1247):
        import torch
        from omegaconf import OmegaConf
        from DeepCache import DeepCacheSDHelper
        from resident_vram_probe import load_latentsync
        self.torch = torch; self.cfg = OmegaConf.load(LS / "configs/unet/stage2_512.yaml")
        self.steps, self.guidance, self.seed = steps, guidance, seed
        t0 = time.perf_counter(); self.pipe = load_latentsync()
        h = DeepCacheSDHelper(pipe=self.pipe); h.set_params(cache_interval=3, cache_branch_id=0); h.enable()
        self.load_s = round(time.perf_counter() - t0, 2)

    def __call__(self, video: pathlib.Path, audio: pathlib.Path, out: pathlib.Path, temp: pathlib.Path) -> float:
        from accelerate.utils import set_seed
        set_seed(self.seed); old = os.getcwd(); os.chdir(LS); t0 = time.perf_counter()
        try:
            self.pipe(video_path=str(video), audio_path=str(audio), video_out_path=str(out), num_frames=self.cfg.data.num_frames,
                      num_inference_steps=self.steps, guidance_scale=self.guidance, weight_dtype=self.torch.float16,
                      width=self.cfg.data.resolution, height=self.cfg.data.resolution, mask_image_path=self.cfg.data.mask_image_path,
                      temp_dir=str(temp))
        finally:
            os.chdir(old)
        return round(time.perf_counter() - t0, 2)


def read_frames(p: pathlib.Path) -> list:
    import cv2
    cap = cv2.VideoCapture(str(p)); fr = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        fr.append(f)
    cap.release(); return fr


def assemble(master: pathlib.Path, src: pathlib.Path, segs: list[dict], replacements: dict, out: pathlib.Path, work: pathlib.Path) -> dict:
    """Write master frames in order, substituting frames from `replacements[seg index] = list_of_frames`; mux original audio."""
    import cv2, imageio
    tmp = work / (out.stem + "_video.mp4")
    cap = cv2.VideoCapture(str(master)); n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = imageio.get_writer(str(tmp), fps=FPS, codec="libx264", macro_block_size=None, ffmpeg_params=["-crf", "13"], ffmpeg_log_level="error")
    written = 0; padded = {}
    for s in segs:
        rep = replacements.get(s["index"])
        for k in range(s["frames"]):
            ok, fr = cap.read()
            if not ok:
                break
            if rep is not None and k < len(rep):
                fr = rep[k]
            elif rep is not None:
                padded[s["index"]] = padded.get(s["index"], 0) + 1   # LatentSync returned fewer frames: keep original
            writer.append_data(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); written += 1
    writer.close(); cap.release()
    run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", str(tmp), "-i", str(src), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy", str(out)])
    return {"frames_written": written, "master_frames": n_total, "padded_with_original": padded}


def crop_experiment(master: pathlib.Path, wav: pathlib.Path, seg: dict, work: pathlib.Path, runner: LatentSyncRunner, target_face_h: int, pad: float) -> tuple[list, dict]:
    """SMALL_FACE segment: fixed ROI around the median face, upscaled so face height ~= target_face_h, LatentSync, paste back."""
    import cv2
    st = seg["face_bbox_stats"]; frames = read_frames(cut_segment(master, wav, seg, work, "roi_src")[0])
    H, W = frames[0].shape[:2]
    side = int(max(st["w_median"], st["h_median"]) * (1 + 2 * pad)); side = min(side, W, H)
    cx, cy = st["cx_median"], st["cy_median"]
    x1 = int(min(max(cx - side / 2, 0), W - side)); y1 = int(min(max(cy - side / 2, 0), H - side)); x2, y2 = x1 + side, y1 + side
    scale = max(1.0, target_face_h / max(st["h_median"], 1)); roi_size = int(math.ceil(side * scale / 16) * 16)
    roi_v = work / f"seg{seg['index']:03d}_roi_in.mp4"
    import imageio
    w = imageio.get_writer(str(roi_v), fps=FPS, codec="libx264", macro_block_size=None, ffmpeg_params=["-crf", "13"], ffmpeg_log_level="error")
    for fr in frames:
        w.append_data(cv2.cvtColor(cv2.resize(fr[y1:y2, x1:x2], (roi_size, roi_size), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB))
    w.close()
    _, roi_a = cut_segment(master, wav, seg, work, "roi_aud")
    roi_out = work / f"seg{seg['index']:03d}_roi_ls.mp4"
    info = {"roi": [x1, y1, x2, y2], "roi_side_px": side, "scale": round(scale, 3), "roi_input_px": roi_size,
            "face_h_median_px": st["h_median"], "face_h_after_scale_px": round(st["h_median"] * scale, 1)}
    try:
        info["latentsync_s"] = runner(roi_v, roi_a, roi_out, work / f"ls_temp_roi_{seg['index']}")
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"; return [], info
    ls_frames = read_frames(roi_out); info["roi_frames_out"] = len(ls_frames); pasted = []
    for k, fr in enumerate(frames):
        if k < len(ls_frames):
            fr = fr.copy(); fr[y1:y2, x1:x2] = cv2.resize(ls_frames[k], (side, side), interpolation=cv2.INTER_AREA)
        pasted.append(fr)
    info["preview_roi"] = str(roi_out)
    return pasted, info


def syncnet_on_segment(final: pathlib.Path, seg: dict, work: pathlib.Path, tag: str) -> dict:
    from syncnet_common import eval_syncnet
    if seg["frames"] < 50:
        return {"status": "N/A_TOO_SHORT", "note": "SyncNet detector needs a track of >= 50 frames"}
    cut = work / f"seg{seg['index']:03d}_{tag}_final_cut.mp4"
    run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-ss", f"{seg['start_frame'] / FPS:.4f}", "-i", str(final), "-t", f"{seg['frames'] / FPS:.4f}",
         "-c:v", "libx264", "-crf", "18", "-c:a", "aac", str(cut)])
    try:
        r = eval_syncnet(cut); return {"status": "PASS", "confidence": r["confidence"], "av_offset_frames": r["av_offset_frames"], "seconds": r["seconds"]}
    except Exception as e:
        msg = str(e)
        if "Face not detected" in msg:
            return {"status": "N/A_SYNCNET_NO_FACE", "error": "SyncNet S3FD detector found no >=50-frame face track in this cut"}
        tail = [l for l in msg.splitlines() if l.strip() and "S3FD" not in l][-2:]
        return {"status": "FAIL", "error": " | ".join(tail)[:300]}


def process(src: pathlib.Path, out_dir: pathlib.Path, runner: LatentSyncRunner | None, a) -> dict:
    name = src.stem; work = out_dir / f"work_{name}"; work.mkdir(parents=True, exist_ok=True)
    t_all = time.perf_counter()
    master, wav = make_master(src, work)
    frames = classify_frames(master)
    segs = segment(frames, a.min_ls_frames)
    counts = {c: sum(1 for f in frames if f["cls"] == c) for c in ("VALID_FACE", "SMALL_FACE", "NO_FACE")}
    print(f"[{name}] {len(frames)} frames@25fps  {counts}  segments={len(segs)}", flush=True)
    replacements = {}; ls_runs = []; verify_info = []
    reuse = pathlib.Path(a.reuse[name]) if name in a.reuse else None
    queue = [s for s in segs if s["action"] == "LATENT_SYNC"]
    while queue:
        s = queue.pop(0)
        if reuse and len(segs) == 1 and reuse.exists():
            frs = read_frames(reuse); s["latentsync"] = {"reused": str(reuse), "frames_out": len(frs)}; replacements[s["index"]] = frs; continue
        if runner is None:
            s["action"] = "PASS_THROUGH_NO_RUNNER"; continue
        try:
            v, au = cut_segment(master, wav, s, work, f"ls{s['index']:03d}")
            if s.get("verify_depth", 0) < a.max_verify_depth:
                ok = verify_segment(v); s["verify_depth"] = s.get("verify_depth", 0) + 1
                if not all(ok):
                    subs = split_by_mask(s, ok, a.min_ls_frames, next_index=len(segs))
                    verify_info.append({"segment": s["index"], "rejected_frames": ok.count(False), "split_into": [x["index"] for x in subs]})
                    print(f"[{name}] seg {s['index']} verify: {ok.count(False)}/{len(ok)} frames rejected by LatentSync detector -> split into {len(subs)}", flush=True)
                    s["action"] = "SPLIT"; s["split_into"] = [x["index"] for x in subs]
                    for x in subs:
                        x["verify_depth"] = s["verify_depth"]; segs.append(x)   # sub-cuts are re-verified up to --max-verify-depth
                        if x["action"] == "LATENT_SYNC":
                            queue.append(x)
                    continue
            o = work / f"seg{s['index']:03d}_ls_out.mp4"
            dt = runner(v, au, o, work / f"ls_temp_{s['index']}")
            frs = read_frames(o); replacements[s["index"]] = frs
            s["latentsync"] = {"seconds": dt, "frames_in": s["frames"], "frames_out": len(frs), "s_per_video_min": round(dt / (s["frames"] / FPS) * 60, 1)}
            ls_runs.append(dt); print(f"[{name}] seg {s['index']} {s['start_s']}-{s['end_s']}s LatentSync {dt}s -> {len(frs)} frames", flush=True)
        except Exception as e:
            s["action"] = "PASS_THROUGH_LS_FAILED"; s["error"] = f"{type(e).__name__}: {e}"; print(f"[{name}] seg {s['index']} LS FAILED: {s['error']}", flush=True)
    # timeline order for assembly / manifest: split parents are dropped, children take their place
    segs[:] = sorted([s for s in segs if s["action"] != "SPLIT"], key=lambda s: s["start_frame"])
    final = out_dir / f"facesync_{name}.mp4"
    asm = assemble(master, src, segs, replacements, final, work)
    rep = {"video": str(src), "input_meta": probe(src), "master_frames": len(frames), "frame_classes": counts, "segments": segs,
           "output": str(final), "output_meta": probe(final), "assembly": asm, "latentsync_total_s": round(sum(ls_runs), 1),
           "verify_splits": verify_info,
           "thresholds": {"valid_w": VALID_W, "valid_h": VALID_H, "valid_score": VALID_SCORE, "latentsync_filter": "w>=50 h>=80 score>=0.5", "min_ls_frames": a.min_ls_frames}}
    # SyncNet only where we lip-synced
    if not a.skip_syncnet:
        for s in segs:
            s["syncnet"] = syncnet_on_segment(final, s, work, "main") if s["action"] == "LATENT_SYNC" else {"status": "N/A", "reason": s["action"]}
            print(f"[{name}] seg {s['index']} {s['classification']}/{s['action']} syncnet={s['syncnet']}", flush=True)
    # crop+resize experiment for SMALL_FACE
    if a.crop_experiment and runner is not None and counts["SMALL_FACE"]:
        rep2 = dict(replacements); exp = []
        for s in segs:
            if s["classification"] != "SMALL_FACE" or s["frames"] < a.min_ls_frames:
                continue
            pasted, info = crop_experiment(master, wav, s, work, runner, a.target_face_h, a.pad)
            info.update(segment=s["index"], start_s=s["start_s"], end_s=s["end_s"], frames=s["frames"])
            if pasted:
                rep2[s["index"]] = pasted
            exp.append(info); print(f"[{name}] crop-exp seg {s['index']} {info}", flush=True)
        final2 = out_dir / f"facesync_{name}_smallface_crop.mp4"
        asm2 = assemble(master, src, segs, rep2, final2, work)
        if not a.skip_syncnet:
            for info in exp:
                s = segs[info["segment"]]
                if "error" not in info:
                    info["syncnet_pasted"] = syncnet_on_segment(final2, s, work, "crop")
                    try:   # the ROI clip itself (face upscaled, audio included) is what SyncNet can actually track
                        from syncnet_common import eval_syncnet
                        r = eval_syncnet(pathlib.Path(info["preview_roi"]))
                        info["syncnet_roi_clip"] = {"status": "PASS", "confidence": r["confidence"], "av_offset_frames": r["av_offset_frames"], "seconds": r["seconds"]}
                    except Exception as e:
                        info["syncnet_roi_clip"] = {"status": "N/A_SYNCNET_NO_FACE" if "Face not detected" in str(e) else "FAIL"}
                    print(f"[{name}] crop-exp seg {s['index']} syncnet pasted={info['syncnet_pasted']} roi={info['syncnet_roi_clip']}", flush=True)
        rep["crop_experiment"] = {"output": str(final2), "output_meta": probe(final2), "assembly": asm2, "segments": exp,
                                  "params": {"target_face_h": a.target_face_h, "pad": a.pad}}
    rep["wall_s_total"] = round(time.perf_counter() - t_all, 1)
    write_json(out_dir / f"manifest_{name}.json", rep)
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--out-dir", default="/tmp/dabai_pilot_08fa")
    ap.add_argument("--min-ls-frames", type=int, default=25, help="VALID_FACE runs shorter than this are passed through")
    ap.add_argument("--reuse", nargs="*", default=[], metavar="NAME:MP4", help="whole-video LatentSync output to reuse when the video is one VALID segment")
    ap.add_argument("--crop-experiment", action="store_true")
    ap.add_argument("--target-face-h", type=int, default=160, help="crop experiment: upscale so the median face height reaches this many px")
    ap.add_argument("--pad", type=float, default=0.6, help="crop experiment: padding around the face as a fraction of face size, each side")
    ap.add_argument("--max-verify-depth", type=int, default=3, help="how many times a cut may be re-checked/split before LatentSync is called")
    ap.add_argument("--skip-syncnet", action="store_true")
    ap.add_argument("--no-latentsync", action="store_true", help="classify + assemble only (everything passes through)")
    ap.add_argument("--json", default=str(PILOT / "0.8_face_aware.json"))
    a = ap.parse_args(); a.reuse = {x.split(":", 1)[0]: x.split(":", 1)[1] for x in a.reuse}
    out_dir = pathlib.Path(a.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    runner = None if a.no_latentsync else LatentSyncRunner()
    reps = []
    for v in a.videos:
        reps.append(process(pathlib.Path(v).resolve(), out_dir, runner, a))
    rep = {"task": "0.8-face-aware", "gpu": gpu_info(), "latentsync_commit": sh(f"git -C {LS} rev-parse HEAD"),
           "config": {"stage2_512": True, "steps": 20, "guidance": 1.5, "deepcache": True, "fp16": True, "seed": 1247},
           "videos": reps, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(pathlib.Path(a.json), rep); print(f"-> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
