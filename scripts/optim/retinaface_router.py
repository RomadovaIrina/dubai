#!/usr/bin/env python3
"""RetinaFace skip router: fast per-frame routing decision BEFORE LatentSync (facexlib RetinaFace ResNet50, project weights).

Frame classes (compatible with scripts/pilot/face_aware_latentsync.py::segment()):
  VALID_FACE   largest face passes  w >= --face-min-w (50), h >= --face-min-h (80), 0.2 < w/h < 1.5, score >= --retina-conf
  SMALL_FACE   a face is detected but fails a size/aspect test          reason: width_below_threshold | height_below_threshold | aspect_out_of_range
  NO_FACE      no detection at all, or best detection below --retina-conf   reason: no_face | low_confidence

The defaults mirror LatentSync's own FaceDetector filter (latentsync/utils/face_detector.py: w<50 or h<80 -> rejected),
so a VALID_FACE frame is one LatentSync itself would accept; the business wish "<50 px" is covered by --face-min-w/-h.
This router is the routing decision only; LatentSync's own detector is still run on every cut segment right before
inference (face_aware verify_segment) as the safety check. Disagreement -> the segment is passed through, never a crash.

CLI (classification only, no LatentSync):
    source scripts/env.sh
    python scripts/optim/retinaface_router.py test_videos/01.MP4 ... --json reports/pilot/optim/retinaface_routing.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot"))
from pilot_common import MODELS, gpu_info, write_json  # noqa: E402

DEFAULT_MIN_W, DEFAULT_MIN_H, DEFAULT_CONF = 50, 80, 0.8   # 0.8 = scripts/check_retinaface.py threshold
ASPECT = (0.2, 1.5)                                         # LatentSync FaceDetector aspect filter
FPS = 25


class RetinaFaceRouter:
    def __init__(self, min_w: int = DEFAULT_MIN_W, min_h: int = DEFAULT_MIN_H, conf: float = DEFAULT_CONF,
                 device: str = "cuda", half: bool = True):
        from facexlib.detection import init_detection_model
        self.min_w, self.min_h, self.conf = min_w, min_h, conf
        self.det = init_detection_model("retinaface_resnet50", half=half, device=device, model_rootpath=str(MODELS / "facexlib"))

    def classify_frame(self, frame_bgr) -> dict:
        import torch
        with torch.no_grad():
            dets = self.det.detect_faces(frame_bgr, conf_threshold=0.02)   # everything; our confidence gate is applied below
        if len(dets) == 0:
            return {"cls": "NO_FACE", "reason": "no_face", "bbox": None}
        best = max(dets, key=lambda d: float(d[4]))                          # highest confidence first
        x1, y1, x2, y2, score = [float(v) for v in best[:5]]
        if score < self.conf:
            return {"cls": "NO_FACE", "reason": "low_confidence", "bbox": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2),
                                                                           "w": int(x2 - x1), "h": int(y2 - y1), "score": round(score, 3)}}
        # among confident detections take the largest (LatentSync picks the largest accepted face)
        conf_dets = [d for d in dets if float(d[4]) >= self.conf]
        big = max(conf_dets, key=lambda d: (float(d[2]) - float(d[0])) * (float(d[3]) - float(d[1])))
        x1, y1, x2, y2, score = [float(v) for v in big[:5]]
        w, h = int(x2 - x1), int(y2 - y1)
        bbox = {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "w": w, "h": h, "score": round(score, 3)}
        if w < self.min_w:
            return {"cls": "SMALL_FACE", "reason": "width_below_threshold", "bbox": bbox}
        if h < self.min_h:
            return {"cls": "SMALL_FACE", "reason": "height_below_threshold", "bbox": bbox}
        ar = w / max(h, 1)
        if not (ASPECT[0] < ar < ASPECT[1]):
            return {"cls": "SMALL_FACE", "reason": "aspect_out_of_range", "bbox": bbox}
        return {"cls": "VALID_FACE", "reason": None, "bbox": bbox}

    def classify_video(self, path: pathlib.Path) -> list[dict]:
        """Rows in the schema face_aware_latentsync.segment() consumes: {frame, cls, bbox} (+ reason)."""
        import cv2
        cap = cv2.VideoCapture(str(path)); rows = []; i = 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            r = self.classify_frame(fr); r["frame"] = i; rows.append(r); i += 1
        cap.release()
        return rows

    def params(self) -> dict:
        return {"detector": "facexlib retinaface_resnet50 fp16", "weights": str(MODELS / "facexlib"), "face_min_w": self.min_w,
                "face_min_h": self.min_h, "retina_conf": self.conf, "aspect_range": list(ASPECT)}


def routing_stats(frames: list[dict], segs: list[dict]) -> dict:
    n = len(frames)
    counts = {c: sum(1 for f in frames if f["cls"] == c) for c in ("VALID_FACE", "SMALL_FACE", "NO_FACE")}
    reasons = {}
    for f in frames:
        if f.get("reason"):
            reasons[f["reason"]] = reasons.get(f["reason"], 0) + 1
    ls_frames = sum(s["frames"] for s in segs if s["action"] == "LATENT_SYNC")
    return {"total_frames": n, "frame_classes": counts, "skip_reasons": reasons, "segments": len(segs),
            "ls_eligible_frames": ls_frames, "ls_eligible_seconds": round(ls_frames / FPS, 3),
            "pass_through_frames": n - ls_frames, "pass_through_seconds": round((n - ls_frames) / FPS, 3),
            "eligible_fraction": round(ls_frames / n, 4) if n else None,
            "by_action": {a: sum(s["frames"] for s in segs if s["action"] == a) for a in sorted({s["action"] for s in segs})}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--face-min-w", type=int, default=DEFAULT_MIN_W)
    ap.add_argument("--face-min-h", type=int, default=DEFAULT_MIN_H)
    ap.add_argument("--retina-conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--min-ls-frames", type=int, default=25)
    ap.add_argument("--work-dir", default="/tmp/dabai_optim/routing")
    ap.add_argument("--json", default="reports/pilot/optim/retinaface_routing.json")
    ap.add_argument("--compare", default="reports/pilot/0.8_face_aware.json", help="previous insightface-based face-aware result to compare with")
    a = ap.parse_args()
    import face_aware_latentsync as fa
    work = pathlib.Path(a.work_dir); work.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter(); router = RetinaFaceRouter(a.face_min_w, a.face_min_h, a.retina_conf); load_s = round(time.perf_counter() - t0, 3)
    prev = {}
    if a.compare and pathlib.Path(a.compare).exists():
        for v in json.load(open(a.compare))["videos"]:
            prev[pathlib.Path(v["video"]).stem] = {"frame_classes": v["frame_classes"], "master_frames": v["master_frames"],
                                                   "by_action": {}}
            for s in v["segments"]:
                prev[pathlib.Path(v["video"]).stem]["by_action"][s["action"]] = prev[pathlib.Path(v["video"]).stem]["by_action"].get(s["action"], 0) + s["frames"]
    out = []
    for v in a.videos:
        src = pathlib.Path(v).resolve(); w = work / src.stem; w.mkdir(exist_ok=True)
        master, _ = fa.make_master(src, w)                       # 25 fps CFR master, exactly what LatentSync sees
        t0 = time.perf_counter(); frames = router.classify_video(master); cls_s = round(time.perf_counter() - t0, 3)
        segs = fa.segment(frames, a.min_ls_frames)
        st = routing_stats(frames, segs)
        row = {"video": src.name, "master": str(master), "classify_seconds": cls_s, "ms_per_frame": round(cls_s / max(len(frames), 1) * 1000, 2), **st,
               "segments_detail": [{k: s[k] for k in ("index", "start_s", "end_s", "frames", "classification", "action")} for s in segs],
               "previous_insightface_result": prev.get(src.stem)}
        out.append(row)
        print(f"{src.name}: {st['total_frames']} frames {st['frame_classes']} reasons={st['skip_reasons']} eligible {st['ls_eligible_seconds']}s "
              f"({st['eligible_fraction']}) pass-through {st['pass_through_seconds']}s | {cls_s}s ({row['ms_per_frame']} ms/frame)", flush=True)
        shutil.rmtree(w, ignore_errors=True)                     # runtime media is not kept
    rep = {"task": "optim/retinaface-routing", "router": router.params(), "router_load_s": load_s, "min_ls_frames": a.min_ls_frames,
           "gpu": gpu_info(), "videos": out, "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(pathlib.Path(a.json), rep); print(f"-> {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
