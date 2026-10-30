#!/usr/bin/env python3
"""r9 E2E pre-check: what the input is and what will be hard, BEFORE any GPU stage. Never fails on a hard video: it only reports.
  ffprobe: duration, fps (avg / r), resolution, display rotation, codec / pix_fmt, audio present;
  MediaPipe Face Landmarker (3 faces, image mode, every N-th frame ~ 4 per second): face presence, faces per frame, face height in px.
    python scripts/r9/precheck.py --input X.mp4 --out precheck.json [--work DIR]
Flags (list of strings, used by e2e.sh and copied into manifest.json):
  no_audio (-> ERROR), no_face, face_partly_absent (< 80 % of probed frames), two_faces (>= 2 faces on >= 10 % of probed frames),
  small_face (median face height < 90 px), tiny_face (< 60 px: the generator gate --face-min-h 60 drops it), high_fps (> 31), odd_fps
  (not ~24/25/30/50/60), rotated_source (display matrix != 0), landscape, low_res (short side < 600), hdr_or_10bit, long_video (> 60 s)."""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot"))


def probe(p):
    r = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(p)], capture_output=True, text=True, check=True).stdout)
    v = next((s for s in r["streams"] if s["codec_type"] == "video"), None); a = next((s for s in r["streams"] if s["codec_type"] == "audio"), None)
    fr = lambda s: (lambda n, d: float(n) / float(d) if float(d) else 0.0)(*s.split("/"))
    rot = 0
    for sd in (v or {}).get("side_data_list", []):
        if "rotation" in sd: rot = int(round(float(sd["rotation"])))
    if "rotate" in (v or {}).get("tags", {}): rot = int(v["tags"]["rotate"])
    w, h = int(v["width"]), int(v["height"])
    if abs(rot) % 180 == 90: w, h = h, w   # as displayed
    return {"duration_s": float(r["format"]["duration"]), "width": w, "height": h, "coded_width": int(v["width"]), "coded_height": int(v["height"]),
            "rotation": rot, "fps_avg": round(fr(v.get("avg_frame_rate", "0/1")), 3), "fps_r": round(fr(v.get("r_frame_rate", "0/1")), 3),
            "nb_frames": int(v.get("nb_frames", 0) or 0), "vcodec": v.get("codec_name"), "pix_fmt": v.get("pix_fmt"), "color_transfer": v.get("color_transfer"),
            "audio": bool(a), "acodec": (a or {}).get("codec_name"), "a_rate": int((a or {}).get("sample_rate", 0) or 0), "a_channels": int((a or {}).get("channels", 0) or 0)}


def faces(p, work, fps):
    import clean_face_stack as cfs
    every = max(1, int(round(fps / 4)))
    out = cfs.run_mediapipe(pathlib.Path(p), pathlib.Path(work) / "precheck_mp3.npz", mode="image", num_faces=3, every=every)
    d = np.load(out); W, H = [int(x) for x in d["wh"]]
    la = d["lm_all"] if "lm_all" in d.files else d["lm"][:, None]
    idx = np.arange(0, len(la), every); la = la[idx]                      # the probed frames only
    ok = np.isfinite(la[..., 0]).all(axis=-1)                            # (T, F) face present
    n = ok.sum(axis=1); hts = []
    for t, f in zip(*np.nonzero(ok)):
        y = la[t, f, :, 1]; hts.append(float(y.max() - y.min()) * (H if np.nanmax(np.abs(y)) <= 1.5 else 1.0))
    big = []   # the largest face per frame (the speaker candidate)
    for t in range(len(la)):
        hs = [float(la[t, f, :, 1].max() - la[t, f, :, 1].min()) for f in range(la.shape[1]) if ok[t, f]]
        if hs: big.append(max(hs) * (H if max(hs) <= 1.5 else 1.0))
    return {"probed_frames": int(len(la)), "every": every, "face_frac": round(float((n >= 1).mean()), 3) if len(n) else 0.0,
            "two_face_frac": round(float((n >= 2).mean()), 3) if len(n) else 0.0, "max_faces": int(n.max()) if len(n) else 0,
            "face_h_px_median": round(float(np.median(big)), 1) if big else 0.0, "face_h_px_p10": round(float(np.percentile(big, 10)), 1) if big else 0.0}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--input", required=True); ap.add_argument("--out", required=True); ap.add_argument("--work", default=None)
    a = ap.parse_args(); work = pathlib.Path(a.work or pathlib.Path(a.out).parent); work.mkdir(parents=True, exist_ok=True)
    r = {"input": str(a.input)}; flags = []
    try:
        r["probe"] = pr = probe(a.input)
    except Exception as e:   # unreadable input
        r["error"] = f"ffprobe: {e}"; r["flags"] = ["unreadable_input"]; pathlib.Path(a.out).write_text(json.dumps(r, indent=1)); print("== precheck ERROR", e); return
    fps = pr["fps_avg"] or pr["fps_r"]
    if not pr["audio"]: flags.append("no_audio")
    if fps > 31: flags.append("high_fps")
    if min(abs(fps - f) for f in (23.976, 24, 25, 29.97, 30, 50, 59.94, 60)) > 0.2: flags.append("odd_fps")
    if pr["fps_r"] and abs(pr["fps_r"] - fps) > 0.5: flags.append("variable_fps")
    if pr["rotation"]: flags.append("rotated_source")
    if pr["width"] > pr["height"]: flags.append("landscape")
    if min(pr["width"], pr["height"]) < 600: flags.append("low_res")
    if (pr["pix_fmt"] or "").endswith("10le") or pr["color_transfer"] in ("arib-std-b67", "smpte2084"): flags.append("hdr_or_10bit")
    if pr["duration_s"] > 60: flags.append("long_video")
    try:
        r["faces"] = fc = faces(a.input, work, fps)
        if fc["face_frac"] == 0: flags.append("no_face")
        elif fc["face_frac"] < 0.8: flags.append("face_partly_absent")
        if fc["two_face_frac"] >= 0.10: flags.append("two_faces")
        if 0 < fc["face_h_px_median"] < 60: flags.append("tiny_face")
        elif 0 < fc["face_h_px_median"] < 90: flags.append("small_face")
    except Exception as e:
        r["faces_error"] = str(e)[-800:]; flags.append("face_probe_failed")
    r["flags"] = flags; pathlib.Path(a.out).write_text(json.dumps(r, indent=1))
    print(f"== precheck {pathlib.Path(a.input).name}: {pr['width']}x{pr['height']} {fps} fps {pr['duration_s']:.1f} s rot {pr['rotation']}; faces {r.get('faces')}; flags {flags}")


if __name__ == "__main__":
    main()
