#!/usr/bin/env python3
"""Round 3 (track A), spec 8.2: synthetic two-face clip from two test videos side by side (same height), source audio + dub of ONE of them
-> a manifest in the runner's schema so candidate_video_stage.py runs the clean stack's two-face path (probe -> tracks -> diarization turn
<-> mouth-motion assignment -> lip-sync only the active speaker).  Then `check` measures, per half of the frame, the change vs the master:
the inactive face must be untouched (encode noise only), the active one lip-synced.

    r3_twoface.py build --left 03:46:12 --right 04:25:12 --speaker left --out DIR      (id:t0:dur)
    r3_twoface.py check --out DIR --video RUN.mp4
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np
CAN = pathlib.Path("/workspace/dub/canonical/artifacts/baseline"); RAW = pathlib.Path("/workspace/dub/inputs/test_videos_raw")


def run(c): subprocess.run(c, check=True)


def src_of(v): return next(RAW.glob(f"{v}.*"))


def build(a):
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); work = out / "work"; work.mkdir(exist_ok=True)
    L = a.left.split(":"); R = a.right.split(":"); dur = float(L[2])
    spk = L if a.speaker == "left" else R; v, t0 = spk[0], float(spk[1])
    clip = out / "twoface.mp4"
    run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-ss", L[1], "-t", str(dur), "-i", str(src_of(L[0])), "-ss", R[1], "-t", str(dur), "-i", str(src_of(R[0])),
         "-filter_complex", "[0:v]fps=30,setpts=PTS-STARTPTS[a];[1:v]fps=30,setpts=PTS-STARTPTS[b];[a][b]hstack=inputs=2[v]", "-map", "[v]",
         "-map", f"{0 if a.speaker == 'left' else 1}:a:0", "-c:v", "libx264", "-crf", "12", "-c:a", "aac", "-b:a", "192k", "-shortest", str(clip)])
    man = json.loads((CAN / v / "final" / f"{v}_final.manifest.json").read_text()); W = CAN / v / "work" / f"work_{v}"
    for name, srcf in (("own_audio_16k.wav", W / "own_audio_16k.wav"), ("dubbed_16k.wav", W / "dubbed_16k.wav"), ("final_mix_44k.wav", W / "final_mix_44k.wav")):
        run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-ss", str(t0), "-t", str(dur), "-i", str(srcf), str(work / name)])
    for name, srcf in (("dubbed_16k.wav", "dubbed_aac.m4a"), ("final_mix_44k.wav", "final_mix_aac.m4a")):
        run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", str(work / name), "-c:a", "aac", "-b:a", "192k", str(work / srcf)])
    sh = lambda lst, ks=("start", "end"): [dict(x, **{ks[0]: round(max(0, x[ks[0]] - t0), 3), ks[1]: round(min(dur, x[ks[1]] - t0), 3)}) for x in lst if x[ks[1]] > t0 and x[ks[0]] < t0 + dur]
    pr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height,avg_frame_rate", "-select_streams", "v:0", "-of", "json", str(clip)], text=True, capture_output=True).stdout)["streams"][0]
    n, d = pr["avg_frame_rate"].split("/")
    m2 = {"task": "r3 two-face synthetic (spec 8.2)", "input": str(clip), "output": str(out / "twoface_out.mp4"), "work_dir": str(work),
          "source": {"duration_s": dur, "video": {"width": pr["width"], "height": pr["height"], "avg_fps": int(n) / int(d)}},
          "speech_intervals": sh(man["speech_intervals"]), "diarization": {"turns": sh(man["diarization"]["turns"])}, "units": [],
          "dubbed_track": {"wav16k": str(work / "dubbed_16k.wav"), "aac": str(work / "dubbed_aac.m4a")},
          "twoface": {"left": a.left, "right": a.right, "speaker": a.speaker, "half_width": pr["width"] // 2}}
    (out / "twoface.manifest.json").write_text(json.dumps(m2, indent=1)); print(out / "twoface.manifest.json")


def check(a):
    import cv2
    out = pathlib.Path(a.out); m2 = json.loads((out / "twoface.manifest.json").read_text()); hw = m2["twoface"]["half_width"]
    rm = json.loads(pathlib.Path(a.video).with_suffix(".manifest.json").read_text())
    master = pathlib.Path(rm["work_dir"]) / "master_25fps.mp4"
    cv, cm = cv2.VideoCapture(a.video), cv2.VideoCapture(str(master)); dl, dr = [], []
    while True:
        o1, f1 = cv.read(); o2, f2 = cm.read()
        if not (o1 and o2): break
        d = np.abs(f1.astype(np.int16) - f2.astype(np.int16)).max(2)
        dl.append([float(d[:, :hw].mean()), float((d[:, :hw] > 12).mean())]); dr.append([float(d[:, hw:].mean()), float((d[:, hw:] > 12).mean())])
    dl, dr = np.array(dl), np.array(dr)
    fs = rm.get("face_stack", {})
    res = {"speaker": m2["twoface"]["speaker"], "frames": len(dl), "left_mean_absdiff": round(float(dl[:, 0].mean()), 3), "right_mean_absdiff": round(float(dr[:, 0].mean()), 3),
           "left_px_changed_gt12": round(float(dl[:, 1].mean()), 5), "right_px_changed_gt12": round(float(dr[:, 1].mean()), 5),
           "left_frames_changed": int((dl[:, 1] > 0.002).sum()), "right_frames_changed": int((dr[:, 1] > 0.002).sum()),
           "faces": fs.get("faces"), "two_face_probe": fs.get("two_face_probe"), "ls_fraction": rm["lipsync"]["ls_fraction"], "status": rm.get("video_status")}
    pathlib.Path(a.video).with_suffix(".twoface.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1)[:1500])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["build", "check"]); ap.add_argument("--left"); ap.add_argument("--right"); ap.add_argument("--speaker", default="left")
    ap.add_argument("--out", required=True); ap.add_argument("--video"); a = ap.parse_args()
    build(a) if a.cmd == "build" else check(a)
