#!/usr/bin/env python3
"""Round 3 (track A) evaluation of full-video runs (artifacts/r3/runs) and window arms (artifacts/r3/win) -> <base>.r3eval.json.
  sync      upstream SyncNet (LatentSync eval_sync_conf, private cwd): dub-only + final mix, conf + AV offset
  syncpf    sync_perframe.py on the dub-only clip: per-second conf_at0 / conf_best / offset, lip gap vs dub log-RMS r (MediaPipe on the
            output), dub / source activity per second
  slide     slide_metric.py (nose base / cheeks vs the master, mask centre vs head)
  v4        common metric v4 (/workspace/dub/common/mouth_jitter_hr.py): mouth resid / sharp, lower-face shear / jerk, frozen_mouth per second,
            LatentSync window-boundary jumps (spec 8.5)
  speed     LatentSync s per video-minute (of the run's own duration), video stage s per video-minute
    python scripts/quality/r3_eval.py BASE [BASE...]   (BASE = .../03_r2best or .../03_wB_base, without extension)
"""
from __future__ import annotations
import json, os, pathlib, subprocess, sys
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot")); os.environ.setdefault("DUB_ROOT", str(HERE.parents[1]))
PY = "/venv/dabai/bin/python"; NOTHP = str(HERE.parent / "nothp"); MASTERS = pathlib.Path("/workspace/dub/artifacts/r3/masters")


def sh(cmd, **kw):
    return subprocess.run([NOTHP] + cmd, **kw)


def run_eval(base: pathlib.Path, parts=("sync", "syncpf", "slide", "v4")) -> dict:
    man_p = base.with_suffix(".manifest.json"); out = base.with_suffix(".r3eval.json"); mp4 = base.with_suffix(".mp4")
    if not man_p.exists() or not mp4.exists(): return None
    ev = json.loads(out.read_text()) if out.exists() else {}
    man = json.loads(man_p.read_text()); vid = pathlib.Path(man["input"]).stem; M = MASTERS / vid
    win = man.get("window"); f0 = int(win["f0"]) if win else 0
    n = int(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(mp4)],
                           text=True, capture_output=True).stdout.strip())
    work = base.parent / f"work_{base.name}"   # the arm's own work dir (window runs: drive_16k.wav)
    base_work = pathlib.Path(f"/workspace/dub/artifacts/baseline/{vid}/work/work_{vid}")
    drv = (work / "drive_16k.wav") if win else pathlib.Path(man["dubbed_track"]["wav16k"])   # the run's OWN driving dub track (audio-r2 runs: track C's)
    src16 = base_work / "own_audio_16k.wav"
    dub = pathlib.Path(str(base) + "_dubonly.mp4")
    if not dub.exists():
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", str(mp4), "-i", str(drv), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", str(dub)], check=True)
    ref = M / "master_25fps.mp4"
    if win:   # master frames of the window as their own clip (v4 reference)
        ref = base.parent / f"{vid}_ref_{f0}_{f0 + n}.mp4"
        if not ref.exists():
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", str(M / "master_25fps.mp4"), "-vf", f"select=between(n\\,{f0}\\,{f0 + n - 1})", "-vsync", "0",
                            "-c:v", "libx264", "-crf", "0", "-an", str(ref)], check=True)
    if "sync" in parts and "sync" not in ev:
        from syncnet_common import eval_syncnet
        ev["sync"] = {}
        for k, p in (("mix", mp4),):   # dub-only global conf / offset comes from sync_perframe (same detector / crop / model, one pass less of CPU)
            try: x = eval_syncnet(p); ev["sync"][k] = {"conf": x["confidence"], "av_offset": x["av_offset_frames"]}
            except Exception as e: ev["sync"][k] = {"error": str(e)[-200:]}
    if "syncpf" in parts and "syncpf" not in ev:
        import clean_face_stack as cfs
        mpz = pathlib.Path(str(base) + "_mp.npz")
        try: cfs.run_mediapipe(mp4, mpz)
        except Exception as e: print("mediapipe on output failed", e); mpz = None
        sj = pathlib.Path(str(base) + ".syncpf.json")
        # source activity: slice the source speech to the window
        src_arg = []
        if src16.exists():
            if win:
                sw = work / "src_16k.wav"
                if not sw.exists():
                    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-ss", f"{f0 / 25:.3f}", "-t", f"{n / 25:.3f}", "-i", str(src16), str(sw)], check=True)
                src_arg = ["--src-audio", str(sw)]
            else: src_arg = ["--src-audio", str(src16)]
        sh([PY, str(HERE / "sync_perframe.py"), "--video", str(mp4), "--audio", str(drv), "--out", str(sj)] + src_arg + (["--mp-npz", str(mpz)] if mpz else []),
           check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if sj.exists():
            d = json.loads(sj.read_text()); ev.setdefault("sync", {}).setdefault("dub_only", {"conf": d["global"]["conf"], "av_offset": d["global"]["av_offset"], "via": "sync_perframe"})
            ev["syncpf"] = {"global": d["global"], "lip_env_r": d.get("lip_env_r_global"), "lip_env_r_by_lag": d.get("lip_env_r_by_lag"), "t0": f0 / 25}
    if "slide" in parts and "slide" not in ev:
        sj = base.with_suffix(".slide.json")
        cmd = [PY, str(HERE / "slide_metric.py"), "--video", str(mp4), "--master", str(M / "master_25fps.mp4"), "--npz", str(M / "master_mediapipe.npz"),
               "--offset", str(f0), "--frames", f"{f0}:{f0 + n}", "--manifest", str(man_p), "--rigid", str(M / "master_rigid.npz"), "--series", "--out", str(sj)]
        bx = pathlib.Path(str(base) + "_maskboxes.npz")
        if bx.exists(): cmd += ["--boxes", str(bx)]
        sh(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if sj.exists(): ev["slide"] = json.loads(sj.read_text()); ev["slide"].pop("series", None)
    if "v4" in parts and "v4" not in ev:
        vm = man_p
        if win:
            m2 = json.loads(man_p.read_text())
            for s in m2["lipsync"]["segments"]: s["start_frame"] -= f0; s["end_frame"] -= f0
            m2["lipsync"]["master_frames"] = n; vm = base.with_suffix(".v4man.json"); vm.write_text(json.dumps(m2))
        vj = base.with_suffix(".v4.json")
        sh([PY, "/workspace/dub/common/mouth_jitter_hr.py", "--video", str(mp4), "--reference", str(ref), "--manifest", str(vm), "--tag", base.name, "--out", str(vj)],
           check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if vj.exists():
            a = json.loads(vj.read_text()); al = a["all"]
            ev["v4"] = {"mouth_resid": al["resid"]["mouth"]["ratio"], "mouth_sharp": al["sharp"]["mouth"]["ratio"], "lf_shear": al["shear"]["lowerface"]["ratio"],
                        "lf_jerk": al["jerk"]["lowerface"]["ratio"], "mouth_jerk": al["jerk"]["mouth"]["ratio"], "mouth_vel": al["vel"]["mouth"]["ratio"],
                        "upper_resid": al["resid"]["upper"]["ratio"]}
            for k in ("twitch", "impulse"):
                if k in al: ev["v4"][k] = al[k]
            for k in ("latentsync_window_boundaries",):
                if k in a: ev["v4"]["window_boundaries"] = a[k]; break
    lsr = man["lipsync"]; dur_s = (n / 25)
    ev["speed"] = {"ls_s": lsr.get("latentsync_inference_total_s"), "ls_s_per_video_min": round(lsr.get("latentsync_inference_total_s", 0) / (dur_s / 60), 1),
                   "ls_frames": sum(s["end_frame"] - s["start_frame"] for s in lsr["segments"] if s["action"] == "LATENT_SYNC"),
                   "ls_s_per_ls_frame": round(lsr.get("latentsync_inference_total_s", 0) / max(1, sum(s["end_frame"] - s["start_frame"] for s in lsr["segments"] if s["action"] == "LATENT_SYNC")), 4),
                   "stage_seconds": man.get("stage_seconds")}
    ev["frames"] = n; ev["f0"] = f0
    out.write_text(json.dumps(ev, indent=1)); return ev


if __name__ == "__main__":
    for b in sys.argv[1:]:
        e = run_eval(pathlib.Path(b))
        print(b, json.dumps({k: e.get(k) for k in ("sync", "syncpf")} if e else None)[:400], flush=True)
