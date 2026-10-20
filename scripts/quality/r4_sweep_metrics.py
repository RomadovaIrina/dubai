#!/usr/bin/env python3
"""r4 generator sweep (on the audio agent's GPU): metrics of ONE window clip of ls_window_ab.py vs the same window of the 25 fps master.
  syncnet    sync_perframe.py on the dub-only clip: global conf / AV offset, mean conf_at0 over seconds where the dub speaks, worst second
  lip_r      inner-lip gap vs the driving audio log-RMS (Pearson, lag 0, whole window) = mouth opening follows the speech energy
  flash      r3_flash.py (single-frame lip spikes) of the clip with the master window as the reference
  jaw        jaw-contour jitter: MediaPipe jaw points relative to the nose bridge / IOD, mean |2nd difference|, clip / master (1.0 = as steady)
  s_per_ls_frame   LatentSync seconds / generated frames (window manifest)
    python scripts/quality/r4_sweep_metrics.py --win-dir OUT --id 04 --tag A0_w1 --master MASTER_25fps.mp4 --t0 12 --t1 20 --json x.json"""
import argparse, json, pathlib, subprocess, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; PY = sys.executable; sys.path.insert(0, str(HERE.parent / "pilot"))
JAW = [132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361]; NB, EL, ER = 168, 33, 263


def last_json(out):
    for l in reversed(out.strip().splitlines()):
        if l.startswith("{"): return json.loads(l)
    return {}


def mp(video, npz):
    import clean_face_stack as cfs   # MediaPipe runs in its own env (/venv/mediapipe)
    return np.load(cfs.run_mediapipe(pathlib.Path(video), pathlib.Path(npz)))["lm"]


def jaw_jitter(lm):
    lm = lm[..., :2].astype(float); iod = np.linalg.norm(lm[:, EL] - lm[:, ER], axis=1)
    rel = (lm[:, JAW] - lm[:, NB][:, None]) / iod[:, None, None]
    d2 = np.abs(rel[2:] - 2 * rel[1:-1] + rel[:-2]).mean((1, 2)); return float(np.nanmean(d2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--win-dir", required=True); ap.add_argument("--id", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--master", required=True); ap.add_argument("--t0", type=float, required=True); ap.add_argument("--t1", type=float, required=True); ap.add_argument("--json", required=True)
    a = ap.parse_args(); d = pathlib.Path(a.win_dir); clip = d / f"{a.id}_{a.tag}_dubonly.mp4"; man = json.loads((d / f"{a.id}_{a.tag}.manifest.json").read_text())
    drv = next(d.glob(f"work_*_{a.tag}/drive_16k.wav")); mw = d / f"master_{a.id}_{a.t0:g}.mp4"
    if not mw.exists():
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{a.t0}", "-t", f"{a.t1 - a.t0}", "-i", a.master, "-an", "-c:v", "libx264", "-crf", "12", str(mw)], check=True)
    lm_c = mp(clip, d / f"{a.id}_{a.tag}_mp.npz"); lm_m = mp(mw, d / f"master_{a.id}_{a.t0:g}_mp.npz")
    sp = d / f"{a.id}_{a.tag}.sync.json"
    subprocess.run([PY, str(HERE / "sync_perframe.py"), "--video", str(clip), "--audio", str(drv), "--mp-npz", str(d / f"{a.id}_{a.tag}_mp.npz"), "--out", str(sp)], capture_output=True)
    s = json.loads(sp.read_text()) if sp.exists() else {}
    rows = [r for r in s.get("per_second", []) if r.get("conf_at0") is not None and (r.get("dub_act") or 0) >= 0.3]
    fl = last_json(subprocess.run([PY, str(HERE / "r3_flash.py"), "--video", str(clip), "--src", str(mw), "--out", str(d / f"{a.id}_{a.tag}.flash.json")], capture_output=True, text=True).stdout)
    calls = ((man.get("lipsync") or {}).get("calls")) or []
    ls_s = (man.get("stage_seconds") or {}).get("latentsync"); ls_f = sum(c.get("frames", 0) for c in calls) or (man.get("window", {}).get("f1", 0) - man.get("window", {}).get("f0", 0))
    r = {"id": a.id, "tag": a.tag, "t0": a.t0, "t1": a.t1, "sync_conf": (s.get("global") or {}).get("conf"), "av_offset": (s.get("global") or {}).get("av_offset"),
         "conf_at0_speech_mean": round(float(np.mean([x["conf_at0"] for x in rows])), 3) if rows else None,
         "conf_at0_worst_s": round(float(min(x["conf_at0"] for x in rows)), 3) if rows else None, "lip_r": s.get("lip_env_r_global"),
         "flash": fl.get("flash_frames"), "jaw_ratio": round(jaw_jitter(lm_c) / max(jaw_jitter(lm_m), 1e-9), 3),
         "s_per_ls_frame": round(ls_s / ls_f, 3) if ls_s and ls_f else None}
    pathlib.Path(a.json).write_text(json.dumps(r, indent=1)); print(json.dumps(r))


if __name__ == "__main__":
    main()
