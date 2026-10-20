#!/usr/bin/env python3
"""r4 generator sweep: the K worst ~W s lip-sync windows of a full base run (25 fps output of candidate_video_stage.py), ranked by the per-second
dub-only SyncNet conf_at0 (sync_perframe.py) over seconds where the dub speaks and a face is tracked; windows do not overlap.
    python scripts/quality/r4_sweep_windows.py --video BASE_25fps.mp4 --dub dubbed_16k.wav --out wins.json [--k 3] [--w 8]"""
import argparse, json, pathlib, subprocess, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--video", required=True); ap.add_argument("--dub", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=3); ap.add_argument("--w", type=int, default=8); a = ap.parse_args()
    out = pathlib.Path(a.out); dv = out.with_name(out.stem + "_dubonly.mp4"); sj = out.with_name(out.stem + "_sync.json")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", a.video, "-i", a.dub, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-shortest", str(dv)], check=True)
    subprocess.run([sys.executable, str(HERE / "sync_perframe.py"), "--video", str(dv), "--audio", a.dub, "--out", str(sj)], capture_output=True)
    rows = json.loads(sj.read_text())["per_second"]; n = len(rows)
    c = np.array([r.get("conf_at0", np.nan) if (r.get("dub_act") or 0) >= 0.3 and r.get("n", 0) >= 10 else np.nan for r in rows])
    score = [(np.nanmean(c[i:i + a.w]) if np.isfinite(c[i:i + a.w]).sum() >= a.w * 0.6 else np.inf, i) for i in range(0, max(1, n - a.w + 1))]
    wins = []
    for sc, i in sorted(score):
        if not np.isfinite(sc) or len(wins) >= a.k: break
        if all(abs(i - j) >= a.w for _, j in wins): wins.append((round(float(sc), 3), i))
    res = [{"t0": float(rows[i]["t"]), "t1": float(rows[i]["t"]) + a.w, "conf_at0_mean": sc} for sc, i in sorted(wins, key=lambda x: x[1])]
    out.write_text(json.dumps({"video": a.video, "windows": res, "global": json.loads(sj.read_text()).get("global")}, indent=1)); print(json.dumps(res))


if __name__ == "__main__":
    main()
