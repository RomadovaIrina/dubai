#!/usr/bin/env python3
"""r6 G: extra window metrics for one ls_window_ab.py clip (run after r4_sweep_metrics.py, which writes master_<id>_<t0>.mp4 + _mp.npz):
  twitch       r5_twitch.py score (output vs the master window = the source at 25 fps, master landmarks): twitch_sum, n_events, top20_mean, p99
  chin wave    r4_chin_wave.py (rigid variant included): mean / worst 2-s window
  mouth/dub    r4_mouth_audio.py on the dub-only clip: mouth_audio_mismatch_s
Merged into <id>_<tag>.metrics.json.
    python scripts/r6/r6_win_extra.py --win-dir OUT --id 04 --tag G0_base_10 --t0 10"""
import argparse, json, pathlib, subprocess, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; PY = sys.executable; FPS = 25


def run(args):
    p = subprocess.run([PY, *map(str, args)], capture_output=True, text=True)
    if p.returncode: print(p.stdout[-800:], p.stderr[-1500:], file=sys.stderr)
    return p.returncode == 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--win-dir", required=True); ap.add_argument("--id", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--t0", type=float, required=True); a = ap.parse_args(); d = pathlib.Path(a.win_dir)
    mw = d / f"master_{a.id}_{a.t0:g}.mp4"; mnpz = d / f"master_{a.id}_{a.t0:g}_mp.npz"; clip = d / f"{a.id}_{a.tag}_dubonly.mp4"
    lmt = d / f"master_{a.id}_{a.t0:g}_lmt.npz"
    if not lmt.exists():
        lm = np.load(mnpz)["lm"]; np.savez(lmt, lm=lm, t=np.arange(len(lm)) / FPS)
    r = {}
    tj = d / f"{a.id}_{a.tag}.twitch.json"
    if run([HERE.parent / "r5" / "r5_twitch.py", "score", "--video", clip, "--src", mw, "--lm", lmt, "--out", tj]):
        s = json.loads(tj.read_text()); s = s.get("summary", s)
        r.update({f"twitch_{k}": s.get(k) for k in ("twitch_sum", "n_events", "top20_mean", "p99")})
    cj = d / f"{a.id}_{a.tag}.chin.json"
    if run([HERE.parent / "r4" / "r4_chin_wave.py", "--video", clip, "--src", mw, "--lm", lmt, "--out", cj]):
        c = json.loads(cj.read_text()); r.update({f"chin_{k}": v for k, v in c.items() if isinstance(v, (int, float))})
    drv = next(d.glob(f"work_*_{a.tag}/drive_16k.wav"), None); mj = d / f"{a.id}_{a.tag}.mouth.json"
    if drv and run([HERE.parent / "r4" / "r4_mouth_audio.py", "--video", clip, "--dub", drv, "--out", mj]):
        m = json.loads(mj.read_text()); r.update({k: v for k, v in m.items() if isinstance(v, (int, float)) and "mismatch" in k})
    mf = d / f"{a.id}_{a.tag}.metrics.json"; base = json.loads(mf.read_text()) if mf.exists() else {}
    base.update(r); mf.write_text(json.dumps(base, indent=1)); print(json.dumps(r))


if __name__ == "__main__":
    main()
