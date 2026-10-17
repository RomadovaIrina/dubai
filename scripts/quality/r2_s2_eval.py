#!/usr/bin/env python3
"""Round 2 stage 2: metrics for whole-video 25 fps runs in r2/s2 (slide / nose base / mask-vs-head, SyncNet mix + dub-only, common metric
v4, LatentSync / MediaPipe / rigid-motion s per video-minute) -> r2/s2/<id>_<tag>.eval.json, and a markdown table.
    python scripts/quality/r2_s2_eval.py --tags s2_clean0 s2_flowhm s2_safe --ids 04 05 03 01 [--no-v4] [--table OUT.md]
"""
from __future__ import annotations
import argparse, json, os, pathlib, subprocess, sys
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot")); os.environ.setdefault("DUB_ROOT", str(HERE.parents[1]))
R2 = pathlib.Path("/workspace/dub/artifacts/lipsync-temporal/r2"); S2 = R2 / "s2"
PY = "/venv/dabai/bin/python"


def run_eval(vid, tag, v4=True):
    base = S2 / f"{vid}_{tag}"; man_p = base.with_suffix(".manifest.json"); out = base.with_suffix(".eval.json")
    if not man_p.exists(): return None
    ev = json.loads(out.read_text()) if out.exists() else {}
    man = json.loads(man_p.read_text()); mp4 = base.with_suffix(".mp4"); n = man["lipsync"]["master_frames"]
    if "slide" not in ev:
        sj = base.with_suffix(".slide.json")
        cmd = [PY, str(HERE / "slide_metric.py"), "--video", str(mp4), "--master", str(R2 / f"masters/{vid}/master_25fps.mp4"), "--npz", str(R2 / f"masters/{vid}/master_mediapipe.npz"),
               "--offset", "0", "--frames", f"0:{n}", "--manifest", str(man_p), "--rigid", str(R2 / f"anchor/{vid}_rigid.npz"), "--series", "--out", str(sj)]
        bx = pathlib.Path(str(base) + "_maskboxes.npz")
        if bx.exists(): cmd += ["--boxes", str(bx)]
        subprocess.run(cmd, check=True); ev["slide"] = json.loads(sj.read_text()); ev["slide"].pop("series", None)
    if "sync" not in ev:
        from syncnet_common import eval_syncnet
        dub = pathlib.Path(str(base) + "_dubonly.mp4")
        if not dub.exists():
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", str(mp4), "-i", man["dubbed_track"]["aac"], "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", str(dub)], check=True)
        ev["sync"] = {}
        for k, p in (("dub_only", dub), ("mix", mp4)):
            try: x = eval_syncnet(p); ev["sync"][k] = {"conf": x["confidence"], "av_offset": x["av_offset_frames"]}
            except Exception as e: ev["sync"][k] = {"error": str(e)[-200:]}
    if v4 and "v4" not in ev:
        vj = base.with_suffix(".v4.json")
        subprocess.run([PY, "/workspace/dub/common/mouth_jitter_hr.py", "--video", str(mp4), "--reference", str(R2 / f"masters/{vid}/master_25fps.mp4"), "--manifest", str(man_p),
                        "--tag", tag, "--out", str(vj)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if vj.exists():
            a = json.loads(vj.read_text())["all"]
            ev["v4"] = {"mouth_resid": a["resid"]["mouth"]["ratio"], "mouth_sharp": a["sharp"]["mouth"]["ratio"], "lf_sharp": a["sharp"]["lowerface"]["ratio"],
                        "lf_shear": a["shear"]["lowerface"]["ratio"], "lf_shear95": a["shear95"]["lowerface"]["ratio"], "lf_jerk": a["jerk"]["lowerface"]["ratio"],
                        "mouth_jerk": a["jerk"]["mouth"]["ratio"], "upper_resid": a["resid"]["upper"]["ratio"]}
    st = man["stage_seconds"]; dur_min = man["source"]["duration_s"] / 60
    ls_frames = sum(s["end_frame"] - s["start_frame"] for s in man["lipsync"]["segments"] if s["action"] == "LATENT_SYNC")
    ev["speed"] = {"latentsync_s": man["lipsync"].get("latentsync_inference_total_s"), "ls_s_per_video_min": round(man["lipsync"].get("latentsync_inference_total_s", 0) / dur_min, 1),
                   "ls_frames": ls_frames, "mediapipe_s_per_video_min": round(st.get("prep_mediapipe", 0) / dur_min, 1),
                   "rigid_s_per_video_min": round(st.get("prep_rigid_motion", 0) / dur_min, 1), "video_stage_total_s": st.get("total"),
                   "video_stage_s_per_video_min": round(st.get("total", 0) / dur_min, 1)}
    ev["validation"] = man["validation"]["pass"]
    out.write_text(json.dumps(ev, indent=1)); return ev


def f(x, n=3):
    return "–" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--ids", nargs="+", default=["04", "05", "03", "01"])
    ap.add_argument("--no-v4", action="store_true"); ap.add_argument("--table", default=None)
    a = ap.parse_args(); rows = []
    for vid in a.ids:
        for tag in a.tags:
            ev = run_eval(vid, tag, not a.no_v4)
            if ev: rows.append((vid, tag, ev)); print(vid, tag, "done", flush=True)
    L = ["| video | arm | nose slide mean / p95 | steps > 0.3 px | nose abs mean | cheek slide | mask vs head mean / p95 | SyncNet mix / dub (AV) | lf shear / p95 | mouth resid | mouth / lf sharp | LS s/video-min | MediaPipe + rigid s/video-min | PASS |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    import numpy as np
    for vid, tag, ev in rows:
        sl = ev["slide"]; nb = sl["regions"]["nosebase"]; ch = sl["regions"]["cheeks"]; mk = sl.get("mask_centre_resid_vs_head_px") or {}
        ser = json.loads((S2 / f"{vid}_{tag}.slide.json").read_text()).get("series", {}).get("nosebase", [])
        x = np.array(ser, float); dd = np.linalg.norm(np.diff(x, axis=0), axis=1) if len(x) > 2 else np.array([]); dd = dd[np.isfinite(dd)]
        j = f"{100 * (dd > 0.3).mean():.1f} %" if len(dd) else "–"
        sy = ev["sync"]; v4 = ev.get("v4", {}); sp = ev["speed"]
        L.append(f"| {vid} | {tag} | {f(nb['slide_px']['mean'])} / {f(nb['slide_px']['p95'])} | {j} | {f(nb['abs_px']['mean'])} | {f(ch['slide_px']['mean'])} | "
                 f"{f(mk.get('mean'),2)} / {f(mk.get('p95'),2)} | {f(sy.get('mix',{}).get('conf'),2)} / {f(sy.get('dub_only',{}).get('conf'),2)} ({f(sy.get('dub_only',{}).get('av_offset'))}) | "
                 f"{f(v4.get('lf_shear'),2)} / {f(v4.get('lf_shear95'),2)} | {f(v4.get('mouth_resid'),2)} | {f(v4.get('mouth_sharp'),2)} / {f(v4.get('lf_sharp'),2)} | "
                 f"{f(sp['ls_s_per_video_min'],0)} | {f(sp['mediapipe_s_per_video_min'],0)} + {f(sp['rigid_s_per_video_min'],0)} | {'PASS' if ev['validation'] else 'FAIL'} |")
    txt = "\n".join(L); print(txt)
    if a.table: pathlib.Path(a.table).write_text(txt + "\n")


if __name__ == "__main__":
    main()
