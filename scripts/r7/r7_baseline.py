#!/usr/bin/env python3
"""r7 F: the phase-2 QUALITY BASELINE (m11) and the gate every speed change must pass against it.
  freeze: collect the per-video metrics of a merged candidate (artifacts/r4m/<TAG>) into one json
      python scripts/r7/r7_baseline.py freeze --tag m11 --audio /workspace/dub/handoff/in/audio-r7-v3a-all5 --out reports/experiments/r7-face/quality_baseline.json
  gate:   compare a new arm with the baseline; a metric fails when it is worse than the baseline by more than its tolerance
      python scripts/r7/r7_baseline.py gate --tag m12 --audio <dir> --baseline reports/experiments/r7-face/quality_baseline.json [--out F.md]
Metrics (per video): twitch_sum, skin_sum (r5/r6 twitch json), SyncNet on dub speech + dub-only, lip flash strong, whole-frame gate
(worst %, frames > 0.2 %), pressed lips = closed output mouth while the source talks and the dub is silent (s), mouth / dub mismatch (s),
window-joint seam ratios (resid, low-freq), mouth sharpness vs source, chin flicker / jaw jitter ratios, ring colour |dL| worst 1 s,
spec pass (resolution / fps / frames / h264), speed s/video-min (face stage total)."""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np

B = pathlib.Path("/workspace/dub/artifacts/r4m"); TW = pathlib.Path("/workspace/dub/artifacts/r6f/twitch"); FG = pathlib.Path("/workspace/dub/artifacts/r7f/frame_gate")
HERE = pathlib.Path(__file__).resolve().parent
# metric -> (direction, tolerance): "low" = lower is better, "high" = higher is better; tolerance is absolute unless it ends with '%'
TOL = {"twitch_sum": ("low", "15%"), "skin_sum": ("low", 1.0), "sync_speech": ("high", 0.3), "sync_dub_only": ("high", 0.3), "flash_strong": ("low", 3),
       "frame_gate_worst_pct": ("low", 0.2), "frame_gate_over": ("low", 0), "pressed_lips_s": ("low", 0.5), "mouth_audio_mismatch_s": ("low", 0.5),
       "joints_resid": ("low", 0.05), "joints_lowfreq": ("low", 0.05), "sharp_mouth": ("high", 0.03), "chin_flicker": ("low", 0.1), "jaw_jitter": ("low", 0.1),
       "colour_worst_1s_dL": ("low", 1.0), "spec_pass": ("high", 0), "s_per_video_min": ("low", 600)}   # speed: absolute cap, not relative


def g(d, *ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d: return None
        d = d[k]
    return d


def collect(tag, audio, ids):
    R = {}
    for id_ in ids:
        d = B / tag; s = json.loads((d / f"{id_}_{tag}.score.json").read_text()); m = json.loads((d / f"{id_}_{tag}.manifest.json").read_text())
        try: t = json.loads((TW / tag / f"{id_}.twitch.json").read_text())["summary"]
        except Exception: t = {}
        try: f = json.loads((FG / tag / f"{id_}.json").read_text())
        except Exception: f = {}
        r = {"twitch_sum": t.get("twitch_sum"), "skin_sum": t.get("skin_sum"), "sync_speech": g(s, "sync_speech", "conf_speech"), "sync_dub_only": g(s, "syncnet", "dub_only", "conf"),
             "flash_strong": s.get("flash_strong"), "frame_gate_worst_pct": f.get("worst_pct"), "frame_gate_over": f.get("frames_over"), "frame_gate_verdict": f.get("verdict"),
             "mouth_audio_mismatch_s": g(s, "mouth_audio", "mouth_audio_mismatch_s"), "joints_resid": g(s, "joints", "raw", "resid", "joint_over_inner"), "joints_lowfreq": g(s, "joints", "raw", "lowfreq", "joint_over_inner"),
             "sharp_mouth": g(s, "face_v4", "sharp_mouth_ls_core"), "chin_flicker": g(s, "r4face", "chin_flicker", "ratio_out_over_src"), "jaw_jitter": g(s, "r4face", "jaw_jitter", "ratio_out_over_src"),
             "colour_worst_1s_dL": g(s, "r4face", "colour_worst_1s_abs_dL"), "spec_pass": 1 if g(s, "spec_9_4", "pass") else 0, "wer_final_mix": g(s, "asr", "wer_final_mix"),
             "s_per_video_min": g(m, "r4v", "s_per_video_min", "total"), "face_frames": g(s, "r4face", "frames_with_face"), "ls_frames": g(s, "coverage", "ls_frames"), "audio_md5": s.get("audio_md5")}
        R[id_] = r
    if audio:   # pressed lips: closed output mouth while the source talks and the dub is silent
        out = subprocess.run([sys.executable, str(HERE / "r7_closed_mouth.py"), "--ids", *ids, "--variants", f"{tag}:{audio}", "--out", "/tmp/r7_closed_mouth_tmp.json"], capture_output=True, text=True)
        try:
            cm = json.loads(pathlib.Path("/tmp/r7_closed_mouth_tmp.json").read_text())
            for id_ in ids:
                if id_ in cm and tag in cm[id_]: R[id_]["pressed_lips_s"] = cm[id_][tag]["closed_while_source_talks_dub_silent_s"]; R[id_]["open_mouth_dub_silent_s"] = cm[id_][tag]["open_while_dub_silent_s"]
        except Exception as e: print("closed_mouth failed:", e, out.stderr[-300:])
    return R


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("cmd", choices=["freeze", "gate"])
    ap.add_argument("--tag", required=True); ap.add_argument("--audio", default=""); ap.add_argument("--ids", nargs="+", default=["05", "04", "03", "01", "02"])
    ap.add_argument("--baseline", default="reports/experiments/r7-face/quality_baseline.json"); ap.add_argument("--out", default="")
    a = ap.parse_args(); R = collect(a.tag, a.audio, a.ids)
    if a.cmd == "freeze":
        doc = {"tag": a.tag, "audio": a.audio, "tolerances": {k: {"better": v[0], "tol": v[1]} for k, v in TOL.items()}, "videos": R}
        p = pathlib.Path(a.out or a.baseline); p.write_text(json.dumps(doc, indent=1)); print("== baseline frozen:", p)
        for id_, r in R.items(): print(id_, {k: r[k] for k in ("twitch_sum", "sync_speech", "flash_strong", "pressed_lips_s", "joints_resid", "sharp_mouth", "s_per_video_min")})
        return
    base = json.loads(pathlib.Path(a.baseline).read_text()); fails = []; lines = [f"# gate {a.tag} vs quality baseline {base['tag']}", "", "| video | metric | baseline | new | tol | verdict |", "|---|---|---:|---:|---|---|"]
    for id_ in a.ids:
        b = base["videos"].get(id_, {}); n = R[id_]
        for k, (better, tol) in TOL.items():
            bv, nv = b.get(k), n.get(k)
            if bv is None or nv is None: continue
            if k == "s_per_video_min": ok = nv <= tol; cmp_ = f"<= {tol}"
            else:
                t = abs(bv) * float(tol[:-1]) / 100 if isinstance(tol, str) and tol.endswith("%") else float(tol)
                ok = (nv <= bv + t) if better == "low" else (nv >= bv - t); cmp_ = f"{'+' if better == 'low' else '-'}{tol}"
            v = "ok" if ok else "**FAIL**"; lines.append(f"| {id_} | {k} | {bv} | {nv} | {cmp_} | {v} |")
            if not ok: fails.append(f"{id_} {k}: {nv} vs baseline {bv}")
    lines += ["", "## Verdict: " + ("**PASS** (no metric worse than the baseline beyond its tolerance; the user WATCH still decides)" if not fails else "**FAIL**"), *[f"- {x}" for x in fails]]
    txt = "\n".join(lines); print(txt)
    if a.out: pathlib.Path(a.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
