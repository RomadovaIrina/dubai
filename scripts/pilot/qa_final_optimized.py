#!/usr/bin/env python3
"""Visual-QA helper for the final optimized pipeline outputs (no pipeline parameters are touched here).

For every <NN>_optimized.mp4 + manifest produced by run_clean_pipeline_05.py --video-backend optimized:
  * independent validation: streams, resolution, duration delta, frame count vs the 25 fps master, every frame
    decodable / non-blank, dubbed audio really muxed (correlation with the dubbed 16 kHz track of the run),
    no LatentSync / face-not-detected failure, CodeFormer not invoked;
  * manifest["qa"] block: speech-gated seconds, pass-through seconds, gate transitions, LatentSync segments,
    VALID / SMALL / NO_FACE / NO_SPEECH frame counts, validation checks;
  * up to --max-clips boundary clips (~5 s, centred on NO_SPEECH <-> LATENT_SYNC transitions) with a burnt-in label
    and a green top bar on frames where LatentSync output is active -> <out-dir>/boundaries/<NN>_boundary_<k>.mp4.

    source scripts/env.sh
    python scripts/pilot/qa_final_optimized.py --out-dir qa_outputs/final_optimized --work-dir /path/to/qa_work 01 02 03 04 05
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
FPS = 25
SR = 16000


def sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True).stdout


def probe(p: pathlib.Path) -> dict:
    j = json.loads(sh(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(p)]))
    v = next((s for s in j["streams"] if s["codec_type"] == "video"), None)
    a = next((s for s in j["streams"] if s["codec_type"] == "audio"), None)
    return {"duration_s": round(float(j["format"].get("duration", 0)), 3), "has_video": v is not None, "has_audio": a is not None,
            "width": int(v["width"]) if v else None, "height": int(v["height"]) if v else None,
            "fps": eval(v["avg_frame_rate"]) if v and v.get("avg_frame_rate", "0/0") != "0/0" else None,
            "audio_codec": a["codec_name"] if a else None, "size_bytes": p.stat().st_size}


def decode_check(p: pathlib.Path) -> dict:
    import cv2
    cap = cv2.VideoCapture(str(p)); n = blank = bad = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        n += 1
        if fr is None or fr.size == 0 or not np.isfinite(fr.astype(np.float32)).all():
            bad += 1
        elif float(fr.std()) < 1e-3:
            blank += 1
    cap.release()
    return {"decoded_frames": n, "undecodable_frames": bad, "blank_frames": blank}


def audio16k(p: pathlib.Path) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-i", str(p), "-vn", "-ac", "1", "-ar", str(SR), "-f", "f32le", "-"],
                         check=True, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def corr(a: np.ndarray, b: np.ndarray) -> float:
    n = min(len(a), len(b)); a = a[:n].astype(np.float64); b = b[:n].astype(np.float64)
    if n == 0 or a.std() == 0 or b.std() == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def transitions(segs: list[dict]) -> list[dict]:
    """Adjacent segment pairs where one side is LATENT_SYNC and the other a NO_SPEECH pass-through."""
    out = []
    segs = sorted(segs, key=lambda s: s["start_frame"])
    for a, b in zip(segs, segs[1:]):
        la, lb = a["action"] == "LATENT_SYNC", b["action"] == "LATENT_SYNC"
        na, nb = a.get("classification") == "NO_SPEECH", b.get("classification") == "NO_SPEECH"
        if (la and nb) or (na and lb):
            out.append({"frame": b["start_frame"], "t": round(b["start_frame"] / FPS, 3),
                        "direction": "LATENT_SYNC -> NO_SPEECH" if la else "NO_SPEECH -> LATENT_SYNC",
                        "left": {"index": a["index"], "action": a["action"], "frames": a["frames"]},
                        "right": {"index": b["index"], "action": b["action"], "frames": b["frames"]}})
    return out


def pick(items: list, k: int) -> list:
    if len(items) <= k:
        return items
    if k <= 1:
        return [items[len(items) // 2]]
    idx = sorted({round(i * (len(items) - 1) / (k - 1)) for i in range(k)})
    return [items[i] for i in idx]


def boundary_clip(out_video: pathlib.Path, dst: pathlib.Path, n: str, tr: dict, duration: float, ls_intervals: list, clip_s: float) -> dict:
    start = max(0.0, tr["t"] - clip_s / 2); end = min(duration, start + clip_s); start = max(0.0, end - clip_s)
    parts = []
    for a, b in ls_intervals:                       # green bar while LatentSync output is on screen (clip-relative times)
        a_, b_ = max(a, start) - start, min(b, end) - start
        if b_ > a_:
            parts.append(f"drawbox=x=0:y=0:w=iw:h=10:color=lime@0.85:t=fill:enable='between(t,{a_:.3f},{b_:.3f})'")
    label = f"{n} boundary {tr['t']:.2f}s  {tr['direction'].replace('LATENT_SYNC', 'LS').replace('->', '>')}"
    parts.append(f"drawtext=text='{label}':x=8:y=16:fontsize=17:fontcolor=white:box=1:boxcolor=black@0.55")
    parts.append("drawtext=text='green bar = LatentSync active':x=8:y=40:fontsize=15:fontcolor=lime:box=1:boxcolor=black@0.55")
    parts.append(f"drawtext=text='%{{eif\\:t+{start:.3f}\\:d}}.%{{eif\\:mod(t*100\\,100)\\:d\\:2}}s':x=8:y=h-30:fontsize=18:fontcolor=white:box=1:boxcolor=black@0.55")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-nostdin", "-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(out_video),
                    "-vf", ",".join(parts), "-c:v", "libx264", "-crf", "18", "-c:a", "aac", "-b:a", "128k", str(dst)], check=True)
    return {"file": str(dst), "clip_start_s": round(start, 3), "clip_end_s": round(end, 3), **tr}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", help="video ids, e.g. 01 02 03 04 05")
    ap.add_argument("--out-dir", default="qa_outputs/final_optimized")
    ap.add_argument("--work-dir", required=True, help="--work-dir given to run_clean_pipeline_05.py (work_<id>/dubbed_16k.wav is used)")
    ap.add_argument("--max-clips", type=int, default=3)
    ap.add_argument("--clip-seconds", type=float, default=5.0)
    ap.add_argument("--run-log", default=None, help="run_all.log with '===== <file> exit <code>' lines")
    a = ap.parse_args()
    out_dir = pathlib.Path(a.out_dir); bdir = out_dir / "boundaries"; bdir.mkdir(parents=True, exist_ok=True)
    exit_codes = {}
    if a.run_log and pathlib.Path(a.run_log).exists():
        for line in pathlib.Path(a.run_log).read_text().splitlines():
            if line.startswith("=====") and " exit " in line:
                parts = line.split(); exit_codes[parts[1].split(".")[0]] = int(parts[3])
    rows = []
    for n in a.videos:
        out = out_dir / f"{n}_optimized.mp4"; mp = out_dir / f"{n}_optimized.manifest.json"
        if not out.exists() or not mp.exists():
            rows.append({"video": n, "output": str(out), "validation": "MISSING"}); print(f"{n}: missing output/manifest"); continue
        man = json.loads(mp.read_text()); ls = man["lipsync"]; src = man["source"]
        op = probe(out); dec = decode_check(out)
        dub = pathlib.Path(a.work_dir) / f"work_{n}" / "dubbed_16k.wav"; own = pathlib.Path(a.work_dir) / f"work_{n}" / "audio16k.wav"
        oa = audio16k(out)
        c_dub = corr(oa, audio16k(dub)) if dub.exists() else None
        c_own = corr(oa, audio16k(own)) if own.exists() else None
        segs = sorted(ls["segments"], key=lambda s: s["start_frame"])
        trs = transitions(segs)
        ls_iv = [(s["start_frame"] / FPS, s["end_frame"] / FPS) for s in segs if s["action"] == "LATENT_SYNC"]
        failed = [s for s in segs if s["action"] == "PASS_THROUGH_LS_FAILED"]
        fnd = [s for s in segs if "Face not detected" in str(s.get("error", ""))]
        tol = man["validation"]["tolerance_s"]; delta = round(op["duration_s"] - src["duration_s"], 3)
        checks = {
            "output_exists_nonempty": op["size_bytes"] > 0,
            "video_and_audio_streams": op["has_video"] and op["has_audio"],
            "dubbed_audio_present": c_dub is not None and c_dub > 0.95,
            "source_resolution_preserved": (op["width"], op["height"]) == (src["video"]["width"], src["video"]["height"]),
            "duration_delta_within_tolerance": abs(delta) <= tol,
            "frame_count_preserved": dec["decoded_frames"] == ls["master_frames"] == ls["assembly"]["frames_written"],
            "all_frames_decodable": dec["undecodable_frames"] == 0,
            "no_blank_frames": dec["blank_frames"] == 0,
            "no_latentsync_crash": len(failed) == 0 and exit_codes.get(n, 0) == 0 and "error" not in man,
            "no_face_not_detected_crash": len(fnd) == 0,
            "codeformer_invoked_no": man.get("codeformer") == "DISABLED (not invoked)" and str(ls.get("codeformer", "")).startswith("OFF"),
            "runner_validation_pass": bool(man["validation"]["pass"]),
        }
        gate = ls.get("speech_gate", {})
        qa = {"speech_gated_seconds": gate.get("gated_in_seconds"), "gate_out_seconds": gate.get("gated_out_seconds"),
              "latentsync_seconds": ls["latentsync_seconds_of_video"], "pass_through_seconds": ls["pass_through_seconds_of_video"],
              "speech_gate_transitions": len(trs), "latentsync_segments": ls["by_action"].get("LATENT_SYNC", {}).get("segments", 0),
              "frames": {c: ls["frame_classes"].get(c, 0) for c in ("VALID_FACE", "SMALL_FACE", "NO_FACE", "NO_SPEECH")},
              "by_action": ls["by_action"], "output_probe": op, "decode_check": dec, "duration_delta_s": delta, "tolerance_s": tol,
              "audio_corr_output_vs_dubbed_track": None if c_dub is None else round(c_dub, 4),
              "audio_corr_output_vs_original_audio": None if c_own is None else round(c_own, 4),
              "runner_exit_code": exit_codes.get(n), "checks": checks, "pass": all(checks.values()), "transitions": trs, "boundary_clips": []}
        for k, tr in enumerate(pick(trs, a.max_clips), 1):
            qa["boundary_clips"].append(boundary_clip(out, bdir / f"{n}_boundary_{k:02d}.mp4", n, tr, op["duration_s"], ls_iv, a.clip_seconds))
        man["qa"] = qa; mp.write_text(json.dumps(man, indent=1, default=str))
        rows.append({"video": n, "output": str(out), "duration": op["duration_s"], "frames": dec["decoded_frames"], "transitions": len(trs),
                     "ls_s": qa["latentsync_seconds"], "pt_s": qa["pass_through_seconds"], "validation": "PASS" if qa["pass"] else "FAIL", "checks": checks})
        print(f"{n}: {'PASS' if qa['pass'] else 'FAIL'}  frames {dec['decoded_frames']}  transitions {len(trs)}  clips {len(qa['boundary_clips'])}  "
              f"corr dubbed {qa['audio_corr_output_vs_dubbed_track']} / original {qa['audio_corr_output_vs_original_audio']}", flush=True)
        if not qa["pass"]:
            print("   failed checks:", [k for k, v in checks.items() if not v])
    print("\n| video | output path | duration s | frame count | gate transitions | LS seconds | pass-through seconds | validation |")
    print("|---|---|---:|---:|---:|---:|---:|---|")
    for r in rows:
        if r["validation"] == "MISSING":
            print(f"| {r['video']} | {r['output']} | - | - | - | - | - | MISSING |"); continue
        print(f"| {r['video']} | {r['output']} | {r['duration']} | {r['frames']} | {r['transitions']} | {r['ls_s']} | {r['pt_s']} | {r['validation']} |")
    (out_dir / "qa_summary.json").write_text(json.dumps(rows, indent=1, default=str))
    return 0 if all(r["validation"] == "PASS" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
