#!/usr/bin/env python3
"""r8 speed: production seconds per stage of one scripts/r8/r8_audio.sh run from the raw input, per video and in s / video-minute.
Production = what the dub needs: the r6 pipeline (ASR, diarization, segmentation, Qwen translation, Chatterbox TTS + QA, separation, track),
retake, clause takes + burst_align, close_inner + soften, lip_place, hush + final soften, remix. NOT counted: the faith judge, metrics, coverage rows.
    python scripts/r8/speed_table.py <tag> [ids]  -> markdown table on stdout"""
import os
import json, pathlib, subprocess, sys

A = pathlib.Path("/workspace/dub/artifacts/r4a")


def dur(id_):
    src = sorted(pathlib.Path(os.environ.get("DUB_INPUTS", "/workspace/dub/inputs/test_videos_raw")).glob(f"{id_}.*"))[0]
    return float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(src)]))


def main():
    tag = sys.argv[1]; ids = sys.argv[2:] or ["01", "02", "03", "04", "05"]; rows = []; tot = {}; dtot = 0.0
    for id_ in ids:
        r = {}
        pre = A / (__import__("os").environ.get("PRE") or f"{tag}pre")
        if pre.exists():
            st = json.loads((pre / "runs" / f"{id_}.steps.json").read_text()); r["pipeline"] = st.get("audio_or_reuse", 0.0)
            man = next((pre / "runs" / id_).glob("work_*/*_r4a.manifest.json"), None) or next((pre / "runs").glob(f"{id_}_r4a.manifest.json"), None)
            if man:
                ss = json.loads(man.read_text()).get("stage_seconds") or {}
                for k in ("asr", "diarization", "translation", "tts", "tts_load", "qa_load", "segmentation", "background_sep", "background_mix"):
                    if isinstance(ss.get(k), (int, float)): r["  " + k] = ss[k]
        s = A / f"{tag}src" / f"{id_}.steps.json"
        if s.exists():
            st = json.loads(s.read_text()); r["retake"] = st.get("retake", 0.0); r["clause+burst_align"] = st.get("burst_align", 0.0); r["close_inner+soften"] = st.get("close_inner", 0.0)
            op = A / f"{tag}src" / "oneproc_jobs.times.json"   # R8_ONEPROC: the one-process stage's own per-video timing
            if op.exists():
                for x in json.loads(op.read_text()):
                    if x["id"] != id_: continue
                    if x["step"] == "retake": r["retake"] = x["seconds"]
                    else: r["clause+burst_align"] = r.get("clause+burst_align", 0.0) + x["seconds"]
        c = A / tag / "runs" / f"{id_}.steps.json"
        if c.exists():
            st = json.loads(c.read_text()); r["lip_place"] = st.get("lip_place", 0.0); r["hush+soften"] = st.get("hush", 0.0); r["remix"] = st.get("remix", 0.0)
        r["TOTAL"] = sum(v for k, v in r.items() if not k.startswith("  ")); d = dur(id_); dtot += d
        rows.append((id_, d, r))
        for k, v in r.items(): tot[k] = tot.get(k, 0.0) + v
    keys = list(tot)
    print(f"| stage | " + " | ".join(f"{i} ({d:.0f} s)" for i, d, _ in rows) + " | all | s / video-min |")
    print("|---|" + "---:|" * (len(rows) + 2))
    for k in keys:
        print(f"| {k} | " + " | ".join(f"{r.get(k, 0):.1f}" for _, _, r in rows) + f" | {tot[k]:.1f} | {tot[k] / (dtot / 60):.1f} |")


if __name__ == "__main__":
    main()
