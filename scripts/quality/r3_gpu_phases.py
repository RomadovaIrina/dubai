#!/usr/bin/env python3
"""Round 3 D: phase table of an r3_gpu_audit run (stage walls from the run manifest, VRAM peaks / means from the 200 ms nvidia-smi
samples between the stage-log marks): audio-pre (VAD, ASR, Qwen segmentation, diarization, translation), TTS (Chatterbox + QA + fit),
post-TTS audio (align, bed, mix, gate), face stack (CPU), LatentSync (load + inference + assembly).
    python scripts/quality/r3_gpu_phases.py --dir AUDIT/<variant> [--json]"""
import argparse, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); import r3_gpu_audit as ga
ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--json", action="store_true"); a = ap.parse_args(); D = pathlib.Path(a.dir)
res = {}
for sl in sorted(D.glob("*.stages.jsonl")):
    tag = sl.name.split(".")[0]; T, M, U = ga.nvsmi(D / f"{tag}.nvsmi.csv"); recs = [json.loads(l) for l in open(sl) if l.strip()]
    for job in dict.fromkeys(r.get("job") or tag for r in recs):
        R = [r for r in recs if (r.get("job") or tag) == job]
        mp = next(iter(D.glob(f"**/{job}_final.manifest.json")), None); man = json.loads(mp.read_text()) if mp else {}; st = man.get("stage_seconds", {})
        vm = man.get("source", {}).get("duration_s", 60) / 60
        def tt(name, which="t1", last=False):
            x = [r for r in R if r["name"] == name]; x = x[-1:] if last else x[:1]; return x[0][which] if x else None
        jstart = min(r["t0"] for r in R); nxt = [min(r["t0"] for r in recs if (r.get("job") or tag) == j) for j in dict.fromkeys(r.get("job") or tag for r in recs) if j != job]
        jend = min([t for t in nxt if t > jstart], default=T[-1] if len(T) else jstart)
        b = {"start": jstart, "tts0": tt("translation") , "tts1": tt("qwen_released", "t0") or tt("alignment", "t0"), "post1": tt("speech_gate"),
             "face1": tt("face_stack_mediapipe"), "end": jend}
        phases = [("audio-pre: VAD, ASR, Qwen seg, pyannote, translation", "start", "tts0"), ("TTS: Chatterbox + QA + fit", "tts0", "tts1"),
                  ("post-TTS audio: align, Spleeter bed, mix, gate", "tts1", "post1"), ("face stack (MediaPipe, CPU)", "post1", "face1"),
                  ("LatentSync: load + inference + assembly", "face1", "end")]
        rows = []
        for name, x, y in phases:
            if b.get(x) is None or b.get(y) is None: continue
            sel = (T >= b[x]) & (T <= b[y])
            rows.append({"phase": name, "wall_s": round(b[y] - b[x], 1), "s_per_vmin": round((b[y] - b[x]) / vm, 1),
                         "peak_mib": int(M[sel].max()) if sel.any() else None, "mean_mib": int(M[sel].mean()) if sel.any() else None,
                         "util_mean": round(float(U[sel].mean()), 1) if sel.any() else None})
        res[job] = {"video_s": round(vm * 60, 1), "phases": rows, "manifest_stage_s": {k: st.get(k) for k in ("tts", "latentsync", "latentsync_load", "segmentation", "translation", "asr", "diarization", "total")},
                    "job_wall_s": round(jend - jstart, 1), "job_peak_mib": int(M[(T >= jstart) & (T <= jend)].max()) if len(T) else None}
if a.json: print(json.dumps(res, indent=1)); sys.exit(0)
for job, r in res.items():
    print(f"### {D.name} / {job} ({r['video_s']} s video): job wall {r['job_wall_s']} s = {r['job_wall_s'] / (r['video_s'] / 60):.0f} s/video-min, peak {r['job_peak_mib']} MiB")
    print("| phase | wall s | s/video-min | peak VRAM MiB | mean VRAM MiB | GPU util % |\n|---|---:|---:|---:|---:|---:|")
    for p in r["phases"]: print(f"| {p['phase']} | {p['wall_s']} | {p['s_per_vmin']} | {p['peak_mib']} | {p['mean_mib']} | {p['util_mean']} |")
    print(f"manifest stage s: {r['manifest_stage_s']}\n")
