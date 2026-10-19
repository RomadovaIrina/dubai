#!/usr/bin/env python3
"""Round 3 D: GPU audit table from r3_gpu_audit.sh runs: per stage / model load -> wall s, device peak VRAM (nvidia-smi, 200 ms),
torch CUDA peak (allocator, since stage start), mean GPU utilisation, s per video-minute.
    python scripts/quality/r3_gpu_audit.py --dir /workspace/dub/artifacts/r3d/audit/gpu [--json out.json]"""
from __future__ import annotations
import argparse, datetime, glob, json, pathlib
import numpy as np

# runner stage -> (pipeline step, device of its model). Stages not listed are CPU / IO glue.
DEVICE = {"vad": "Silero VAD", "asr": "faster-whisper large-v3", "diarization": "pyannote 3.1", "segmentation": "Qwen2.5-7B (segmentation)",
          "translation_load": "Qwen2.5-7B load", "translation": "Qwen2.5-7B (translation)", "tts_load": "Chatterbox load", "qa_load": "TTS QA load (Whisper + wav2vec2)",
          "tts": "Chatterbox TTS + QA + fit (Qwen refit)", "background_sep": "Spleeter 2stems (CPU, TF venv)", "background_mix": "bed mix",
          "speech_gate": "Silero VAD (dub)", "face_stack_mediapipe": "MediaPipe face stack (CPU)", "latentsync_load": "LatentSync load",
          "latentsync": "LatentSync 1.6", "final_assembly": "composite + mux (25 fps)"}


def nvsmi(p):
    t, m, u = [], [], []
    for line in open(p):
        x = [s.strip() for s in line.split(",")]
        if len(x) < 3: continue
        try:
            ts = datetime.datetime.strptime(x[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc).timestamp()
            t.append(ts); m.append(float(x[1])); u.append(float(x[2]))
        except ValueError: continue
    return np.array(t), np.array(m), np.array(u)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--json", default=None); a = ap.parse_args()
    D = pathlib.Path(a.dir); out = {}
    durs = {}
    for mp in glob.glob(str(D / "**" / "*_final.manifest.json"), recursive=True):
        m = json.loads(pathlib.Path(mp).read_text()); durs[pathlib.Path(mp).name.split("_")[0]] = m["source"]["duration_s"]
    for sl in sorted(D.glob("*.stages.jsonl")):
        tag = sl.name.split(".")[0]; nv = D / f"{tag}.nvsmi.csv"
        T, M, U = nvsmi(nv) if nv.exists() else (np.array([]),) * 3
        # nvidia-smi timestamps are local time of the container (UTC here); clock offset vs time.time() checked below
        recs = [json.loads(l) for l in open(sl) if l.strip()]
        for r in recs:
            job = r.get("job") or tag; sel = (T >= r["t0"] - 0.1) & (T <= r["t1"] + 0.1)
            r["dev_peak_mib"] = int(M[sel].max()) if sel.any() else None; r["util_mean"] = round(float(U[sel].mean()), 1) if sel.any() else None
            r["vmin"] = durs.get(job, 0) / 60
            out.setdefault(job, []).append(r)
        out.setdefault("_device_peak", {})[tag] = int(M.max()) if len(M) else None
    if a.json: pathlib.Path(a.json).write_text(json.dumps(out, indent=1))
    print(f"# GPU audit: {D.name}\n")
    for job, recs in out.items():
        if job.startswith("_"): continue
        vm = recs[0]["vmin"] or float("nan")
        print(f"## {job} ({vm * 60:.1f} s video)\n\n| stage | model | kind | wall s | s/video-min | device peak MiB (nvidia-smi) | torch peak MiB | GPU util % |\n|---|---|---|---:|---:|---:|---:|---:|")
        tot = None
        for r in recs:
            if r["name"] == "total": tot = r; continue
            print(f"| {r['name']} | {DEVICE.get(r['name'], '') if r['kind'] == 'stage' else 'model load' + (' (resident)' if r.get('resident') else '')} | {r['kind']} | {r['wall_s']:.1f} | "
                  f"{r['wall_s'] / vm:.1f} | {r['dev_peak_mib'] if r['dev_peak_mib'] is not None else '–'} | {r.get('torch_peak_alloc_mib', r.get('torch_alloc_delta_mib', '–'))} | {r['util_mean'] if r['util_mean'] is not None else '–'} |")
        if tot is None and recs:
            t0, t1 = min(r["t0"] for r in recs), max(r["t1"] for r in recs); pk = [r["dev_peak_mib"] for r in recs if r["dev_peak_mib"] is not None]
            tot = {"wall_s": t1 - t0, "dev_peak_mib": max(pk) if pk else None, "util_mean": None}
        if tot: print(f"| **total** | | | **{tot['wall_s']:.1f}** | **{tot['wall_s'] / vm:.0f}** | {tot['dev_peak_mib']} | | {tot['util_mean']} |")
        print()
    print("device peak per run (nvidia-smi):", out.get("_device_peak"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
