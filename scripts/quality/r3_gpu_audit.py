#!/usr/bin/env python3
"""Round 3 D: GPU audit table from r3_gpu_audit.sh runs: per stage / model load -> wall s, device peak VRAM (nvidia-smi, 200 ms; sees llama.cpp),
torch CUDA peaks (allocated / reserved, since stage start), device VRAM at the END of the stage (after its unload = baseline check), mean GPU
utilisation, s per video-minute; per-job model load costs; the pipeline peak (max over the whole run) and a VRAM-over-time plot.
    python scripts/quality/r3_gpu_audit.py --dir /workspace/dub/artifacts/r3d/audit/gpu [--json out.json] [--plot-dir DIR]"""
from __future__ import annotations
import argparse, datetime, glob, json, pathlib
import numpy as np

DEVICE = {"vad": "Silero VAD", "asr": "faster-whisper large-v3", "diarization": "pyannote 3.1", "segmentation": "Qwen2.5-7B (segmentation)",
          "translation_load": "Qwen2.5-7B load", "translation": "Qwen2.5-7B (translation)", "tts_load": "Chatterbox load", "qa_load": "TTS QA load (Whisper + wav2vec2)",
          "tts": "Chatterbox TTS + QA + fit (Qwen refit)", "background_sep": "Spleeter 2stems (torch port, GPU; DUB_SPLEETER_BACKEND=tf = CPU)", "background_mix": "bed mix",
          "speech_gate": "Silero VAD (dub)", "face_stack_mediapipe": "MediaPipe face stack (CPU)", "latentsync_load": "LatentSync load",
          "latentsync": "LatentSync 1.6", "final_assembly": "composite + mux (25 fps)", "qwen_released": "Qwen released (mark)"}


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


def plot(tag, T, M, U, recs, out):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    if not len(T): return None
    t0 = T[0]; fig, ax = plt.subplots(figsize=(14, 5)); ax2 = ax.twinx()
    ax2.plot(T - t0, U, color="#bbbbbb", lw=0.6, label="GPU util %"); ax2.set_ylim(0, 105); ax2.set_ylabel("GPU util %")
    ax.plot(T - t0, M / 1024, color="#1f5fbf", lw=1.4, label="device VRAM used (nvidia-smi)"); ax.set_ylabel("GiB"); ax.set_xlabel("s since run start")
    ax.axhline(32607 / 1024, color="#c03030", ls="--", lw=0.8); ax.text(1, 32607 / 1024 - 1.2, "RTX 5090 32 GiB", color="#c03030", fontsize=8)
    ax.axhline(24 * 0.9766, color="#d08000", ls=":", lw=0.8); ax.text(1, 24 * 0.9766 - 1.2, "24 GB card", color="#d08000", fontsize=8)
    cols = ["#e8f0ff", "#fff3e0"]; k = 0
    for r in recs:
        if r["kind"] != "stage" or r["wall_s"] < 2: continue
        ax.axvspan(r["t0"] - t0, r["t1"] - t0, color=cols[k % 2], zorder=0); k += 1
        ax.text((r["t0"] + r["t1"]) / 2 - t0, 0.5 + (k % 3) * 1.1, r["name"], rotation=90, fontsize=7, ha="center", va="bottom")
    pk = int(np.argmax(M)); ax.annotate(f"peak {M[pk] / 1024:.1f} GiB", (T[pk] - t0, M[pk] / 1024), xytext=(10, 10), textcoords="offset points", fontsize=9)
    ax.set_ylim(0, 34); ax.set_title(f"GPU audit {tag}: device VRAM over time (200 ms nvidia-smi samples), stages shaded")
    fig.tight_layout(); fig.savefig(out, dpi=110); plt.close(fig); return out


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--json", default=None); ap.add_argument("--plot-dir", default=None)
    a = ap.parse_args(); D = pathlib.Path(a.dir); out = {}; plots = {}
    durs = {}
    for mp in glob.glob(str(D / "**" / "*_final.manifest.json"), recursive=True):
        m = json.loads(pathlib.Path(mp).read_text()); durs[pathlib.Path(mp).name.split("_")[0]] = m["source"]["duration_s"]
    for sl in sorted(D.glob("*.stages.jsonl")):
        tag = sl.name.split(".")[0]; nv = D / f"{tag}.nvsmi.csv"
        T, M, U = nvsmi(nv) if nv.exists() else (np.array([]),) * 3
        recs = [json.loads(l) for l in open(sl) if l.strip()]
        for r in recs:
            job = r.get("job") or tag; sel = (T >= r["t0"] - 0.1) & (T <= r["t1"] + 0.1)
            r["dev_peak_mib"] = int(M[sel].max()) if sel.any() else None; r["util_mean"] = round(float(U[sel].mean()), 1) if sel.any() else None
            r["vmin"] = durs.get(job, 0) / 60
            out.setdefault(job, []).append(r)
        out.setdefault("_device_peak", {})[tag] = int(M.max()) if len(M) else None
        out.setdefault("_baseline", {})[tag] = int(M[:5].min()) if len(M) else None
        if a.plot_dir and len(T): plots[tag] = plot(f"{D.name}/{tag}", T, M, U, recs, pathlib.Path(a.plot_dir) / f"vram_{D.name}_{tag}.png")
    if a.json: pathlib.Path(a.json).write_text(json.dumps(out, indent=1, default=str))
    print(f"# GPU audit: {D.name}\n")
    for job, recs in out.items():
        if job.startswith("_"): continue
        vm = recs[0]["vmin"] or float("nan")
        print(f"## {job} ({vm * 60:.1f} s video)\n\n| stage | model / device | kind | wall s | s/video-min | device peak MiB | torch peak alloc / reserved MiB | device MiB at stage end | GPU util % |\n|---|---|---|---:|---:|---:|---:|---:|---:|")
        tot = None
        for r in recs:
            if r["name"] == "total": tot = r; continue
            kind = r["kind"] if r["kind"] != "load" else ("load (resident)" if r.get("resident") else "load")
            tp = f"{r.get('torch_peak_alloc_mib', '–')} / {r.get('torch_peak_reserved_mib', '–')}" if r["kind"] != "load" else f"+{r.get('torch_alloc_delta_mib', '–')} alloc"
            print(f"| {r['name']} | {DEVICE.get(r['name'], '')} | {kind} | {r['wall_s']:.1f} | {r['wall_s'] / vm:.1f} | {r['dev_peak_mib'] if r['dev_peak_mib'] is not None else '–'} | "
                  f"{tp} | {r.get('dev_used_end_mib', '–')} | {r['util_mean'] if r['util_mean'] is not None else '–'} |")
        if tot is None and recs:
            t0, t1 = min(r["t0"] for r in recs), max(r["t1"] for r in recs); pk = [r["dev_peak_mib"] for r in recs if r["dev_peak_mib"] is not None]
            tot = {"wall_s": t1 - t0, "dev_peak_mib": max(pk) if pk else None, "util_mean": None}
        if tot: print(f"| **span of timed stages** | | | **{tot['wall_s']:.1f}** | **{tot['wall_s'] / vm:.0f}** | **{tot['dev_peak_mib']}** | | | |")
        loads = [r for r in recs if r["kind"] == "load"]
        if loads:
            ls = sum(r["wall_s"] for r in loads)
            print(f"\nmodel loads in this job: {len(loads)}, {ls:.1f} s = {ls / vm:.1f} s/video-min = {100 * ls / vm / 300:.1f} % of the 300 s/min budget: "
                  + ", ".join(f"{r['name']} {r['wall_s']:.1f} s" for r in loads))
        print()
    print("pipeline peak per run (nvidia-smi, max over the whole run, MiB):", out.get("_device_peak"), " baseline at start:", out.get("_baseline"))
    if plots: print("plots:", {k: str(v) for k, v in plots.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
