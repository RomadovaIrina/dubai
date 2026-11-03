#!/usr/bin/env python3
"""r13 12.3: the regression summary table from /workspace/dub/r13s/reg (dabai_job status.json + e2e timing / gpu_time + SyncNet + tempo audit)."""
import json, pathlib, re, sys
R = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/workspace/dub/r13s/reg"); rows = []
R12 = {"m18": 8.33, "m03": 8.21, "old03": 7.57, "old05": 4.23}   # k5 SyncNet, r11 REPORT (the r12 baseline)
for st in sorted(R.glob("*/status.json")):
    i = st.parent.name; s = json.loads(st.read_text()); e = st.parent / "e2e"
    t = json.loads((e / "timing.json").read_text()) if (e / "timing.json").exists() else {}
    g = json.loads((e / "gpu_time.json").read_text()) if (e / "gpu_time.json").exists() else {}
    def sync(f):
        p = st.parent / "sync" / f
        m = re.search(r"sync_speech \S+: (\S+) over (\d+) s", p.read_text()) if p.exists() else None
        return (None if not m or m.group(1) == "nan" else float(m.group(1)), int(m.group(2)) if m else 0)
    so, ss = sync("out.txt"), sync("src.txt")
    ta = json.loads((R / f"{i}.tempo.json").read_text()) if (R / f"{i}.tempo.json").exists() else {}
    sw = t.get("stage_wall_s") or {}
    rows.append({"id": i, "status": s["status"], "flags": s["flags"], "vmin": s["video_minutes"], "wall_s": t.get("total_wall_s"),
                 "wall_s_per_vmin": t.get("total_s_per_video_min"), "gpu_busy_nvsmi_s_per_vmin": t.get("gpu_busy_total_s_per_video_min"),
                 "gpu_busy_nvml_s_per_vmin": (g.get("total") or {}).get("gpu_busy_s_per_vmin"), "gpu_weighted_nvml_s_per_vmin": (g.get("total") or {}).get("gpu_weighted_s_per_vmin"),
                 "stage_wall_s": sw, "vram_peak_mib": t.get("vram_peak_mib"), "sync_out": so[0], "sync_out_s": so[1], "sync_src": ss[0], "sync_src_s": ss[1],
                 "sync_r12_k5": R12.get(i), "tempo_p50": (ta.get("tempo_p5_p50_p95") or [None, None])[1], "tempo_phrase_viol": len(ta.get("phrase_violations") or []),
                 "output_matches_source": (s.get("output_format") or {}).get("matches_source"), "messages": [m["en"] for m in s.get("messages", [])]})
(R / "regression.json").write_text(json.dumps(rows, indent=1))
f = lambda x, n=1: "–" if x is None else (f"{x:.{n}f}" if isinstance(x, float) else str(x))
print("| video | min | status | flags | wall s/vmin | GPU-busy s/vmin (NVML busy / weighted) | VRAM peak GB | SyncNet out (r12 k5) | SyncNet source | tempo p50 / phrases outside |")
print("|---|---:|---|---|---:|---|---:|---|---:|---|")
for r in rows:
    print(f"| {r['id']} | {r['vmin']:.2f} | {r['status']} | {', '.join(r['flags']) or '–'} | {f(r['wall_s_per_vmin'], 0)} | {f(r['gpu_busy_nvml_s_per_vmin'], 0)} / {f(r['gpu_weighted_nvml_s_per_vmin'], 0)} | "
          f"{f(r['vram_peak_mib'] / 1024 if r['vram_peak_mib'] else None)} | {f(r['sync_out'], 2)} ({f(r['sync_r12_k5'], 2)}) | {f(r['sync_src'], 2)} | {f(r['tempo_p50'], 2)} / {r['tempo_phrase_viol']} |")
V = sum(r["vmin"] for r in rows); fb = [r for r in rows if r["gpu_busy_nvml_s_per_vmin"]]
print(f"\nall {len(rows)} SUCCESS: {all(r['status'] == 'SUCCESS' for r in rows)}; total {V:.2f} video-min; GPU-busy (NVML) over all: "
      f"{sum(r['gpu_busy_nvml_s_per_vmin'] * r['vmin'] for r in fb) / sum(r['vmin'] for r in fb):.0f} s/vmin; with a face only: "
      f"{sum(r['gpu_busy_nvml_s_per_vmin'] * r['vmin'] for r in fb if 'no_face' not in r['flags']) / sum(r['vmin'] for r in fb if 'no_face' not in r['flags']):.0f} s/vmin")
