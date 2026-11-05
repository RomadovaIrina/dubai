#!/usr/bin/env python3
"""r15 K: the results table from fast/<arm>/{run.log KS_JOB lines, eval_<id>/eval.json} -> markdown + JSON.
Face GPU-s/vmin = (wall - landmarks) / video s * 60, warm (the last KS_JOB of each video); the per-stage split per video-minute (m18)."""
import json, os, sys
K = "/workspace/dub/keysync/fast"; ARMS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["prod", "trt", "fast4", "fast4v", "fast2v", "lcm4"]
IDS = [("q18", "m18"), ("m03", "m03"), ("qo3", "old03")]
rows, js = [], {}
for a in ARMS:
    if not os.path.exists(f"{K}/{a}/run.log"): continue
    jobs = {}
    for l in open(f"{K}/{a}/run.log", errors="ignore"):
        if "KS_JOB " in l: j = json.loads(l.split("KS_JOB ", 1)[1]); jobs[j["tag"]] = j
    r = {"arm": a}
    for i, v in IDS:
        j = jobs.get(i); e = {}
        try: e = json.load(open(f"{K}/{a}/eval_{i}/eval.json"))
        except Exception: pass
        if j:
            st = j["stages_s"]; m = 60 / j["video_s"]
            r[v] = {"gpu_s_vmin": round((j["wall_s"] - st.get("extract_video_landmarks", 0)) / j["video_s"] * 60, 1),
                    "interp_unet": round((st.get("sample_interpolation", 0) - st.get("vae_decode_interp", 0)) * m, 1), "interp_dec": round(st.get("vae_decode_interp", 0) * m, 1),
                    "kf": round(st.get("sample_keyframes", 0) * m, 1), "enc": round(st.get("compute_video_embedding", 0) * m, 1), "save": round(st.get("save_audio_video", 0) * m, 1),
                    "peak_gb": j.get("peak_vram_gb")}
        else: r[v] = {}
        r[v].update({k: e.get(k) for k in ("sync_speech_at0", "flicker_lips_mean", "flicker_lips_p95", "flicker_low_mean")})
    rows.append(r); js[a] = r
f = lambda x, d=1: "–" if x is None else (f"{x:.{d}f}" if isinstance(x, float) else str(x))
print("| arm | s/vmin m18 / m03 / old03 | SyncNet m18 / m03 / old03 | lips flicker m18 / m03 / old03 | m18 split: interp UNet / interp decode / kf / enc / save |")
print("|---|---|---|---|---|")
for r in rows:
    g = lambda k, d=1: " / ".join(f(r[v].get(k), d) for _, v in IDS)
    s = r["m18"]; sp = " / ".join(f(s.get(k), 0) for k in ("interp_unet", "interp_dec", "kf", "enc", "save")) if s else "–"
    print(f"| {r['arm']} | {g('gpu_s_vmin', 0)} | {g('sync_speech_at0', 3)} | {g('flicker_lips_mean', 2)} | {sp} |")
json.dump(js, open("/workspace/dub/keysync/fast/table.json", "w"), indent=1)
