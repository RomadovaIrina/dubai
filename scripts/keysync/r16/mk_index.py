#!/usr/bin/env python3
"""r16 T: final_0811 INDEX.md + copies (faststart mp4 + status.json) for Irina's eye check.
    python3 mk_index.py OUT_DIR name=JOBDIR[:audio_dir] ...      (speech@0 = Q's eval_final.sh on the final vs the job's OWN dub, speaker track)"""
import glob, json, os, re, shutil, subprocess, sys
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); R = "/workspace/dub/dubai"; rows = []
for arg in sys.argv[2:]:
    name, jd = arg.split("=", 1)
    st = json.load(open(f"{jd}/status.json")); e2e = f"{jd}/e2e"; log = open(f"{e2e}/logs/e2e.log", errors="ignore").read()
    fk = open(f"{e2e}/logs/face_keysync.log", errors="ignore").read() if os.path.exists(f"{e2e}/logs/face_keysync.log") else ""
    mp4 = st.get("output") or next(iter(sorted(glob.glob(f"{jd}/*.mp4"))), None)
    dst = f"{OUT}/{name}.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", mp4, "-map", "0", "-c", "copy", "-movflags", "+faststart", dst], check=True)
    json.dump(st, open(f"{OUT}/{name}.status.json", "w"), indent=1, ensure_ascii=False)
    for extra in glob.glob(f"{jd}/*.ass"): shutil.copy(extra, f"{OUT}/{name}.ass")
    probes = re.findall(r"KS_PROBE f\d+ av_offset (-?\d+) conf ([\d.]+) -> dub delay (-?\d+) ms", fk + log)
    sd = re.findall(r"KS_SHOTS delay (-?\d+) ms", fk)
    delay = (f"{probes[0][2]} ms (offset {probes[0][0]}, conf {probes[0][1]})" if probes else "-") + (f"; shots use {sd[0]} ms" if sd else "")
    gpu = re.findall(r"NVML GPU-busy ([\d.]+) s/vmin", log); nshots = re.findall(r"KS_SHOTS_RUN \S+: (\d+) segments, (\d+) with a face", fk)
    ad = glob.glob(f"/workspace/dub/artifacts/r4a/r9_{st['job_id']}/{st['job_id']}/dubbed_16k.wav")
    sp = "-"
    if ad:
        r = subprocess.run(["bash", f"{R}/scripts/r16q/eval_final.sh", dst, ad[0], f"{jd}/eval_final", "--noise", "0", "--label", name], capture_output=True, text=True, cwd=R)
        m = re.search(r'"speech_at0": ([\d.\-]+).*?"n_speech_sec": (\d+)', r.stdout); sp = f"{m.group(1)} ({m.group(2)} s)" if m else "eval failed"
    p = st.get("params", {}); fl = st.get("flags", [])
    rows.append({"name": name, "status": st["status"], "commit": st.get("versions", {}).get("pipeline", "?"), "face": p.get("face"), "subs": p.get("subs", {}).get("enabled"),
                 "wm": p.get("watermark", {}).get("enabled"), "delay": delay, "shots": f"{nshots[0][0]} / {nshots[0][1]} with a face" if nshots else "1",
                 "warn": ", ".join(sorted({w["code"] for w in st.get("warnings", [])})) or "-", "flags": ", ".join(f for f in fl if f.startswith(("ks_", "face_", "beard", "codeformer", "audio_reused"))),
                 "speech": sp, "gpu": gpu[0] if gpu else "-", "wall": st.get("wall_s"), "vmin": st.get("video_minutes")})
    print("INDEXED", name, rows[-1]["status"], rows[-1]["speech"], flush=True)
with open(f"{OUT}/INDEX.md", "w") as f:
    f.write("# r16 T final e2e set (08–09.11): full raw-source runs for Irina's eye\n\n"
            "Defaults: face=auto → KeySync (beards too), KeySync ref 0 + gate (thr 0.12, cap 0.25, hi 8, lo 2), delay probe as is, paste shift 0, "
            "multi-shot (ks_shots) + α-paste through clean crossfades, Spleeter bed (htdemucs not on this box), 25 fps output.\n"
            "speech@0 = Q's eval_final.sh on the final vs the job's OWN fresh dub (speaker track; not comparable with the golden runs on the accepted dubs).\n\n"
            "| video | status | commit | delay probe | shots | warnings | key flags | speech@0 | GPU-s/vmin (NVML) | wall s / video min |\n|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r['name']} | {r['status']} | {r['commit']} | {r['delay']} | {r['shots']} | {r['warn']} | {r['flags']} | {r['speech']} | {r['gpu']} | {r['wall']} / {r['vmin']} |\n")
print("INDEX written", OUT)
