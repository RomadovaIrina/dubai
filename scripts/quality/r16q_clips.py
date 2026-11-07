#!/usr/bin/env python3
"""r16 Q: 0.5x review clips of ONLY the flagged spots: source | ours, the face region, source time + the detector names burnt in.
Reads the r16q_detect.py JSON (+ its --npz geometry cache for the face box). The ranges of all detectors are padded (--pad), merged,
ranked by severity and capped (--max). Writes <out>/<id>_flag<NN>_<t0>-<t1>_<dets>.mp4, one <id>_ALL_FLAGS_05x.mp4 (all spots in a row)
and <id>_flags.tsv.
    python3 r16q_clips.py --det det.json --npz det.npz --id clip01 --out /workspace/dub/outbox/r16_q/clip01 [--pad 0.6 --max 10]"""
import argparse, json, pathlib, subprocess
import numpy as np

ap = argparse.ArgumentParser(); ap.add_argument("--det", required=True); ap.add_argument("--npz", required=True); ap.add_argument("--id", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--pad", type=float, default=0.6); ap.add_argument("--max", type=int, default=10)
ap.add_argument("--maxlen", type=float, default=6.0); ap.add_argument("--skip", default="", help="detectors to leave out of the clips")
a = ap.parse_args(); R = json.load(open(a.det)); z = np.load(a.npz, allow_pickle=True); boxes = z["boxes"]; W, H = R["size"]
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
pr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", R["ours"]], capture_output=True, text=True).stdout)
dur = float(pr["format"]["duration"])
SEV = {"black": 1.0, "slide": 1.2, "overlap": 1.5, "texture": 0.6, "explode": 1.3, "edge": 0.8, "flicker": 0.7}
ev = []
for d, v in R["det"].items():
    if d in a.skip.split(","): continue
    for t0, t1, pk in v["ranges"]:
        ev.append([max(0, t0 - a.pad), min(dur, t1 + a.pad), {d}, SEV.get(d, 1) * (t1 - t0 + 0.2) * (1 + min(float(pk or 0), 5) / 5)])
ev.sort(key=lambda e: e[0]); M = []
for e in ev:
    if M and e[0] <= M[-1][1] + 0.4 and e[1] - M[-1][0] <= a.maxlen: M[-1][1] = max(M[-1][1], e[1]); M[-1][2] |= e[2]; M[-1][3] += e[3]
    else: M.append(e)
for m in M:
    if m[1] - m[0] > a.maxlen: m[1] = m[0] + a.maxlen
M = sorted(sorted(M, key=lambda e: -e[3])[: a.max], key=lambda e: e[0])
sw = W / float(json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", R["src"]],
                                         capture_output=True, text=True).stdout)["streams"][0]["width"])
files = []; rows = ["n\tt0\tt1\tdetectors\tscore\tfile"]
for n, (t0, t1, ds, sc) in enumerate(M, 1):
    k0, k1 = int(t0 * 25), max(int(t0 * 25) + 1, int(t1 * 25)); b = np.nanmedian(boxes[k0:min(k1, len(boxes))], 0)
    s = b[2] * 1.15; cx, cy = b[0] + b[2] / 2, b[1] + b[2] / 2; s = min(s, W, H); x0 = int(np.clip(cx - s / 2, 0, W - s)) // 2 * 2; y0 = int(np.clip(cy - s / 2, 0, H - s)) // 2 * 2; s = int(s) // 2 * 2
    lab = "+".join(sorted(ds)); fn = out / f"{a.id}_flag{n:02d}_{t0:05.2f}-{t1:05.2f}_{lab}.mp4"
    tc = f"drawtext=text='%{{pts\\:hms\\:{t0:.3f}}}':x=8:y=8:fontsize=26:fontcolor=white:box=1:boxcolor=black@0.6"
    fc = (f"[0:v]fps=25,scale={W}:{H},crop={s}:{s}:{x0}:{y0},scale=540:540,{tc},drawtext=text='source':x=8:y=h-36:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.6[a];"
          f"[1:v]fps=25,scale={W}:{H},crop={s}:{s}:{x0}:{y0},scale=540:540,drawtext=text='ours  {lab}':x=8:y=h-36:fontsize=24:fontcolor=yellow:box=1:boxcolor=black@0.6[b];"
          f"[a][b]hstack,setpts=2*PTS[v];[1:a]atempo=0.5[au]")
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", R["src"], "-ss", f"{t0:.3f}", "-t", f"{t1 - t0:.3f}", "-i", R["ours"],
           "-filter_complex", fc, "-map", "[v]", "-map", "[au]", "-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p", "-r", "25", "-c:a", "aac", "-b:a", "128k", str(fn)]
    if subprocess.run(cmd).returncode == 0: files.append(fn)
    rows.append(f"{n}\t{t0:.2f}\t{t1:.2f}\t{lab}\t{sc:.2f}\t{fn.name}")
(out / f"{a.id}_flags.tsv").write_text("\n".join(rows) + "\n")
if files:
    lst = out / "concat.txt"; lst.write_text("".join(f"file '{f.name}'\n" for f in files))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(out / f"{a.id}_ALL_FLAGS_05x.mp4")]); lst.unlink()
print(f"{a.id}: {len(files)} flag clips -> {out}")
