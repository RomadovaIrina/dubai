#!/usr/bin/env python3
"""r16 TZ 5.6: variant B for a finished KeySync job + the eye material.
  1. B video  = fps_native_merge.py (native source fps; the face layer motion-compensated) muxed with A's AUDIO STREAM COPIED (bit-identical)
  2. windows  = the 3-s spots with the most head motion WHILE the dub speaks (master landmarks speed / inter-ocular distance x dub RMS)
  3. clips    = 0.5x "A 25 fps | B <fps> | source" side by side (each panel shown at the source fps: A's 25 fps judder stays visible), audio 0.5x
  4. numbers  = audio md5 A vs B; judder = coefficient of variation of the frame-to-frame face-region change at the source fps (duplicated
                25 fps frames -> zeros + jumps -> high; the source itself is the reference)
    python fps_ab.py --e2e JOB/e2e --a JOB/<id>.mp4 --src SRC_NATIVE.mp4 --out DIR [--modes flow,nearest] [--n 2] [--len 3]"""
import argparse, json, pathlib, subprocess, sys
import numpy as np
H = pathlib.Path(__file__).resolve().parent


def run(cmd): r = subprocess.run(cmd, capture_output=True, text=True); return r.stdout + r.stderr


def probe(p):
    s = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", p],
                                  capture_output=True, text=True).stdout)["streams"][0]
    n, d = s["r_frame_rate"].split("/"); return s["width"], s["height"], float(n) / float(d), s["r_frame_rate"]


def amd5(p): return run(["ffmpeg", "-loglevel", "error", "-i", p, "-map", "0:a", "-c", "copy", "-f", "md5", "-"]).strip()


def windows(e2e, n, ln):
    import soundfile as sf
    z = np.load(e2e / "work/ks/master_mp.npz"); lm = z["lm"]; pr = z["presence"]
    c = lm.mean(1); iod = np.linalg.norm(lm[:, 33] - lm[:, 263], axis=1) + 1e-6   # mediapipe outer eye corners
    sp = np.r_[0, np.linalg.norm(np.diff(c, axis=0), axis=1) / iod[1:]]; sp[~pr] = 0
    y, sr = sf.read(str(e2e / "work/ks/dub_f0.wav"), dtype="float32", always_2d=True); y = y.mean(1); h = sr // 25
    rms = np.sqrt(np.array([np.mean(y[i * h:(i + 1) * h] ** 2) if i * h < len(y) else 0 for i in range(len(sp))]) + 1e-12)
    voiced = rms > 0.1 * np.percentile(rms, 95); w = int(ln * 25); score = np.convolve(sp * voiced, np.ones(w), "valid")
    out = []
    for i in np.argsort(-score):
        if all(abs(i - j) >= w for j in out): out.append(int(i))
        if len(out) == n: break
    return [(j / 25, float(score[j] / w)) for j in sorted(out)]


def judder(path, t0, ln, fps, box):
    x0, y0, x1, y1 = box
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-ss", f"{t0}", "-t", f"{ln}", "-i", path, "-vf", f"fps={fps},crop={x1 - x0}:{y1 - y0}:{x0}:{y0},scale=160:-2",
                          "-pix_fmt", "gray", "-f", "rawvideo", "-"], capture_output=True).stdout
    fh = int(round(160 * (y1 - y0) / (x1 - x0) / 2)) * 2; f = np.frombuffer(raw, np.uint8); nf = len(f) // (160 * fh)
    if nf < 3: return None
    fr = f[: nf * 160 * fh].reshape(nf, fh, 160).astype(np.float32); d = np.abs(np.diff(fr, axis=0)).mean((1, 2))
    return round(float(d.std() / (d.mean() + 1e-6)), 3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--e2e", required=True); ap.add_argument("--a", required=True); ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--modes", default="flow,nearest"); ap.add_argument("--n", type=int, default=2); ap.add_argument("--len", type=float, default=3.0)
    a = ap.parse_args(); e2e = pathlib.Path(a.e2e); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); res = {"src": a.src}
    W, Hh, fps, rate = probe(a.src); res["src_fps"] = fps; res["a_fps"] = probe(a.a)[2]
    for m in a.modes.split(","):
        bv = out / f"B_{m}_video.mp4"; log = run([sys.executable, str(H / "fps_native_merge.py"), "--hi", a.src, "--lo-src", str(e2e / "work/ks/master_25fps.mp4"),
                                                 "--lo-comp", str(e2e / "work/ks_face.mp4"), "--out", str(bv), "--mode", m, "--threads", "16"])
        res[f"merge_{m}"] = json.loads(log.split("== fps_native_merge ", 1)[1].splitlines()[0]) if "== fps_native_merge {" in log else log[-300:]
        bf = out / f"B_{m}.mp4"; run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(bv), "-i", a.a, "-map", "0:v", "-map", "1:a", "-c", "copy", "-movflags", "+faststart", str(bf)])
        res[f"audio_md5_equal_{m}"] = amd5(str(bf)) == amd5(a.a) and amd5(a.a) != ""
    wins = windows(e2e, a.n, a.len); res["windows"] = wins
    z = np.load(e2e / "work/ks/master_mp.npz"); lm = z["lm"]; wh = z["wh"]
    mouth = lm[:, [61, 291, 13, 14, 0, 17]]; cx, cy = np.nanmedian(mouth[..., 0]), np.nanmedian(mouth[..., 1])
    iod = float(np.nanmedian(np.linalg.norm(lm[:, 33] - lm[:, 263], axis=1))); sx, sy = W / wh[0], Hh / wh[1]   # landmarks in master px -> source px
    half = int(0.9 * iod * sx); box = (max(0, int(cx * sx) - half), max(0, int(cy * sy) - half), min(W, int(cx * sx) + half), min(Hh, int(cy * sy) + half))
    res["judder"] = {}
    names = {"flow": "B1 flow blend", "flownear": "B2 flow nearest", "nearest": "B nearest", "blend": "B blend"}; ms = a.modes.split(",")
    for i, (t0, sc) in enumerate(wins):
        t0 = max(0.0, t0 - 0.2); bfs = [out / f"B_{m}.mp4" for m in ms]
        res["judder"][f"w{i}"] = {"t0": round(t0, 2), "A_25fps": judder(a.a, t0, a.len, fps, box), **{f"B_{m}": judder(str(b), t0, a.len, fps, box) for m, b in zip(ms, bfs)},
                                  "source": judder(a.src, t0, a.len, fps, box)}
        ph = 640 if Hh >= W else 400   # vertical: 640 px panels; landscape: 400 px high
        lab = lambda s: f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:text='{s}':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6"
        ins = [a.a] + [str(b) for b in bfs] + [a.src]; labs = ["A 25 fps (now)"] + [f"{names.get(m, m)} {fps:g} fps" for m in ms] + [f"source {fps:g} fps"]
        vf = "".join(f"[{j}:v]fps={rate},scale=-2:{ph},{lab(l)}[p{j}];" for j, l in enumerate(labs)) + "".join(f"[p{j}]" for j in range(len(ins))) + \
             f"hstack=inputs={len(ins)},setpts=2*PTS[v];[1:a]atempo=0.5[au]"
        clip = out / f"ab_w{i}_{t0:.1f}s_half_speed.mp4"; cmd = ["ffmpeg", "-loglevel", "error", "-y"]
        for p in ins: cmd += ["-ss", f"{t0}", "-t", f"{a.len}", "-i", p]
        run(cmd + ["-filter_complex", vf, "-map", "[v]", "-map", "[au]", "-c:v", "libx264", "-crf", "20", "-preset", "fast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                   "-movflags", "+faststart", str(clip)])
        res["judder"][f"w{i}"]["clip"] = str(clip); res["judder"][f"w{i}"]["clip_mb"] = round(clip.stat().st_size / 1e6, 2) if clip.exists() else None
    (out / "fps_ab.json").write_text(json.dumps(res, indent=1)); print("FPS_AB", json.dumps(res))


if __name__ == "__main__":
    main()
