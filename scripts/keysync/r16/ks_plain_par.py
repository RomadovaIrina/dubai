#!/usr/bin/env python3
"""r16 T: ks_plain.py on N CPU workers (the paste was the e2e bottleneck: 3.5-6.3 s per video-second serial, 24 s/s on a 4K crop).
The master is cut into chunks; each worker gets its chunk + MARGIN frames of context on both sides (every temporal step of ks_plain - lfm
sigma 6 -> +-18 frames, the align smoothing, the tracked box - sees the same neighbourhood as the serial run), writes only its own frames
(--write-range), and the chunks are joined by stream copy (same x264 settings) + the audio. The global colour match is exact: phase 1 runs
--cm-only per chunk (Lab sums of the sampled skin pixels), combined here into one gain / offset that every chunk loads (--cm-file).
Inputs are sliced per chunk: master-indexed (--master video, --lm / --others / --lip-keep / --scale-fix arrays, box xs / ys) and
KeySync-indexed (--ks video, --ks-lm, --align-ecc), the KeySync window = the chunk window - shift (a local shift keeps the pairing).
    python ks_plain_par.py --jobs N --out OUT.mp4 -- <every ks_plain.py argument except --out>       (--jobs 0 = auto: cores / 6, >= 120 frames per chunk)"""
import argparse, json, math, os, shutil, subprocess, sys, tempfile, time
import concurrent.futures as cf
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=0); ap.add_argument("--out", required=True)
ap.add_argument("--margin", type=int, default=30); ap.add_argument("--min-chunk", type=int, default=120); ap.add_argument("--tmp", default="")
ap.add_argument("rest", nargs=argparse.REMAINDER); a = ap.parse_args()
R = [x for x in a.rest if x != "--"]
def val(flag, default=None):
    return R[R.index(flag) + 1] if flag in R else default
def setv(args, flag, v):
    args = list(args)
    if flag in args: args[args.index(flag) + 1] = str(v)
    else: args += [flag, str(v)]
    return args
nframes = lambda v: int(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", v],
                                       capture_output=True, text=True).stdout.strip())
t0 = time.time(); MST, KSV, SH = val("--master"), val("--ks"), int(val("--shift", 0))
n, nk = nframes(MST), nframes(KSV)
J = a.jobs or max(1, (os.cpu_count() or 8) // 6)
J = max(1, min(J, n // a.min_chunk)); L = math.ceil(n / J)
T = a.tmp or tempfile.mkdtemp(prefix="ksp_", dir=os.path.dirname(os.path.abspath(a.out))); os.makedirs(T, exist_ok=True)
print(f"ks_plain_par: {n} master / {nk} KeySync frames, shift {SH}, {J} chunks of {L} (+-{a.margin} context)", flush=True)
if J == 1:   # short clip: the serial paste
    sys.exit(subprocess.run([sys.executable, os.path.join(HERE, "ks_plain.py")] + R + ["--out", a.out]).returncode)


def cut(src, k0, cnt, out):   # frames k0 .. k0+cnt-1 of a 25 fps CFR video, losslessly (FFV1)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, (k0 - 0.5) / 25):.4f}", "-i", src, "-frames:v", str(cnt), "-an", "-c:v", "ffv1", "-level", "3",
                    "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", out], check=True)


def chunk_args(c):
    wa, wb = c * L, min(n, (c + 1) * L)
    if wa >= wb: return None
    a0, a1 = max(0, wa - a.margin), min(n, wb + a.margin)
    i0, i1 = max(0, a0 - SH), min(nk, a1 - SH + 1)   # the KeySync frames shown in [a0, a1)
    d = f"{T}/c{c:03d}"; os.makedirs(d, exist_ok=True); args = list(R)
    cut(MST, a0, a1 - a0, f"{d}/master.mkv"); args = setv(args, "--master", f"{d}/master.mkv")
    cut(KSV, i0, i1 - i0, f"{d}/ks.mkv"); args = setv(args, "--ks", f"{d}/ks.mkv")
    args = setv(args, "--shift", SH + i0 - a0)   # local: KeySync frame (k - a0) - shift' == global (k - SH) - i0
    z = np.load(val("--lm")); has_j = "j" in z.files
    if has_j: np.savez(f"{d}/lm.npz", **{k: z[k] for k in z.files if k != "j"}, j=z["j"][a0:a1])
    else: np.savez(f"{d}/lm.npz", **{k: (z[k][a0:a1] if z[k].ndim and len(z[k]) == n else z[k]) for k in z.files})
    args = setv(args, "--lm", f"{d}/lm.npz")
    if val("--others") and not has_j:
        o = np.load(val("--others")); np.savez(f"{d}/others.npz", **{k: (o[k][a0:a1] if o[k].ndim and len(o[k]) >= a1 else o[k]) for k in o.files}); args = setv(args, "--others", f"{d}/others.npz")
    for fl, key in (("--ks-lm", "lm"), ("--align-ecc", None)):
        if val(fl):
            q = np.load(val(fl)); np.savez(f"{d}/{fl[2:]}.npz", **{k: (q[k][i0:i1] if q[k].ndim and len(q[k]) == nk else q[k]) for k in q.files}); args = setv(args, fl, f"{d}/{fl[2:]}.npz")
    for fl in ("--lip-keep",):
        if val(fl): np.save(f"{d}/lk.npy", np.load(val(fl))[a0:a1]); args = setv(args, fl, f"{d}/lk.npy")
    if val("--scale-fix"):
        q = np.load(val("--scale-fix")); np.savez(f"{d}/sf.npz", **{k: (q[k][a0:a1] if q[k].ndim and len(q[k]) == n else q[k]) for k in q.files}); args = setv(args, "--scale-fix", f"{d}/sf.npz")
    b = json.load(open(val("--box")))
    if b.get("xs"): b["xs"], b["ys"] = b["xs"][a0:a1], b["ys"][a0:a1]
    json.dump(b, open(f"{d}/box.json", "w")); args = setv(args, "--box", f"{d}/box.json")
    return {"c": c, "dir": d, "args": args, "wr": f"{wa - a0}:{wb - a0}"}


def run(job, extra, log):
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENCV_NUM_THREADS": "2"}
    r = subprocess.run([sys.executable, os.path.join(HERE, "ks_plain.py")] + job["args"] + ["--write-range", job["wr"]] + extra,
                       stdout=open(f"{job['dir']}/{log}", "w"), stderr=subprocess.STDOUT, env=env)
    if r.returncode: raise RuntimeError(f"chunk {job['c']} {log} rc {r.returncode}: {open(job['dir'] + '/' + log).read()[-800:]}")
    return job


with cf.ThreadPoolExecutor(J) as ex: jobs = [j for j in ex.map(chunk_args, range(J)) if j]
t1 = time.time()
if "--no-cm" not in R:   # phase 1: the global colour match from every chunk's samples (exact: sums of the same sampled pixels)
    with cf.ThreadPoolExecutor(len(jobs)) as ex: list(ex.map(lambda j: run(j, ["--cm-only", "--cm-file", f"{j['dir']}/cm.npz", "--out", f"{j['dir']}/none.mp4"], "cm.log"), jobs))
    S = [np.load(f"{j['dir']}/cm.npz") for j in jobs]; S = [x for x in S if int(x["n"]) > 0]; N_ = sum(int(x["n"]) for x in S)
    mk = sum(x["s1k"] for x in S) / N_; mm = sum(x["s1m"] for x in S) / N_
    sk = np.sqrt(np.maximum(sum(x["s2k"] for x in S) / N_ - mk ** 2, 0)); sm = np.sqrt(np.maximum(sum(x["s2m"] for x in S) / N_ - mm ** 2, 0))
    gain = np.clip(sm / np.maximum(sk, 1e-3), 0.9, 1.1).astype(np.float32); off = (mm - mk * gain).astype(np.float32)
    np.savez(f"{T}/cm.npz", gain=gain, off=off); print(f"global colour match (Lab, {N_} px from {len(S)} chunks): gain {gain.round(3).tolist()} offset {off.round(2).tolist()}", flush=True)
t2 = time.time()
with cf.ThreadPoolExecutor(len(jobs)) as ex: list(ex.map(lambda j: run(j, (["--cm-file", f"{T}/cm.npz"] if "--no-cm" not in R else []) + ["--out", f"{j['dir']}/out.mp4"], "paste.log"), jobs))
t3 = time.time()
open(f"{T}/list.txt", "w").write("".join(f"file '{j['dir']}/out.mp4'\n" for j in jobs))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", f"{T}/list.txt", "-i", val("--audio"), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", a.out], check=True)
got = nframes(a.out)
print(f"plain paste: KeySync {nk} frames, master {n}, shift {SH} -> {a.out} ({got} frames; {J} workers: slicing {t1 - t0:.0f} s, colour match {t2 - t1:.0f} s, "
      f"paste {t3 - t2:.0f} s, total {time.time() - t0:.0f} s)", flush=True)
if got != n: print(f"WARNING: {got} frames written != master {n}", flush=True); sys.exit(1)
if not os.environ.get("KS_PAR_KEEP"): shutil.rmtree(T, ignore_errors=True)
