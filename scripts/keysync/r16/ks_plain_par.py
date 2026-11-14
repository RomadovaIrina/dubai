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
    # r16 K1: the source landmarks / others stay WHOLE (they are indexed by the landmark row J[k]); only the master -> row map j is cut, whatever len(lm) is
    z = np.load(val("--lm")); nl = len(z["lm"]); jj = z["j"] if "j" in z.files else np.arange(nl)
    jj = np.asarray(jj)[np.minimum(np.arange(a0, a1), len(jj) - 1)]
    np.savez(f"{d}/lm.npz", **{k: z[k] for k in z.files if k != "j"}, j=jj); args = setv(args, "--lm", f"{d}/lm.npz")
    # KeySync-indexed arrays (--ks-lm rows, the k_ecc M / scale): cut by the KeySync index WHATEVER their length (k_ecc has N rows, KeySync N + 60)
    for fl in ("--ks-lm", "--align-ecc"):
        if val(fl):
            q = np.load(val(fl)); np.savez(f"{d}/{fl[2:]}.npz", **{k: (q[k][i0:i1] if q[k].ndim else q[k]) for k in q.files}); args = setv(args, fl, f"{d}/{fl[2:]}.npz")
    for fl in ("--lip-keep",):
        if val(fl): np.save(f"{d}/lk.npy", np.load(val(fl))[a0:a1]); args = setv(args, fl, f"{d}/lk.npy")
    if val("--scale-fix"):
        q = np.load(val("--scale-fix")); np.savez(f"{d}/sf.npz", **{k: (q[k][a0:a1] if q[k].ndim and len(q[k]) == n else q[k]) for k in q.files}); args = setv(args, "--scale-fix", f"{d}/sf.npz")
    b = json.load(open(val("--box")))
    if b.get("xs"): b["xs"], b["ys"] = b["xs"][a0:a1], b["ys"][a0:a1]
    json.dump(b, open(f"{d}/box.json", "w")); args = setv(args, "--box", f"{d}/box.json")
    args += ["--k-base", str(a0), "--ks-base", str(i0), "--ks-total", str(nk), "--n-total", str(n), "--video-only"]
    return {"c": c, "dir": d, "args": args, "wr": f"{wa - a0}:{wb - a0}"}


def run(job, extra, log):
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENCV_NUM_THREADS": "2"}
    r = subprocess.run([sys.executable, os.path.join(HERE, "ks_plain.py")] + job["args"] + ["--write-range", job["wr"]] + extra,
                       stdout=open(f"{job['dir']}/{log}", "w"), stderr=subprocess.STDOUT, env=env)
    if r.returncode: raise RuntimeError(f"chunk {job['c']} {log} rc {r.returncode}: {open(job['dir'] + '/' + log).read()[-800:]}")
    return job


with cf.ThreadPoolExecutor(J) as ex: jobs = [j for j in ex.map(chunk_args, range(J)) if j]
t1 = time.time()
NOCM, CAVITY, SHARP_AUTO = "--no-cm" in R, float(val("--cavity", 0)), val("--sharp", "0") == "auto"
P1 = (not NOCM) or CAVITY > 0 or SHARP_AUTO
if P1:   # phase 1: every GLOBAL estimate from every chunk's samples (exact: the same sampled pixels / pairs as the serial run, on the global k grids)
    with cf.ThreadPoolExecutor(len(jobs)) as ex: list(ex.map(lambda j: run(j, ["--cm-only", "--cm-file", f"{j['dir']}/cm.npz", "--out", f"{j['dir']}/none.mp4"], "cm.log"), jobs))
    S0 = [np.load(f"{j['dir']}/cm.npz") for j in jobs]; S = [x for x in S0 if int(x["n"]) > 0]; N_ = sum(int(x["n"]) for x in S); G = {}
    if not NOCM and N_ > 0:
        mk = sum(x["s1k"] for x in S) / N_; mm = sum(x["s1m"] for x in S) / N_
        sk = np.sqrt(np.maximum(sum(x["s2k"] for x in S) / N_ - mk ** 2, 0)); sm = np.sqrt(np.maximum(sum(x["s2m"] for x in S) / N_ - mm ** 2, 0))
        G["gain"] = np.clip(sm / np.maximum(sk, 1e-3), 0.9, 1.1).astype(np.float32); G["off"] = (mm - mk * G["gain"]).astype(np.float32)
        print(f"global colour match (Lab, {N_} px from {len(S)} chunks): gain {G['gain'].round(3).tolist()} offset {G['off'].round(2).tolist()}", flush=True)
    elif not NOCM:
        G["gain"], G["off"] = np.ones(3, np.float32), np.zeros(3, np.float32); print("WARNING: no skin samples in any chunk: colour match = identity", flush=True)
    if SHARP_AUTO:
        h_ = np.concatenate([x["hfr"] for x in S0]) if S0 else np.zeros((0, 2)); h_ = h_[np.isfinite(h_).all(1) & (h_[:, 1] > 0)] if len(h_) else h_
        r_ = float(np.median(h_[:, 0] / h_[:, 1])) if len(h_) else 1.0; G["sharp"] = np.float64(np.clip((r_ - 1) * 1.5, 0, 1.5)) if len(h_) else np.float64(0.0)
        print(f"sharp auto (all chunks): skin HF source / KeySync {r_:.3f} -> unsharp {float(G['sharp']):.2f}", flush=True)
    if CAVITY > 0:
        sc = np.concatenate([x["sc"] for x in S0]) if S0 else np.zeros((0, 3))
        G["cav_l"] = np.float64(np.percentile(sc[:, 0], float(val("--cavity-p", 30)))) if len(sc) else np.float64(-1); print(f"KS_CAVITY tone: source open-mouth interior L p{float(val('--cavity-p', 30)):g} = {float(G['cav_l']):.1f} ({len(sc)} px)", flush=True); G["cav_ab"] = sc[:, 1:].mean(0).astype(np.float32) if len(sc) else np.zeros(2, np.float32)
    np.savez(f"{T}/cm.npz", **G)
t2 = time.time()
with cf.ThreadPoolExecutor(len(jobs)) as ex: list(ex.map(lambda j: run(j, (["--cm-file", f"{T}/cm.npz"] if P1 else []) + ["--out", f"{j['dir']}/out.mp4"], "paste.log"), jobs))
for j in jobs:   # surface every chunk WARNING (e.g. an end fade) in the parent log
    for ln in open(f"{j['dir']}/paste.log"):
        if "WARNING" in ln: print(f"chunk {j['c']}: {ln.rstrip()}", flush=True)
t3 = time.time()
open(f"{T}/list.txt", "w").write("".join(f"file '{j['dir']}/out.mp4'\n" for j in jobs))
pr = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_name:format=duration", "-of", "json", val("--audio")], capture_output=True, text=True).stdout or "{}")
acod = (pr.get("streams") or [{}])[0].get("codec_name", ""); adur = float(pr.get("format", {}).get("duration", 0) or 0)
NOUT = min(n, int(math.floor(adur * 25 + 1e-3))) or n   # frames out = min(master, floor(audio * 25)), the same rule as the serial ks_plain
if abs(adur - n / 25) > 0.04: print(f"WARNING: audio {adur:.3f} s vs master {n / 25:.3f} s: the output is cut to {NOUT} frames", flush=True)
AC = ["-c:a", "copy"] if acod in ("aac", "mp3", "ac3", "eac3", "alac") else ["-c:a", "aac", "-b:a", "192k"]
if AC[1] != "copy": print(f"WARNING: audio codec '{acod}' cannot be stream-copied into mp4: re-encoded to AAC 192k", flush=True)
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", f"{T}/list.txt", "-i", val("--audio"), "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-frames:v", str(NOUT)] + AC +
               ["-t", f"{NOUT / 25:.6f}", "-movflags", "+faststart", a.out], check=True)
got = nframes(a.out)
print(f"plain paste: KeySync {nk} frames, master {n}, shift {SH} -> {a.out} ({got} frames; {J} workers: slicing {t1 - t0:.0f} s, colour match {t2 - t1:.0f} s, "
      f"paste {t3 - t2:.0f} s, total {time.time() - t0:.0f} s)", flush=True)
if got != NOUT: print(f"WARNING: {got} frames written != expected {NOUT} (master {n}, audio {adur:.3f} s)", flush=True)
if got < n - 13: sys.exit(1)
if not os.environ.get("KS_PAR_KEEP"): shutil.rmtree(T, ignore_errors=True)
