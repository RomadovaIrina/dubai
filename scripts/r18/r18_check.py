#!/usr/bin/env python3
"""r18: the automatic proof before Irina sees a render (autopsy 08.11 vs 10.11, 6.4). NEW vs the accepted 08.11 file REF:
  audio    md5 of the decoded PCM (s16le) must be EQUAL (the accepted mix, stream-copied)
  colour   the colr box (nclx 1/1/1 = BT.709) + ffprobe colour tags, size, fps, frame count
  outside  pixels OUTSIDE the paste box (the job's box json(s), grown by --grow x side) must equal REF up to codec noise: per frame the mean
           |NEW - REF| and the signed mean per channel (a paler / greener frame = the 10.11 defect: -2..-7 levels); pass: the mean |gray diff| at half size
           (median over frames) <= --tol (two independent x264 crf 16 encodes of the same source: ~0.7-1.0; the 10.11 builds: 2.3-5.4) and every
           |signed channel mean| <= --tol-signed (the paler-frame test proper)
  inside   PSNR inside the box (info only: KeySync is unseeded, the floor between two identical runs is ~44 dB whole-frame)
    r18_check.py --new N.mp4 --ref R.mp4 --work /data/jobs/<id>/e2e/work [--step 3] --out check.json"""
import argparse, glob, hashlib, json, os, subprocess, sys
import numpy as np, cv2

ap = argparse.ArgumentParser(); ap.add_argument("--new", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--work", default="")
ap.add_argument("--step", type=int, default=3); ap.add_argument("--grow", type=float, default=0.06); ap.add_argument("--tol", type=float, default=1.2)
ap.add_argument("--tol-signed", type=float, default=0.6); ap.add_argument("--out", default=""); a = ap.parse_args()


def probe(p):
    o = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
                                   "stream=width,height,r_frame_rate,nb_read_packets,pix_fmt,color_space,color_transfer,color_primaries,color_range,codec_name",
                                   "-of", "json", p], capture_output=True, text=True).stdout)["streams"][0]
    b = open(p, "rb").read(); i = b.find(b"colr"); colr = None
    if i >= 0 and b[i + 4:i + 8] in (b"nclx", b"nclc"): colr = b[i + 4:i + 8].decode() + " " + "/".join(str(int.from_bytes(b[i + 8 + 2 * k:i + 10 + 2 * k], "big")) for k in range(3))
    return {**o, "colr": colr}


def pcm(p):
    b = subprocess.run(["ffmpeg", "-v", "error", "-i", p, "-map", "0:a:0", "-f", "s16le", "-"], capture_output=True).stdout
    return hashlib.md5(b).hexdigest(), len(b)


def frames(p, w, h):
    pr = subprocess.Popen(["ffmpeg", "-v", "error", "-i", p, "-vf", f"scale={w}:{h}:flags=bicubic", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=w * h * 3 * 4)
    while True:
        b = pr.stdout.read(w * h * 3)
        if len(b) < w * h * 3: break
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)


def boxes(work, n):
    """per master frame: the list of paste boxes (x0, y0, x1, y1) as fractions of the master frame"""
    B = [[] for _ in range(n)]
    def add(bj, k0, k1):
        b = json.load(open(bj)); s, W, H = b["side"], b["W"], b["H"]; xs, ys = b.get("xs"), b.get("ys")
        for k in range(k0, min(k1, n)):
            j = k - k0; x = xs[min(j, len(xs) - 1)] if xs else b["x"]; y = ys[min(j, len(ys) - 1)] if ys else b["y"]
            g = a.grow * s; B[k].append(((x - g) / W, (y - g) / H, (x + s + g) / W, (y + s + g) / H))
    kw = os.path.join(work, "ks"); segs = json.load(open(os.path.join(kw, "shots.json")))["segments"] if os.path.exists(os.path.join(kw, "shots.json")) else []
    if len(segs) > 1:
        for i, sg in enumerate(segs):
            k0, k1 = (sg[0], sg[1]) if isinstance(sg, (list, tuple)) else (sg.get("k0", sg.get("a")), sg.get("k1", sg.get("b")))
            for bj in glob.glob(os.path.join(work, "shots", f"s{i}", "ks", "data", "box_*.json")): add(bj, k0, k1)
    else:
        for bj in glob.glob(os.path.join(kw, "data", "box_*.json")): add(bj, 0, n)
    return B


pn, pr_ = probe(a.new), probe(a.ref); an, ar = pcm(a.new), pcm(a.ref)
res = {"new": a.new, "ref": a.ref, "audio_md5_equal": an[0] == ar[0] and an[1] > 0, "audio_md5": [an[0], ar[0]], "audio_bytes": [an[1], ar[1]],
       "probe_new": pn, "probe_ref": pr_, "colr_ok": pn["colr"] is not None and pn["colr"] == pr_["colr"],
       "format_equal": all(pn[k] == pr_[k] for k in ("width", "height", "r_frame_rate", "nb_read_packets", "pix_fmt"))}
w, h = int(pr_["width"]), int(pr_["height"]); n = int(pr_["nb_read_packets"])
BX = boxes(a.work, n) if a.work else [[] for _ in range(n)]
ad, sg, ps_in, gm, k = [], [], [], [], 0
for k, (fn, fr) in enumerate(zip(frames(a.new, w, h), frames(a.ref, w, h))):
    if k % a.step: continue
    out = np.ones((h, w), bool); ins = np.zeros((h, w), bool)
    for (x0, y0, x1, y1) in (BX[k] if k < len(BX) else []):
        X0, Y0, X1, Y1 = max(0, int(x0 * w)), max(0, int(y0 * h)), min(w, int(x1 * w) + 1), min(h, int(y1 * h) + 1); out[Y0:Y1, X0:X1] = False; ins[Y0:Y1, X0:X1] = True
    d = fn.astype(np.int16) - fr.astype(np.int16)
    if out.any(): ad.append(float(np.abs(d[out]).mean())); sg.append(d[out].mean(0))
    if out.any():   # the autopsy's measure (pic_cmp.py): the MEDIAN pixel of |gray, half size| outside (Q's oval vs 08.11: 0.74; the 10.11 builds: 2.3-5.4)
        g = np.abs(cv2.resize(cv2.cvtColor(fn, cv2.COLOR_RGB2GRAY), (w // 2, h // 2), interpolation=cv2.INTER_AREA).astype(np.float32) -
                   cv2.resize(cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY), (w // 2, h // 2), interpolation=cv2.INTER_AREA).astype(np.float32))
        gm.append(float(np.mean(g[cv2.resize(out.astype(np.uint8), (w // 2, h // 2), interpolation=cv2.INTER_NEAREST) > 0])))
    if ins.any(): m = float((d[ins].astype(np.float32) ** 2).mean()); ps_in.append(99.0 if m == 0 else float(10 * np.log10(255 ** 2 / m)))
ad = np.array(ad); sg = np.array(sg) if sg else np.zeros((1, 3))
res.update({"frames_compared": len(ad), "frames_with_box": len(ps_in), "outside_absdiff_median": round(float(np.median(ad)), 3), "outside_absdiff_p95": round(float(np.percentile(ad, 95)), 3),
            "outside_absdiff_max": round(float(ad.max()), 3), "outside_signed_rgb": [round(float(x), 3) for x in sg.mean(0)],
            "inside_psnr_mean": round(float(np.mean(ps_in)), 2) if ps_in else None, "inside_psnr_min": round(float(np.min(ps_in)), 2) if ps_in else None})
res["outside_gray_half_mean"] = round(float(np.median(gm)), 3) if gm else None
res["outside_ok"] = bool(res["outside_gray_half_mean"] is not None and res["outside_gray_half_mean"] <= a.tol and max(abs(x) for x in res["outside_signed_rgb"]) <= a.tol_signed)
res["PASS"] = bool(res["audio_md5_equal"] and res["colr_ok"] and res["format_equal"] and res["outside_ok"])
if a.out: json.dump(res, open(a.out, "w"), indent=1)
print("R18_CHECK " + json.dumps({k_: res[k_] for k_ in ("PASS", "audio_md5_equal", "colr_ok", "format_equal", "outside_ok", "outside_gray_half_mean", "outside_absdiff_median", "outside_absdiff_p95",
                                                         "outside_signed_rgb", "inside_psnr_mean", "inside_psnr_min", "frames_compared")} | {"colr": [pn["colr"], pr_["colr"]]}))
sys.exit(0 if res["PASS"] else 1)
