#!/usr/bin/env python3
"""r16b K2: lip-sync THROUGH a cross-dissolve instead of switching source -> KeySync on a visible face (q03 22.76 s / 30.4 s, Irina: no pixel
switch on a visible face). A dissolve frame is S_k = a_k B_k + (1 - a_k) T (B = the face shot, T = the other shot / title card, a linear
crossfade: q03 residual <= 4.7 grey levels). Then
  unmix    the KeySync input for the dissolve frames = the face alone, B^_k = (S_k - (1 - a_k) T) / a_k (a_k >= --amin; below it the nearest
           valid B^ is held: those lips are weighted by a_k < amin in the output anyway); clean frames are copied. Writes the lossless master of
           the segment [k0, k1) + <out>.alpha.json (a_k per frame, fitted against the T frame and the clean B frame, least squares).
  assemble out_k = S_k + a_k (P_k - B^_k) inside each part (P = the pasted KeySync render of the unmixed master): only the LIP CHANGE goes in,
           scaled by how visible the face is (a_k = 1 -> P_k; a_k -> 0 -> the source), the rest of the frame stays the source; outside the
           parts the source. Colour-managed (ks_cms), x264 crf 16 like ks_plain, the final mix stream-COPIED.
  check    is a transition a clean linear crossfade? Fits a_k over [d0, d1) against the T and B frames; ok when the worst residual is
           < --max-rel (0.10) x mean |B - T| AND a_k is monotone (Spearman |rho| > 0.9) and spans >= 0.8. Prints KS_DISSOLVE_CHECK {...}
           and writes <out> (json). ok -> alpha-paste; not ok -> keep the hard switch AND show the flag (KS WARN line + "flag" in the
           json: q03 37.72 s, the fresco pans during its fade: residual 0.285 of |B - T|). q03 22.76 s / 30.4 s: 0.029 / 0.031 -> ok.
  Accepted by Irina's eye 08.11 on q03: "альфа пейст не плавает не дергается" (it doesn't float, doesn't jerk).
  Wiring (T): per face-segment edge that ks_shots.py marked as a transition zone: T frame = the first frame outside the zone on the other
  side, B frame = the first clean face frame; check -> unmix (segment extended over the zone, KS_MASTER_IN = the unmixed master) ->
  ks_face.sh -> assemble. Paste shift / delay as for any segment (q03 shot B needed shift +1 on its own run).
    python ks_dissolve.py check --master M.mp4 --d0 D0 --d1 D1 --t-frame T --b-frame B --out check.json
    python ks_dissolve.py unmix --master M.mp4 --k0 K0 --k1 K1 --d0 D0 --d1 D1 --t-frame T --b-frame B --out seg_unmixed.mp4
    KS_CMS=1 python ks_dissolve.py assemble --master M.mp4 --audio MIX --part P.mp4:K0:seg_unmixed.mp4 [...] --out OUT.mp4"""
import argparse, json, os, subprocess, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.realpath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.realpath(__file__)), ".."))
ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["check", "unmix", "assemble"]); ap.add_argument("--max-rel", type=float, default=0.10, dest="max_rel"); ap.add_argument("--master", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--k0", type=int); ap.add_argument("--k1", type=int); ap.add_argument("--t-frame", type=int, dest="tf"); ap.add_argument("--b-frame", type=int, dest="bf")
ap.add_argument("--d0", type=int, default=-1); ap.add_argument("--d1", type=int, default=-1)   # the dissolve frames [d0, d1); a = 1 elsewhere
ap.add_argument("--amin", type=float, default=0.3); ap.add_argument("--audio"); ap.add_argument("--part", nargs="*", default=[]); a = ap.parse_args()
pr = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", a.master], capture_output=True, text=True).stdout.strip().split(",")
W, H = int(pr[0]), int(pr[1]); FB = W * H * 3


def reader(path, cms=False):
    import ks_cms
    dec = ks_cms.dec(None, "rgb24") if cms else []
    p = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-i", path] + dec + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE, bufsize=FB * 2)
    while True:
        b = p.stdout.read(FB)
        if len(b) < FB: return
        yield np.frombuffer(b, np.uint8).reshape(H, W, 3)


if a.cmd == "check":
    lo, hi = min(a.d0, a.tf, a.bf), max(a.d1 - 1, a.tf, a.bf); F = {}
    for k, f in enumerate(reader(a.master)):
        if lo <= k <= hi: F[k] = f[::4, ::4].astype(np.float32)
        if k >= hi: break
    T, B = F[a.tf], F[a.bf]; D = (B - T).ravel(); DD = float(np.dot(D, D)); dbt = float(np.abs(B - T).mean())
    ks = list(range(a.d0, a.d1)); al = [float(np.dot((F[k] - T).ravel(), D) / DD) for k in ks]
    res = [float(np.abs(F[k] - (x * B + (1 - x) * T)).mean()) for k, x in zip(ks, al)]
    rk = lambda v: np.argsort(np.argsort(v)).astype(float)
    rho = float(np.corrcoef(rk(ks), rk(al))[0, 1]) if len(ks) > 2 else 0.0
    rel = max(res) / max(dbt, 1e-6); span = max(al) - min(al)
    ok = rel < a.max_rel and abs(rho) > 0.9 and span >= 0.8
    why = [] if ok else [w for w, c in (("residual %.2f of |B-T| >= %.2f (not a linear crossfade: camera / content moves)" % (rel, a.max_rel), rel >= a.max_rel),
                                         ("alpha not monotone (rho %.2f)" % rho, abs(rho) <= 0.9), ("alpha span %.2f < 0.8" % span, span < 0.8)) if c]
    out = {"ok": ok, "d0": a.d0, "d1": a.d1, "t_frame": a.tf, "b_frame": a.bf, "max_rel_resid": round(rel, 3), "rho": round(rho, 3), "span": round(span, 3),
           "mean_BT": round(dbt, 1), "alpha": [round(x, 3) for x in al], "flag": None if ok else "hard switch kept at frames %d-%d (%.2f s): %s" % (a.d0, a.d1, a.d0 / 25, "; ".join(why))}
    json.dump(out, open(a.out, "w"), indent=1)
    print("KS_DISSOLVE_CHECK " + json.dumps({k: out[k] for k in ("ok", "d0", "d1", "max_rel_resid", "rho", "span", "flag")}), flush=True)
    if not ok: print("KS WARN " + out["flag"], flush=True)

elif a.cmd == "unmix":
    # streaming (Q audit P3: a 5-min 1080p shot held whole = ~46 GB): pass 1 keeps only T, B and the dissolve window [d0, d1) (+1 frame each side
    # for the held frames); pass 2 streams the segment, copying every frame outside the window
    d0, d1 = max(a.d0, a.k0), min(a.d1, a.k1); F = {}
    for k, f in enumerate(reader(a.master)):
        if k in (a.tf, a.bf) or d0 <= k < d1: F[k] = f
        if k >= max(d1, a.tf, a.bf): break
    T = F[a.tf].astype(np.float32); B = F[a.bf].astype(np.float32); D = (B - T)[::4, ::4].ravel(); DD = float(np.dot(D, D))
    al = {k: (float(np.clip(np.dot((F[k].astype(np.float32) - T)[::4, ::4].ravel(), D) / DD, 0, 1)) if d0 <= k < d1 else 1.0) for k in range(a.k0, a.k1)}
    U = {}
    for k in range(d0, d1):
        if al[k] >= 0.995: U[k] = F[k]
        elif al[k] >= a.amin: U[k] = np.clip((F[k].astype(np.float32) - (1 - al[k]) * T) / al[k], 0, 255).astype(np.uint8)
    ok = sorted(U)
    for k in range(d0, d1):   # a_k < amin: hold the nearest valid unmixed frame (inside the window; else the nearest clean frame outside it)
        if k not in U: U[k] = U[min(ok, key=lambda q: abs(q - k))] if ok else (F[a.bf])
    del F
    w = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "25", "-i", "-",
                          "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
    for k, f in enumerate(reader(a.master)):
        if k < a.k0: continue
        if k >= a.k1: break
        w.stdin.write((U[k] if k in U else f).tobytes())
    w.stdin.close(); w.wait()
    json.dump({"k0": a.k0, "k1": a.k1, "t_frame": a.tf, "b_frame": a.bf, "alpha": {str(k): round(v, 4) for k, v in al.items()}}, open(a.out + ".alpha.json", "w"))
    dz = [k for k in al if al[k] < 0.995]
    print(f"KS_DISSOLVE unmix [{a.k0}, {a.k1}): dissolve frames {min(dz) if dz else '-'}..{max(dz) if dz else '-'}, "
          f"held (a < {a.amin}) {sum(1 for k in al if al[k] < a.amin)}", flush=True)

else:
    import ks_cms
    parts = []
    for s in a.part:
        p, k0, um = s.split(":"); k0 = int(k0); aj = json.load(open(um + ".alpha.json"))
        n = int(subprocess.run(["ffprobe", "-v", "error", "-count_packets", "-select_streams", "v:0", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", p],
                               capture_output=True, text=True).stdout.strip())
        parts.append({"k0": k0, "k1": k0 + n, "P": reader(p, True), "U": reader(um, True), "al": {int(k): v for k, v in aj["alpha"].items()}})
    tmp = a.out + ".v.mp4"
    w = subprocess.Popen(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", "25", "-i", "-",
                          "-c:v", "libx264", "-crf", "16", "-preset", "medium"] + ks_cms.enc() + [tmp], stdin=subprocess.PIPE)
    n = 0; mixed = 0
    for k, S in enumerate(reader(a.master, True)):
        out = S
        for q in parts:
            if q["k0"] <= k < q["k1"]:
                P = next(q["P"]).astype(np.float32); U = next(q["U"]).astype(np.float32); al = q["al"].get(k, 1.0)
                out = np.clip(S.astype(np.float32) + al * (P - U), 0, 255).astype(np.uint8); mixed += al < 0.995
        w.stdin.write(out.tobytes()); n += 1
    w.stdin.close(); w.wait()
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", tmp, "-i", a.audio, "-map", "0:v", "-map", "1:a", "-c:v", "copy", "-c:a", "copy", "-movflags", "+faststart", a.out], check=True)
    os.remove(tmp); print(f"KS_DISSOLVE assemble {a.out}: {n} frames, {mixed} dissolve frames alpha-pasted", flush=True)
