#!/usr/bin/env python3
"""r16 Q day 4: tail-like breaks ANYWHERE in a final (not just the last second). Reads r16q_defects' <id>.series.npz.
A break = a sudden jump of a per-frame signal against the 2 s before it (the same local rule that catches the m03 / m18 / q01 tails):
  res_out  lower-face residual outside the mouth (ours vs source, Lab)    4-frame mean / median of the 2 s before  >= 1.5  (and +2 L)
  res_low  the whole lower face                                            same
  jaw_dev  ours' jaw landmarks vs the source's, % IOD                      ratio >= 1.5 and +3 % IOD
  lm_dev   ours' stable landmarks (nose, eyes) vs the source's             ratio >= 1.5 and +2 % IOD   ("Momo": the whole face moves)
  soft     mouth sharpness ours / src                                      4-frame mean / before <= 0.65
  lip      lip ring area ours / src                                        <= 0.65 or >= 1.5 of before
  seam     jaw-band residual wobble minus the forehead floor               4-frame mean - before >= 2.0 L / frame
  ghost    second chin (secondary parallel edge / jaw edge, final - source) 4-frame mean >= 0.25 and +0.15 over before
Context per event: a pause in the final's audio starting within 0.6 s (silencedetect -35 dB, 0.25 s), a shot cut within 1 s (scene > 0.3
on the source), the last 1.5 s of the video.
    python3 scan_breaks.py --series X.series.npz --final FINAL [--src SRC] --id ID --out DIR [--png N]"""
import argparse, json, pathlib, re, subprocess
import numpy as np

FPS = 25.0


def m4(x):
    y = np.where(np.isfinite(x), x, np.nan); out = np.full(len(y), np.nan)
    for i in range(len(y) - 3):
        v = y[i:i + 4]
        if np.isfinite(v).sum() >= 3: out[i] = np.nanmean(v)
    return out


def before(x, w=50, gap=2):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        v = x[max(0, i - w):max(0, i - gap)]; v = v[np.isfinite(v)]
        if len(v) >= 20: out[i] = np.median(v)
    return out


def runs(fl, sc, minlen=3, gap=3):
    idx = np.flatnonzero(fl); out = []
    if not len(idx): return out
    s = p = idx[0]
    for i in list(idx[1:]) + [None]:
        if i is not None and i - p <= gap + 1: p = i; continue
        if p - s + 1 >= minlen:
            seg = sc[s:p + 1]; k = s + int(np.nanargmax(seg)); out.append((s, p, k, float(sc[k])))
        if i is not None: s = p = i
    return out


def silences(v):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", v, "-vn", "-af", "silencedetect=noise=-35dB:d=0.25", "-f", "null", "-"], capture_output=True, text=True).stderr
    st = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r)]; en = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r)]
    return list(zip(st, en + [1e9] * (len(st) - len(en))))


def cuts(v):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-threads", "2", "-i", v, "-an", "-vf", "fps=25,scale=320:-2,select='gt(scene,0.3)',showinfo", "-f", "null", "-"], capture_output=True, text=True).stderr
    return [float(x) for x in re.findall(r"pts_time:([\d.]+)", r)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--series", required=True); ap.add_argument("--final", required=True); ap.add_argument("--src", default=None)
    ap.add_argument("--id", required=True); ap.add_argument("--out", required=True); ap.add_argument("--png", type=int, default=0)
    a = ap.parse_args(); O = pathlib.Path(a.out); O.mkdir(parents=True, exist_ok=True)
    z = np.load(a.series); S = {k: z[k].astype(float) for k in z.files}; n = len(S["res_out"])
    sx = np.fmax(S["seam_in"], S["seam_out"]) - np.where(np.isfinite(S["seam_up"]), S["seam_up"], 0)
    gx = S["gh_Bo"] / np.fmax(S["gh_Ao"], 1.0) - S["gh_Bs"] / np.fmax(S["gh_As"], 1.0) if "gh_Bo" in S else np.full(n, np.nan)
    ev = []
    def up(nm, x, rthr, dthr, cls):
        M = m4(x); B = before(x); r = np.where(np.isfinite(B), M / np.fmax(B, 1e-3), np.nan); fl = (r >= rthr) & ((M - B) >= dthr)
        for s, e, k, v in runs(fl, r): ev.append(dict(cls=cls, sig=nm, t0=s / FPS, t1=(e + 4) / FPS, t=k / FPS, k=int(k), score=round(v / rthr, 2), val=round(float(M[k]), 2), base=round(float(B[k]), 2)))
    up("res_out", S["res_out"], 1.5, 2.0, "tail-like break")
    up("res_low", S["res_low"], 1.5, 2.0, "tail-like break")
    up("jaw_dev", S["jaw_dev"], 1.5, 3.0, "tail-like break")
    up("lm_dev", S["lm_dev"], 1.5, 2.0, "face moves (Momo)")
    for nm, x, lo, hi, cls in (("mouth_sharp", S["mouth_sharp"], 0.65, None, "soft mouth spike"), ("lip_thick", S["lip_thick"], 0.65, 1.5, "lip shape spike")):
        M = m4(x); B = before(x); r = np.where(np.isfinite(B) & (B > 0.05), M / np.fmax(B, 1e-3), np.nan); r = np.minimum(r, 5.0)
        for s, e, k, v in runs(r <= lo, lo / np.fmax(r, 1e-3)): ev.append(dict(cls=cls, sig=nm, t0=s / FPS, t1=(e + 4) / FPS, t=k / FPS, k=int(k), score=round(v, 2), val=round(float(M[k]), 3), base=round(float(B[k]), 3)))
        if hi:
            for s, e, k, v in runs(r >= hi, r / hi): ev.append(dict(cls=cls, sig=nm + "_fat", t0=s / FPS, t1=(e + 4) / FPS, t=k / FPS, k=int(k), score=round(v, 2), val=round(float(M[k]), 3), base=round(float(B[k]), 3)))
    M = m4(sx); B = before(sx); d = M - B
    for s, e, k, v in runs(d >= 2.0, d): ev.append(dict(cls="seam spike", sig="seam", t0=s / FPS, t1=(e + 4) / FPS, t=k / FPS, k=int(k), score=round(v / 2.0, 2), val=round(float(M[k]), 2), base=round(float(B[k]), 2)))
    M = m4(gx); B = before(gx); fl = (M >= 0.25) & ((M - np.nan_to_num(B, nan=0)) >= 0.15) & np.isfinite(B)
    for s, e, k, v in runs(fl, M): ev.append(dict(cls="ghost chin", sig="ghost", t0=s / FPS, t1=(e + 4) / FPS, t=k / FPS, k=int(k), score=round(v / 0.25, 2), val=round(float(M[k]), 3), base=round(float(np.nan_to_num(B[k])), 3)))
    sil = silences(a.final); cu = cuts(a.src or a.final); dur = n / FPS
    for e in ev:
        ctx = []
        p = [s for s, _ in sil if -0.3 <= s - e["t"] <= 0.6]
        if p: ctx.append(f"pause @{p[0]:.2f}")
        if any(s - 0.2 <= e["t"] <= en for s, en in sil): ctx.append("in pause")
        c = [x for x in cu if abs(x - e["t"]) <= 1.0]
        if c: ctx.append(f"cut @{c[0]:.2f}")
        if e["t"] >= dur - 1.5: ctx.append("video end")
        e["ctx"] = ", ".join(ctx); e["t0"], e["t1"], e["t"] = round(e["t0"], 2), round(e["t1"], 2), round(e["t"], 2)
    ev.sort(key=lambda e: -e["score"])
    # merge per moment: events within 0.5 s of each other -> one moment, classes listed
    mom = []
    for e in sorted(ev, key=lambda e: e["t"]):
        if mom and e["t"] - mom[-1]["t1"] <= 0.5:
            m = mom[-1]; m["t1"] = max(m["t1"], e["t1"]); m["ev"].append(e)
            if e["score"] > m["score"]: m.update(score=e["score"], t=e["t"], k=e["k"])
        else: mom.append(dict(t0=e["t0"], t1=e["t1"], t=e["t"], k=e["k"], score=e["score"], ev=[e]))
    for m in mom:
        m["n_signals"] = len({x["sig"] for x in m["ev"]}); m["classes"] = sorted({x["cls"] for x in m["ev"]}); m["ctx"] = "; ".join(sorted({x["ctx"] for x in m["ev"] if x["ctx"]}))
        m["rank"] = round(m["score"] * (1 + 0.5 * (m["n_signals"] - 1)), 2)   # several signals at once = more likely a visible break
    mom.sort(key=lambda m: -m["rank"])
    res = dict(id=a.id, frames=n, dur=round(dur, 2), n_events=len(ev), moments=mom, pauses=[[round(s, 2), round(e, 2)] for s, e in sil], cuts=cu,
               ghost=dict(median=round(float(np.nanmedian(gx)), 3) if np.isfinite(gx).any() else None, p95=round(float(np.nanpercentile(gx, 95)), 3) if np.isfinite(gx).any() else None))
    json.dump(res, open(O / f"{a.id}.breaks.json", "w"), indent=1)
    print(f"{a.id}: {len(ev)} events, {len(mom)} moments; top: " + "; ".join(f"{m['t']:.2f}s r{m['rank']} {'/'.join(m['classes'])} [{m['ctx']}]" for m in mom[:6]))


if __name__ == "__main__":
    main()
