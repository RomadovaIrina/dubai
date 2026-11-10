#!/usr/bin/env python3
"""r16 Q day 4: per-segment "best variant" selector + composer.
The default variant (A2) stays wherever Q's detectors are quiet. Where they fire above their calibrated thresholds, it switches locally
to the variant with the lowest badness there (A8 / A9 ...; the source only inside dub silence), with hysteresis + >= 0.3 s crossfades.

Per frame, per variant: bad = max of (each normalised so that 1.0 = the calibrated threshold)
  ghost   ghost-chin excess (final - source), 0.5-s rolling median / 0.25
  black   r16q_detect black flag (ours black cavity, source not)                 -> 1.5 while flagged (3-frame dilation)
  seam    jaw-band residual wobble minus the forehead floor, 4-frame mean / 3.5
  soft    mouth sharpness ours / src, 7-frame median: 0.70 / x
  lip     lip ring area ours / src, 7-frame median: max(0.65 / x, x / 1.5)
  break   tail-like jump: jaw_dev or res_out 4-frame mean / median of the 2 s before, / 1.5 (only where it also rose by +3 % IOD / +2 L)
Sync per second (K1 eval sync_main.json per_second conf_at0, dub_act) guards the switch: an alternative may not lose > SYNC_DROP of
conf_at0 on the seconds with speech in the segment.

  python3 select_compose.py --src MASTER --default A2=FINAL:QGPREFIX[:SYNC.json] --alt A8=FINAL:QGPREFIX[:SYNC] ... --out OUT.mp4 [--dry]
  QGPREFIX = path without extension of r16q_defects' output (QGPREFIX.series.npz, QGPREFIX.det.npz.det.series.npz)"""
import argparse, json, pathlib, subprocess, sys
import numpy as np

FPS = 25.0
THR = dict(bad=1.0, pad_s=0.2, merge_s=1.0, min_s=0.6, xfade_s=0.32, gain=0.8, sync_drop=0.3, silence_act=0.1, max_s=3.0)   # v4: sync guard 1.5 -> 0.3 (K1: v1 lost 0.2-0.8 on speech), pieces <= 3 s


def roll_med(x, h):
    y = np.where(np.isfinite(x), x, np.nan); out = np.full(len(y), np.nan)
    for i in range(len(y)):
        v = y[max(0, i - h):i + h + 1]; v = v[np.isfinite(v)]
        if len(v): out[i] = np.median(v)
    return out


def m4(x):
    out = np.full(len(x), np.nan)
    for i in range(len(x) - 3):
        v = x[i:i + 4]; v = v[np.isfinite(v)]
        if len(v) >= 3: out[i] = v.mean()
    return out


def before(x, w=50, gap=2):
    out = np.full(len(x), np.nan)
    for i in range(len(x)):
        v = x[max(0, i - w):max(0, i - gap)]; v = v[np.isfinite(v)]
        if len(v) >= 20: out[i] = np.median(v)
    return out


def badness(prefix):
    z = np.load(prefix + ".series.npz"); S = {k: z[k].astype(float) for k in z.files}; n = len(S["res_out"])
    d = np.load(prefix + ".det.npz.det.series.npz"); M = {k: d[k].astype(float) for k in d.files}
    parts = {}
    gx = S["gh_Bo"] / np.fmax(S["gh_Ao"], 1.0) - S["gh_Bs"] / np.fmax(S["gh_As"], 1.0)
    parts["ghost"] = roll_med(gx, 6) / 0.25
    bo = (M["cavL_o"] < 15) & (M["cavB_o"] < 0.02); bs = (M["cavL_s"] < 15) & (M["cavB_s"] < 0.02)
    bo = bo & ~bs & ~(np.isfinite(M["cavL_s"]) & (M["cavL_s"] - M["cavL_o"] < 8.0))
    bl = np.convolve(bo.astype(float), np.ones(3), "same") > 0
    parts["black"] = np.where(bl, 1.5, 0.0)
    sx = np.fmax(S["seam_in"], S["seam_out"]) - np.where(np.isfinite(S["seam_up"]), S["seam_up"], 0)
    parts["seam"] = m4(sx) / 3.5
    parts["soft"] = 0.70 / np.fmax(roll_med(S["mouth_sharp"], 3), 1e-3)
    lt = roll_med(S["lip_thick"], 3); parts["lip"] = np.fmax(0.65 / np.fmax(lt, 1e-3), lt / 1.5)
    br = np.zeros(n)
    for nm, dthr in (("jaw_dev", 3.0), ("res_out", 2.0)):
        x = S[nm]; Mx = m4(x); B = before(x); r = np.where(np.isfinite(B), Mx / np.fmax(B, 1e-3), np.nan)
        br = np.fmax(br, np.where((Mx - B) >= dthr, r / 1.5, 0.0))
    parts["break"] = br
    for k in parts: parts[k] = np.nan_to_num(np.asarray(parts[k], float)[:n], nan=0.0)
    bad = np.max(np.stack(list(parts.values())), 0)
    return bad, parts, n


def sync_per_s(p):
    if not p or not pathlib.Path(p).exists(): return {}
    s = json.load(open(p)); return {int(round(x["t"])): (x.get("conf_at0"), x.get("dub_act", 0.0)) for x in s.get("per_second", [])}


def segments(flag, n):
    pad, merge, mn = int(THR["pad_s"] * FPS), int(THR["merge_s"] * FPS), int(THR["min_s"] * FPS)
    idx = np.flatnonzero(flag); segs = []
    for i in idx:
        a, b = max(0, i - pad), min(n - 1, i + pad)
        if segs and a - segs[-1][1] <= merge: segs[-1][1] = max(segs[-1][1], b)
        else: segs.append([a, b])
    out = []
    for a, b in segs:
        if b - a + 1 < mn: c = (a + b) // 2; a, b = max(0, c - mn // 2), min(n - 1, c + mn // 2)
        if n - 1 - b <= int(FPS): b = n - 1 + int(FPS)   # never switch back in the last second (a jump there = a new tail); ramp past the end
        if a <= int(FPS): a = -int(FPS)
        # split long segments into <= MAX_S pieces so each piece is judged (and sync-guarded) on its own
        L = int(THR["max_s"] * FPS); k = a
        while b - k + 1 > L * 1.5: out.append([k, k + L - 1]); k += L
        out.append([k, b])
    return out


def parse(spec):
    name, rest = spec.split("=", 1); p = rest.split(":")
    return dict(name=name, final=p[0], qg=p[1], sync=p[2] if len(p) > 2 else None)


def reader(v, W, H):
    cmd = ["ffmpeg", "-v", "error", "-threads", "2", "-i", v, "-vf", f"fps={FPS},scale={W}:{H}:flags=bicubic", "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stdin=subprocess.DEVNULL, bufsize=W * H * 3 * 2); nb = W * H * 3
    while True:
        b = p.stdout.read(nb)
        if len(b) < nb: break
        yield np.frombuffer(b, np.uint8).reshape(H, W, 3)
    p.stdout.close(); p.wait()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--src", required=True); ap.add_argument("--default", required=True); ap.add_argument("--alt", action="append", default=[])
    ap.add_argument("--out", required=True); ap.add_argument("--dry", action="store_true"); ap.add_argument("--no-source", action="store_true")
    a = ap.parse_args()
    D = parse(a.default); ALT = [parse(s) for s in a.alt]
    bD, pD, n = badness(D["qg"]); sD = sync_per_s(D["sync"])
    B = {D["name"]: bD}; P = {D["name"]: pD}; SY = {D["name"]: sD}
    for v in ALT:
        b, p, nv = badness(v["qg"]); m = min(n, nv); b2 = np.full(n, 9.0); b2[:m] = b[:m]; B[v["name"]] = b2; P[v["name"]] = p; SY[v["name"]] = sync_per_s(v["sync"])
    segs = segments(bD >= THR["bad"], n)
    plan = []
    for a0, b0 in segs:
        secs = sorted({int(k / FPS) for k in range(max(0, a0), min(n - 1, b0) + 1)})
        speech = [s for s in secs if sD.get(s, (None, 0.0))[1] is not None and (sD.get(s, (None, 0.0))[1] or 0) > THR["silence_act"]]
        silent = len(speech) == 0
        cand = {}
        def sy(name):
            vals = [SY[name].get(s, (None, 0))[0] for s in speech]; vals = [x for x in vals if x is not None]
            return float(np.mean(vals)) if vals else None
        a0c, b0c = max(0, a0), min(n - 1, b0)
        core = np.arange(a0c, b0c + 1)[bD[a0c:b0c + 1] >= THR["bad"]]   # compare on the frames that fired, not the padding
        if not len(core): core = np.arange(a0c, b0c + 1)
        base_b = float(np.mean(bD[core])); base_s = sy(D["name"])
        for v in ALT:
            mb = float(np.mean(B[v["name"]][core])); vs = sy(v["name"])
            ok_sync = base_s is None or vs is None or vs >= base_s - THR["sync_drop"]
            if mb <= THR["gain"] * base_b and ok_sync: cand[v["name"]] = (mb, vs)
        if silent and not a.no_source: cand["source"] = (0.0, None)
        pick = min(cand, key=lambda k: cand[k][0]) if cand else None
        worst = max(P[D["name"]], key=lambda k: (float((P[D["name"]][k][core] >= 1).mean()), float(np.mean(P[D["name"]][k][core]))))
        plan.append(dict(t0=round(a0 / FPS, 2), t1=round((b0 + 1) / FPS, 2), shown=f"{max(0, a0) / FPS:.2f}-{min(n, b0 + 1) / FPS:.2f}", why=worst, bad_default=round(base_b, 2), sync_default=None if base_s is None else round(base_s, 2),
                         silent=silent, pick=pick or D["name"], bad_pick=round(cand[pick][0], 2) if pick else round(base_b, 2),
                         sync_pick=None if not pick or cand[pick][1] is None else round(cand[pick][1], 2),
                         cands={k: [round(x, 2) if x is not None else None for x in v] for k, v in cand.items()}))
    # per-frame weights of each variant (crossfade ramps of XF frames centred on the segment edges)
    XF = int(round(THR["xfade_s"] * FPS)); names = [D["name"]] + [v["name"] for v in ALT] + ["source"]
    W = {k: np.zeros(n) for k in names}
    runs_ = []   # touching pieces with the same pick -> one run (no crossfade dip between them)
    for s in plan:
        a0, b0 = int(round(s["t0"] * FPS)), int(round(s["t1"] * FPS)) - 1
        if runs_ and runs_[-1][2] == s["pick"] and a0 - runs_[-1][1] <= 1: runs_[-1][1] = b0
        else: runs_.append([a0, b0, s["pick"]])
    for a0, b0, pk in runs_:
        s = {"pick": pk}
        if s["pick"] == D["name"]: continue
        w = np.zeros(n); k = np.arange(n)
        up = np.clip((k - (a0 - XF / 2)) / XF, 0, 1); dn = np.clip(((b0 + XF / 2) - k) / XF, 0, 1); w = np.minimum(up, dn)
        W[s["pick"]] = np.fmax(W[s["pick"]], w)
    tot_alt = sum(W[k] for k in names if k != D["name"]); over = tot_alt > 1
    for k in names:
        if k != D["name"]: W[k] = np.where(over, W[k] / np.fmax(tot_alt, 1e-6), W[k])
    W[D["name"]] = 1 - sum(W[k] for k in names if k != D["name"])
    sw = float((W[D["name"]] < 0.999).sum() / FPS)
    comp_bad = sum(W[k] * (B[k] if k in B else 0.0) for k in names)
    rep = dict(default=D["name"], alts=[v["name"] for v in ALT], frames=n, thr=THR, switched_s=round(sw, 2), n_segments=sum(s["pick"] != D["name"] for s in plan),
               flagged_s_default=round(float((bD >= THR["bad"]).sum() / FPS), 2), flagged_s_composed_est=round(float((comp_bad >= THR["bad"]).sum() / FPS), 2),
               per_part_default={k: round(float((v >= 1).sum() / FPS), 2) for k, v in pD.items()}, plan=plan,
               use_s={k: round(float((W[k] > 0.5).sum() / FPS), 2) for k in names})
    out = pathlib.Path(a.out); json.dump(rep, open(out.with_suffix(".plan.json"), "w"), indent=1)
    with open(out.with_suffix(".cuts.txt"), "w") as f:
        for s in plan: f.write(f"{s['t0']:7.2f} {s['t1']:7.2f}  {s['pick']:8s} why={s['why']:6s} bad {s['bad_default']}->{s['bad_pick']} sync {s['sync_default']}->{s['sync_pick']} {'silent' if s['silent'] else ''}\n")
    print(json.dumps({k: rep[k] for k in ("switched_s", "n_segments", "flagged_s_default", "flagged_s_composed_est", "per_part_default", "use_s")}))
    if a.dry: return
    probe = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "json", D["final"]], capture_output=True, text=True).stdout)["streams"][0]
    Wd, Hd = probe["width"], probe["height"]
    used = [k for k in names if k != D["name"] and W[k].max() > 0]
    paths = {D["name"]: D["final"], "source": a.src, **{v["name"]: v["final"] for v in ALT}}
    rd = {k: reader(paths[k], Wd, Hd) for k in [D["name"]] + used}
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{Wd}x{Hd}", "-r", str(FPS), "-i", "-", "-i", D["final"],
                            "-map", "0:v", "-map", "1:a?", "-c:v", "libx264", "-crf", "16", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", str(out)], stdin=subprocess.PIPE)
    for k in range(n):
        fr = {}
        try: fr[D["name"]] = next(rd[D["name"]])
        except StopIteration: break
        for nm in used:   # a variant that ends early falls back to the default frame
            try: fr[nm] = next(rd[nm])
            except StopIteration: fr[nm] = fr[D["name"]]
        if all(W[nm][k] == 0 for nm in used): img = fr[D["name"]]
        else:
            acc = fr[D["name"]].astype(np.float32) * W[D["name"]][k]
            for nm in used: acc += fr[nm].astype(np.float32) * W[nm][k]
            img = np.clip(acc + 0.5, 0, 255).astype(np.uint8)
        try: enc.stdin.write(img.tobytes())
        except BrokenPipeError: break
    try: enc.stdin.close()
    except BrokenPipeError: pass   # ffmpeg -shortest stopped at the audio end
    enc.wait(); print("WROTE", out)


if __name__ == "__main__":
    main()
