#!/usr/bin/env python3
"""r10 lever A/B clips: python diag_arm.py <id> <arm> <start_s,start_s,...> [--final-label X]
Mouth + chin crop (~2x), panels SOURCE | RAW r9 (ls_pre) | RAW <arm> | final <arm>, 4 s from each start, 1x + 0.5x, + a 12-frame contact sheet.
RAW panels show the nearest 25 fps master frame. Arm = artifacts r9g_<id><arm> (layers) / r9f_<id><arm> (final); arm "lc" etc. may give a final only (--final PATH)."""
import sys, json, pathlib, argparse, numpy as np, cv2, av
ap = argparse.ArgumentParser(); ap.add_argument("id"); ap.add_argument("arm"); ap.add_argument("starts"); ap.add_argument("--final", default=None); ap.add_argument("--raw-arm", default=None)
ap.add_argument("--also", action="append", default=[], help="LABEL=PATH extra source-rate panel before the final"); ap.add_argument("--label", default=None); ap.add_argument("--dur", type=float, default=4.0)
a = ap.parse_args(); id_ = a.id; arm = a.arm
R = pathlib.Path("/workspace/dub/r10"); B = pathlib.Path("/workspace/dub/artifacts/r4m"); O = pathlib.Path("/workspace/dub/outbox/r10_face/diag/levers"); O.mkdir(parents=True, exist_ok=True)
src = sorted((R / f"runs/{id_}/work/in").glob(f"{id_}.*"))[0]; G = f"r9g_{id_}"
def raw(g): return str(B / g / f"work_{id_}_{g}" / f"{id_}_{g}_25fps_ls_pre.mp4")
ra = a.raw_arm if a.raw_arm is not None else arm
panels = [("SOURCE", str(src), False), ("RAW r9", raw(G), True)]
if ra: panels.append((f"RAW {ra}", raw(f"{G}{ra}"), True))
panels += [(x.split("=", 1)[0], x.split("=", 1)[1], False) for x in a.also]
panels.append((a.label or f"final {arm}", a.final or str(B / f"r9f_{id_}{arm}/{id_}_r9f_{id_}{arm}.mp4"), False))
lm = np.load(B / G / f"timing/{id_}_source_lm_F.npz")["lm"]
tau_s = np.asarray(json.load(open(B / G / f"timing/{id_}_master_timing_F.json"))["tau_s"])
c = av.open(str(src)); s = c.streams.video[0]; fps = float(s.average_rate); tsrc = np.array([f.pts for f in c.decode(s)], np.float64) * float(s.time_base); c.close()
def grab(path, idx):
    out = {}; want = set(idx); cap = cv2.VideoCapture(path); i = 0
    while want:
        ok, f = cap.read()
        if not ok: break
        if i in want: out[i] = f; want.discard(i)
        i += 1
    return out
for wi, st in enumerate(float(x) for x in a.starts.split(",")):
    k0 = int(np.argmin(np.abs(tsrc - st))); ks = list(range(k0, min(len(tsrc), k0 + int(round(a.dur * fps)))))
    ms = [int(np.argmin(np.abs(tau_s - tsrc[k]))) for k in ks]
    p = np.nanmedian(lm[ks], 0); iod = np.linalg.norm(p[33, :2] - p[263, :2])
    cx = float(p[[13, 14, 152], 0].mean()); cy = float(0.5 * (p[0, 1] + p[152, 1])); w = int(1.5 * iod) // 2 * 2; h = int(1.4 * iod) // 2 * 2; x0, y0 = int(cx - w / 2), int(cy - h / 2)
    data = {}
    for name, path, is_raw in panels:
        fs = grab(path, ms if is_raw else ks)
        data[name] = [fs[m if is_raw else k][max(0, y0):y0 + h, max(0, x0):x0 + w] for k, m in zip(ks, ms)]
    sc = max(1, int(round(360 / w))); tiles = []
    for t in range(len(ks)):
        row = []
        for name, _, _ in panels:
            im = cv2.resize(data[name][t], None, fx=sc, fy=sc, interpolation=cv2.INTER_LANCZOS4); cv2.putText(im, name, (6, 22), 0, 0.6, (255, 255, 255), 2); row.append(im)
        f_ = np.hstack(row); cv2.putText(f_, f"{id_} t={tsrc[ks[t]]:.2f}s", (6, f_.shape[0] - 8), 0, 0.55, (0, 255, 255), 2); tiles.append(f_)
    for speed, tag in ((1.0, "1x"), (0.5, "0.5x")):
        outp = O / f"{id_}_{arm or 'x'}_{st:.1f}s_{tag}.mp4"; o = av.open(str(outp), "w"); stv = o.add_stream("libx264", rate=round(fps * speed))
        stv.width, stv.height = tiles[0].shape[1] // 2 * 2, tiles[0].shape[0] // 2 * 2; stv.pix_fmt = "yuv420p"; stv.options = {"crf": "16"}
        for im in tiles:
            for pk in stv.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(im[:stv.height, :stv.width]), format="bgr24")): o.mux(pk)
        for pk in stv.encode(): o.mux(pk)
        o.close()
    mid = len(ks) // 2; idx = list(range(max(0, mid - 6), min(len(ks), mid + 6)))
    cv2.imwrite(str(O / f"{id_}_{arm or 'x'}_{st:.1f}s_sheet.jpg"), np.vstack([np.hstack([data[name][i] for i in idx]) for name, _, _ in panels]), [cv2.IMWRITE_JPEG_QUALITY, 92])
print("== diag_arm", id_, arm, a.starts)
