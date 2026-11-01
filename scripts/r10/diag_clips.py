#!/usr/bin/env python3
"""r10 diagnosis: worst 4 s windows (lip-ring brightness pops + chin slide, from face_metrics per-frame series of the r9+fix final), zoomed
mouth+chin crops SOURCE | RAW LatentSync (ls_pre, nearest master frame) | r9 + fix | F1, 1x and 0.5x, + contact sheets of consecutive frames."""
import sys, json, subprocess, pathlib, numpy as np, cv2, av
id_ = sys.argv[1]; nwin = int(sys.argv[2]) if len(sys.argv) > 2 else 2
R = pathlib.Path("/workspace/dub/r10"); B = pathlib.Path("/workspace/dub/artifacts/r4m"); O = pathlib.Path("/workspace/dub/outbox/r10_face/diag"); O.mkdir(parents=True, exist_ok=True)
src = sorted((R / f"runs/{id_}/work/in").glob(f"{id_}.*"))[0]; G = f"r9g_{id_}"; W = B / G / f"work_{id_}_{G}"
vids = {"SOURCE": str(src), "RAW LatentSync": str(W / f"{id_}_{G}_25fps_ls_pre.mp4"), "r9 + fix": str(B / f"r9f_{id_}/{id_}_r9f_{id_}.mp4"), "F1 (r10)": str(R / f"abl/{id_}/{id_}_f1.mp4")}
lm = np.load(B / G / f"timing/{id_}_source_lm_F.npz")["lm"]; fr = np.load(R / f"metrics/{id_}/F_fx.frames.npz")
c = av.open(str(src)); fps = float(c.streams.video[0].average_rate); c.close()
n = len(lm); T = fr["touched"][:n]
brr = np.nan_to_num(fr["brr"][:n, 0] - fr["brr"][:n, 1]); vel = np.zeros(n)
df = fr["dflow"][:n]; ok = np.isfinite(df).all(1)
for k in range(1, n):
    if ok[k] and ok[k - 1]: vel[k] = np.linalg.norm(df[k] - df[k - 1])
score = np.clip(brr / (np.percentile(brr[T], 90) + 1e-6), 0, 5) + np.clip(vel / (np.percentile(vel[T], 90) + 1e-6), 0, 5)
L = int(round(4 * fps)); cs = np.convolve(score * T, np.ones(L), "valid"); picks = []
for i in np.argsort(-cs):
    if all(abs(i - j) > L for j in picks): picks.append(int(i))
    if len(picks) >= nwin: break
tau = json.load(open(B / G / f"timing/{id_}_master_timing_F.json")); tau_s = np.asarray(tau["tau_s"])
c = av.open(str(src)); s = c.streams.video[0]; tsrc = np.array([f.pts for f in c.decode(s)], np.float64) * float(s.time_base); c.close()
def frames(path, ks, master=False):
    out = {}; want = set(ks); cap = cv2.VideoCapture(path); i = 0
    while want:
        ok_, f = cap.read()
        if not ok_: break
        if i in want: out[i] = f; want.discard(i)
        i += 1
    return out
rep = []
for wi, k0 in enumerate(picks):
    ks = list(range(k0, min(n, k0 + L))); p = np.nanmedian(lm[ks], 0); iod = np.linalg.norm(p[33, :2] - p[263, :2])
    cx = float(p[[13, 14, 152], 0].mean()); cy = float(0.5 * (p[0, 1] + p[152, 1])); w = int(1.5 * iod) // 2 * 2; h = int(1.4 * iod) // 2 * 2
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    ms = [int(np.argmin(np.abs(tau_s - tsrc[k]))) for k in ks]   # nearest master frame (true time) for the raw layer
    data = {}
    for name, path in vids.items():
        fs = frames(path, ms if name.startswith("RAW") else ks)
        data[name] = [fs[m if name.startswith("RAW") else k][max(0, y0):y0 + h, max(0, x0):x0 + w] for k, m in zip(ks, ms)]
    sc = max(1, int(round(360 / w)))   # >= 2x on these faces
    tiles = []
    for t in range(len(ks)):
        row = []
        for name in vids:
            im = cv2.resize(data[name][t], None, fx=sc, fy=sc, interpolation=cv2.INTER_LANCZOS4); cv2.putText(im, name, (6, 22), 0, 0.6, (255, 255, 255), 2)
            row.append(im)
        fr_ = np.hstack(row); cv2.putText(fr_, f"{id_} src frame {ks[t]}  t={tsrc[ks[t]]:.2f}s", (6, fr_.shape[0] - 8), 0, 0.55, (0, 255, 255), 2); tiles.append(fr_)
    for speed, tag in ((1.0, "1x"), (0.5, "0.5x")):
        outp = O / f"{id_}_diag{wi + 1}_{tsrc[k0]:.1f}s_{tag}.mp4"
        o = av.open(str(outp), "w"); st = o.add_stream("libx264", rate=round(fps * speed)); st.width, st.height = tiles[0].shape[1] // 2 * 2, tiles[0].shape[0] // 2 * 2; st.pix_fmt = "yuv420p"; st.options = {"crf": "16"}
        for im in tiles:
            for pk in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(im[:st.height, :st.width]), format="bgr24")): o.mux(pk)
        for pk in st.encode(): o.mux(pk)
        o.close()
    # contact sheet: 12 consecutive frames around the window's worst frame, one row per panel
    kw = ks[int(np.argmax(score[ks]))]; idx = [ks.index(k) for k in range(max(ks[0], kw - 5), min(ks[-1], kw + 6) + 1)]
    sheet = np.vstack([np.hstack([cv2.resize(data[name][i], None, fx=1, fy=1) for i in idx]) for name in vids])
    cv2.imwrite(str(O / f"{id_}_diag{wi + 1}_sheet_frames{ks[idx[0]]}-{ks[idx[-1]]}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    rep.append({"window": wi + 1, "start_s": round(float(tsrc[k0]), 2), "frames": [ks[0], ks[-1]], "worst_frame": int(kw), "brr_excess_max": round(float(brr[ks].max()), 2), "chin_vel_max": round(float(vel[ks].max()), 4)})
print(json.dumps(rep))
