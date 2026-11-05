#!/usr/bin/env python3
"""r15 F: for every repaired run (b, e) of a layer repair, how trustworthy the 2-neighbour flow interpolation was: in the mouth box of the
layer, the flow p -> q magnitude (IOD), its forward-backward error (IOD, p90), and the mouth-box difference between the redrawn middle frame
and the original layer frame. Also the master's face motion over the span.  python scripts/r15/interp_conf.py <id> [tag]"""
import sys, json, pathlib
import numpy as np, cv2
idn = sys.argv[1]; tag = sys.argv[2] if len(sys.argv) > 2 else "q2"
W = pathlib.Path(f"/workspace/dub/artifacts/r4m/r9g_{tag}_{idn}/work_{tag}_{idn}_r9g_{tag}_{idn}"); P = W / f"{tag}_{idn}_r9g_{tag}_{idn}_25fps"
def read(p):
    c = cv2.VideoCapture(str(p)); F = []
    while True:
        ok, f = c.read()
        if not ok: return F
        F.append(f)
L = read(str(P) + "_ls_cfd_ia04_srt1.mp4"); R = read(str(P) + "_ls_cfd_ia04_srt1_rep2s.mp4")
lm = np.load(W / "video/master_mediapipe.npz")["lm"].astype(np.float64)
rj = json.loads(pathlib.Path(str(P) + "_ls_cfd_ia04_srt1_rep2s.repair.json").read_text())
dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM); mot = np.load(f"/workspace/dub/r15f/motion_{idn}.npy")
for b, e in rj["repaired"]:
    p, q = b - 1, e + 1; k = (b + e) // 2; f = lm[k]; iod = np.linalg.norm(f[33, :2] - f[263, :2]); cx, cy = f[[61, 291, 13, 14], :2].mean(0); r = int(0.9 * iod)
    y0, x0 = max(0, int(cy - r)), max(0, int(cx - r)); box = lambda im: np.ascontiguousarray(cv2.cvtColor(im[y0:y0 + 2 * r, x0:x0 + 2 * r], cv2.COLOR_BGR2GRAY))
    gp, gq = box(L[p]), box(L[q]); fpq, fqp = dis.calc(gp, gq, None), dis.calc(gq, gp, None)
    H, Wd = gp.shape; gy, gx = np.mgrid[0:H, 0:Wd].astype(np.float32)
    back = cv2.remap(fqp, gx + fpq[..., 0], gy + fpq[..., 1], cv2.INTER_LINEAR)   # fqp sampled where p's pixels land in q
    fbe = np.linalg.norm(fpq + back, axis=2)
    mag = np.linalg.norm(fpq, axis=2)
    d_rep = np.abs(box(R[k]).astype(np.float32) - box(L[k]).astype(np.float32)).mean()
    d_pq = np.abs(gp.astype(np.float32) - gq.astype(np.float32)).mean()
    print(f"run {b}-{e} ({b / 25:.2f}s): flow p90 {np.percentile(mag, 90) / iod:.3f} IOD, fb-err p90 {np.percentile(fbe, 90) / iod:.3f} IOD, |p-q| {d_pq:.1f}, |rep-orig| {d_rep:.1f}, face motion {mot[max(0, b - 3):e + 4].max():.3f}", flush=True)
