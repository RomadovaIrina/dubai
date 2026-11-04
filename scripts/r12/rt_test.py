#!/usr/bin/env python3
"""r13 LS: Lab drift of ONE PyAV decode -> bgr -> x264 (qp 0) round trip, default conversions vs explicit BT.709 tv range both ways.
    python scripts/r12/rt_test.py SOURCE.mp4 [frames]"""
import sys, tempfile, pathlib
import numpy as np, av, cv2
src = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 30


def dec(path, cms):
    c = av.open(path); s = c.streams.video[0]; out = []
    for fr in c.decode(s):
        out.append(fr.reformat(format="bgr24", src_colorspace="ITU709", src_color_range="MPEG").to_ndarray() if cms else fr.to_ndarray(format="bgr24"))
        if len(out) >= N: break
    c.close(); return out


def enc(frames, path, cms):
    c = av.open(path, "w"); st = c.add_stream("libx264", rate=25); st.width, st.height = frames[0].shape[1], frames[0].shape[0]; st.pix_fmt = "yuv420p"
    st.options = {"qp": "0", "preset": "ultrafast"} | ({"x264-params": "colorprim=bt709:transfer=bt709:colormatrix=bt709"} if cms else {})
    for f in frames:
        vf = av.VideoFrame.from_ndarray(f, format="bgr24")
        if cms: vf = vf.reformat(format="yuv420p", dst_colorspace="ITU709", dst_color_range="MPEG")
        for p in st.encode(vf): c.mux(p)
    for p in st.encode(): c.mux(p)
    c.close()


def lab(x): y = cv2.cvtColor(x, cv2.COLOR_BGR2LAB).astype(np.float32); y[..., 0] *= 100 / 255; y[..., 1:] -= 128; return y


with tempfile.TemporaryDirectory() as td:
    for cms in (False, True):
        a = dec(src, cms); o = str(pathlib.Path(td) / f"rt{int(cms)}.mp4"); enc(a, o, cms); b = dec(o, cms)
        d = np.mean([(lab(y) - lab(x)).reshape(-1, 3).mean(0) for x, y in zip(a, b)], 0)
        # the same output read back the way the reference reader reads the SOURCE (default decode of both)
        a0, b0 = dec(src, False), dec(o, False); d0 = np.mean([(lab(y) - lab(x)).reshape(-1, 3).mean(0) for x, y in zip(a0, b0)], 0)
        print(f"== cms={cms}: self-consistent dL da db {np.round(d, 2).tolist()} | read back by default decode {np.round(d0, 2).tolist()}", flush=True)
