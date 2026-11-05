#!/usr/bin/env python3
"""r15 F3: CodeFormer at a chosen fidelity weight on the pre-CodeFormer LatentSync layer (GPU, NC model: DUB_ALLOW_NC=1 only).
The production layer is CodeFormer w = 1.0 (generator stage, insightface 5 points). This re-runs scripts/optim/codeformer_accel.CodeFormerAccel
(no ParseNet, as production) with w = --w on <run>_25fps_ls_pre.mp4, LatentSync frames only (the manifest's LATENT_SYNC segments),
5 points from the layer's MediaPipe landmarks (eye centres, nose tip, mouth corners). Output = a CF layer for r8_cf_detail.py --cf.
    flock /workspace/dub/gpu.lock python scripts/r15/cf_layer.py --work WORK --pre LS_PRE.mp4 --lm LS.mediapipe.npz --w 0.7 --out LS_cf07.mp4
"""
import argparse, json, pathlib, subprocess, sys, time
import numpy as np, cv2
R = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(R / "scripts/optim")); sys.path.insert(0, str(R / "scripts/pilot"))
EYE_L = [154, 155, 33, 7, 246, 161, 159, 158, 144, 145, 173, 133, 157, 163, 153, 160]
EYE_R = [374, 380, 390, 373, 249, 385, 384, 263, 466, 387, 386, 381, 382, 398, 388, 362]


def main():
    ap = argparse.ArgumentParser(); [ap.add_argument("--" + k, required=True) for k in ("work", "pre", "lm", "out")]
    ap.add_argument("--w", type=float, required=True); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--crf", type=int, default=10)
    a = ap.parse_args(); t0 = time.perf_counter()
    from codeformer_accel import CodeFormerAccel
    cf = CodeFormerAccel(fidelity_weight=a.w, batch_size=a.batch, use_parse=False, autocast=True, parse_autocast=True)
    W_ = pathlib.Path(a.work); man = json.loads(next(W_.glob("*_25fps.manifest.json")).read_text())
    lm = np.load(a.lm)["lm"][..., :2].astype(np.float32); n = len(lm); ls = np.zeros(n, bool)
    for s in man["lipsync"]["segments"]:
        if s["action"] == "LATENT_SYNC": ls[s["start_frame"]:min(n, s["end_frame"])] = True
    cap = cv2.VideoCapture(a.pre); Wd, Hd, fps = int(cap.get(3)), int(cap.get(4)), cap.get(cv2.CAP_PROP_FPS) or 25.0
    enc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{Wd}x{Hd}", "-r", f"{fps}", "-i", "-",
                            "-c:v", "libx264", "-crf", str(a.crf), "-preset", "medium", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
    k = 0; restored = 0
    while True:
        frs, idx = [], []
        for _ in range(a.chunk):
            ok, f = cap.read()
            if not ok: break
            frs.append(f); idx.append(k); k += 1
        if not frs: break
        l5 = []
        for i in idx:
            p = lm[i] if i < n else None
            if p is None or not ls[i] or not np.isfinite(p[EYE_L + EYE_R + [4, 61, 291]]).all(): l5.append(None); continue
            l5.append(np.array([p[EYE_L].mean(0), p[EYE_R].mean(0), p[4], p[61], p[291]], np.float32))
        out, st = cf.restore_frames(frs, l5); restored += st["faces_restored"]
        for f in out: enc.stdin.write(np.ascontiguousarray(f).tobytes())
    enc.stdin.close(); enc.wait(); cap.release(); cf.close()
    print("== cf_layer", json.dumps({"frames": k, "restored": restored, "w": a.w, "seconds": round(time.perf_counter() - t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
