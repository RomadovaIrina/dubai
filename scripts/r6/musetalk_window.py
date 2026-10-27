#!/usr/bin/env python3
"""r6 G: MuseTalk 1.5 (TMElyralab, weights MIT: HF card `license: mit`, "The trained model are available for any purpose, even commercially")
as an alternative lip-sync backend for ls_window_ab.py --precomputed. Licence-clean inference only:
  - crop box from OUR MediaPipe landmarks (Apache-2.0) instead of MuseTalk's DWPose + S3FD; same geometry as MuseTalk v1.5's
    get_landmark_and_bbox: x = jaw-contour extent, y2 = chin (+ extra_margin), y1 = nose(iBUG 29) - (chin - nose) -> the box is centred on the
    lower nose; box smoothed over time (median 5 + mean 3) like our LatentSync anchors are
  - NO face-parse-bisent (CelebAMask-HQ, non-commercial data): the raw 256^2 MuseTalk crop is pasted as a rectangle and ls_window_ab.py's own
    mouth-only mask / feather / crossfade decides what reaches the output (identical to the LatentSync arms)
  - VAE = our models/sd-vae-ft-mse (MIT), audio = whisper-tiny (MIT)
    python scripts/r6/musetalk_window.py --master MASTER_25fps.mp4 --t0 10 --t1 18 --audio drive_16k.wav --out mt.mp4 [--mt-root DIR]"""
import argparse, json, math, pathlib, subprocess, sys, time
import numpy as np, cv2, torch

HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "pilot"))
FPS = 25
JAW = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454]
NOSE29, CHIN = 5, 152


def boxes_from_lm(lm, H, W, margin):
    ok = np.isfinite(lm[:, 0, 0]); b = np.full((len(lm), 4), np.nan)
    for i in np.where(ok)[0]:
        p = lm[i]; x1, x2 = p[JAW, 0].min(), p[JAW, 0].max(); y2 = p[CHIN, 1]; ny = p[NOSE29, 1]
        b[i] = [x1, max(0, ny - (y2 - ny)), x2, min(H, y2 + margin)]
    idx = np.arange(len(b))
    for k in range(4):   # fill gaps, then median 5 + mean 3 over time
        v = b[:, k]; g = np.isfinite(v)
        if g.sum() == 0: return None
        b[:, k] = np.interp(idx, idx[g], v[g])
    from scipy.ndimage import median_filter, uniform_filter1d
    b = uniform_filter1d(median_filter(b, size=(5, 1), mode="nearest"), 3, axis=0, mode="nearest")
    return np.round(b).astype(int)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--master", required=True); ap.add_argument("--t0", type=float, required=True); ap.add_argument("--t1", type=float, required=True)
    ap.add_argument("--audio", required=True); ap.add_argument("--out", required=True); ap.add_argument("--mt-root", default="/workspace/dub/alt/MuseTalk")
    ap.add_argument("--mt-models", default="/workspace/dub/alt/mt_models"); ap.add_argument("--margin", type=int, default=10); ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args(); sys.path.insert(0, a.mt_root)
    from musetalk.models.vae import VAE
    from musetalk.models.unet import UNet, PositionalEncoding
    from musetalk.utils.audio_processor import AudioProcessor
    from transformers import WhisperModel
    import clean_face_stack as cfs
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    f0, f1 = int(round(a.t0 * FPS)), int(round(a.t1 * FPS))
    cap = cv2.VideoCapture(a.master); cap.set(cv2.CAP_PROP_POS_FRAMES, f0); frames = []
    for _ in range(f1 - f0):
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release(); H, W = frames[0].shape[:2]
    clip = out.with_name(out.stem + "_src.mp4")
    wr = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-qp", "0", str(clip)], stdin=subprocess.PIPE)
    for f in frames: wr.stdin.write(f.tobytes())
    wr.stdin.close(); wr.wait()
    lm = np.load(cfs.run_mediapipe(clip, out.with_name(out.stem + "_src_mp.npz")))["lm"]
    bx = boxes_from_lm(lm, H, W, a.margin)
    dev = torch.device("cuda"); t0 = time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    vae = VAE(model_path=str(ROOT / "models" / "sd-vae-ft-mse"), use_float16=True)
    unet = UNet(unet_config=f"{a.mt_models}/musetalkV15/musetalk.json", model_path=f"{a.mt_models}/musetalkV15/unet.pth", device=dev)
    unet.model = unet.model.half().to(dev); pe = PositionalEncoding(d_model=384).half().to(dev)
    ap_ = AudioProcessor(feature_extractor_path=f"{a.mt_models}/whisper")
    wh = WhisperModel.from_pretrained(f"{a.mt_models}/whisper").to(dev, torch.float16).eval()
    t_load = time.perf_counter() - t0; t0 = time.perf_counter()
    with torch.no_grad():
        feats, ln = ap_.get_audio_feature(a.audio)
        chunks = ap_.get_whisper_chunk(feats, dev, torch.float16, wh, ln, fps=FPS, audio_padding_length_left=2, audio_padding_length_right=2)
        n = min(len(frames), len(chunks)); lat = []
        for i in range(n):
            x1, y1, x2, y2 = bx[i]; crop = cv2.resize(frames[i][y1:y2, x1:x2], (256, 256), interpolation=cv2.INTER_LANCZOS4)
            lat.append(vae.get_latents_for_unet(crop))
        res = []
        for s in range(0, n, a.batch):
            wb = torch.stack(list(chunks[s:s + a.batch])).to(dev); lb = torch.cat(lat[s:s + a.batch]).to(dev, torch.float16)
            pred = unet.model(lb, torch.tensor([0], device=dev), encoder_hidden_states=pe(wb)).sample
            res.extend(vae.decode_latents(pred))
    torch.cuda.synchronize(); t_gen = time.perf_counter() - t0
    wr = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-qp", "0", str(out)], stdin=subprocess.PIPE)
    for i, f in enumerate(frames):
        g = f.copy()
        if i < n:
            x1, y1, x2, y2 = bx[i]; g[y1:y2, x1:x2] = cv2.resize(res[i], (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)
        wr.stdin.write(g.tobytes())
    wr.stdin.close(); wr.wait()
    r = {"backend": "musetalk_v15", "frames": n, "load_s": round(t_load, 2), "gen_s": round(t_gen, 2), "s_per_frame": round(t_gen / max(n, 1), 4),
         "peak_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20), "crop_px_mean": float(np.mean(bx[:, 2] - bx[:, 0]))}
    out.with_suffix(".json").write_text(json.dumps(r, indent=1)); print(json.dumps(r))


if __name__ == "__main__":
    main()
