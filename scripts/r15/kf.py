#!/usr/bin/env python3
"""r15 W2: KEYFRAMES + INTERPOLATION in the stabilised aligned face-crop space, behind LS_KF=k (default off; ls_stable.StableRunner).

LatentSync (teacher or student, any overlap mode) generates the mouth on every k-th master frame of the segment only (keyframes 0, k, 2k, ...
+ the last frame), each with ITS OWN 25 fps audio window (chunk selection, as LS_RATE): option (a), the temporal layers see a k x subsampled
clip. The keyframes' generated 512 crops (after LatentSync's own paste of the upper crop) are kept in the ALIGNED crop space (the restore
is intercepted); every in-between frame's mouth is interpolated there between its two keyframes (KF_INTERP=dis, default: bidirectional
DIS optical flow, OpenCV Apache-2.0, ls_stable._flow_mid; KF_INTERP=rife: Practical-RIFE 4.25, MIT code + weights, needs the user's OK to
run its downloaded model code), and composited over THAT frame's own aligned crop with LatentSync's lower-face mask (the upper face, the head pose / position =
the real frame; only the lower face is interpolated); then every frame goes through LatentSync's normal restore (inverse affine) with its
own affine. Head motion is never interpolated (the LS_RATE tremble came from flow-filling whole frames).
    LS_KF=2|3|4   KF_RIFE=<dir with train_log/> (default /workspace/dub/rife/Practical-RIFE)   KF_RIFE_SCALE (1.0)
    KF_VOICED=1   option (b): every frame where the dub is voiced (+-2 frames) is generated (keyframes only on the dub-silent stretches:
                  plain subsampling dropped the 1-2 frame visemes / closures, SyncNet old03 kf2 -0.5, kf3 -1.7 vs s4p1500c8)"""
import os, pathlib, sys, time
import numpy as np
import torch

_R = {}


def _rife():
    if "m" not in _R:
        root = os.environ.get("KF_RIFE", "/workspace/dub/rife/Practical-RIFE"); sys.path.insert(0, root)
        from train_log.RIFE_HDv3 import Model
        m = Model(); sd = torch.load(f"{root}/train_log/flownet.pkl", map_location="cuda", weights_only=True)
        m.flownet.load_state_dict({k.replace("module.", ""): v for k, v in sd.items() if "module." in k}, False); m.eval(); _R["m"] = m
    return _R["m"]


def acc_plain(runner, frames, audio_wav, detections, fps):
    return runner.acc.run_frames(frames, str(audio_wav), detections, video_fps=fps)


@torch.no_grad()
def run(runner, frames, audio_wav, detections, fps, k):
    """frames (n, H, W, 3) RGB uint8 -> (n, H, W, 3): keyframes generated, in-between lower faces RIFE-interpolated in crop space"""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "quality"))
    from ls_stable import _paste_w, LS, _ReplayDetector
    acc = runner.acc; pipe = acc.pipe; n = len(frames); t0 = time.perf_counter()
    K = set(range(0, n, k)) | {n - 1}
    if os.environ.get("KF_VOICED") == "1":   # option (b): dense where the dub is VOICED (RMS above the segment max - 30 dB, dilated +-2 frames:
        import soundfile as sf               # fast visemes / closures live there), keyframes every k frames only on the dub-silent stretches
        y, sr = sf.read(str(audio_wav), dtype="float32"); y = y if y.ndim == 1 else y.mean(1); hop = sr // 25
        rms = np.array([np.sqrt(np.mean(y[i * hop:(i + 1) * hop] ** 2)) if i * hop < len(y) else 0.0 for i in range(n)])
        v = rms > rms.max() * 10 ** (-30 / 20) if rms.max() > 0 else np.zeros(n, bool)
        v = np.convolve(v.astype(float), np.ones(5), "same") > 0; K |= set(np.flatnonzero(v).tolist())
    K = np.array(sorted(K))
    if len(K) == n:   # nothing to interpolate: the plain generator
        return acc_plain(runner, frames, audio_wav, detections, fps)
    # 1) the keyframes alone through the generator; restore intercepted -> their synced ALIGNED crops (f, 3, 512, 512) in [-1, 1]
    ss, hk = acc.silent_seg, acc.chunk_hook; cap = {}

    def sel(enc, ch, hk=hk):
        ch = hk(enc, ch) if hk is not None else ch
        return [ch[i] for i in K if i < len(ch)]

    def restore_cap(faces, video_frames, boxes, aff):
        cap["synced"] = faces.detach().clone(); return np.asarray(video_frames[:len(faces)])
    acc.chunk_hook = sel
    if ss is not None: acc.silent_seg = np.asarray(ss)[K[K < len(ss)]]
    own = pipe.__dict__.get("restore_video"); pipe.restore_video = restore_cap
    try:
        acc.run_frames(np.ascontiguousarray(frames[K]), str(audio_wav), [detections[i] for i in K] if detections is not None else None, video_fps=fps)
    finally:
        if own is None: del pipe.restore_video
        else: pipe.restore_video = own
        acc.chunk_hook = hk; acc.silent_seg = ss
    syn = cap["synced"]; t_gen = time.perf_counter() - t0; t1 = time.perf_counter()
    # 2) every frame's aligned crop / box / affine (the same replayed detections and aligner as the generator), its lower-face paste weight
    old = os.getcwd(); os.chdir(LS)
    try:
        ip = acc.image_processor(); ip.restorer.p_bias = None; real = ip.face_detector
        if detections is not None: ip.face_detector = _ReplayDetector(real, detections)
        try: vf, faces, boxes, aff = pipe.loop_video(list(range(n)), frames)
        finally: ip.face_detector = real
        dev = syn.device; out = torch.empty((n,) + tuple(syn.shape[1:]), device=dev, dtype=syn.dtype); out[torch.as_tensor(K, device=dev)] = syn
        mid = [i for i in range(n) if i not in set(K.tolist())]; how = os.environ.get("KF_INTERP", "dis"); sc = float(os.environ.get("KF_RIFE_SCALE", "1.0"))
        if how == "dis":   # bidirectional DIS optical flow (OpenCV, Apache-2.0; ls_stable._flow_mid, as LS_RATE_EVEN) on the aligned crops
            import cv2
            from ls_stable import _flow_mid
            dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
            u8 = ((syn.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()   # (k, 512, 512, 3)
        for a in range(0, len(mid), 16):
            ii = mid[a:a + 16]; j = np.searchsorted(K, ii) - 1; ka, kb = K[j], K[j + 1]; tt = (np.array(ii) - ka) / (kb - ka)
            if how == "dis":
                mi = torch.from_numpy(np.stack([_flow_mid(dis, u8[j_], u8[j_ + 1], float(t_)) for j_, t_ in zip(j, tt)])).to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
            else:   # Practical-RIFE 4.25 (MIT code + weights)
                rife = _rife(); ts = torch.tensor(tt, device=dev, dtype=torch.float32).view(-1, 1, 1, 1)
                i0 = (syn[torch.as_tensor(j, device=dev)].float() + 1) / 2; i1 = (syn[torch.as_tensor(j + 1, device=dev)].float() + 1) / 2
                mi = rife.inference(i0, i1, ts.expand(-1, 1, *i0.shape[2:]).contiguous(), sc) * 2 - 1
            p, _, m = ip.prepare_masks_and_masked_images(faces[ii], affine_transform=False)
            w = _paste_w(m).to(dev, torch.float32)
            out[torch.as_tensor(ii, device=dev)] = (mi * w + p.to(dev, torch.float32) * (1 - w)).to(out.dtype)
        t_int = time.perf_counter() - t1; t2 = time.perf_counter()
        res = pipe.restore_video(out, vf, boxes, aff)
    finally:
        os.chdir(old)
    runner.last_kf = {"k": k, "frames": n, "keyframes": int(len(K)), "gen_s": round(t_gen, 2), "interp_s": round(t_int, 2), "restore_s": round(time.perf_counter() - t2, 2)}
    print(f"  [kf] k={k}: {len(K)} keyframes of {n} frames, gen {t_gen:.1f} s, RIFE + crops {t_int:.1f} s", flush=True)
    return res
