#!/usr/bin/env python3
"""r15 W1: BEST-OF-N per window for the overlapping 'continue' generator (ls_stable.StableLatentSync, LS_STRIDE=S LS_OVERLAP=continue),
behind LS_BON=N (default 1 = off, the code path is not entered). Built for the 4-step student (no CFG), works with any step count / CFG.

Each window (16 frames, stride S) is denoised for N candidate initial noises AT ONCE (batch N): candidate 0 = the trajectory's own noise
(so a pick of 0 everywhere = the plain c8 output), candidate k = LatentSync-style noise (one draw repeated over the frames) from seed
1247 + 7919 k. The frames a window shares with the previous window are pinned to the CHOSEN previous result (re-noised with the
candidate's own noise), so the pick is sequential and compatible with c8. Each candidate's NEW frames are decoded (the lower 288 rows of
the 512 aligned crop only) and scored; the best one is kept:
  sync    stable-SyncNet (LatentSync's training SyncNet, 16 frames, lower half, 256 px) cosine of the full window vs its mel   / 0.03
  open    mouth-opening proxy (dark-cavity share in the open box, scripts/r13/distill/distill2.py opening()) on the dub-VOICED new frames
          minus that on the dub-SILENT ones (penalises the lazy mouth AND a mouth open in silence)                               / 0.05 x 0.5
  sharp   log Laplacian variance of the luma in the lip box (penalises smear)                                                  / 0.15 x 0.5
  jump    log( lip-box change first-new vs last-pinned frame / median change between the new frames ) (continuity)            / 0.30 x -0.5
Fixed generic scales (no per-video tuning); a tie keeps candidate 0. Stats -> acc.last_bon (+ BON_LOG jsonl when set).
LS_BON_EARLY=1 (cheap): the N candidates run only the FIRST denoising step; the pick is scored on their x0 preview (the scheduler's
pred_original_sample) and only the winner runs the remaining steps: UNet cost (N + steps - 1) / steps instead of N."""
import json, math, os, pathlib, sys, time
import numpy as np
import torch
import torch.nn.functional as F


_CFG_RESCALE = float(os.environ.get("LS_CFG_RESCALE", "0") or 0)
_CFG_TSMOOTH = float(os.environ.get("LS_CFG_TSMOOTH", "0") or 0)   # r15 Z: temporal smoothing of the CFG amplification term


def _ls_cfg(nu, na, g):
    """classifier-free guidance, the same as ls_stable._cfg (r15 Z LS_CFG_RESCALE=phi: per-frame guidance rescale, Lin et al. 2023)"""
    out = nu + g * (na - nu)
    if _CFG_TSMOOTH > 0 and out.ndim == 5 and out.shape[2] >= 3:   # (b, c, f, h, w): smooth (na - nu) over f, keep na exact
        d = (na - nu).float(); dp = torch.cat([d[:, :, :1], d[:, :, :-1]], 2); dn = torch.cat([d[:, :, 1:], d[:, :, -1:]], 2)
        out = (na.float() + (g - 1) * ((1 - 2 * _CFG_TSMOOTH) * d + _CFG_TSMOOTH * (dp + dn))).to(out.dtype)
    if _CFG_RESCALE <= 0: return out
    dims = (1, 3, 4) if out.ndim == 5 else tuple(range(1, out.ndim)); o = out.float()
    s_c = na.float().std(dim=dims, keepdim=True); s_o = o.std(dim=dims, keepdim=True)
    return (_CFG_RESCALE * o * (s_c / s_o.clamp_min(1e-6)) + (1 - _CFG_RESCALE) * o).to(out.dtype)

ROOT = pathlib.Path(__file__).resolve().parents[2]
OBOX = (0.35, 0.56, 0.65, 0.72); LBOX = (0.25, 0.538, 0.747, 0.787); Y0 = 224   # decoded crop rows [224, 512) of the 512 crop = latent rows [28, 64)
W_SYNC, W_OPEN, W_SHARP, W_JUMP = 1.0, 0.5, 0.5, 0.5; S_SYNC, S_OPEN, S_SHARP, S_JUMP = 0.03, 0.05, 0.15, 0.30
_SN = {}


def _syncnet():
    if "m" not in _SN:
        sys.path.insert(0, str(ROOT / "scripts" / "r13" / "distill")); import distill as D
        _SN["m"], _SN["c"] = D.load_syncnet()
    return _SN["m"], _SN["c"]


def _audio(audio_path, n):
    """(mel (80, T) of LatentSync's melspectrogram, per-frame RMS (n,) at 25 fps, voiced mask (RMS above max - 30 dB))"""
    import librosa
    from latentsync.utils.audio import melspectrogram
    y, _ = librosa.load(str(audio_path), sr=16000); mel = torch.from_numpy(melspectrogram(y)).float()
    hop = 640; rms = np.array([np.sqrt(np.mean(y[i * hop:(i + 1) * hop] ** 2)) if i * hop < len(y) else 0.0 for i in range(n)])
    return mel, rms, rms > rms.max() * 10 ** (-30 / 20) if rms.max() > 0 else np.zeros(n, bool)


def _decode_low(pipe, lat):
    """(K, 4, F, 64, 64) latents -> (K, F, 3, 288, 512) pixels in [-1, 1] of crop rows [224, 512)"""
    K, C, Fn = lat.shape[:3]; z = lat[:, :, :, Y0 // 8:].permute(0, 2, 1, 3, 4).reshape(K * Fn, C, -1, lat.shape[-1])
    z = z / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor; out = []
    vae = pipe.vae; dec = vae.decoder; inner = getattr(dec, "inner", None)   # LS_VAE_COMPILE's _Pad16 is compiled for the full 64 x 64 latent only:
    if inner is not None: vae.decoder = inner                                # the cropped decode runs eager (no ~60 s re-autotune, no padding to 16)
    try:
        for i in range(0, z.shape[0], 24): out.append(vae.decode(z[i:i + 24].to(vae.dtype)).sample.float())
    finally:
        vae.decoder = dec
    return torch.cat(out).reshape(K, Fn, 3, out[0].shape[-2], out[0].shape[-1])


def _luma(px): return (0.299 * px[:, 0] + 0.587 * px[:, 1] + 0.114 * px[:, 2] + 1) / 2   # (F, h, w) in [0, 1]


def _box(y, b, W=512):
    return y[:, int(b[1] * 512) - Y0:int(b[3] * 512) - Y0, int(b[0] * W):int(b[2] * W)]


def _opening(px):
    y = _luma(px); skin = y[:, 32:].flatten(1).median(1).values   # lower half of the 512 crop = rows [256, 512) = our rows [32, 288)
    return torch.sigmoid((0.55 * skin[:, None, None] - _box(y, OBOX)) / 0.04).mean((1, 2))


def _sharp(px):
    g = _box(_luma(px), LBOX)[:, None]; k = torch.tensor([[[[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]]], device=g.device)
    return torch.log(F.conv2d(g, k).flatten(1).var(1) + 1e-8)


def _score(pipe, xs, a_, o, b_, dec, voiced, mel, video_fps, sn, sc, weights=None):
    """xs (K, 4, L, h, w) candidate latents of window [a_, b_) (o pinned frames) -> (pick, z, raw (K, 4), px (K, Fn, 3, 288, 512))"""
    K = xs.shape[0]; px = _decode_low(pipe, xs[:, :, o:]); vo = voiced[a_ + o:b_]; raw = np.zeros((K, 4))
    for k in range(K):
        p = px[k]; op = _opening(p).cpu().numpy()
        raw[k, 1] = (op[vo].mean() if vo.any() else 0.0) - (op[~vo].mean() if (~vo).any() else 0.0)
        raw[k, 2] = float(_sharp(p).mean())
        if o > 0 and (a_ + o - 1) in dec:
            yl = _box(_luma(torch.cat([dec[a_ + o - 1][None], p])), LBOX); d = (yl[1:] - yl[:-1]).abs().mean((1, 2)).cpu().numpy()
            raw[k, 3] = math.log((d[0] + 1e-4) / (np.median(d[1:]) + 1e-4)) if len(d) > 1 else 0.0
        full = torch.cat([torch.stack([dec[i] for i in range(a_, a_ + o)]), p]) if o > 0 and all(i in dec for i in range(a_, a_ + o)) else (p if o == 0 else None)
        m0 = int(80 * a_ / video_fps)
        if full is not None and full.shape[0] == 16 and m0 + 52 <= mel.shape[1]:
            v = F.interpolate(full[:, :, 32:], size=(sc.data.resolution // 2, sc.data.resolution), mode="bilinear", align_corners=False)
            ve, ae = sn(v.reshape(1, -1, v.shape[2], v.shape[3]).half(), mel[None, None, :, m0:m0 + 52].to(xs.device).half())
            raw[k, 0] = float(F.cosine_similarity(ve.float(), ae.float()))
        else: raw[k, 0] = np.nan
    z = np.zeros(K); ws = weights or (W_SYNC, W_OPEN, W_SHARP, -W_JUMP)
    for j, (w, s) in enumerate(zip(ws, (S_SYNC, S_OPEN, S_SHARP, S_JUMP))):
        col = raw[:, j]
        if np.isfinite(col).all(): z += w * (col - col.mean()) / s
    return (int(np.argmax(z)) if z.max() > z[0] + 1e-6 else 0), z, raw, px


def _src_open_ref(cur_px):
    """BON_REF=src: the median dark-cavity opening proxy of the SOURCE's own aligned crops of this segment (the same proxy / box as the
    generated frames): the speaker's natural opening level, an absolute reference a uniformly damped student can be measured against"""
    vals = []
    for i in range(0, cur_px.shape[0], 32):
        p = cur_px[i:i + 32, :, Y0:].float().to("cuda"); vals.append(_opening(p).cpu().numpy())
    return float(np.median(np.concatenate(vals))) if vals else None


def continue_bon_sel(acc, pipe, windows, eps0, mask_lat, masked_lat, ref_lat, audio_all, timesteps, extra, do_cfg, audio_path, video_fps, cur_px=None):
    """LS_BON_SEL=1 (with LS_BON=N): SELECTIVE best-of-N. Every window runs the first denoising step with the CURRENT noise only; its x0
    preview is checked: LAZY = voiced (>= half the new frames dub-voiced) and the voiced-frame opening proxy < BON_LAZY (0.7) x this
    speaker's running median of window previews; SMEAR = the lip-box log Laplacian variance < running median - BON_SMEAR (0.3). Only a
    flagged window tries the other N-1 noises (first step each, batch); a candidate is taken only if it improves the flagged criterion and
    wins z = sync + open + 0.5 sharp - 2 jump (strong continuity: the first new frame vs the last pinned one); the chosen noise STAYS
    the current noise for the next windows (no switching back and forth: the eye saw more tremble with bon3's per-window seeds)."""
    N = int(os.environ.get("LS_BON", "3")); dev = eps0.device; n = eps0.shape[2]; t_start = time.perf_counter(); t_score = 0.0
    LAZY = float(os.environ.get("BON_LAZY", "0.7")); SMEAR = float(os.environ.get("BON_SMEAR", "0.3")); WSEL = (1.0, 1.0, 0.5, -2.0)
    E = [eps0]
    for k in range(1, N):
        g = torch.Generator(device=dev).manual_seed(1247 + 7919 * k)
        E.append(torch.randn((1,) + tuple(eps0.shape[1:2]) + (1,) + tuple(eps0.shape[3:]), generator=g, device=dev, dtype=eps0.dtype).repeat(1, 1, n, 1, 1) * pipe.scheduler.init_noise_sigma)
    E = torch.cat(E); final = torch.empty_like(eps0); done = 0; dec = {}; rows = []; cur = 0; h_open, h_sharp = [], []; nflag = nlazy = nsmear = nswitch = 0
    sn, sc = _syncnet(); mel, rms, voiced = _audio(audio_path, n)
    src_ref = _src_open_ref(cur_px) if os.environ.get("BON_REF") == "src" and cur_px is not None else None   # absolute lazy reference (see _src_open_ref)

    def conds(a_, b_, K):
        mk, mi, rl = [v[:, :, a_:b_].expand(K, *v.shape[1:2], b_ - a_, *v.shape[3:]).contiguous() for v in (mask_lat, masked_lat, ref_lat)]
        au = audio_all[a_:b_].repeat(K, 1, 1)
        if do_cfg: mk, mi, rl, au = torch.cat([mk] * 2), torch.cat([mi] * 2), torch.cat([rl] * 2), torch.cat([torch.zeros_like(au), au])
        return mk, mi, rl, au

    def run_step(x, Ew, t, cs, K, o, known):
        mk, mi, rl, au = cs; x = x.clone()
        if o > 0: x[:, :, :o] = pipe.scheduler.add_noise(known.expand(K, *known.shape[1:]), Ew[:, :, :o], t.reshape(1))
        x2 = torch.cat([x] * 2) if do_cfg else x
        npred = acc.unet(torch.cat([pipe.scheduler.scale_model_input(x2, t), mk, mi, rl], dim=1), t, encoder_hidden_states=au).sample
        if do_cfg:
            nu, na = npred.chunk(2); npred = _ls_cfg(nu, na, acc.guidance)
        st_ = pipe.scheduler.step(npred, t, x, **extra); x0 = st_.pred_original_sample.clone()
        if o > 0: x0[:, :, :o] = known.expand(K, *known.shape[1:])
        return st_.prev_sample, x0

    with torch.no_grad():
        for a_, b_ in windows:
            o = max(0, done - a_); known = final[:, :, a_:a_ + o] if o > 0 else None; t0_ = timesteps[0]
            Ew = E[cur:cur + 1, :, a_:b_]; x, x0 = run_step(Ew.clone(), Ew, t0_, conds(a_, b_, 1), 1, o, known)
            ts_ = time.perf_counter(); p0 = _decode_low(pipe, x0[:, :, o:])[0]; vo = voiced[a_ + o:b_]
            op = _opening(p0).cpu().numpy(); shp = float(_sharp(p0).mean()); ov = float(op[vo].mean()) if vo.sum() * 2 >= max(1, len(vo)) else None
            if src_ref is not None: lazy = ov is not None and ov < LAZY * src_ref   # BON_REF=src: below the speaker's own source opening level
            else: lazy = ov is not None and len(h_open) >= 3 and ov < LAZY * float(np.median(h_open))
            smear = len(h_sharp) >= 3 and shp < float(np.median(h_sharp)) - SMEAR
            if ov is not None: h_open.append(ov)
            h_sharp.append(shp); t_score += time.perf_counter() - ts_
            if lazy or smear:
                nflag += 1; nlazy += int(lazy); nsmear += int(smear); alts = [k for k in range(N) if k != cur]; Ea = E[alts][:, :, a_:b_]
                xa, x0a = run_step(Ea.clone(), Ea, t0_, conds(a_, b_, len(alts)), len(alts), o, known)
                ts_ = time.perf_counter(); _, z, raw, _ = _score(pipe, torch.cat([x0, x0a]), a_, o, b_, dec, voiced, mel, video_fps, sn, sc, weights=WSEL); t_score += time.perf_counter() - ts_
                ok = [j for j in range(1, len(alts) + 1) if (not lazy or raw[j, 1] > raw[0, 1]) and (not smear or raw[j, 2] > raw[0, 2]) and z[j] > z[0] + 1e-6]
                j = max(ok, key=lambda j_: z[j_]) if ok else 0
                if j:
                    cur = alts[j - 1]; x = xa[j - 1:j]; nswitch += 1
                rows.append({"w": [a_, b_], "lazy": bool(lazy), "smear": bool(smear), "pick": int(cur), "switched": bool(j), "z": [round(float(v), 2) for v in z],
                             "raw": [[round(float(v), 4) for v in r] for r in raw]})
            Ew = E[cur:cur + 1, :, a_:b_]; cs = conds(a_, b_, 1)
            for t in timesteps[1:]: x, _ = run_step(x, Ew, t, cs, 1, o, known)
            if o > 0: x[:, :, :o] = known
            final[:, :, a_:b_] = x; done = b_
            ts_ = time.perf_counter(); pxc = _decode_low(pipe, x[:, :, o:])[0]; t_score += time.perf_counter() - ts_
            for i, f_ in enumerate(range(a_ + o, b_)): dec[f_] = pxc[i]
            for f_ in [f_ for f_ in dec if f_ < a_ - 16]: del dec[f_]
    st = {"N": N, "selective": True, "src_ref": src_ref, "windows": len(windows), "flagged": nflag, "lazy": nlazy, "smear": nsmear, "switched": nswitch,
          "share_flagged": round(nflag / max(1, len(windows)), 3), "share_switched": round(nswitch / max(1, len(windows)), 3),
          "not_seed0": nswitch, "share_not_seed0": round(nswitch / max(1, len(windows)), 3), "pick_hist": [], "early": True,
          "seconds": round(time.perf_counter() - t_start, 2), "score_s": round(t_score, 2)}
    lg = os.environ.get("BON_LOG")
    if lg:
        with open(lg, "a") as f_: f_.write(json.dumps({**st, "audio": str(audio_path), "rows": rows}) + "\n")
    print(f"  [bon-sel] N={N} windows {len(windows)}, flagged {nflag} (lazy {nlazy}, smear {nsmear}), switched {nswitch}, {st['seconds']} s (scoring {st['score_s']} s)", flush=True)
    return final, st


def continue_ens(acc, pipe, windows, eps0, mask_lat, masked_lat, ref_lat, audio_all, timesteps, extra, do_cfg):
    """r15 Z LS_ENS=K (default off; the 'continue' overlap path): a K-noise LATENT ENSEMBLE against per-frame sampling texture noise
    (the lip colour / brightness jumps born in the raw generator layer, not on window seams). The K trajectories (noise 0 = the
    trajectory's own, k = seed 1247 + 7919 k) are denoised as one batch; at EVERY step their x0 predictions are averaged and each
    trajectory steps with the epsilon implied by the shared x0 (DDIM, epsilon prediction), so they agree on one x0 = the average of K
    independent samples' estimates; the window keeps trajectory 0. Cost x K UNet rows (pair it with LS_KF=2)."""
    K = int(os.environ.get("LS_ENS", "1")); dev = eps0.device; n = eps0.shape[2]; t0_ = time.perf_counter()
    E = [eps0]
    for k in range(1, K):
        g = torch.Generator(device=dev).manual_seed(1247 + 7919 * k)
        E.append(torch.randn((1,) + tuple(eps0.shape[1:2]) + (1,) + tuple(eps0.shape[3:]), generator=g, device=dev, dtype=eps0.dtype).repeat(1, 1, n, 1, 1) * pipe.scheduler.init_noise_sigma)
    E = torch.cat(E); final = torch.empty_like(eps0); done = 0; ac = pipe.scheduler.alphas_cumprod.to(dev)
    with torch.no_grad():
        for a_, b_ in windows:
            o = max(0, done - a_); x = E[:, :, a_:b_].clone(); known = final[:, :, a_:a_ + o] if o > 0 else None
            mk, mi, rl = [v[:, :, a_:b_].expand(K, *v.shape[1:2], b_ - a_, *v.shape[3:]).contiguous() for v in (mask_lat, masked_lat, ref_lat)]
            au = audio_all[a_:b_].repeat(K, 1, 1)
            if do_cfg: mk, mi, rl, au = torch.cat([mk] * 2), torch.cat([mi] * 2), torch.cat([rl] * 2), torch.cat([torch.zeros_like(au), au])
            for t in timesteps:
                if o > 0: x[:, :, :o] = pipe.scheduler.add_noise(known.expand(K, *known.shape[1:]), E[:, :, a_:a_ + o], t.reshape(1))
                x2 = torch.cat([x] * 2) if do_cfg else x
                npred = acc.unet(torch.cat([pipe.scheduler.scale_model_input(x2, t), mk, mi, rl], dim=1), t, encoder_hidden_states=au).sample
                if do_cfg:
                    nu, na = npred.chunk(2); npred = _ls_cfg(nu, na, acc.guidance)
                a_t = ac[int(t)].to(npred.dtype); x0 = (x - (1 - a_t).sqrt() * npred) / a_t.sqrt()
                x0m = x0.float().mean(0, keepdim=True).to(x0.dtype)
                eps = (x - a_t.sqrt() * x0m) / (1 - a_t).sqrt()          # the epsilon each trajectory needs to land on the shared x0
                x = pipe.scheduler.step(eps, t, x, **extra).prev_sample
            if o > 0: x[:, :, :o] = known.expand(K, *known.shape[1:])
            final[:, :, a_:b_] = x[:1]; done = b_
    print(f"  [ens] K={K} windows {len(windows)}, {time.perf_counter() - t0_:.1f} s", flush=True)
    return final, {"K": K, "windows": len(windows)}


def continue_bon(acc, pipe, windows, eps0, mask_lat, masked_lat, ref_lat, audio_all, timesteps, extra, do_cfg, audio_path, video_fps, cur_px=None):
    if os.environ.get("LS_BON_SEL") == "1":
        return continue_bon_sel(acc, pipe, windows, eps0, mask_lat, masked_lat, ref_lat, audio_all, timesteps, extra, do_cfg, audio_path, video_fps, cur_px=cur_px)
    N = int(os.environ.get("LS_BON", "1")); early = os.environ.get("LS_BON_EARLY") == "1"; dev = eps0.device; n = eps0.shape[2]
    t_start = time.perf_counter(); t_score = 0.0
    E = [eps0]
    for k in range(1, N):
        g = torch.Generator(device=dev).manual_seed(1247 + 7919 * k)
        E.append(torch.randn((1,) + tuple(eps0.shape[1:2]) + (1,) + tuple(eps0.shape[3:]), generator=g, device=dev, dtype=eps0.dtype).repeat(1, 1, n, 1, 1) * pipe.scheduler.init_noise_sigma)
    E = torch.cat(E); final = torch.empty_like(eps0); done = 0; dec = {}; picks = []; rows = []
    sn, sc = _syncnet(); mel, rms, voiced = _audio(audio_path, n)

    def conds(a_, b_, K):
        mk, mi, rl = [v[:, :, a_:b_].expand(K, *v.shape[1:2], b_ - a_, *v.shape[3:]).contiguous() for v in (mask_lat, masked_lat, ref_lat)]
        au = audio_all[a_:b_].repeat(K, 1, 1)   # (K * frames, 50, 384): the UNet folds "(b f)" batch-major, as the CFG pair in ls_stable
        if do_cfg: mk, mi, rl, au = torch.cat([mk] * 2), torch.cat([mi] * 2), torch.cat([rl] * 2), torch.cat([torch.zeros_like(au), au])
        return mk, mi, rl, au

    with torch.no_grad():
        for a_, b_ in windows:
            o = max(0, done - a_); Ew = E[:, :, a_:b_]; x = Ew.clone(); known = final[:, :, a_:a_ + o] if o > 0 else None
            K = N; mk, mi, rl, au = conds(a_, b_, K); pick = None; z = raw = px = None
            for si, t in enumerate(timesteps):
                if o > 0: x[:, :, :o] = pipe.scheduler.add_noise(known.expand(K, *known.shape[1:]), Ew[:, :, :o], t.reshape(1))
                x2 = torch.cat([x] * 2) if do_cfg else x
                npred = acc.unet(torch.cat([pipe.scheduler.scale_model_input(x2, t), mk, mi, rl], dim=1), t, encoder_hidden_states=au).sample
                if do_cfg:
                    nu, na = npred.chunk(2); npred = _ls_cfg(nu, na, acc.guidance)
                st_ = pipe.scheduler.step(npred, t, x, **extra); x = st_.prev_sample
                if early and si == 0 and K > 1 and len(timesteps) > 1 and getattr(st_, "pred_original_sample", None) is not None:   # pick on the first step's x0 preview
                    x0 = st_.pred_original_sample.clone()
                    if o > 0: x0[:, :, :o] = known.expand(K, *known.shape[1:])
                    ts_ = time.perf_counter(); pick, z, raw, _ = _score(pipe, x0, a_, o, b_, dec, voiced, mel, video_fps, sn, sc); t_score += time.perf_counter() - ts_
                    x = x[pick:pick + 1].clone(); Ew = Ew[pick:pick + 1]; K = 1; mk, mi, rl, au = conds(a_, b_, K)
            if o > 0: x[:, :, :o] = known.expand(K, *known.shape[1:])
            ts_ = time.perf_counter()
            if pick is None:   # full mode (or one step): score the finished candidates
                pick, z, raw, px = _score(pipe, x, a_, o, b_, dec, voiced, mel, video_fps, sn, sc); xc = x[pick:pick + 1]; pxc = px[pick]
            else:              # early mode: decode the winner's finished new frames for the next windows' sync / continuity
                xc = x[:1]; pxc = _decode_low(pipe, xc[:, :, o:])[0]
            t_score += time.perf_counter() - ts_
            final[:, :, a_:b_] = xc; done = b_
            for i, f_ in enumerate(range(a_ + o, b_)): dec[f_] = pxc[i]
            for f_ in [f_ for f_ in dec if f_ < a_ - 16]: del dec[f_]
            picks.append(pick); rows.append({"w": [a_, b_], "pick": pick, "z": [round(float(v), 2) for v in z], "raw": [[round(float(v), 4) for v in r] for r in raw]})
    st = {"N": N, "early": early, "windows": len(picks), "not_seed0": int(sum(p != 0 for p in picks)), "share_not_seed0": round(sum(p != 0 for p in picks) / max(1, len(picks)), 3),
          "pick_hist": [int(sum(p == k for p in picks)) for k in range(N)], "seconds": round(time.perf_counter() - t_start, 2), "score_s": round(t_score, 2)}
    lg = os.environ.get("BON_LOG")
    if lg:
        with open(lg, "a") as f_: f_.write(json.dumps({**st, "audio": str(audio_path), "rows": rows}) + "\n")
    print(f"  [bon] N={N}{' early' if early else ''} windows {st['windows']}, pick != seed0 {st['not_seed0']} ({100 * st['share_not_seed0']:.0f} %), hist {st['pick_hist']}, {st['seconds']} s (scoring {st['score_s']} s)", flush=True)
    return final, st
