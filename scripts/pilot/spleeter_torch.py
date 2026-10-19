#!/usr/bin/env python3
"""Spleeter 2stems inference in PyTorch on the GPU (Agent C r3, 2026-10-19): a numerically faithful port of the Deezer Spleeter v1.4.0
2stems model (MIT code, weights = MIT repo release asset; scripts/setup_spleeter.sh) so the background bed no longer needs the TF 2.9 CPU venv.

Why: on server A the TF/librosa path ran on ~1 core of a saturated ~31-CPU cgroup quota: 160-470 s per video (r2 box: 5 s). Here: STFT,
both U-Nets, ratio masks and iSTFT on the GPU, ~1 s per video, < 1 GB VRAM.

Port = spleeter/separator.py `_separate_librosa` + model/__init__.py (masks) + model/functions/unet.py, inference mode:
  * STFT per channel: N zeros padded both sides, frame 4096 / hop 1024, periodic Hann, no centering (= librosa stft, center=False);
  * |STFT| partitioned in T=512 frame segments (zero padded), first F=1024 bins, NHWC -> NCHW;
  * U-Net per instrument: 6x [conv 5x5 s2 TF-"same" (pad 1/2) -> BN -> LeakyReLU 0.2]; the skips use the RAW conv outputs and the
    decoder starts from the raw conv6 (as in the TF graph); 6x [deconv 5x5 s2 (crop 1) -> ReLU -> BN], concat [skip, up];
    sigmoid(conv 4x4 dilation 2, pad 3) x input magnitude;
  * ratio mask (out_i^2 + eps/2) / (sum out^2 + eps), eps 1e-10, extended with zeros to 2049 bins, applied to the complex STFT;
  * iSTFT: windowed overlap-add / sum of squared windows (librosa istft, center=False), crop [N, N + length).
Weights: TF checkpoint -> npz once (`convert`, in the TF venv), then plain torch. Verified against the TF output: `verify`.

    <spleeter venv>/bin/python -I scripts/pilot/spleeter_torch.py convert --ckpt models/spleeter/pretrained_models/2stems/model --out W.npz
    python scripts/pilot/spleeter_torch.py separate --weights W.npz --wav source_44k.wav --bg bg.wav --vocals voc.wav
"""
from __future__ import annotations
import argparse, pathlib, sys, time
import numpy as np

N_FFT, HOP, T_SEG, F_BINS, EPS = 4096, 1024, 512, 1024, 1e-10
INSTRUMENTS = ("vocals", "accompaniment")       # graph construction order in the checkpoint (vocals = conv2d..conv2d_6)


def convert(ckpt: str, out: str) -> None:
    import tensorflow as tf   # only in the Spleeter TF venv
    r = tf.train.load_checkpoint(ckpt)
    keep = {k: r.get_tensor(k) for k in r.get_variable_to_shape_map() if not any(s in k for s in ("Adam", "_power", "global_step"))}
    np.savez(out, **{k.replace("/", "__"): v for k, v in keep.items()})
    print(f"{len(keep)} tensors -> {out}")


def _build(torch):
    nn, F = torch.nn, torch.nn.functional

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            ch = [2, 16, 32, 64, 128, 256, 512]
            self.enc = nn.ModuleList([nn.Conv2d(ch[i], ch[i + 1], 5, 2) for i in range(6)])
            self.enc_bn = nn.ModuleList([nn.BatchNorm2d(ch[i + 1], eps=1e-3) for i in range(6)])
            dec_in = [512, 512, 256, 128, 64, 32]; dec_out = [256, 128, 64, 32, 16, 1]
            self.dec = nn.ModuleList([nn.ConvTranspose2d(i, o, 5, 2) for i, o in zip(dec_in, dec_out)])
            self.dec_bn = nn.ModuleList([nn.BatchNorm2d(o, eps=1e-3) for o in dec_out])
            self.last = nn.Conv2d(1, 2, 4, dilation=2)

        def forward(self, x):                                   # x: (B, 2, T, F) magnitude
            skips, h = [], x
            for i in range(6):
                c = self.enc[i](F.pad(h, (1, 2, 1, 2)))           # TF "same", stride 2, k 5: pad 1 before / 2 after
                skips.append(c); h = F.leaky_relu(self.enc_bn[i](c), 0.2)
            h = skips[5]                                         # decoder starts from the raw conv6 (TF graph)
            for i in range(6):
                u = self.dec[i](h); n2 = h.shape[-2] * 2, h.shape[-1] * 2
                u = self.dec_bn[i](F.relu(u[..., 1:1 + n2[0], 1:1 + n2[1]]))   # TF "same" transpose: crop 1
                h = torch.cat([skips[4 - i], u], 1) if i < 5 else u
            m = torch.sigmoid(self.last(F.pad(h, (3, 3, 3, 3))))
            return m * x

        def load_tf(self, W: dict, base: int, bn_base: int, tr_base: int):
            t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
            name = lambda p, k: p if k == 0 else f"{p}_{k}"
            with torch.no_grad():
                for i in range(6):
                    c = name("conv2d", base + i); self.enc[i].weight.copy_(t(W[f"{c}__kernel"]).permute(3, 2, 0, 1)); self.enc[i].bias.copy_(t(W[f"{c}__bias"]))
                    d = name("conv2d_transpose", tr_base + i); self.dec[i].weight.copy_(t(W[f"{d}__kernel"]).permute(3, 2, 0, 1)); self.dec[i].bias.copy_(t(W[f"{d}__bias"]))
                c = name("conv2d", base + 6); self.last.weight.copy_(t(W[f"{c}__kernel"]).permute(3, 2, 0, 1)); self.last.bias.copy_(t(W[f"{c}__bias"]))
                for j, bn in enumerate(list(self.enc_bn) + list(self.dec_bn)):
                    b = name("batch_normalization", bn_base + j)
                    bn.weight.copy_(t(W[f"{b}__gamma"])); bn.bias.copy_(t(W[f"{b}__beta"]))
                    bn.running_mean.copy_(t(W[f"{b}__moving_mean"])); bn.running_var.copy_(t(W[f"{b}__moving_variance"]))
            return self
    return UNet


class Spleeter2Stems:
    def __init__(self, weights: str | pathlib.Path, device: str = "cuda"):
        import torch
        self.torch, self.device = torch, device
        W = dict(np.load(weights)); UNet = _build(torch)
        self.nets = {inst: UNet().load_tf(W, 7 * k, 12 * k, 6 * k).to(device).eval() for k, inst in enumerate(INSTRUMENTS)}
        self.win = torch.hann_window(N_FFT, periodic=True, dtype=torch.float64, device=device)

    def separate(self, y: np.ndarray, batch: int = 4) -> dict:
        """y: (n,) or (n, c) float waveform at 44.1 kHz -> {"vocals": (n, 2), "accompaniment": (n, 2)} float32."""
        torch = self.torch
        y = np.asarray(y, dtype=np.float32)
        y = np.stack([y, y], 1) if y.ndim == 1 else (np.concatenate([y, y], 1) if y.shape[1] == 1 else y[:, :2])
        n = y.shape[0]
        with torch.no_grad():
            x = torch.from_numpy(y.T.copy()).to(self.device, torch.float64)                  # (2, n)
            x = torch.nn.functional.pad(x, (N_FFT, N_FFT))
            S = torch.stft(x, N_FFT, HOP, window=self.win, center=False, return_complex=True)   # (2, 2049, frames)
            frames = S.shape[-1]; seg = -(-frames // T_SEG)
            mag = torch.nn.functional.pad(S.abs().float(), (0, seg * T_SEG - frames))[:, :F_BINS]  # (2, 1024, seg*T)
            mag = mag.reshape(2, F_BINS, seg, T_SEG).permute(2, 0, 3, 1).contiguous()          # (seg, 2, T, F)
            outs = {k: torch.cat([net(mag[i:i + batch]) for i in range(0, seg, batch)]) for k, net in self.nets.items()}
            tot = sum(o.double() ** 2 for o in outs.values()) + EPS
            res = {}
            for k, o in outs.items():
                m = (o.double() ** 2 + EPS / len(outs)) / tot                                 # (seg, 2, T, F)
                m = m.permute(1, 3, 0, 2).reshape(2, F_BINS, seg * T_SEG)[..., :frames]
                m = torch.nn.functional.pad(m, (0, 0, 0, N_FFT // 2 + 1 - F_BINS))           # zeros above F
                res[k] = self._istft(S * m, n)
        return res

    def _istft(self, Z, n: int) -> np.ndarray:
        torch = self.torch
        fr = torch.fft.irfft(Z, n=N_FFT, dim=1) * self.win[None, :, None]                   # (2, N_FFT, frames)
        L = N_FFT + HOP * (fr.shape[-1] - 1)
        y = torch.nn.functional.fold(fr, (1, L), (1, N_FFT), stride=(1, HOP))[:, 0, 0]      # overlap-add
        w2 = torch.nn.functional.fold((self.win ** 2)[None, :, None].expand(1, N_FFT, fr.shape[-1]), (1, L), (1, N_FFT), stride=(1, HOP))[0, 0, 0]
        y = torch.where(w2 > np.finfo(np.float32).tiny, y / w2, y)
        return y[:, N_FFT:N_FFT + n].T.float().cpu().numpy()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert"); c.add_argument("--ckpt", required=True); c.add_argument("--out", required=True)
    s = sub.add_parser("separate"); s.add_argument("--weights", required=True); s.add_argument("--wav", required=True)
    s.add_argument("--bg", required=True); s.add_argument("--vocals", required=True); s.add_argument("--device", default="cuda")
    v = sub.add_parser("verify", help="compare with TF stems of the same wav"); v.add_argument("--weights", required=True); v.add_argument("--wav", required=True)
    v.add_argument("--tf-bg", required=True); v.add_argument("--tf-vocals", required=True); v.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.cmd == "convert":
        convert(a.ckpt, a.out); return 0
    import soundfile as sf
    y, sr = sf.read(a.wav, dtype="float32"); assert sr == 44100, sr
    t0 = time.perf_counter(); sp = Spleeter2Stems(a.weights, a.device); t1 = time.perf_counter(); out = sp.separate(y); t2 = time.perf_counter()
    print(f"load {t1 - t0:.2f}s separate {t2 - t1:.2f}s for {len(y) / sr:.1f}s audio")
    if a.cmd == "separate":
        sf.write(a.bg, out["accompaniment"], sr, subtype="FLOAT"); sf.write(a.vocals, out["vocals"], sr, subtype="FLOAT"); return 0
    for k, f in (("accompaniment", a.tf_bg), ("vocals", a.tf_vocals)):
        ref, _ = sf.read(f, dtype="float32"); o = out[k][: len(ref)]
        err = o - ref; snr = 10 * np.log10(np.sum(ref.astype(np.float64) ** 2) / max(np.sum(err.astype(np.float64) ** 2), 1e-30))
        print(f"{k}: SNR vs TF {snr:.1f} dB, max abs err {np.abs(err).max():.2e}, ref rms {np.sqrt(np.mean(ref ** 2)):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
