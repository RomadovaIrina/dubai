"""r20 (Irina 15.11, old03 boss final: "лицо микромигает и микродергается"): the KEYFRAME BLINK.
The pipeline overwrites the first / last frame of every 14-frame interpolation segment with the KEYFRAME model's latent
(samples_z[:, 0] = embbedings[:, :, 0]; samples_z[:, -1] = ...), then decodes. The keyframe model (8 steps, 3-way guidance 5 / 2) and the 2-step
interpolation student do not render the same brightness: on old03 every 13th frame of the raw output is ~4 levels darker over the whole face for
ONE frame (twitch.py: mean offset by phase +0.6 x12, -3.7 at the keyframe), ks_guard substitutes dozens of them, ks_seam (built for a step, not
a one-frame blink) leaves ~2 levels in the final = a blink every 0.52 s.
KS_KFOUT=student  the boundary frames keep the interpolation model's OWN rendition (it draws all 14 frames; the keyframes still condition it);
                  the shared boundary frame is the pipeline's average of the two segments' renditions (merge_overlapping_segments).
KS_KFOUT=mix:W    W x keyframe latent + (1 - W) x the student's (W=1 = the authors' overwrite).
ks_fast.py installs it last (it wraps the interpolation model's sampler and its final decode). Prints KS_KFOUT lines."""
import os
import torch

MODE = os.environ.get("KS_KFOUT", "")
W = 0.0 if MODE == "student" else float(MODE.split(":")[1]) if MODE.startswith("mix:") else 1.0
ST = {"z": None, "n": 0, "d": 0.0}


class _Sampler:
    def __init__(self, base): self._b = base
    def __getattr__(self, k): return getattr(self._b, k)
    def __call__(self, *a, **k):
        r = self._b(*a, **k); ST["z"] = r.detach().clone(); return r


def install(model, num_frames=14):
    model.sampler = _Sampler(model.sampler); dec = model.decode_first_stage
    def decode(z):
        s = ST["z"]; ST["z"] = None
        if s is not None and s.shape == z.shape and z.shape[0] % num_frames == 0:
            zz = z.reshape(-1, num_frames, *z.shape[1:]); ss = s.reshape(-1, num_frames, *s.shape[1:]).to(zz.dtype)
            ST["n"] += zz.shape[0]; ST["d"] += float((zz[:, [0, -1]] - ss[:, [0, -1]]).abs().mean())   # how far the keyframe latents are from the student's own boundary frames
            zz = zz.clone(); zz[:, 0] = W * zz[:, 0] + (1 - W) * ss[:, 0]; zz[:, -1] = W * zz[:, -1] + (1 - W) * ss[:, -1]; z = zz.reshape(z.shape)
        return dec(z)
    model.decode_first_stage = decode
    import atexit
    atexit.register(lambda: print(f"KS_KFOUT {MODE}: keyframe weight {W}, segments {ST['n']}", flush=True))
    print(f"KS_KFOUT on: boundary frames = {W} x keyframe latent + {1 - W} x the interpolation model's own", flush=True)
