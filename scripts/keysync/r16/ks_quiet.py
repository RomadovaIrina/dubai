"""r20 (Irina 15.11: "кривая концовка", the lips press into a thin line / stick after the last phrase): KeySync's SILENT mouth.
The keyframe model runs 3-way guidance e = e_uc + ref_ratio (e_ref - e_uc) + audio_ratio (c - e_ref) with audio_ratio 5: in speech that makes the
articulation, in silence it pushes "no sound" to the extreme = pressed, widened lips with no lip line, in EVERY pause >= ~2 s (old03 26-30 s,
m03 34-36 s) and in the tail. KS_QUIET lowers the audio guidance only for the keyframes that sit in silence:
  KS_QUIET="ar=1.0,gap=0.3,ramp=0.4[,rr=2.0][,thr=0.05]"
  q(frame) = clip((distance to the nearest dub speech - gap) / ramp, 0, 1), from the 16 kHz dub this KeySync run hears (25 fps RMS > thr x p99);
  per keyframe row: audio_ratio -> 5 + q (ar - 5) [and ref_ratio -> 2 + q (rr - 2)]. ar=1 = the plain "silence" condition, ar=0 = no audio
  condition at all (the mouth of the reference). Speech keyframes (q = 0) are bit-for-bit the default guider. The zero-flag reference slots keep
  the default. The interpolation student has its guidance distilled in: it only follows the keyframes.
ks_fast.py installs it (after the models are loaded). Prints KS_QUIET lines."""
import os, wave
import numpy as np, torch

cfg = dict(x.split("=") for x in os.environ.get("KS_QUIET", "").split(",") if "=" in x)
AR = float(cfg.get("ar", 1.0)); RR = float(cfg["rr"]) if "rr" in cfg else None
GAP, RAMP, THR = float(cfg.get("gap", 0.3)), float(cfg.get("ramp", 0.4)), float(cfg.get("thr", 0.05))
HOLD = cfg.get("hold", "0") == "1"   # r20: every keyframe of one silent run starts from the SAME noise (Irina 15.11 on ar=2: the lips flutter in the pause = each silent keyframe drew its own mouth)
ST = {"q": None, "idx": None, "hn": None, "held": 0, "chunk": -1, "off": 0, "sig": None, "in_kf": False, "rows": 0, "quiet": 0.0, "at": []}


def quiet_curve(path, fps=25.0):
    w = wave.open(path); sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
    if w.getnchannels() > 1: x = x.reshape(-1, w.getnchannels()).mean(1)
    hop = int(round(sr / fps)); n = len(x) // hop
    if n < 3: return None
    e = np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1)); sp = np.convolve(e, np.ones(3) / 3, mode="same") > THR * max(float(np.percentile(e, 99)), 1.0)
    k = np.arange(n); on = np.flatnonzero(sp)
    if not len(on): return np.ones(n, np.float32)
    d = np.abs(k[:, None] - on[None, :]).min(1) if n * len(on) < 4e7 else np.array([np.abs(on - i).min() for i in k])
    return np.clip((d / fps - GAP) / max(RAMP, 1e-3), 0, 1).astype(np.float32)


def install(d, model_kf):
    g = model_kf.sampler.guider; cls = type(g); call0 = cls.__call__
    if getattr(g, "use_normalized", False): print("KS_QUIET off: the keyframe guider is the normalized variant", flush=True); return
    s0, c0, k0 = d.sample, d.create_pipeline_inputs, d.sample_keyframes
    def sample(*a, **k):
        try: ST["q"] = quiet_curve(k.get("audio_path") or a[3])
        except Exception as ex: ST["q"] = None; print(f"KS_QUIET: no quiet curve ({ex}) -> default guidance", flush=True)
        ST["rows"] = 0; ST["quiet"] = 0.0; ST["off"] = 0; ST["at"] = []; ST["hn"] = None; ST["held"] = 0
        r = s0(*a, **k)
        q = ST["q"]; print("KS_QUIET job: audio frames %d quiet share %.2f | keyframe rows %d, mean q %.2f, ar -> %.1f%s" % (0 if q is None else len(q), 0 if q is None else float((q > 0.5).mean()), ST["rows"], ST["quiet"] / max(ST["rows"], 1), AR, "" if RR is None else f", rr -> {RR}") + (" | held noise rows %d" % ST["held"] if HOLD else "") + " | quiet keyframes at s: " + " ".join("%.1f" % t for t in ST["at"]), flush=True)
        return r
    def cpi(*a, **k):
        r = c0(*a, **k); ST["idx"] = list(r[-1]); ST["rem"] = list(r[8]); return r
    def skf(*a, **k):
        ST["in_kf"] = True; ST["chunk"] = -1; ST["sig"] = None
        try: return k0(*a, **k)
        finally: ST["in_kf"] = False; ST["off"] += ST["chunk"] + 1   # sample_keyframes runs once per 2 batches of 14 keyframes: the slot offset runs on over the whole video (r20-mouth's catch)
    def call(self, x, sigma):
        q_, idx = ST["q"], ST["idx"]
        if self is not g or not ST["in_kf"] or q_ is None or idx is None: return call0(self, x, sigma)
        sv = float(sigma.flatten()[0])
        if ST["sig"] is None or sv > ST["sig"] * 1.0001: ST["chunk"] += 1; new = True   # a new sampling run = the next 14-keyframe chunk
        else: new = False
        ST["sig"] = sv
        e_uc, e_ref, c = x.chunk(3); n = e_uc.shape[0]; base = (ST["off"] + ST["chunk"]) * n; q = torch.zeros(n)
        for j in range(n):
            s = base + j
            if s < len(idx) and idx[s] is not None and not ST["rem"][s]: q[j] = float(q_[min(int(idx[s]), len(q_) - 1)])
        if new: ST["rows"] += n; ST["quiet"] += float(q.sum()); ST["at"] += [int(idx[base + j]) / 25.0 for j in range(n) if q[j] > 0.5]
        if float(q.max()) <= 0: return call0(self, x, sigma)
        q = q.to(device=x.device, dtype=x.dtype).view(-1, *([1] * (x.dim() - 1)))
        ar = self.audio_ratio + q * (AR - self.audio_ratio); rr = self.ref_ratio if RR is None else self.ref_ratio + q * (RR - self.ref_ratio)
        return ar * (c - e_ref) + rr * (e_ref - e_uc) + e_uc
    d.sample, d.create_pipeline_inputs, d.sample_keyframes = sample, cpi, skf; cls.__call__ = call
    if HOLD:
        class _KfSampler:   # the keyframe sampler's initial noise: rows of one silent run share the first silent row's noise
            def __init__(self, b): self._b = b
            def __getattr__(self, k): return getattr(self._b, k)
            def __call__(self, denoiser, x, *a, **k):
                q_, idx = ST["q"], ST["idx"]
                if ST["in_kf"] and q_ is not None and idx is not None:
                    n = x.shape[0]; base = (ST["off"] + ST["chunk"] + 1) * n; x = x.clone()
                    for j in range(n):
                        s_ = base + j
                        if s_ >= len(idx) or idx[s_] is None or ST["rem"][s_]: continue
                        if float(q_[min(int(idx[s_]), len(q_) - 1)]) >= 0.99:
                            if ST["hn"] is None: ST["hn"] = x[j].clone()
                            else: x[j] = ST["hn"]; ST["held"] += 1
                        else: ST["hn"] = None
                return self._b(denoiser, x, *a, **k)
        model_kf.sampler = _KfSampler(model_kf.sampler)
    print(f"KS_QUIET on: silent keyframes audio_ratio {g.audio_ratio} -> {AR}" + ("" if RR is None else f", ref_ratio {g.ref_ratio} -> {RR}") + f" (gap {GAP} s, ramp {RAMP} s, thr {THR})" + (", silent runs share their keyframe noise" if HOLD else ""), flush=True)
