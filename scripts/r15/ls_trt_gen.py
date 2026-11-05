#!/usr/bin/env python3
"""r15 K: the PRODUCTION face generator (scripts/quality/candidate_video_stage.py) with the LatentSync student (= scripts/r13/distill/student_gen.py:
the LoRA merged, R13_STUDENT_STEPS steps, guidance 1.0, no DeepCache, bridge-steps 1) and, optionally, its UNet replaced by TensorRT FP16 engines.
Nothing in the generator is edited; the UNet loader is wrapped at runtime.
  LS_TRT_CAPTURE=<dir>  record the UNet inputs (sample, timestep, encoder_hidden_states) of the first call of each new input shape + the first 3 timesteps
  LS_TRT_DIR=<dir>      <dir>/<shape key>/unet.plan replace unet.forward for those shapes (other shapes fall back to PyTorch, counted);
                        LS_TRT_FREE=1 also frees the PyTorch UNet weights (VRAM) - only when every call shape has an engine
  LS_TRT_TIMING=1       per-call CUDA-synchronised UNet time -> "[lstrt] unet_s ..." at exit
    R10_GEN_PY=scripts/r15/ls_trt_gen.py R13_STUDENT_LORA=<ckpt> R13_STUDENT_STEPS=4 [LS_TRT_DIR=...] python ... (the r4_video.sh generator args)"""
import atexit, os, pathlib, runpy, sys, time
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "pilot")); sys.path.insert(0, str(ROOT / "scripts" / "quality")); sys.path.insert(0, str(ROOT / "third_party" / "latentsync")); sys.path.insert(1, str(HERE))
import torch
from latentsync.models.unet import UNet3DConditionModel

LORA = os.environ.get("R13_STUDENT_LORA", "none"); orig = UNet3DConditionModel.from_pretrained.__func__
ST = {"calls": 0, "trt": 0, "fallback": 0, "t": 0.0, "shapes": {}}
shape_key = lambda s: "x".join(str(int(v)) for v in s.shape)


def _report():
    print(f"[lstrt] unet calls {ST['calls']} trt {ST['trt']} fallback {ST['fallback']} unet_s {ST['t']:.1f} shapes {ST['shapes']} "
          f"peak_vram_gb {torch.cuda.max_memory_allocated() / 2**30:.2f}", flush=True)


atexit.register(_report)


class _Out:
    def __init__(self, s): self.sample = s


def wrap(unet):
    fwd0 = unet.forward; cap = os.environ.get("LS_TRT_CAPTURE"); D = os.environ.get("LS_TRT_DIR"); timing = os.environ.get("LS_TRT_TIMING") == "1"
    engines = {}; rec = {}
    if D:
        from ks_trt import Engine
        for p in pathlib.Path(D).glob("*/unet.plan"): engines[p.parent.name] = Engine(str(p)); print(f"[lstrt] engine {p.parent.name}", flush=True)

    def fwd(sample, timestep, encoder_hidden_states=None, *a, **k):
        if timing: torch.cuda.synchronize(); t0 = time.perf_counter()
        key = shape_key(sample) + "_a" + shape_key(encoder_hidden_states); ST["calls"] += 1; ST["shapes"][key] = ST["shapes"].get(key, 0) + 1
        if cap:
            r = rec.setdefault(key, [])
            tv = float(torch.as_tensor(timestep).flatten()[0])
            if len(r) < 3 and (not r or r[-1]["tv"] != tv):
                r.append({"tv": tv, "sample": sample.detach().cpu(), "timestep": torch.as_tensor(timestep).detach().cpu(), "ehs": encoder_hidden_states.detach().cpu(), "a": a, "k": {kk: vv for kk, vv in k.items() if not isinstance(vv, torch.Tensor)}})
                if len(r) == 3: pathlib.Path(cap).mkdir(parents=True, exist_ok=True); torch.save(r, f"{cap}/{key}.pt"); print(f"[lstrt] captured {key}", flush=True)
        E = engines.get(key)
        if E is not None and not a and not k.get("down_block_additional_residuals"):
            tt = torch.as_tensor(timestep, device=sample.device).reshape(-1)[:1].float()
            out = E(sample=sample, timestep=tt, encoder_hidden_states=encoder_hidden_states)["out"].to(sample.dtype); ST["trt"] += 1; res = _Out(out)
        else:
            if E is None and engines: ST["fallback"] += 1
            res = fwd0(sample, timestep, encoder_hidden_states=encoder_hidden_states, *a, **k)
        if timing: torch.cuda.synchronize(); ST["t"] += time.perf_counter() - t0
        return res
    unet.forward = fwd
    if D and os.environ.get("LS_TRT_FREE") == "1":
        for p_ in unet.parameters(): p_.data = torch.empty(0, dtype=p_.dtype, device=p_.device)
        torch.cuda.empty_cache(); print("[lstrt] PyTorch UNet weights freed", flush=True)
    return unet


def from_pretrained(cls, *a, **k):
    unet, info = orig(cls, *a, **k)
    if LORA != "none":   # = student_gen.py: W <- W + B @ A (alpha = rank: scale 1)
        d = torch.load(LORA, map_location="cpu"); sd = d["lora"]; mods = dict(unet.named_modules()); n = 0
        for name in d["names"]:
            lin = mods[name]; A = sd[f"{name}.A"].float(); B = sd[f"{name}.B"].float()
            with torch.no_grad(): lin.weight.data = (lin.weight.data.float() + B @ A).to(lin.weight.dtype)
            n += 1
        print(f"[student] merged LoRA ({n} projections, step {d.get('step')}) from {LORA}", flush=True)
    return wrap(unet), info


UNet3DConditionModel.from_pretrained = classmethod(from_pretrained)
target = ROOT / "scripts" / "quality" / "candidate_video_stage.py"
sys.argv[0] = str(target); N = os.environ.get("R13_STUDENT_STEPS", "4")
sys.argv += ["--steps", N, "--steps-silent", N, "--guidance", "1.0", "--no-deepcache", "--bridge-steps", "1"] + os.environ.get("R13_STUDENT_EXTRA", "").split()
runpy.run_path(str(target), run_name="__main__")
