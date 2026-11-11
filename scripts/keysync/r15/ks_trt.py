#!/usr/bin/env python3
"""r15 K: TensorRT FP16 engines for the KeySync UNets (keyframe + interpolation), with the production DeepCache split (KS_DC=2:0):
  full.plan    (x, timesteps, context) -> (out, h)  = the whole VideoUNet + the deep feature cached for the shallow steps (input to output block L-1)
  shallow.plan (x, timesteps, context, h) -> out    = input block 0 + output block L-1 + out, reusing h (the patch's DeepCache path)
Fixed shapes = the production calls (keyframes 42 = 3 CFG rows x 14 frames; interpolation 84 = 3 rows x 2 segments x 14); shorter tail calls are padded.
ONNX via the TorchScript exporter (a plain fp32 graph; the references are PyTorch under fp16 autocast = prod), repacked into one external-data file, then TensorRT with FP16
and the normalisation layers kept in fp32. --no-dc: a full.plan only (the LCM student runs without DeepCache).
    capture: KS_CAPTURE=<dir> python ks_fast.py ...           (the inputs of the first chunk at the first 3 sigmas, per model)
    make:    python ks_trt.py --cap <dir> --which interp|kf --out <dir> [--lora L --no-dc]   (export + build + verify + per-call timing)
    use:     python ks_fast.py --trt <dir> ...                 (<dir>/{interp,kf}/full.plan [+ shallow.plan])"""
import argparse, json, math, os, sys, time
import torch, torch.nn as nn

T2T = {}


def _dt(trt_dt):
    import tensorrt as trt
    return {trt.float32: torch.float32, trt.float16: torch.float16, trt.bfloat16: torch.bfloat16, trt.int32: torch.int32, trt.int64: torch.int64}[trt_dt]


# ---------------------------------------------------------------- capture
class Capture(nn.Module):
    def __init__(self, inner, tag, path):
        super().__init__(); self.inner = inner; self.tag = tag; self.path = path; self.rec = []; self.last = None; self.done = False

    def __getattr__(self, name):   # the pipeline reads VideoUNet attributes (additional_audio_frames, ...)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["inner"], name)

    def forward(self, x, timesteps=None, **kw):
        tc = float(timesteps.flatten()[0])
        if not self.done and tc != self.last:
            self.last = tc
            self.rec.append({"x": x.detach().cpu(), "timesteps": timesteps.detach().cpu(), **{k: (v.detach().cpu() if isinstance(v, torch.Tensor) else v) for k, v in kw.items()}})
            if len(self.rec) == 3:
                self.done = True; os.makedirs(self.path, exist_ok=True); torch.save(self.rec, os.path.join(self.path, f"{self.tag}.pt"))
                print(f"CAPTURED {self.tag}", {k: (tuple(v.shape) if isinstance(v, torch.Tensor) else v) for k, v in self.rec[0].items()}, flush=True)
        return self.inner(x, timesteps=timesteps, **kw)


def add_capture(m, tag, path):
    m.model.diffusion_model = Capture(m.model.diffusion_model, tag, path)


# ---------------------------------------------------------------- runtime
class Engine:
    def __init__(self, path):
        import tensorrt as trt
        self.logger = trt.Logger(trt.Logger.WARNING); rt = trt.Runtime(self.logger); self.path = path
        self.eng = rt.deserialize_cuda_engine(open(path, "rb").read()); self.ctx = self.eng.create_execution_context()
        self.names = [self.eng.get_tensor_name(i) for i in range(self.eng.num_io_tensors)]
        self.inputs = [n for n in self.names if self.eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outputs = [n for n in self.names if n not in self.inputs]
        self.shape = {n: tuple(self.eng.get_tensor_shape(n)) for n in self.names}; self.dtype = {n: _dt(self.eng.get_tensor_dtype(n)) for n in self.names}
        self.batch = self.shape[self.inputs[0]][0]

    def park(self):   # r17 resident worker: free the engine's device memory (weights + context) between jobs; the plan file stays in the page cache
        self.ctx = None; self.eng = None

    def unpark(self):   # r17: re-deserialize the same plan (seconds from the page cache; the I/O names / shapes / dtypes are unchanged)
        if self.eng is not None: return
        import tensorrt as trt
        self.eng = trt.Runtime(self.logger).deserialize_cuda_engine(open(self.path, "rb").read()); self.ctx = self.eng.create_execution_context()

    def __call__(self, **inp):
        keep = []
        for n in self.inputs:
            t = inp[n].to(self.dtype[n]).contiguous(); keep.append(t); self.ctx.set_tensor_address(n, t.data_ptr())
        out = {n: torch.empty(self.shape[n], dtype=self.dtype[n], device="cuda") for n in self.outputs}
        for n, t in out.items(): self.ctx.set_tensor_address(n, t.data_ptr())
        self.ctx.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        return out


class TRTUNet(nn.Module):
    """Drop-in for VideoUNet.forward with the DeepCache (KS_DC=N:0) state machine of the keysync patch; engines at a fixed batch, short calls padded."""
    _ks_trt = True

    def __init__(self, d, dc=True):
        super().__init__(); self.full = Engine(os.path.join(d, "full.plan"))
        sp = os.path.join(d, "shallow.plan"); self.shallow = Engine(sp) if dc and os.path.exists(sp) else None
        _dc = os.environ.get("KS_DC", "") if self.shallow is not None else ""
        self.dcn = int(_dc.split(":")[0]) if _dc else 0; self.st = None
        print(f"TRT {d}: batch {self.full.batch} dcn {self.dcn} inputs {self.full.inputs} outputs {self.full.outputs}", flush=True)

    def park(self):   # r17: engines off the GPU + the DeepCache state (it holds CUDA feature tensors) dropped
        self.st = None
        for e in (self.full, self.shallow):
            if e is not None: e.park()

    def unpark(self):
        for e in (self.full, self.shallow):
            if e is not None: e.unpark()

    def forward(self, x, timesteps=None, context=None, y=None, audio_emb=None, image_only_indicator=None, num_video_frames=14, **kw):
        assert y is None and audio_emb is None, "TRT engines were exported without y / audio_emb"
        B = x.shape[0]; EB = self.full.batch; full = True
        if self.dcn > 1:
            tc = float(timesteps.flatten()[0]); st = self.st
            if st is None or tc > st["t"]: st = {"n": 0, "k": 0, "c": {}}
            elif tc == st["t"]: st["k"] += 1
            else: st["n"] += 1; st["k"] = 0
            st["t"] = tc; self.st = st; ck = st["c"].get(st["k"])
            full = st["n"] % self.dcn == 0 or ck is None or ck["B"] != B
        if B < EB:   # pad by repeating whole 14-frame videos (frames: x / timesteps; context: per frame or per video)
            fi = torch.arange(EB, device=x.device) % B; x, timesteps = x[fi], timesteps[fi]
            ec = self.full.shape["context"][0]
            context = context[fi] if context.shape[0] == B else context[torch.arange(ec, device=x.device) % context.shape[0]]
        assert x.shape[0] == EB, f"call batch {B} > engine batch {EB}"
        if full:
            o = self.full(x=x, timesteps=timesteps, context=context)
            if self.dcn > 1: self.st["c"][self.st["k"]] = {"h": o["h"], "B": B}
        else:
            o = self.shallow(x=x, timesteps=timesteps, context=context, h=ck["h"])
        return o["out"][:B].float()


def use_engines(m, d):
    dm = m.model.diffusion_model; off = getattr(dm, "_ks_dc_off", False)
    t = TRTUNet(d, dc=not off); t.additional_audio_frames = getattr(dm, "additional_audio_frames", 0)
    m.model.diffusion_model = t; del dm; torch.cuda.empty_cache()


# ---------------------------------------------------------------- export / build / verify
class Full(nn.Module):
    def __init__(self, dm, ioi, nvf, dc):
        super().__init__(); self.dm = dm; self.ioi = ioi; self.nvf = nvf; self.dc = dc

    def forward(self, x, timesteps, context):
        self.dm._ks_dc = None
        out = self.dm(x, timesteps=timesteps, context=context, image_only_indicator=self.ioi, num_video_frames=self.nvf)
        return (out, self.dm._ks_dc["c"][0]["h"].half()) if self.dc else out   # the cached deep feature in fp16 (as under the production autocast)


class Shallow(nn.Module):
    def __init__(self, dm, ioi, nvf, ci):
        super().__init__(); self.dm = dm; self.ioi = ioi; self.nvf = nvf; self.ci = ci

    def forward(self, x, timesteps, context, h):
        self.dm._ks_dc = {"n": 0, "k": 0, "t": 1e30, "c": {0: {"h": h.to(x.dtype), "ci": self.ci, "shape": tuple(x.shape)}}}
        return self.dm(x, timesteps=timesteps, context=context, image_only_indicator=self.ioi, num_video_frames=self.nvf)


def build(onnx_path, plan_path, cache_path, opt_level=3):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING); builder = trt.Builder(logger); net = builder.create_network(0); parser = trt.OnnxParser(net, logger)
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors): print("ONNX_PARSE", parser.get_error(i), flush=True)
        raise SystemExit("parse failed")
    cfg = builder.create_builder_config(); cfg.set_flag(trt.BuilderFlag.FP16); cfg.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 12 << 30); cfg.builder_optimization_level = opt_level
    tc = cfg.create_timing_cache(open(cache_path, "rb").read() if os.path.exists(cache_path) else b""); cfg.set_timing_cache(tc, ignore_mismatch=False)
    nn_ = 0
    for i in range(net.num_layers):
        L = net.get_layer(i)
        if L.type == trt.LayerType.NORMALIZATION: L.precision = trt.float32; nn_ += 1
    print(f"BUILD {onnx_path}: layers {net.num_layers}, fp32 norm {nn_}", flush=True); t0 = time.perf_counter()
    ser = builder.build_serialized_network(net, cfg)
    if ser is None: raise SystemExit("build failed")
    open(plan_path, "wb").write(ser); open(cache_path, "wb").write(memoryview(cfg.get_timing_cache().serialize()))
    print(f"BUILT {plan_path} {os.path.getsize(plan_path) / 2**30:.2f} GB in {time.perf_counter() - t0:.0f}s", flush=True)


def psnr(a, b):
    a, b = a.float(), b.float(); rng = float(b.max() - b.min()); mse = float(((a - b) ** 2).mean())
    return round(10 * math.log10(rng * rng / max(mse, 1e-20)), 2), round(float((a - b).norm() / b.norm().clamp(min=1e-12)), 5)


def timeit(f, n=10):
    f(); torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(n): f()
    torch.cuda.synchronize(); return round((time.perf_counter() - t0) / n * 1000, 1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cap", required=True); ap.add_argument("--which", required=True, choices=["interp", "kf"])
    ap.add_argument("--out", required=True); ap.add_argument("--cfg", default="_s8"); ap.add_argument("--lora", default=""); ap.add_argument("--no-dc", action="store_true")
    ap.add_argument("--skip-build", action="store_true"); ap.add_argument("--opt-level", type=int, default=3); ap.add_argument("--batch", type=int, default=0)
    a = ap.parse_args(); od = os.path.join(a.out, a.which); os.makedirs(od, exist_ok=True); dc = not a.no_dc
    sys.path.insert(0, os.getcwd()); sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
    os.environ["KS_DC"] = "2:0" if dc else ""
    import scripts.sampling.dubbing_pipeline_raw as d
    ck = {"interp": "interpolation", "kf": "keyframe"}[a.which]
    model, _, _ = d.load_model(f"scripts/sampling/configs/{ck}{a.cfg}.yaml", "cuda", 14, "latents", f"pretrained_models/checkpoints/{ck}_dub.pt")
    dm = model.model.diffusion_model; dm.eval()
    for mod in dm.modules():   # no torch.utils.checkpoint wrappers in the traced graph (inference only)
        if isinstance(getattr(mod, "use_checkpoint", None), bool): mod.use_checkpoint = False
        if isinstance(getattr(mod, "checkpoint", None), bool): mod.checkpoint = False
    if a.lora:
        import ks_student; ks_student.merge_lora(dm, a.lora)
    cap = torch.load(os.path.join(a.cap, f"{a.which}.pt"), weights_only=False)
    for r in cap:
        for k in ("y", "audio_emb", "reference_context", "landmarks", "time_context"): assert r.get(k) is None, f"captured {k} is not None"
    g = lambda r, k: r[k].cuda() if isinstance(r[k], torch.Tensor) else r[k]
    ioi, nvf = g(cap[0], "image_only_indicator"), cap[0]["num_video_frames"]; nvf = nvf if isinstance(nvf, int) else int(torch.as_tensor(nvf).flatten()[0])
    X = [(g(r, "x").float(), g(r, "timesteps").float(), g(r, "context").float()) for r in cap]
    if a.batch and a.batch != X[0][0].shape[0]:   # tile whole 14-frame videos to the engine batch (run it with KS_ICHUNK = batch / 14 / ...)
        B0 = X[0][0].shape[0]; fi = torch.arange(a.batch, device="cuda") % B0; nv = a.batch // nvf
        tile = lambda c: c[fi] if c.shape[0] == B0 else c[torch.arange(nv, device="cuda") % c.shape[0]]
        X = [(x[fi], t[fi], tile(c)) for x, t, c in X]; ioi = torch.zeros(nv, nvf, device="cuda", dtype=ioi.dtype if ioi is not None else torch.float32)
    print("CAP", a.which, [tuple(t.shape) for t in X[0]], [t.dtype for t in X[0]], "sigmas(c_noise)", [round(float(r[1][0]), 3) for r in X], flush=True)
    full, shallow = Full(dm, ioi, nvf, dc), None
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):   # the PyTorch references (production numerics)
        ref = []
        o0 = full(*X[0]); ref.append(o0)
        if dc:
            ci = dm._ks_dc["c"][0]["ci"]; shallow = Shallow(dm, ioi, nvf, ci)
            ref.append(shallow(*X[1], o0[1])); ref.append(full(*X[2]))
        else:
            ref.append(full(*X[1])); ref.append(full(*X[2]))
        t_full = timeit(lambda: full(*X[0])); t_sh = timeit(lambda: shallow(*X[1], o0[1])) if dc else None
    print(f"TORCH eager autocast ms/call: full {t_full} shallow {t_sh}", flush=True)
    raw = os.path.join(od, "raw"); os.makedirs(raw, exist_ok=True)   # the exporter writes ~1000 loose external tensors -> repacked into one .data file
    paths = {"full": os.path.join(raw, "full.onnx"), "shallow": os.path.join(raw, "shallow.onnx")}
    if not a.skip_build:
        with torch.no_grad():   # a plain fp32 graph (autocast casts do not export reliably); TensorRT picks FP16 per layer, the norms pinned to fp32
            t0 = time.perf_counter()
            torch.onnx.export(full, X[0], paths["full"], input_names=["x", "timesteps", "context"], output_names=["out", "h"] if dc else ["out"], opset_version=17, do_constant_folding=True, dynamo=False)
            print(f"ONNX full {time.perf_counter() - t0:.0f}s", flush=True)
            if dc:
                torch.onnx.export(shallow, (*X[1], o0[1]), paths["shallow"], input_names=["x", "timesteps", "context", "h"], output_names=["out"], opset_version=17, do_constant_folding=True, dynamo=False)
                print("ONNX shallow done", flush=True)
        del full, shallow; model.cpu(); torch.cuda.empty_cache()
        import onnx, shutil
        for k in (["full", "shallow"] if dc else ["full"]):   # TensorRT 10.16 failed to import the loose 'onnx__*' tensors: one external-data file instead
            mo = onnx.load(paths[k], load_external_data=True); dst = os.path.join(od, f"{k}.onnx")
            onnx.save_model(mo, dst, save_as_external_data=True, all_tensors_to_one_file=True, location=f"{k}.onnx.data", size_threshold=1024); paths[k] = dst; del mo
        shutil.rmtree(raw, ignore_errors=True)
        cache = os.path.join(a.out, "timing.cache")
        build(paths["full"], os.path.join(od, "full.plan"), cache, a.opt_level)
        if dc: build(paths["shallow"], os.path.join(od, "shallow.plan"), cache, a.opt_level)
    E = Engine(os.path.join(od, "full.plan")); S = Engine(os.path.join(od, "shallow.plan")) if dc else None
    r0 = E(x=X[0][0], timesteps=X[0][1], context=X[0][2]); res = {"which": a.which, "lora": a.lora, "dc": dc}
    res["full_s0"] = psnr(r0["out"], ref[0][0] if dc else ref[0])
    if dc:
        res["h_s0"] = psnr(r0["h"], ref[0][1]); r1 = S(x=X[1][0], timesteps=X[1][1], context=X[1][2], h=r0["h"]); res["shallow_s1"] = psnr(r1["out"], ref[1])
        r2 = E(x=X[2][0], timesteps=X[2][1], context=X[2][2]); res["full_s2"] = psnr(r2["out"], ref[2][0])
        res["trt_ms_shallow"] = timeit(lambda: S(x=X[1][0], timesteps=X[1][1], context=X[1][2], h=r0["h"]))
    else:
        r2 = E(x=X[2][0], timesteps=X[2][1], context=X[2][2]); res["full_s2"] = psnr(r2["out"], ref[2])
    res["trt_ms_full"] = timeit(lambda: E(x=X[0][0], timesteps=X[0][1], context=X[0][2])); res["torch_eager_ms_full"] = t_full; res["torch_eager_ms_shallow"] = t_sh
    res["note"] = "psnr = [dB over the reference range, relative L2]"
    json.dump(res, open(os.path.join(od, "verify.json"), "w")); print("TRT_VERIFY " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
