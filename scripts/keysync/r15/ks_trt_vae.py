#!/usr/bin/env python3
"""r15 K: a TensorRT FP16 engine for the KeySync VAE decoder of the INTERPOLATION model (the temporal VideoDecoder) at timesteps=1,
i.e. exactly the authors' per-frame decode (decoding_t 1: no temporal mixing), N frames per engine call (the tail call padded).
The interpolation decode is ~90 s per video-minute in prod, the largest stage left once the interpolation UNet is distilled.
    python ks_trt_vae.py --latents dump/<clip>.pt --out DIR [--n 4]   (verify latents = the real video latents of a ks_dump.py clip)
    ks_fast.py --trt-vae DIR                                   (replaces the interpolation model's decode_first_stage)"""
import argparse, os, sys, time
import torch, torch.nn as nn
from einops import rearrange

sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
from ks_trt import Engine, build, psnr, timeit


class Dec(nn.Module):
    def __init__(self, fsm):
        super().__init__(); self.fsm = fsm

    def forward(self, z):
        return self.fsm.decode(z, timesteps=1)


def use_vae_engine(m, d):
    E = Engine(os.path.join(d, "dec.plan")); N = E.batch; sf = m.scale_factor

    def dec(z):
        v = z.dim() == 5
        if v: T_ = z.shape[2]; z = rearrange(z, "b c t h w -> (b t) c h w")
        z = z / sf; outs = []
        for i in range(0, z.shape[0], N):
            c = z[i:i + N]; B = c.shape[0]
            if B < N: c = torch.cat([c, c[-1:].expand(N - B, *c.shape[1:])])
            outs.append(E(z=c)["img"][:B].float())
        out = torch.cat(outs, 0)
        return rearrange(out, "(b t) c h w -> b c t h w", t=T_) if v else out

    m.decode_first_stage = dec; m._ks_trt_vae = True
    print(f"TRT VAE decoder {d}: {N} frames per call", flush=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--latents", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4); ap.add_argument("--cfg", default="_s8")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); sys.path.insert(0, os.getcwd())
    import scripts.sampling.dubbing_pipeline_raw as d
    model, _, _ = d.load_model(f"scripts/sampling/configs/interpolation{a.cfg}.yaml", "cuda", 14, "latents", "pretrained_models/checkpoints/interpolation_dub.pt")
    fsm = model.first_stage_model; fsm.eval()   # (sgm: .eval() does not return the module) decoder only (KeySync encodes with its separate VaeWrapper)
    g = torch.load(a.latents, map_location="cpu", weights_only=False)["chunks"][0]["c"]["gt"]   # (seg, 4, 14, 64, 64), the pipeline's (scaled) latents
    z = (rearrange(g, "b c t h w -> (b t) c h w")[: 3 * a.n].float().cuda() / model.scale_factor)
    x = None
    with torch.no_grad():
        Z = [z[i * a.n:(i + 1) * a.n] for i in range(3)]
        D = Dec(fsm)
        with torch.autocast("cuda", dtype=torch.float16):   # prod --vae-half: decode under fp16 autocast
            ref = [D(zz).float() for zz in Z]; t_torch = timeit(lambda: D(Z[0]))
        raw = os.path.join(a.out, "raw"); os.makedirs(raw, exist_ok=True); t0 = time.perf_counter()
        torch.onnx.export(D, (Z[0],), os.path.join(raw, "dec.onnx"), input_names=["z"], output_names=["img"], opset_version=17, do_constant_folding=True, dynamo=False)
        print(f"ONNX dec {time.perf_counter() - t0:.0f}s", flush=True)
    import onnx, shutil
    mo = onnx.load(os.path.join(raw, "dec.onnx"), load_external_data=True)
    onnx.save_model(mo, os.path.join(a.out, "dec.onnx"), save_as_external_data=True, all_tensors_to_one_file=True, location="dec.onnx.data", size_threshold=1024)
    shutil.rmtree(raw, ignore_errors=True); model.cpu(); torch.cuda.empty_cache()
    build(os.path.join(a.out, "dec.onnx"), os.path.join(a.out, "dec.plan"), os.path.join(a.out, "timing.cache"))
    E = Engine(os.path.join(a.out, "dec.plan"))
    res = {"n": a.n, "psnr_vs_torch_fp16": [psnr(E(z=zz)["img"], r) for zz, r in zip(Z, ref)],
           "trt_ms_per_call": timeit(lambda: E(z=Z[0])), "torch_ms_per_call": t_torch}
    res["trt_ms_per_frame"] = round(res["trt_ms_per_call"] / a.n, 1); res["torch_ms_per_frame"] = round(t_torch / a.n, 1)
    import json; json.dump(res, open(os.path.join(a.out, "verify.json"), "w")); print("TRT_VAE_VERIFY " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()
