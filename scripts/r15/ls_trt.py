#!/usr/bin/env python3
"""r15 K: TensorRT FP16 engines for the LatentSync 1.6 UNet (stage2_512) with the student LoRA merged (the r6 student: 4 steps, no CFG, no DeepCache),
one engine per captured call shape (ls_trt_gen.py LS_TRT_CAPTURE: <shape key>.pt = the first call of that shape at 3 timesteps).
The UNet is loaded as the generator loads it (configs/unet/stage2_512.yaml, checkpoints/latentsync_unet.pt, fp16) and the LoRA merged as student_gen.py.
ONNX = the fp16 graph (TorchScript exporter, no constant folding), repacked into one external-data file; TensorRT FP16 with the normalisation layers in fp32.
Verify: PSNR / relative L2 of the engine output vs PyTorch on the captured inputs + per-call timing.
    python scripts/r15/ls_trt.py --cap /workspace/dub/lstrt/cap --out /workspace/dub/lstrt/eng --lora <ckpt>"""
import argparse, glob, json, os, pathlib, sys, time
import torch, torch.nn as nn
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[1]; LS = ROOT / "third_party" / "latentsync"
sys.path.insert(0, str(LS)); sys.path.insert(1, str(HERE))
from ks_trt import Engine, build, psnr, timeit


class W(nn.Module):
    def __init__(self, unet):
        super().__init__(); self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states):
        return self.unet(sample, timestep, encoder_hidden_states=encoder_hidden_states).sample


def load_unet(lora):
    from omegaconf import OmegaConf
    from latentsync.models.unet import UNet3DConditionModel
    cfg = OmegaConf.load(LS / "configs/unet/stage2_512.yaml"); old = os.getcwd(); os.chdir(LS)
    try:
        unet, _ = UNet3DConditionModel.from_pretrained(OmegaConf.to_container(cfg.model), str(LS / "checkpoints/latentsync_unet.pt"), device="cpu")
    finally:
        os.chdir(old)
    if lora and lora != "none":
        d = torch.load(lora, map_location="cpu"); sd = d["lora"]; mods = dict(unet.named_modules())
        for name in d["names"]:
            lin = mods[name]; A = sd[f"{name}.A"].float(); B = sd[f"{name}.B"].float()
            with torch.no_grad(): lin.weight.data = (lin.weight.data.float() + B @ A).to(lin.weight.dtype)
        print(f"merged {len(d['names'])} LoRA projections, step {d.get('step')}", flush=True)
    return unet.to(dtype=torch.float16).cuda().eval()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--cap", required=True); ap.add_argument("--out", required=True); ap.add_argument("--lora", default="none")
    ap.add_argument("--only", default=""); a = ap.parse_args()
    unet = load_unet(a.lora); m = W(unet); res_all = {}
    for f in sorted(glob.glob(f"{a.cap}/*.pt")):
        key = pathlib.Path(f).stem
        if a.only and key != a.only: continue
        od = pathlib.Path(a.out) / key; od.mkdir(parents=True, exist_ok=True); rec = torch.load(f, weights_only=False)
        X = [(r["sample"].cuda().half(), r["timestep"].reshape(1).cuda(), r["ehs"].cuda().half()) for r in rec]
        print("CAP", key, [tuple(t.shape) for t in X[0]], [t.dtype for t in X[0]], "t", [float(x[1][0]) for x in X], flush=True)
        with torch.no_grad():
            ref = [m(*x).float() for x in X]; t_torch = timeit(lambda: m(*X[0]))
            raw = od / "raw"; raw.mkdir(exist_ok=True); t0 = time.perf_counter()
            torch.onnx.export(m, X[0], str(raw / "unet.onnx"), input_names=["sample", "timestep", "encoder_hidden_states"], output_names=["out"],
                              opset_version=17, do_constant_folding=False, dynamo=False)
            print(f"ONNX {key} {time.perf_counter() - t0:.0f}s", flush=True)
        import onnx, shutil
        mo = onnx.load(str(raw / "unet.onnx"), load_external_data=True)
        onnx.save_model(mo, str(od / "unet.onnx"), save_as_external_data=True, all_tensors_to_one_file=True, location="unet.onnx.data", size_threshold=1024, convert_attribute=True)
        del mo; shutil.rmtree(raw, ignore_errors=True)
        unet.cpu(); torch.cuda.empty_cache()
        build(str(od / "unet.onnx"), str(od / "unet.plan"), str(pathlib.Path(a.out) / "timing.cache"))
        E = Engine(str(od / "unet.plan"))
        res = {"key": key, "lora": a.lora, "psnr": [psnr(E(sample=x[0], timestep=x[1], encoder_hidden_states=x[2])["out"], r) for x, r in zip(X, ref)],
               "trt_ms": timeit(lambda: E(sample=X[0][0], timestep=X[0][1], encoder_hidden_states=X[0][2])), "torch_eager_fp16_ms": t_torch}
        del E; torch.cuda.empty_cache(); unet.cuda()
        json.dump(res, open(od / "verify.json", "w")); print("LS_TRT_VERIFY " + json.dumps(res), flush=True); res_all[key] = res
    json.dump(res_all, open(pathlib.Path(a.out) / "verify_all.json", "w"))


if __name__ == "__main__":
    main()
