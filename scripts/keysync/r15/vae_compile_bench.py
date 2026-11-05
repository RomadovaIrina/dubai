# r15 K: is a torch.compile of the decoder (timesteps fixed to 1 = the per-frame decode) faster than eager fp16? batch 4, the dumped real latents
import os, sys, time, torch; sys.path.insert(0, os.getcwd()); sys.path.insert(1, "/workspace/dub/keysync")
from einops import rearrange
from ks_trt import psnr, timeit
import scripts.sampling.dubbing_pipeline_raw as d
m, _, _ = d.load_model("scripts/sampling/configs/interpolation_s8.yaml", "cuda", 14, "latents", "pretrained_models/checkpoints/interpolation_dub.pt")
fsm = m.first_stage_model; fsm.eval()
g = torch.load("/workspace/dub/keysync/dump/tr_mk5.pt", map_location="cpu", weights_only=False)["chunks"][0]["c"]["gt"]
z = (rearrange(g, "b c t h w -> (b t) c h w")[:8].float().cuda() / m.scale_factor)
dec = lambda x: fsm.decode(x, timesteps=1)
with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
    for n in (4, 8):
        ref = dec(z[:n]); te = timeit(lambda: dec(z[:n]))
        print(f"eager n={n}: {te / n:.1f} ms/frame", flush=True)
    cd = torch.compile(dec, mode=os.environ.get("MODE", "max-autotune-no-cudagraphs"), dynamic=False); t0 = time.perf_counter()
    for n in (4, 8):
        out = cd(z[:n]); print(f"compiled n={n}: first call {time.perf_counter() - t0:.0f}s", flush=True)
        tc = timeit(lambda: cd(z[:n])); print(f"compiled n={n}: {tc / n:.1f} ms/frame psnr {psnr(out, dec(z[:n]))}", flush=True)
