"""r15: CPU check of the q2-matched teacher's DeepCache wiring (distill2 cmd_teacher): a shim pipe around the distill UNet, 10 DDIM steps,
CFG 2.0, interval 5 -> the deep blocks must run on 2 of the 10 steps, and the output must stay close to the no-DeepCache sample."""
import glob, sys, time, pathlib
sys.path.insert(0, "/workspace/dub/dubai/scripts/r13/distill")
import torch
torch.set_num_threads(8)
import distill as D
from omegaconf import OmegaConf
from latentsync.models.unet import UNet3DConditionModel
from diffusers import DDIMScheduler
from DeepCache import DeepCacheSDHelper

cfg = OmegaConf.load(D.LS / "configs/unet/stage2_512.yaml")
unet, _ = UNet3DConditionModel.from_pretrained(OmegaConf.to_container(cfg.model), str(D.LS / "checkpoints/latentsync_unet.pt"), device="cpu"); unet = unet.float().eval()
s = DDIMScheduler.from_pretrained(str(D.LS / "configs")); s.set_timesteps(10, device="cpu")
f = [f for f in sorted(glob.glob("/workspace/dub/r14d/sel/*.pt")) if torch.load(f)["cond"].shape[1] == 8][0]; d = torch.load(f)
cond = d["cond"][None].float(); au = d["audio"][None].float()
z = torch.randn((1, 4, 1) + cond.shape[3:], generator=torch.Generator().manual_seed(1)).repeat(1, 1, cond.shape[2], 1, 1)
calls = [0]; mid = unet.mid_block.forward
def counted(*a, **k): calls[0] += 1; return mid(*a, **k)
unet.mid_block.forward = counted
with torch.no_grad():
    t0 = time.time(); x_ref = D.sample(unet, s, z, cond, au, cfg=2.0); n_ref = calls[0]; t_ref = time.time() - t0
    shim = type("Shim", (), {})(); shim.unet, shim.scheduler = unet, s
    dc = DeepCacheSDHelper(pipe=shim); dc.set_params(cache_interval=5, cache_branch_id=0); dc.enable()
    for rep in range(2):   # twice: the per-batch reset must start a fresh cycle
        dc.cached_output, dc.start_timestep, dc.cur_timestep = {}, None, 0; calls[0] = 0
        t0 = time.time(); x_dc = D.sample(unet, s, z, cond, au, cfg=2.0); t_dc = time.time() - t0
        print(f"rep {rep}: mid-block calls no-DC {n_ref} / DC {calls[0]}; time {t_ref:.0f} s / {t_dc:.0f} s; "
              f"rel diff {((x_dc - x_ref).norm() / x_ref.norm()).item():.4f}", flush=True)
