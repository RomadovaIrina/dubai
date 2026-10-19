#!/usr/bin/env python3
"""Alternative face restorers on the LatentSync raw frames of a decomposition window (track B, candidate C3): writes <dir>/S1_<kind>.mp4,
the counterpart of S1_ls_cf (CodeFormer) for temporal_mouth_stabilize.py --detail S1_<kind> and for the plain per-frame composite.

Same alignment / batching / paste-back as the optimized CodeFormer stage (codeformer_accel: FFHQ 512 template, 5 points from LatentSync's
landmarks of the master frames, smoothed like cand5), only the network is swapped and ParseNet is OFF (its weights are CC BY-NC-SA, see
LICENSES.md): the paste-back uses facelib's square feathered mask; the mouth-only composite downstream limits the change to the ellipse.

  gfpgan14        GFPGAN v1.4 clean arch (Apache-2.0, release v1.3.0), weight 1.0, per-frame RANDOM StyleGAN noise (upstream default)
  gfpgan14_fixed  the same with randomize_noise=False (fixed noise buffers: the same noise pattern in every frame)
  restoreformer   RestoreFormer (Apache-2.0, GFPGAN release v1.3.4)

    python scripts/quality/restore_stage.py --dir DECOMP_DIR --kind gfpgan14_fixed
"""
from __future__ import annotations
import argparse, json, pathlib, sys, time, types
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "optim"))
GF = ROOT / "third_party" / "GFPGAN"; W = ROOT / "models" / "gfpgan"


def gfpgan_archs():
    """Import gfpgan.archs.* from the pinned clone without running gfpgan/__init__.py (it imports training code / extra deps)."""
    if "gfpgan" not in sys.modules:
        pkg = types.ModuleType("gfpgan"); pkg.__path__ = [str(GF / "gfpgan")]; sys.modules["gfpgan"] = pkg
        sub = types.ModuleType("gfpgan.archs"); sub.__path__ = [str(GF / "gfpgan" / "archs")]; sys.modules["gfpgan.archs"] = sub


class Adapter:
    """Callable with CodeFormer's call signature net(x, w=..., adain=...) -> (out,) around another 512 face restorer."""
    def __init__(self, kind: str, device: str = "cuda"):
        import torch
        gfpgan_archs(); self.kind = kind
        if kind.startswith("gfpgan14"):
            from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
            net = GFPGANv1Clean(out_size=512, num_style_feat=512, channel_multiplier=2, decoder_load_path=None, fix_decoder=False, num_mlp=8,
                                input_is_latent=True, different_w=True, narrow=1, sft_half=True)
            ck = torch.load(str(W / "GFPGANv1.4.pth"), map_location="cpu"); ck = ck.get("params_ema", ck.get("params", ck))
            self.randomize = kind == "gfpgan14"
        elif kind == "restoreformer":
            from gfpgan.archs.restoreformer_arch import RestoreFormer
            net = RestoreFormer(); ck = torch.load(str(W / "RestoreFormer.pth"), map_location="cpu"); ck = ck.get("params_ema", ck.get("params", ck)); self.randomize = None
        else:
            raise ValueError(kind)
        net.load_state_dict(ck, strict=True); self.net = net.eval().to(device)

    def __call__(self, x, w=None, adain=None):
        if self.randomize is None: return (self.net(x)[0],)
        return (self.net(x, return_rgb=False, randomize_noise=self.randomize)[0],)


def main() -> int:
    import cv2, torch
    from codeformer_accel import CodeFormerAccel, lm5_from_insightface106
    from decompose_video_stage import write_video
    from temporal_mouth_stabilize import read_video
    import face_aware_latentsync as fa, face_aware_latentsync_accel as faa
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True); ap.add_argument("--kind", required=True, choices=["gfpgan14", "gfpgan14_fixed", "restoreformer"])
    ap.add_argument("--lm-smooth", type=int, default=2); ap.add_argument("--seed", type=int, default=1247)
    a = ap.parse_args(); d = pathlib.Path(a.dir); wm = json.loads((d / "window_manifest.json").read_text()); A = wm["window"][0]
    torch.manual_seed(a.seed)
    t0 = time.perf_counter()
    cf = CodeFormerAccel(fidelity_weight=1.0, batch_size=8, use_parse=False, landmark_source="insightface", autocast=False, parse_autocast=False)
    del cf.net; torch.cuda.empty_cache(); cf.net = Adapter(a.kind); cf.config.update({"network": a.kind, "use_parse": False})
    master = read_video(d / "master.mp4"); raw = read_video(d / "S0_ls_raw.mp4"); out = [f.copy() for f in raw]; st_all = []
    for pc in wm["pieces"]:
        s, e = pc["start"] - A, pc["end"] - A
        dets = fa.detect_frames(np.stack([cv2.cvtColor(master[i], cv2.COLOR_BGR2RGB) for i in range(s, e)]))
        if a.lm_smooth > 0: dets = faa.smooth_detections(dets, a.lm_smooth)
        lms = [lm5_from_insightface106(dets[k][1]) for k in range(e - s)]
        with torch.no_grad():
            res, st = cf.restore_frames([raw[i].copy() for i in range(s, e)], lms)
        for j, i in enumerate(range(s, e)): out[i] = res[j]
        st_all.append({k: v for k, v in st.items() if k != "batches"}); print(f"  piece {pc['start']}:{pc['end']} {st['faces_restored']} faces", flush=True)
    write_video(d / f"S1_{a.kind}.mp4", out)
    meta = {"kind": a.kind, "config": cf.config, "pieces": st_all, "seconds": round(time.perf_counter() - t0, 1)}
    (d / f"S1_{a.kind}.json").write_text(json.dumps(meta, indent=1, default=str)); print(json.dumps({"kind": a.kind, "seconds": meta["seconds"]})); cf.close(); return 0


if __name__ == "__main__":
    raise SystemExit(main())
