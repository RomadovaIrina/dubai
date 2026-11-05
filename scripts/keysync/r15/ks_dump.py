#!/usr/bin/env python3
"""r15 K: distillation data for the KeySync INTERPOLATION UNet. The authors' sample() runs unchanged, except:
  - the keyframe model is skipped: the keyframes = the REAL video latents at the keyframe positions (how the interpolation model was trained);
  - the interpolation sampler is replaced by a dumper: it saves the conditioner outputs c / uc of every interpolation call and returns zeros;
  - the VAE decode and the video save are skipped.
x0 for the consistency loss = c["gt"] (the real latents of each 14-frame segment), with the clip's ORIGINAL audio.
    python ks_dump.py --jobs jobs.txt --out DIR   (run from the keysync code dir; jobs: "<tag> <video 25 fps 512^2> <audio 16 kHz>")"""
import argparse, os, sys, time, types
sys.path.insert(0, os.getcwd())
os.environ.pop("KS_DC", None)
import torch
from einops import rearrange
import scripts.sampling.dubbing_pipeline_raw as d


def kf_stub(model_keyframes, audio_list, gt_list, masks_list, condition, num_frames, *a, **k):
    z = rearrange(gt_list, "b t c h w -> (b t) c h w").float()
    return z, torch.zeros(z.shape[0], 3, 512, 512, device=z.device)   # samples_x: only its H, W (the latent shape) and cond_frames_without_noise (unused by the interp conditioner)


class Dumper:
    def __init__(self, guider):
        self.guider = guider; self.chunks = []

    def __call__(self, denoiser, x, cond, uc=None, num_steps=None, strength=1.0):
        cpu = lambda dd: {k: (v.detach().to("cpu", torch.float16) if v.is_floating_point() else v.detach().cpu()) if isinstance(v, torch.Tensor) else v for k, v in dd.items()}
        self.chunks.append({"c": cpu(cond), "uc": cpu(uc), "n_frames": int(x.shape[0])})
        if len(self.chunks) == 1:
            print("DUMP shapes", {k: tuple(v.shape) for k, v in cond.items() if isinstance(v, torch.Tensor)}, "x", tuple(x.shape), flush=True)
        return torch.zeros_like(x)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", required=True); ap.add_argument("--out", required=True); ap.add_argument("--cfg", default="_s8")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    model, _, n_batch = d.load_model(f"scripts/sampling/configs/interpolation{a.cfg}.yaml", "cuda", 14, "latents", "pretrained_models/checkpoints/interpolation_dub.pt")
    hubert = d.HubertModel.from_pretrained("facebook/hubert-base-ls960").cuda()
    wavlm = d.WavLM_wrapper(model_size="Base+", feed_as_frames=False, merge_type="None", model_path="pretrained_models/checkpoints/WavLM-Base+.pt").cuda()
    vae = d.VaeWrapper("video"); lmk = d.LandmarksExtractor()
    kf = types.SimpleNamespace(model=types.SimpleNamespace(diffusion_model=types.SimpleNamespace(additional_audio_frames=0)), en_and_decode_n_samples_a_time=1)
    d.sample_keyframes = kf_stub
    d.save_audio_video = lambda *x, **k: None
    model.decode_first_stage = lambda z: torch.zeros(z.shape[0], 3, 8, 8, device=z.device)
    for line in open(a.jobs):
        if not line.strip() or line.startswith("#"): continue
        tag, vid, aud = line.split(); t0 = time.perf_counter()
        dump = Dumper(model.sampler.guider); model.sampler = dump
        d.sample(model, kf, video_path=vid, audio_path=aud, num_frames=14, resize_size=512, video_folder="videos", latent_folder="videos",
                 landmark_folder="landmarks", audio_folder="audios", audio_emb_folder="audios", fps_id=24, cond_aug=0.0, decoding_t=1,
                 output_folder=os.path.join(a.out, "_null"), force_uc_zero_embeddings=["cond_frames", "audio_emb"], chunk_size=2, add_zero_flag=True,
                 n_batch=n_batch, n_batch_keyframes=1, extra_audio=None, compute_until="end", audio_emb_type="hubert", extra_naming=tag,
                 what_mask="box", fix_occlusion=False, position=None, start_frame=0, recompute=True,
                 hubert_model=hubert, wavlm_model=wavlm, vae_model=vae, landmarks_model=lmk)
        torch.save({"tag": tag, "n_batch": n_batch, "chunks": dump.chunks}, os.path.join(a.out, f"{tag}.pt"))
        nseg = sum(ch["n_frames"] // 14 for ch in dump.chunks)
        print(f"DUMPED {tag} chunks {len(dump.chunks)} segments {nseg} in {time.perf_counter() - t0:.1f}s", flush=True)
    print("DUMP_DONE", flush=True)


if __name__ == "__main__":
    main()
