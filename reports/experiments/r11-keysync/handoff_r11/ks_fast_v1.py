#!/usr/bin/env python3
"""KeySync speed (r11): ONE resident process for a queue of videos = the models (keyframe + interpolation UNets, SVD VAE, HuBERT, WavLM,
face_alignment) loaded once; each video runs the authors' sample() unchanged, with the same arguments as scripts/infer_raw.sh.
Per-stage GPU timing (cuda-synchronised wall) by wrapping the pipeline's own functions; one JSON line per video.
    python ks_fast.py --jobs jobs.txt --out OUTDIR [--cfg _s6] [--decoding-t 1] [--compile unet|none]
jobs.txt: "<tag> <video 25 fps 512^2> <audio 16 kHz>" per line. Run from the keysync code dir."""
import argparse, json, os, sys, time
sys.path.insert(0, os.getcwd())
import torch
import scripts.sampling.dubbing_pipeline_raw as d

T = {}


def timed(name, f):
    def g(*a, **k):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        try:
            return f(*a, **k)
        finally:
            torch.cuda.synchronize(); T[name] = T.get(name, 0.0) + time.perf_counter() - t0
    return g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True); ap.add_argument("--out", required=True); ap.add_argument("--cfg", default="")
    ap.add_argument("--decoding-t", type=int, default=1, help="frames per VAE decode call (the authors' script: 1)")
    ap.add_argument("--compile", default="none", choices=["none", "unet"])
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    t0 = time.perf_counter()
    model, _, n_batch = d.load_model(f"scripts/sampling/configs/interpolation{a.cfg}.yaml", "cuda", 14, "latents", "pretrained_models/checkpoints/interpolation_dub.pt")
    model_kf, _, n_batch_kf = d.load_model(f"scripts/sampling/configs/keyframe{a.cfg}.yaml", "cuda", 14, "latents", "pretrained_models/checkpoints/keyframe_dub.pt")
    hubert = d.HubertModel.from_pretrained("facebook/hubert-base-ls960").cuda()
    wavlm = d.WavLM_wrapper(model_size="Base+", feed_as_frames=False, merge_type="None", model_path="pretrained_models/checkpoints/WavLM-Base+.pt").cuda()
    vae = d.VaeWrapper("video"); lmk = d.LandmarksExtractor()
    if a.compile == "unet":
        for m in (model, model_kf):
            m.model.diffusion_model = torch.compile(m.model.diffusion_model, mode="max-autotune-no-cudagraphs", dynamic=False)
    torch.cuda.synchronize(); load_s = time.perf_counter() - t0
    print(json.dumps({"load_s": round(load_s, 1), "cfg": a.cfg, "decoding_t": a.decoding_t, "compile": a.compile}), flush=True)
    for n in ("compute_video_embedding", "get_audio_embeddings", "extract_video_landmarks", "sample_keyframes", "sample_interpolation", "save_audio_video"):
        setattr(d, n, timed(n, getattr(d, n)))
    for m, tag in ((model, "interp"), (model_kf, "kf")):
        m.decode_first_stage = timed(f"vae_decode_{tag}", m.decode_first_stage)
    for line in open(a.jobs):
        if not line.strip() or line.startswith("#"): continue
        tag, vid, aud = line.split()
        T.clear(); torch.cuda.synchronize(); t1 = time.perf_counter()
        d.sample(model, model_kf, video_path=vid, audio_path=aud, num_frames=14, resize_size=512, video_folder="videos", latent_folder="videos",
                 landmark_folder="landmarks", audio_folder="audios", audio_emb_folder="audios", fps_id=24, cond_aug=0.0, decoding_t=a.decoding_t,
                 output_folder=os.path.join(a.out, tag), force_uc_zero_embeddings=["cond_frames", "audio_emb"], chunk_size=2, add_zero_flag=True,
                 n_batch=n_batch, n_batch_keyframes=n_batch_kf, extra_audio=None, compute_until="end", audio_emb_type="hubert", extra_naming=tag,
                 what_mask="box", fix_occlusion=False, position=None, start_frame=0, recompute=True,
                 hubert_model=hubert, wavlm_model=wavlm, vae_model=vae, landmarks_model=lmk)
        torch.cuda.synchronize(); wall = time.perf_counter() - t1
        dur = float(os.popen(f"ffprobe -v error -show_entries format=duration -of csv=p=0 {vid}").read())
        r = {"tag": tag, "video_s": round(dur, 2), "wall_s": round(wall, 1), "s_per_vmin": round(wall / dur * 60, 1), "stages_s": {k: round(v, 1) for k, v in T.items()},
             "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        print("KS_JOB " + json.dumps(r), flush=True)
    print("KS_FAST_DONE", flush=True)


if __name__ == "__main__":
    main()
