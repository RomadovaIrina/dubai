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
    ap.add_argument("--compile", default="none", choices=["none", "unet", "all"])
    ap.add_argument("--compile-mode", default="max-autotune-no-cudagraphs", help="torch.compile mode for the UNets (max-autotune = + CUDA graphs)")
    ap.add_argument("--cudnn-bench", action="store_true"); ap.add_argument("--vae-cl", action="store_true", help="VAE decoder in channels_last")
    ap.add_argument("--fp8", action="store_true", help="torchao float8 dynamic activation + weight on the UNets' linear layers")
    ap.add_argument("--vae-half", action="store_true", help="VAE decode under fp16 autocast (the authors disable it: fp32)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    t0 = time.perf_counter()
    model, _, n_batch = d.load_model(f"scripts/sampling/configs/interpolation{a.cfg}.yaml", "cuda", 14, "latents", "pretrained_models/checkpoints/interpolation_dub.pt")
    model_kf, _, n_batch_kf = d.load_model(f"scripts/sampling/configs/keyframe{a.cfg}.yaml", "cuda", 14, "latents", "pretrained_models/checkpoints/keyframe_dub.pt")
    hubert = d.HubertModel.from_pretrained("facebook/hubert-base-ls960").cuda()
    wavlm = d.WavLM_wrapper(model_size="Base+", feed_as_frames=False, merge_type="None", model_path="pretrained_models/checkpoints/WavLM-Base+.pt").cuda()
    vae = d.VaeWrapper("video"); lmk = d.LandmarksExtractor()
    if a.fp8:
        # PerRow needs bf16 weights; the UNets are fp16 -> PerTensor; only the Linear layers with in / out % 16 == 0 (float8 GEMM shapes)
        from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight, PerTensor
        f16 = lambda mod, fqn: isinstance(mod, torch.nn.Linear) and mod.in_features % 16 == 0 and mod.out_features % 16 == 0 and mod.weight.dtype in (torch.float16, torch.bfloat16) and (mod.bias is None or mod.bias.dtype == mod.weight.dtype)   # float8 GEMM: 16-aligned, half bias
        for m in (model, model_kf): print("FP8_LAYERS", sum(1 for n_, mod in m.model.diffusion_model.named_modules() if f16(mod, n_)), "of", sum(1 for mod in m.model.diffusion_model.modules() if isinstance(mod, torch.nn.Linear)), {str(mod.weight.dtype) for mod in m.model.diffusion_model.modules() if isinstance(mod, torch.nn.Linear)}, flush=True)
        for m in (model, model_kf): quantize_(m.model.diffusion_model, float8_dynamic_activation_float8_weight(granularity=PerTensor()), filter_fn=f16)
    if a.compile == "all":   # + the VAE decoders (temporal decoder, one frame per call)
        for m in (model, model_kf): m.first_stage_model.decoder = torch.compile(m.first_stage_model.decoder, mode="max-autotune-no-cudagraphs", dynamic=False)
    if a.compile in ("unet", "all"):
        import torch._dynamo as _dyn; _dyn.config.cache_size_limit = 64; _dyn.config.accumulated_cache_size_limit = 1024
        for m in (model, model_kf):
            m.model.diffusion_model = torch.compile(m.model.diffusion_model, mode=a.compile_mode, dynamic=False)
    if a.cudnn_bench: torch.backends.cudnn.benchmark = True
    if a.vae_cl:
        for m in (model, model_kf):   # the temporal decoder has 5-D (Conv3d) weights: channels_last on the 2-D convs only
            for mod in m.first_stage_model.decoder.modules():
                if isinstance(mod, torch.nn.Conv2d) and mod.weight.dim() == 4:   # own weight only: AE3DConv (a Conv2d) holds a Conv3d child
                    mod.weight.data = mod.weight.data.contiguous(memory_format=torch.channels_last)
    if a.vae_half:
        for m in (model, model_kf): m.disable_first_stage_autocast = False
    torch.cuda.synchronize(); load_s = time.perf_counter() - t0
    print(json.dumps({"load_s": round(load_s, 1), "cfg": a.cfg, "decoding_t": a.decoding_t, "compile": a.compile, "vae_half": a.vae_half, "cudnn_bench": a.cudnn_bench, "fp8": a.fp8, "vae_cl": a.vae_cl, "KS_DC": os.environ.get("KS_DC", "")}), flush=True)
    for n in ("compute_video_embedding", "get_audio_embeddings", "extract_video_landmarks", "sample_keyframes", "sample_interpolation", "save_audio_video"):
        setattr(d, n, timed(n, getattr(d, n)))
    for m, tag in ((model, "interp"), (model_kf, "kf")):
        m.decode_first_stage = timed(f"vae_decode_{tag}", m.decode_first_stage)
    if os.environ.get("KS_PROFILE"):   # profile UNet calls 40..60 of the interpolation model, print the top CUDA ops
        from torch.profiler import profile, ProfilerActivity
        dm = model.model.diffusion_model; fwd = dm.forward; st = {"n": 0, "p": None}
        def pf(*x, **k):
            st["n"] += 1
            if st["n"] == 40: st["p"] = profile(activities=[ProfilerActivity.CUDA]); st["p"].__enter__()
            r = fwd(*x, **k)
            if st["n"] == 60:
                torch.cuda.synchronize(); st["p"].__exit__(None, None, None)
                print(st["p"].key_averages().table(sort_by="cuda_time_total", row_limit=30), flush=True)
            return r
        dm.forward = pf
    ig = os.environ.get("KS_IGUIDE", "")   # interpolation guidance cost: cond | trunc:K | 2way  (keyframes keep the full 3-way CFG)
    if ig:
        g = model.sampler.guider; prep0, call0 = g.prepare_inputs, g.__call__; gs = {"n": 0, "s": None, "mode": "full"}
        def mode_now():
            if ig == "cond": return "cond"
            if ig == "2way": return "2way"
            if ig.startswith("trunc:"): return "full" if gs["n"] < int(ig.split(":")[1]) else "cond"
            if ig.startswith("mix:"): return "full" if gs["n"] < int(ig.split(":")[1]) else "2way"   # 3-way first K, then ref + audio rows
            if ig == "ucc": return "full" if gs["n"] % 2 == 0 else "ucc"   # the uncond row every other step, its (e_ref - e_uc) reused
            raise ValueError(ig)
        def prep(x, sig, c, uc):
            sv = float(sig.flatten()[0])
            gs["n"] = 0 if gs["s"] is None or sv > gs["s"] else gs["n"] + 1; gs["s"] = sv; prev = gs["mode"]; gs["mode"] = mode_now()
            if gs["mode"] != prev:   # the rows per UNet chunk change -> DeepCache's per-chunk features no longer match: start a new cache
                dmo = getattr(model.model.diffusion_model, "_orig_mod", model.model.diffusion_model); dmo._ks_dc = None
            xi, si, co = prep0(x, sig, c, uc)
            if gs["mode"] == "full": return xi, si, co
            def keep(v, n):   # rows: [uncond | ref | audio+ref], each third n long; 2way / ucc keep ref + audio
                return v[2 * n:3 * n] if gs["mode"] == "cond" else v[n:3 * n]
            co = {k: (keep(v, v.shape[0] // 3) if isinstance(v, torch.Tensor) and k in c and isinstance(c[k], torch.Tensor) and v.shape[0] == 3 * c[k].shape[0] else v) for k, v in co.items()}
            return keep(xi, x.shape[0]), keep(si, x.shape[0]), co
        def call(x, sig):
            if gs["mode"] == "full":
                if ig == "ucc": e_uc, e_ref, _ = x.chunk(3); gs["d"] = e_ref - e_uc
                return call0(x, sig)
            if gs["mode"] == "cond": return x
            if gs["mode"] == "ucc":
                e_ref, c_ar = x.chunk(2); return call0(torch.cat([e_ref - gs["d"], e_ref, c_ar]), sig)
            e_ref, c_ar = x.chunk(2)   # 2-way: drop the uncond row = the ref term at ratio 1: e = e_ref + audio_ratio * (c - e_ref)
            return e_ref + g.audio_ratio * (c_ar - e_ref)
        g.prepare_inputs = prep; g.__call__ = call; model.sampler.guider = g
        type(g).__call__ = lambda self, x, sig: call(x, sig) if self is g else call0.__func__(self, x, sig)
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
