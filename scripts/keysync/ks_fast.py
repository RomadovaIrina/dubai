#!/usr/bin/env python3
"""KeySync speed (r11): ONE resident process for a queue of videos = the models (keyframe + interpolation UNets, SVD VAE, HuBERT, WavLM,
face_alignment) loaded once; each video runs the authors' sample() unchanged, with the same arguments as scripts/infer_raw.sh.
Per-stage GPU timing (cuda-synchronised wall) by wrapping the pipeline's own functions; one JSON line per video.
    python ks_fast.py --jobs jobs.txt --out OUTDIR [--cfg _s6] [--decoding-t 1] [--compile unet|none]
jobs.txt: "<tag> <video 25 fps 512^2> <audio 16 kHz> [ref frame]" per line. Run from the keysync code dir."""
import argparse, json, os, sys, time
sys.path.insert(0, os.getcwd()); sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))   # r15: ks_student / ks_trt next to this script
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
    ap.add_argument("--follow", action="store_true", help="r16 T: keep reading --jobs as it grows until a line END (one process for the delay probe AND the full run: one model load / TRT load / VAE compile)")
    ap.add_argument("--serve", default="", help="r17 resident worker: serve requests from this spool dir across VIDEOS (models / TRT / compiled VAE loaded once, parked off the GPU between requests; scripts/keysync/r16/ks_serve.sh)")
    ap.add_argument("--jobs", default=""); ap.add_argument("--out", default=""); ap.add_argument("--cfg", default="")
    ap.add_argument("--decoding-t", type=int, default=1, help="frames per VAE decode call (the authors' script: 1)")
    ap.add_argument("--compile", default="none", choices=["none", "unet", "all"])
    ap.add_argument("--compile-mode", default="max-autotune-no-cudagraphs", help="torch.compile mode for the UNets (max-autotune = + CUDA graphs)")
    ap.add_argument("--cudnn-bench", action="store_true"); ap.add_argument("--vae-cl", action="store_true", help="VAE decoder in channels_last")
    ap.add_argument("--fp8", action="store_true", help="torchao float8 dynamic activation + weight on the UNets' linear layers")
    ap.add_argument("--vae-half", action="store_true", help="VAE decode under fp16 autocast (the authors disable it: fp32)")
    ap.add_argument("--interp-lora", default="", help="r15: the LCM-LoRA student of the interpolation UNet (ks_lcm.py), merged into the weights; 1-row LCM sampler")
    ap.add_argument("--interp-steps", type=int, default=4); ap.add_argument("--lcm-noise", default="fresh", choices=["fresh", "ddim"])
    ap.add_argument("--trt-vae", default="", help="r15: TensorRT VAE decoder engine dir (ks_trt_vae.py) for the interpolation model")
    ap.add_argument("--vae-compile", action="store_true", help="r15: torch.compile the VAE decoder (timesteps=1, fixed batch = --vae-batch)")
    ap.add_argument("--vae-batch", type=int, default=1, help="r15: frames per VAE decode call with timesteps=1 (= the authors' per-frame decode, batched)")
    ap.add_argument("--trt", default="", help="r15: TensorRT engine dir (ks_trt.py): <dir>/{kf,interp}/{full,shallow}.plan replace the UNets")
    a = ap.parse_args()
    if not a.serve:
        assert a.jobs and a.out, "--jobs and --out are required without --serve"
        os.makedirs(a.out, exist_ok=True)
    if a.serve and os.environ.get("KS_SERVE_HOLD"):   # r19 speed: the daemon process is started (python + every import) while the audio stage still
        _h = os.environ["KS_SERVE_HOLD"]; print("KS_SERVE_HOLD waiting for " + _h, flush=True); _t = time.perf_counter()   # owns the GPU; it creates no CUDA
        while not os.path.exists(_h) and not os.path.exists(os.path.join(a.serve, "stop")): time.sleep(0.05)              # context and loads nothing until this
        print(f"KS_SERVE_HOLD released after {time.perf_counter() - _t:.1f} s", flush=True)                                # file appears (e2e.sh, after the audio)
        if os.path.exists(os.path.join(a.serve, "stop")): return
    t0 = time.perf_counter()
    model, _, n_batch = d.load_model(f"scripts/sampling/configs/interpolation{a.cfg}.yaml", "cuda", 14, "latents", "pretrained_models/checkpoints/interpolation_dub.pt")
    model_kf, _, n_batch_kf = d.load_model(f"scripts/sampling/configs/keyframe{a.cfg}.yaml", "cuda", 14, "latents", "pretrained_models/checkpoints/keyframe_dub.pt")
    if a.interp_lora:
        import ks_student
        ks_student.merge_lora(model.model.diffusion_model, a.interp_lora)
        model.model.diffusion_model._ks_dc_off = True   # DeepCache stays on the keyframes only
        model.sampler = ks_student.LCMSampler(model.sampler, a.interp_steps, a.lcm_noise)
    if a.trt_vae:   # r15: the interpolation model's VAE decode through a TensorRT engine (ks_trt_vae.py; per-frame math, N frames per call)
        import ks_trt_vae; ks_trt_vae.use_vae_engine(model, a.trt_vae)
    if a.trt:
        import ks_trt
        for m, tag in ((model, "interp"), (model_kf, "kf")):
            if os.path.exists(os.path.join(a.trt, tag, "full.plan")): ks_trt.use_engines(m, os.path.join(a.trt, tag))
    hubert = d.HubertModel.from_pretrained("facebook/hubert-base-ls960").cuda()
    wavlm = d.WavLM_wrapper(model_size="Base+", feed_as_frames=False, merge_type="None", model_path="pretrained_models/checkpoints/WavLM-Base+.pt").cuda()
    vae = d.VaeWrapper("video"); lmk = d.LandmarksExtractor()
    if a.fp8:
        # PerRow needs bf16 weights; the UNets are fp16 -> PerTensor; only the Linear layers with in / out % 16 == 0 (float8 GEMM shapes)
        from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight, PerTensor, PerRow
        # r13: the UNets load in fp32 (sampling runs under fp16 autocast), so nothing passed the half filter -> KS_FP8_BF16=1 casts the UNets to bf16 first, PerRow
        g8 = PerTensor()
        if os.environ.get("KS_FP8_BF16") == "1":
            for m in (model, model_kf): m.model.diffusion_model.to(torch.bfloat16)
            g8 = PerTensor() if os.environ.get("KS_FP8_GRAN") == "tensor" else PerRow()   # sm120 + this torch: no rowwise scaled_mm
        f16 = lambda mod, fqn: isinstance(mod, torch.nn.Linear) and mod.in_features % 16 == 0 and mod.out_features % 16 == 0 and mod.weight.dtype in (torch.float16, torch.bfloat16) and (mod.bias is None or mod.bias.dtype == mod.weight.dtype)   # float8 GEMM: 16-aligned, half bias
        for m in (model, model_kf): print("FP8_LAYERS", sum(1 for n_, mod in m.model.diffusion_model.named_modules() if f16(mod, n_)), "of", sum(1 for mod in m.model.diffusion_model.modules() if isinstance(mod, torch.nn.Linear)), {str(mod.weight.dtype) for mod in m.model.diffusion_model.modules() if isinstance(mod, torch.nn.Linear)}, flush=True)
        if os.environ.get("KS_FP8_BF16") == "1":   # autocast leaves the time embedding / layernorm outputs in fp32 -> PerRow asserts: cast the quantized linears' input to bf16
            _bf = lambda mod, args: (args[0].to(torch.bfloat16),) + tuple(args[1:])
            for m in (model, model_kf):
                for n_, mod in m.model.diffusion_model.named_modules():
                    if f16(mod, n_): mod.register_forward_pre_hook(_bf)   # the same filter as quantize_ (checked before quantize_ changes the weight dtype)
        for m in (model, model_kf): quantize_(m.model.diffusion_model, float8_dynamic_activation_float8_weight(granularity=g8), filter_fn=f16)
    if a.compile == "all":   # + the VAE decoders (temporal decoder, one frame per call)
        for m in (model, model_kf): m.first_stage_model.decoder = torch.compile(m.first_stage_model.decoder, mode="max-autotune-no-cudagraphs", dynamic=False)
    if a.compile in ("unet", "all"):
        import torch._dynamo as _dyn; _dyn.config.cache_size_limit = 64; _dyn.config.accumulated_cache_size_limit = 1024
        for m in (model, model_kf):
            if getattr(m.model.diffusion_model, "_ks_trt", False): continue   # a TensorRT engine replaces this UNet
            m.model.diffusion_model = torch.compile(m.model.diffusion_model, mode=a.compile_mode, dynamic=False)
    if os.environ.get("KS_CAPTURE"):   # r15: record the UNet inputs of the first chunk at the first 3 sigmas of each model (for the TensorRT export / verify)
        import ks_trt
        for m, tag in ((model, "interp"), (model_kf, "kf")): ks_trt.add_capture(m, tag, os.environ["KS_CAPTURE"])
    if a.cudnn_bench: torch.backends.cudnn.benchmark = True
    if a.vae_cl:
        for m in (model, model_kf):   # the temporal decoder has 5-D (Conv3d) weights: channels_last on the 2-D convs only
            for mod in m.first_stage_model.decoder.modules():
                if isinstance(mod, torch.nn.Conv2d) and mod.weight.dim() == 4:   # own weight only: AE3DConv (a Conv2d) holds a Conv3d child
                    mod.weight.data = mod.weight.data.contiguous(memory_format=torch.channels_last)
    if a.vae_half:
        for m in (model, model_kf): m.disable_first_stage_autocast = False
    torch.cuda.synchronize(); load_s = time.perf_counter() - t0
    print(json.dumps({"load_s": round(load_s, 1), "cfg": a.cfg, "decoding_t": a.decoding_t, "compile": a.compile, "vae_half": a.vae_half, "cudnn_bench": a.cudnn_bench, "fp8": a.fp8, "vae_cl": a.vae_cl, "KS_DC": os.environ.get("KS_DC", ""),
                      "interp_lora": a.interp_lora, "interp_steps": a.interp_steps if a.interp_lora else None, "lcm_noise": a.lcm_noise if a.interp_lora else None, "trt": a.trt, "KS_ICHUNK": os.environ.get("KS_ICHUNK", ""), "vae_batch": a.vae_batch, "vae_compile": a.vae_compile, "trt_vae": a.trt_vae}), flush=True)
    _objs = [model, model_kf, hubert, wavlm, vae, lmk]   # (the resident daemon parks these; the look-ahead extractor joins them when it exists)
    _ahead_q = None
    if os.environ.get("KS_LMK_CACHE"):   # r13 speed lever: landmarks off the critical path (prod: computed during TTS / separation); keyed by the frames' content
        import hashlib, queue, threading, numpy as _np; _xl = d.extract_video_landmarks; _cd = os.environ["KS_LMK_CACHE"]; os.makedirs(_cd, exist_ok=True)
        _fl, _fll = {}, threading.Lock()   # r19: key -> Event of a landmark pass in flight (the job and the look-ahead thread never compute one video twice)
        def _lmk(video_frames, landmarks_model):
            v = video_frames; key = hashlib.md5((str(tuple(v.shape)) + str(v[:: max(1, len(v) // 16)].float().mean().item()) + str(v[-1].float().sum().item())).encode()).hexdigest()
            f = os.path.join(_cd, key + ".npy")
            while True:
                with _fll:
                    if os.path.exists(f): return _np.load(f, allow_pickle=True)
                    ev = _fl.get(key)
                    if ev is None: ev = _fl[key] = threading.Event(); break
                ev.wait()   # the other thread is on this very video
            try:
                r = _xl(video_frames, landmarks_model); _np.save(f + ".tmp.npy", _np.asarray(r)); os.replace(f + ".tmp.npy", f); return r
            finally:
                with _fll: _fl.pop(key, None)
                ev.set()
        d.extract_video_landmarks = _lmk
        if os.environ.get("KS_LMK_AHEAD", "0") == "1":   # OFF by default (VM 14.11, long5: the landmark loop is python-bound, so the thread takes from the
            # sampler what it saves: render sum 746 s without / 757 s with it, and it adds the second extractor's memory). Kept as an option.
            # r19 speed: a job line "LMK <video>" = this video will be rendered soon: its landmarks (face_alignment, a python loop that leaves the GPU
            # mostly idle: 6-8 s per 20 s of video, 118 s of long5's renders) are computed NOW by a second extractor in a background thread, while the
            # GPU samples the job in front; the render then finds them in the cache. The same function on the same frames (d.sample's own read + resize).
            _ahead_q = queue.Queue(); _lm2 = []; _tq0, _pr0, _at = d.tqdm, print, []
            d.tqdm = lambda it, *x, **k: it if threading.current_thread() in _at else _tq0(it, *x, **k)   # no progress bar from the look-ahead thread
            d.print = lambda *x, **k: None if threading.current_thread() in _at else _pr0(*x, **k)         # (its lines would land inside the job's log lines)
            def _ahead():
                _at.append(threading.current_thread())
                while True:
                    path = _ahead_q.get()
                    try:
                        if not _lm2: _lm2.append(d.LandmarksExtractor()); _objs.append(_lm2[0])
                        t_ = time.perf_counter(); v = d.read_video(path, output_format="TCHW")[0]; v = torch.nn.functional.interpolate(v, (512, 512), mode="bilinear")
                        _lmk(v, _lm2[0]); sys.stdout.write(f"\nKS_LMK_AHEAD {path} {len(v)} frames {time.perf_counter() - t_:.1f} s\n"); sys.stdout.flush()
                    except Exception as e: sys.stdout.write("\nKS_LMK_AHEAD failed (the job computes its own): " + repr(e)[:300] + "\n"); sys.stdout.flush()
            threading.Thread(target=_ahead, daemon=True).start()
    if os.environ.get("KS_KF_AR") or os.environ.get("KS_KF_RR"):   # r16 K2: keyframe guidance split (audio vs reference)
        _g = model_kf.sampler.guider
        if os.environ.get("KS_KF_AR"): _g.audio_ratio = float(os.environ["KS_KF_AR"])
        if os.environ.get("KS_KF_RR"): _g.ref_ratio = float(os.environ["KS_KF_RR"])
        print("KS_KF_GUIDE audio", _g.audio_ratio, "ref", _g.ref_ratio, flush=True)
    if os.environ.get("KS_GATE") and a.interp_lora:   # r16 K2: more interp steps only on wide-open segments (ks_gate.py)
        import ks_gate; ks_gate.install(d, model)
    if os.environ.get("KS_NOISE"):   # r16c K2: shared noise across windows / keyframes (ks_noise.py)
        import ks_noise; ks_noise.install(d)
    for n in ("compute_video_embedding", "get_audio_embeddings", "extract_video_landmarks", "sample_keyframes", "sample_interpolation", "save_audio_video"):
        setattr(d, n, timed(n, getattr(d, n)))
    if a.vae_batch > 1:   # r15: the authors decode one frame per call (decoding_t 1 -> timesteps 1: no temporal mixing); the same math batched N frames per call
        from einops import rearrange as _re
        def _mkdec(m):
            N = a.vae_batch; f1 = lambda x: m.first_stage_model.decode(x, timesteps=1)
            # --vae-compile (r15): torch.compile of the decoder with timesteps fixed to 1 (the r12 failure was the dynamic timesteps arg);
            # a fixed batch N (the tail padded with its last frame) = one graph, no recompiles. ~1.9x on the decode, 75 dB vs eager.
            fd = torch.compile(f1, mode=a.compile_mode, dynamic=False) if a.vae_compile else f1
            def dec(z):
                v = z.dim() == 5
                if v: T_ = z.shape[2]; z = _re(z, "b c t h w -> (b t) c h w")
                z = z / m.scale_factor; outs = []
                with torch.autocast("cuda", enabled=not m.disable_first_stage_autocast):
                    for i in range(0, z.shape[0], N):
                        c = z[i:i + N]; B = c.shape[0]
                        if a.vae_compile and B < N: c = torch.cat([c, c[-1:].expand(N - B, *c.shape[1:])])
                        outs.append(fd(c)[:B])
                out = torch.cat(outs, 0)
                return _re(out, "(b t) c h w -> b c t h w", t=T_) if v else out
            return dec
        for m in (model, model_kf):
            if not getattr(m, "_ks_trt_vae", False): m.decode_first_stage = _mkdec(m)
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
    def _hint_tail(path, stop):   # r19 KS_LMK_AHEAD: the "LMK <video>" lines of a follow file are read by THIS thread as soon as they are written
        try:                       # (the job loop below reads a line only when the job in front is done: too late to look ahead)
            f = open(path); buf = ""; seen = set()
            while not stop.is_set():
                ln = f.readline()
                if not ln or not ln.endswith("\n"): buf += ln; time.sleep(0.2); continue
                ln, buf = buf + ln, ""
                if ln.strip() == "END": return
                if ln.startswith("LMK "):
                    v = ln.split(None, 1)[1].strip()
                    if v not in seen: seen.add(v); _ahead_q.put(v)
        except Exception as e: sys.stdout.write("\nKS_LMK_AHEAD hint reader failed: " + repr(e)[:200] + "\n"); sys.stdout.flush()

    def job_lines(path, follow, alive=lambda: True):   # r16 T --follow: the caller appends jobs while this process is resident; "END" stops it
        if not follow:
            yield from open(path); return
        f = open(path); buf = ""; stop = None
        if _ahead_q is not None:
            import threading as _th
            stop = _th.Event(); _th.Thread(target=_hint_tail, args=(path, stop), daemon=True).start()
        try:
            while True:
                ln = f.readline()
                if not ln or not ln.endswith("\n"):
                    buf += ln
                    if not alive(): raise RuntimeError("the requesting job is gone (no END)")   # r17 --serve: never wait forever for a dead client
                    time.sleep(0.2); continue
                ln, buf = buf + ln, ""
                if ln.strip() == "END": return
                if ln.startswith("LMK ") and stop is not None: continue   # (the hint reader has it)
                yield ln
        finally:
            if stop is not None: stop.set()

    def run_line(line, out_dir):
        if not line.strip() or line.startswith("#"): return
        if line.startswith("LMK "):   # r19: a look-ahead hint (see KS_LMK_AHEAD above); ignored when the cache / the look-ahead is off
            if _ahead_q is not None: _ahead_q.put(line.split(None, 1)[1].strip())
            return
        tag, vid, aud, *ex = line.split()   # r16 T: an optional 4th column = this job's KeySync reference frame (KS_REF_IDX); default 0
        os.environ["KS_REF_IDX"] = ex[0] if ex else "0"   # (audit #13: no leak of the previous job's reference)
        if os.environ.get("KS_FAULT") == "crash" and not ("p_" in tag.split("_", 1)[-1] or tag.startswith("p_")):   # r16 TZ 11.1 fault injection
            raise RuntimeError(f"KS_FAULT=crash: injected generator failure in job {tag}")
        T.clear(); a.serve and torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t1 = time.perf_counter()   # (--serve: the peak per request)
        d.sample(model, model_kf, video_path=vid, audio_path=aud, num_frames=14, resize_size=512, video_folder="videos", latent_folder="videos",
                 landmark_folder="landmarks", audio_folder="audios", audio_emb_folder="audios", fps_id=24, cond_aug=0.0, decoding_t=a.decoding_t,
                 output_folder=os.path.join(out_dir, tag), force_uc_zero_embeddings=["cond_frames", "audio_emb"], chunk_size=2, add_zero_flag=True,
                 n_batch=n_batch, n_batch_keyframes=n_batch_kf, extra_audio=None, compute_until="end", audio_emb_type="hubert", extra_naming=tag,
                 what_mask="box", fix_occlusion=False, position=None, start_frame=0, recompute=True,
                 hubert_model=hubert, wavlm_model=wavlm, vae_model=vae, landmarks_model=lmk)
        torch.cuda.synchronize(); wall = time.perf_counter() - t1
        dur = float(os.popen(f"ffprobe -v error -show_entries format=duration -of csv=p=0 {vid}").read())
        r = {"tag": tag, "video_s": round(dur, 2), "wall_s": round(wall, 1), "s_per_vmin": round(wall / dur * 60, 1), "stages_s": {k: round(v, 1) for k, v in T.items()},
             "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1), "peak_reserved_gb": round(torch.cuda.max_memory_reserved() / 2**30, 1)}
        sys.stdout.write("\nKS_JOB " + json.dumps(r) + "\n"); sys.stdout.flush()   # (r19: one write, at a line start whatever the look-ahead thread printed)
        if os.environ.get("KS_JOB_EMPTY_CACHE", "1") == "1":   # r19 VRAM: the allocator's cached blocks go back after every job, so the device reading is one job's
            import gc; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()   # peak, not the sum of what several jobs (and the look-ahead thread) left cached
    if a.serve:
        import ks_serve
        ks_serve.serve(a.serve, load_s, _objs, job_lines, run_line)
        sys.stdout.flush(); sys.stderr.flush(); os._exit(0)   # r19 speed: a stopped daemon leaves at once (the interpreter teardown of the parked models took
        return                                               # seconds while e2e.sh waited for it before the audio stage); nothing is written after KS_SERVE_STOP
    for line in job_lines(a.jobs, a.follow):
        run_line(line, a.out)
    sys.stdout.write("\nKS_FAST_DONE\n"); sys.stdout.flush()


if __name__ == "__main__":
    main()
