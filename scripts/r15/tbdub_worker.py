#!/usr/bin/env python3
"""r15 W0: resident TBDub generator worker (TaoLiveAIGC/TBDub, code + weights Apache-2.0; 2-step student, 512 px) for the DabAI face stage.
Runs in its OWN venv (/workspace/dub/tbdub/venv: diffusers 0.36 / transformers 5.1; torch 2.7.1+cu128 shared from the dabai env via a .pth),
loads the student ONCE and serves segment jobs from scripts/r15/tbdub_gen.py (the R10_GEN_PY drop-in) as JSON lines on stdin / stdout.
  request {"master": <master_25fps.mp4>, "start": s, "n": n, "wav": <the segment's dub, 16 kHz>, "in": <n x H x W x 3 RGB .npy>, "out": <.npy>}
  -> TBDub's own MediaPipe face crop of master frames [s, s + n) (computed ONCE over the whole master, so its bbox smoothing sees the whole
     video), its generation with the segment audio, its own blend + paste-back onto OUR frames ("in") -> "out" (n x H x W x 3 RGB uint8).
  reply   {"ok": true, "n": n, "gen_s": .., "crop_mode": ..}  /  {"ok": false, "error": ..}
env: TBDUB_STEPS (2), TBDUB_SEED (42), TBDUB_OFFLOAD=1 (--cpu-offload, 15 GiB instead of 28), TBDUB_REPO, TBDUB_CKPT."""
import json, os, sys, time, pathlib, tempfile, traceback
TB = pathlib.Path(os.environ.get("TBDUB_REPO", "/workspace/dub/tbdub/TBDub")); CK = os.environ.get("TBDUB_CKPT", "/workspace/dub/tbdub/ckpt")
sys.path.insert(0, str(TB))
import numpy as np, torch
from PIL import Image
import inference as tbi
from preprocessing_mediapipe import preprocess_video_with_mediapipe

_pfd = os.dup(1); os.dup2(2, 1); OUT = os.fdopen(_pfd, "w", buffering=1); sys.stdout = sys.stderr   # the protocol owns the real stdout; every
# other write (the TBDub code, its MediaPipe child process at the OS level: fd 1) goes to the log


def log(*a): print("[tbdub-worker]", *a, flush=True)


def reply(d): OUT.write(json.dumps(d) + "\n"); OUT.flush()


def main():
    argv = ["--video", "unused.mp4", "--audio", "unused.wav", "--checkpoint-dir", CK, "--inference-mode", "student", "--num-student-steps", os.environ.get("TBDUB_STEPS", "2"),
            "--sigma-shift", "1.0", "--motion-from-latents", "--seed", os.environ.get("TBDUB_SEED", "42"), "--output-dir", tempfile.mkdtemp(prefix="tbdub_")]
    if os.environ.get("TBDUB_OFFLOAD") == "1": argv.append("--cpu-offload")
    args = tbi.parse_args(argv); t0 = time.perf_counter()
    pipe, emb = tbi.build_pipeline(args); load_s = round(time.perf_counter() - t0, 2)
    log(f"loaded in {load_s} s, peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB, dit {args.dit_checkpoint}")
    captured = []
    tbi.save_video_with_audio = lambda video, *a, **k: captured.append(video)   # run_inference's paste-back result, kept in memory (no lossy mp4)
    if os.environ.get("TBDUB_FASTCC") == "1":   # speed (default off): TBDub's per-frame Lab colour match with OpenCV instead of skimage (CPU ~0.15 s/frame)
        import cv2

        def cc_cv2(gen, src, strength=1.0):
            out = []
            for g_, s_ in zip(gen, src):
                g = np.asarray(g_.convert("RGB"), np.float32) / 255; s = np.asarray(s_.convert("RGB"), np.float32) / 255
                gl, sl = cv2.cvtColor(g, cv2.COLOR_RGB2Lab), cv2.cvtColor(s, cv2.COLOR_RGB2Lab)
                gm, gs, sm, ss = gl.mean((0, 1)), gl.std((0, 1)), sl.mean((0, 1)), sl.std((0, 1))
                cl = np.where(gs > 0, (gl - gm) * (ss / np.maximum(gs, 1e-12)) + sm, sm)
                rgb = np.clip(cv2.cvtColor(cl.astype(np.float32), cv2.COLOR_Lab2RGB), 0, 1)
                out.append(Image.fromarray((np.clip((1 - strength) * g + strength * rgb, 0, 1) * 255).astype(np.uint8)))
            return out + list(gen[len(out):])
        tbi.color_correction_lab = cc_cv2; log("TBDUB_FASTCC: OpenCV Lab colour match")
    reply({"ok": True, "ready": True, "load_s": load_s}); crops = {}; whole = {}
    paste_orig = tbi.paste_video_back; blends = []
    for line in sys.stdin:
        if not line.strip(): continue
        q = json.loads(line); t1 = time.perf_counter()
        try:
            m = q["master"]
            if m not in crops:   # TBDub's own preprocessing (isolated CPU MediaPipe subprocess) over the whole master, once
                tp = time.perf_counter(); raw, ref, bb, mode = preprocess_video_with_mediapipe(m, model_path=args.mediapipe_model); del raw
                crops[m] = (ref, bb, mode); log(f"preprocess {m}: {len(ref)} frames, crop_mode {mode}, {time.perf_counter() - tp:.1f} s")
            ref, bb, mode = crops[m]; s, n = int(q["start"]), int(q["n"]); bbi = [list(map(int, b)) for b in bb]
            fr = np.load(q["in"]); assert len(fr) == n, (len(fr), n)
            raw = [Image.fromarray(f) for f in fr]; torch.cuda.reset_peak_memory_stats()
            if q.get("full_wav"):   # TBDUB_WHOLE: ONE generation over the whole master with the whole dub (TBDub's own clip / motion continuity), its
                if m not in whole:  # blended 512 crops kept; each segment = its slice pasted back onto OUR frames with TBDub's own paste-back
                    blends.clear(); tbi.paste_video_back = lambda video, raw_, bbs: (blends.append(video), [])[1]
                    try: tbi.run_inference(pipe, emb, tbi.PreprocessedSample(list(ref), list(ref), bbi, q["full_wav"]), "whole", args)
                    finally: tbi.paste_video_back = paste_orig
                    whole[m] = blends[-1]; log(f"whole-video generation {m}: {len(whole[m])} crops in {time.perf_counter() - t1:.1f} s")
                cw = whole[m]; k = max(0, min(n, len(cw) - s))
                out = np.stack([np.asarray(f.convert("RGB")) for f in paste_orig(cw[s:s + k], raw[:k], bbi[s:s + k])]) if k > 0 else fr[:0]
            else:
                sample = tbi.PreprocessedSample(raw, list(ref[s:s + n]), bbi[s:s + n], q["wav"])
                captured.clear()
                tbi.run_inference(pipe, emb, sample, f"seg{s}", args)
                out = np.stack([np.asarray(f.convert("RGB")) for f in captured[-1]])
            torch.cuda.empty_cache()   # the caching allocator would keep TBDub's peak (28 GB whole-video) away from CodeFormer in the generator process
            got = len(out)
            if len(out) < n: out = np.concatenate([out, fr[len(out):]])   # TBDub's length = the audio's (rounded): keep our frames for the tail
            np.save(q["out"], out[:n])
            reply({"ok": True, "n": n, "got": int(got), "gen_s": round(time.perf_counter() - t1, 2), "crop_mode": mode,
                   "peak_mib": round(torch.cuda.max_memory_allocated() / 2**20)})
        except Exception as e:
            log(traceback.format_exc()); reply({"ok": False, "error": f"{type(e).__name__}: {e}"})
    log("stdin closed: exit")


if __name__ == "__main__":
    main()
