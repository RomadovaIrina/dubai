#!/usr/bin/env python3
"""r9 speed (audio lever R, R9_WORKER_SPOOL=<dir>): ONE resident audio worker per GPU, the models loaded once for every video it serves.

A video's audio job is the same code as without the worker (scripts/r9/audio_proc.run_job: the r6 pipeline, the retake / clause / burst_align
stage, the in-process CPU chain) with the job's own argv, env and cwd; r4_audio.sh hands it over through scripts/r9/audio_client.py and gets its
log and exit code back. Resident: Qwen 14B (+ the batched engine), the TTS QA (Whisper + wav2vec2), pyannote, Silero; Chatterbox as ONE fp32 CPU master
copy (each job's pipeline / stage model a fresh GPU copy of it, ~1 s instead of a 5-6 s load); the ASR Whisper = the QA's.
Before every job the state that would make it differ from a cold process is reset: Qwen's llama.cpp prefix (the next call starts from token 0),
the batched engine's KV and resident prefixes, the per-run stats.
    python scripts/r9/audio_worker.py --spool <dir>        (scripts/r9/audio_worker.sh start|stop <dir>)
Spool: jobs/<id>.json -> out/<id>.log (stdout + stderr) + out/<id>.rc; heartbeat = <dir>/alive (touched every second); <dir>/stop ends it."""
import argparse, json, os, pathlib, sys, threading, time, traceback

HERE = pathlib.Path(__file__).resolve().parent
for p in (HERE.parent / "pilot", HERE.parent / "quality", HERE.parent / "r7", HERE.parent / "r8", HERE): sys.path.insert(0, str(p))
RESIDENT_KEYS = "pyannote,silero_vad,tts_qa"   # GPU-resident; Chatterbox: a CPU master copy (chatterbox_from_master); the ASR Whisper = the QA's (share_whisper)


def chatterbox_from_master():
    """ONE fp32 Chatterbox on the CPU for the whole worker; every from_local(<models/chatterbox>, "cuda", t3_model="v3") (the pipeline's, which it
    then casts to fp16, and the stage's fp32 one) gets a fresh copy of it moved to the GPU: the same fp32 weights from the same files as a load
    (the CPU-load -> GPU path of lever K, sample-identical), ~1 s; nothing of it stays on the GPU between jobs"""
    import copy
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS as CB
    fl = CB.from_local.__func__; path = (HERE.parents[1] / "models" / "chatterbox").resolve(); box = {}
    def from_local(cls, ckpt_dir, device, *a, **k):
        if str(device) == "cuda" and pathlib.Path(ckpt_dir).resolve() == path and k == {"t3_model": "v3"} and not a:
            if "m" not in box: box["m"] = fl(cls, ckpt_dir, "cpu", t3_model="v3")
            m = copy.deepcopy(box["m"])
            for mod in (m.t3, m.s3gen, m.ve): mod.to(device)
            m.device = device
            if m.conds is not None: m.conds = m.conds.to(device=device)
            return m
        return fl(cls, ckpt_dir, device, *a, **k)
    CB.from_local = classmethod(from_local)


def share_whisper():
    """the pipeline's ASR WhisperModel(large-v3, cuda, int8_float16) = the TTS QA's (the same model and compute type) once the QA exists"""
    import faster_whisper, audio_proc
    W0 = faster_whisper.WhisperModel; q = str((HERE.parents[1] / "models" / "whisper-large-v3"))
    def WM(model_size_or_path, device="auto", compute_type="default", *a, **k):
        qa = audio_proc.KEEP.get("qa")
        if qa is not None and str(model_size_or_path) == q and device == "cuda" and compute_type == "int8_float16" and not a and not k \
                and getattr(qa, "_r9_compute", None) == "int8_float16":
            return qa.wh
        return W0(model_size_or_path, *a, device=device, compute_type=compute_type, **k)
    faster_whisper.WhisperModel = WM


def cold_reset():
    """what a cold process would not carry: Qwen's prefix KV, the engine's KV / resident prefixes, the stats"""
    import audio_proc
    llm = (audio_proc.KEEP.get("llm") or (None,))[0]
    if llm is not None: llm.reset()
    try:
        import qwen_batch
        for e in qwen_batch._ENGINES.values():
            e.L.llama_memory_clear(e.mem, True); e.cache.clear(); e.recent.clear(); e.tick = 0
        for k in qwen_batch.STATS: qwen_batch.STATS[k] = 0 if isinstance(qwen_batch.STATS[k], int) else 0.0
    except Exception: pass
    tb = sys.modules.get("t3_batch")
    if tb is not None: tb._DECODERS.clear()   # a cold process captures its T3 decode graphs on its own models
    import gc, torch
    gc.collect(); torch.cuda.empty_cache()
    for mod in ("t3_batch", "s3_batch", "qa_batch"):
        m = sys.modules.get(mod)
        if m is not None and hasattr(m, "STATS"): m.STATS.clear(); m.STATS.update({"takes": 0, "chunks": 0} if mod != "qa_batch" else {})


def _gpu_used_gb():
    import torch
    f, t = torch.cuda.mem_get_info(); return round((t - f) / 2**30, 2)


def park():
    """r17 resident worker (R9_WORKER_PARK=1): the GPU handed over to the video stage between jobs (TZ 0.4: the audio peak ~30 GB and KeySync's
    ~28 GB never share the 32 GB card). Qwen is FREED (its GGUF stays in the page cache: the next job's load_llm takes ~2-4 s, model_cache.py note);
    the TTS QA Whisper (CTranslate2) is unloaded to the CPU, wav2vec2 / pyannote / Silero moved to the CPU. -> one JSON line (what, s, GPU used)"""
    import gc, time, torch, audio_proc
    t0 = time.time(); done = []; steps = {"start": _gpu_used_gb()}
    llm = audio_proc.KEEP.pop("llm", None)
    try:
        import qwen_batch; qwen_batch._ENGINES.clear()   # its batched engines hold contexts of this model
    except Exception: pass
    if llm is not None:
        try: type(llm[0]).close(llm[0])   # (the instance's close() is a no-op patch: the class's frees the model + context)
        except Exception: pass
        del llm; gc.collect(); done.append("qwen_freed"); steps["after_qwen"] = _gpu_used_gb()
    qa = audio_proc.KEEP.get("qa")
    if qa is not None:
        try: qa.wh.model.unload_model(to_cpu=True); done.append("qa_whisper_cpu")
        except Exception as e: done.append(f"qa_whisper_keep({type(e).__name__})")
        steps["after_whisper"] = _gpu_used_gb()
        try: qa.w2v.to("cpu"); done.append("qa_w2v_cpu")
        except Exception: pass
    try:
        import model_cache
        for k, v in model_cache._CACHE.items():
            try: v.to(torch.device("cpu")); done.append(f"{model_cache._name(k)}_cpu")
            except Exception: pass
    except Exception: pass
    gc.collect(); torch.cuda.synchronize(); torch.cuda.empty_cache()
    steps["end"] = _gpu_used_gb()
    return {"park": done, "s": round(time.time() - t0, 2), "gpu_used_gb": steps["end"], "steps_gb": steps,
            "torch_alloc_gb": round(torch.cuda.memory_allocated() / 2**30, 2), "torch_reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2)}


def unpark():
    """the inverse of park(), before a job (Qwen is reloaded lazily by the job's own load_llm)"""
    import time, torch, audio_proc
    t0 = time.time(); done = []
    qa = audio_proc.KEEP.get("qa")
    if qa is not None:
        try:
            if not qa.wh.model.model_is_loaded: qa.wh.model.load_model(); done.append("qa_whisper_gpu")
        except Exception as e: done.append(f"qa_whisper({type(e).__name__})")
        try: qa.w2v.to("cuda"); done.append("qa_w2v_gpu")
        except Exception: pass
    try:
        import model_cache
        for k, v in model_cache._CACHE.items():
            try: v.to(torch.device("cuda")); done.append(f"{model_cache._name(k)}_gpu")
            except Exception: pass
    except Exception: pass
    torch.cuda.synchronize()
    return {"unpark": done, "s": round(time.time() - t0, 2), "gpu_used_gb": _gpu_used_gb()}


def install_once():
    """the stage's caches (r8_tts_stage.install_caches) installed ONCE, so its fp32 Chatterbox / QA / aligner survive across jobs"""
    import r8_tts_stage
    if getattr(r8_tts_stage, "_r9_once", False): return
    f = r8_tts_stage.install_caches; done = []
    def once():
        if not done: f(); done.append(1)
    r8_tts_stage.install_caches = once; r8_tts_stage._r9_once = True


PATCHED = [("run_clean_pipeline_05", "load_llm"), ("chatterbox.mtl_tts", "ChatterboxMultilingualTTS.from_local"), ("chatterbox_fp16", "install_speech_token_guard"),
           ("t3_fast", "install_fast_t3"), ("tts_qa", "TtsQA"), ("tts_align", "CtcAligner.from_qa")]   # what r8_tts_stage.install_caches() patches


def snapshot():
    """the module attributes the stage's install_caches() wraps: restored after every job, so its wrappers (and their caches) do not stack up
    across jobs (job 2's pipeline Chatterbox load fell through to job 1's cached stage model)"""
    import importlib
    snap = []
    for mod, attr in PATCHED:
        try: m = importlib.import_module(mod)
        except Exception: continue
        obj, name = (m, attr) if "." not in attr else (getattr(m, attr.split(".")[0]), attr.split(".")[1])
        snap.append((obj, name, obj.__dict__.get(name) if isinstance(obj, type) else getattr(obj, name)))
    return snap


def restore(snap):
    for obj, name, v in snap: setattr(obj, name, v)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--spool", required=True); a = ap.parse_args()
    D = pathlib.Path(a.spool); (D / "jobs").mkdir(parents=True, exist_ok=True); (D / "out").mkdir(parents=True, exist_ok=True)
    def beat():   # atomic (a reader never sees a half-written file)
        while True:
            (D / ".alive.tmp").write_text(str(time.time())); os.replace(D / ".alive.tmp", D / "alive"); time.sleep(1)
    threading.Thread(target=beat, daemon=True).start()
    import audio_proc
    PARK = os.environ.get("R9_WORKER_PARK", "0") == "1"   # r17: the worker's own setting (each job's env replaces os.environ below)
    n = 0
    while not (D / "stop").exists():
        js = sorted((D / "jobs").glob("*.json"), key=lambda p: p.stat().st_mtime)
        if not js: time.sleep(0.1); continue
        jf = js[0]; job = json.loads(jf.read_text()); jid = jf.stem; jf.unlink()
        log = open(D / "out" / f"{jid}.log", "w", buffering=1); out0, err0 = sys.stdout, sys.stderr; rc = 1
        try:
            os.chdir(job["cwd"]); os.environ.clear(); os.environ.update(job["env"])
            os.environ.update({"DUB_RESIDENT": "1", "DUB_RESIDENT_KEYS": os.environ.get("R9_WORKER_KEYS", RESIDENT_KEYS)})
            os.environ["R9_TTS_PRELOAD"] = "off"   # the worker's own CPU master copy instead (chatterbox_from_master)
            sys.stdout = sys.stderr = log
            if not n: chatterbox_from_master(); share_whisper()
            PK = PARK
            if PK and n: print("[worker] " + json.dumps(unpark()), flush=True)
            audio_proc.setup(); cold_reset(); t0 = time.time(); snap = snapshot()   # setup() (the shared QA / Qwen memo) before the snapshot: kept
            try: rc = audio_proc.run_job(job["argv"])
            finally:
                restore(snap)
                if PK: print("[worker] " + json.dumps(park()), flush=True)   # r17: the GPU free for the video stage before the client sees the rc
            rc = 0 if rc in (None, 0) else int(rc) if isinstance(rc, int) else 1
            print(f"[worker] job {n} done in {time.time() - t0:.1f} s (rc {rc})", flush=True)
        except BaseException:
            traceback.print_exc(); rc = 1
        finally:
            sys.stdout, sys.stderr = out0, err0; log.close()
            (D / "out" / f"{jid}.rc").write_text(str(rc)); n += 1


if __name__ == "__main__":
    main()
