#!/usr/bin/env python3
"""r9 speed (audio lever L, R9_ONEPROC=on): the r6 pipeline AND the retake -> clause takes -> burst_align stage in ONE process.

Without it the pipeline process exits and r8_tts_stage.py starts a second one that imports everything again and loads Qwen 14B, the TTS QA
(Whisper large-v3 + wav2vec2) and Chatterbox again (m01: ~14 s of start + loads). Here the pipeline runs as usual (run_clean_pipeline_05.main,
same argv), its Qwen stays resident (its close() is a no-op until the end) and the stage's mains get it through the r8_tts_stage cache
and so does its TTS QA; the batched Qwen engine keeps its resident few-shot prefixes. Chatterbox is NOT shared: the pipeline casts its copy to fp16 in
place, retake / clause takes load their own as before (identical numerics).
    python scripts/r9/audio_proc.py <run_clean_pipeline_05 argv>     env R9_AFTER_JOBS=<dir with oneproc_jobs_<id>.json>
After the pipeline: its manifest copied into the work dir (as r4_audio.sh does), the source face tracks waited for (<npz>.done of the background
MediaPipe, R9_MP_BG), then the jobs exactly as r8_tts_stage.py runs them (logs, <jobs>.times.json)."""
import json, os, pathlib, shutil, sys, time
HERE = pathlib.Path(__file__).resolve().parent
for p in (HERE.parent / "pilot", HERE.parent / "quality", HERE.parent / "r7", HERE.parent / "r8", HERE): sys.path.insert(0, str(p))


def _preload_tts(keep):
    """lever K: ChatterboxMultilingualTTS.from_local(<models/chatterbox>, "cpu", t3_model="v3") in a thread at the start; the FIRST cuda load (the
    pipeline's, which it then casts to fp16) runs as before; a LATER one with the same arguments (the stage's fp32 retake / clause model) gets the
    CPU copy moved to the GPU: the same fp32 weights read from the same files, ~1 s instead of ~5 s"""
    import threading
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS as CB
    fl = CB.from_local.__func__; path = HERE.parents[1] / "models" / "chatterbox"; box = {}
    def load():
        try: box["m"] = fl(CB, path, "cpu", t3_model="v3")
        except Exception as e: box["err"] = repr(e)
    th = threading.Thread(target=load, daemon=True); th.start(); calls = [0]
    def from_local(cls, ckpt_dir, device, *a, **k):
        calls[0] += 1
        if calls[0] > 1 and str(device) == "cuda" and pathlib.Path(ckpt_dir).resolve() == path.resolve() and k == {"t3_model": "v3"} and not a:
            th.join()
            m = box.pop("m", None)
            if m is not None:
                for mod in (m.t3, m.s3gen, m.ve): mod.to(device)
                m.device = device
                if m.conds is not None: m.conds = m.conds.to(device=device)
                print("[oneproc] stage Chatterbox from the CPU preload", file=sys.stderr, flush=True)
                return m
        return fl(cls, ckpt_dir, device, *a, **k)
    CB.from_local = classmethod(from_local)


def main():
    argv = sys.argv[1:]
    import run_clean_pipeline_05 as rp
    keep = {}
    _load = rp.load_llm
    def load_llm(times):   # one Qwen per process; the pipeline's own close() at its end is a no-op (the stage reuses it)
        if "llm" not in keep:
            keep["llm"] = _load(times); keep["llm"][0].close = lambda: None   # load_llm -> (Llama, model name)
        return keep["llm"]
    rp.load_llm = load_llm
    import tts_qa
    _Q = tts_qa.TtsQA
    class SharedQA:   # one TtsQA per process (the pipeline's model_cache.get("tts_qa", TtsQA) and the stage's TtsQA()): a 2nd Whisper next to the resident
        def __new__(cls, *a, **k):   # Qwen + its batched engine runs out of VRAM (02); a stage QA of its own gave bit-identical outputs anyway
            if "qa" not in keep:
                keep["qa"] = _Q(*a, **k); keep["qa"].close = lambda: None; keep["qa"]._r9_compute = os.environ.get("DUB_QA_WHISPER_COMPUTE", "float16")   # (chain_inproc reuse check)
            return keep["qa"]
    tts_qa.TtsQA = SharedQA
    if os.environ.get("R9_TTS_PRELOAD", "off") == "on":   # r9 speed K: the stage's fp32 Chatterbox loaded on the CPU WHILE the GPU pipeline runs
        _preload_tts(keep)                                  # (the pipeline casts its own copy to fp16 in place: not shareable)
    env0 = dict(os.environ)   # the env the pipeline sets for itself (os.environ) is restored before the stage
    sys.argv = [str(HERE.parent / "pilot" / "run_clean_pipeline_05.py")] + argv
    rc = rp.main()
    os.environ.clear(); os.environ.update(env0)
    if os.environ.get("R9_STAGE_ENV"):   # the env r8_audio.sh starts the stage with (r4_audio.sh exports DUB_CUT_SNAP / DUB_EQPOWER_FADE / ... for the pipeline only:
        se = json.loads(pathlib.Path(os.environ["R9_STAGE_ENV"]).read_text())   # retake's cut_parts reads DUB_CUT_SNAP; inheriting it changed the cuts: 02 pause W1 0.26 -> 0.39)
        jd_ = os.environ.get("R9_AFTER_JOBS"); os.environ.clear(); os.environ.update(se)
        if jd_: os.environ["R9_AFTER_JOBS"] = jd_
    if rc not in (None, 0): sys.exit(rc)
    jd = os.environ.get("R9_AFTER_JOBS")
    if not jd: return
    out = pathlib.Path(argv[argv.index("--output") + 1]); wdir = pathlib.Path(argv[argv.index("--work-dir") + 1]); id_ = wdir.name
    J = pathlib.Path(jd) / f"oneproc_jobs_{id_}.json"
    if not J.exists(): print(f"[oneproc] no jobs for {id_}", file=sys.stderr); return
    w = wdir / f"work_{pathlib.Path(argv[argv.index('--input') + 1]).stem}"
    shutil.copy(out.with_name(out.stem + ".manifest.json"), w / f"{id_}_r4a.manifest.json")
    t0 = time.perf_counter()
    for f in json.loads(J.read_text()).get("wait", []):
        while not os.path.exists(f + ".done"): time.sleep(0.2)
    print(f"[oneproc] {id_}: pipeline done, face tracks after {time.perf_counter() - t0:.1f} s; stage in this process", file=sys.stderr, flush=True)
    import r8_tts_stage
    jobs = json.loads(J.read_text())["jobs"]; jf = J.with_name(f"oneproc_{id_}.json"); jf.write_text(json.dumps(jobs, indent=1))
    sys.argv = ["r8_tts_stage", "--jobs", str(jf)]
    r8_tts_stage.main()
    cj = J.with_name(f"chain_{id_}.json")   # r9 speed C (R9_CHAIN_INPROC=on): the CPU chain after the stage in this process (scripts/r9/chain_inproc.py)
    if cj.exists():
        import chain_inproc
        t1 = time.perf_counter(); ok = chain_inproc.run(json.loads(cj.read_text()), keep.get("qa"))
        print(f"[oneproc] {id_}: chain in-process {'done' if ok else 'FAILED (the bash chain runs)'} in {time.perf_counter() - t1:.1f} s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
