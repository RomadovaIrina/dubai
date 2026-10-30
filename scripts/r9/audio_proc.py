#!/usr/bin/env python3
"""r9 speed (audio lever L, R9_ONEPROC=on): the r6 pipeline AND the retake -> clause takes -> burst_align stage in ONE process.

Without it the pipeline process exits and r8_tts_stage.py starts a second one that imports everything again and loads Qwen 14B, the TTS QA
(Whisper large-v3 + wav2vec2) and Chatterbox again (m01: ~14 s of start + loads). Here the pipeline runs as usual (run_clean_pipeline_05.main,
same argv), its Qwen and TTS QA objects stay resident (their close() is a no-op until the end) and the stage's mains get them through the
r8_tts_stage caches; the batched Qwen engine keeps its resident few-shot prefixes. Chatterbox is NOT shared: the pipeline casts its copy to fp16 in
place, retake / clause takes load their own as before (identical numerics).
    python scripts/r9/audio_proc.py <run_clean_pipeline_05 argv>     env R9_AFTER_JOBS=<dir with oneproc_jobs_<id>.json>
After the pipeline: its manifest copied into the work dir (as r4_audio.sh does), the source face tracks waited for (<npz>.done of the background
MediaPipe, R9_MP_BG), then the jobs exactly as r8_tts_stage.py runs them (logs, <jobs>.times.json)."""
import json, os, pathlib, shutil, sys, time
HERE = pathlib.Path(__file__).resolve().parent
for p in (HERE.parent / "pilot", HERE.parent / "quality", HERE.parent / "r7", HERE.parent / "r8", HERE): sys.path.insert(0, str(p))


def main():
    argv = sys.argv[1:]
    import run_clean_pipeline_05 as rp, tts_qa
    keep = {}
    _load = rp.load_llm
    def load_llm(times):   # one Qwen per process; the pipeline's own close() at its end is a no-op (the stage reuses it)
        if "llm" not in keep:
            keep["llm"] = _load(times); keep["llm"].close = lambda: None
        return keep["llm"]
    rp.load_llm = load_llm
    _Q = tts_qa.TtsQA
    class SharedQA:   # one TtsQA per process (the pipeline's model_cache.get("tts_qa", TtsQA) and the stage's TtsQA())
        def __new__(cls, *a, **k):
            if "qa" not in keep:
                keep["qa"] = _Q(*a, **k); keep["qa"].close = lambda: None
            return keep["qa"]
    tts_qa.TtsQA = SharedQA
    sys.argv = [str(HERE.parent / "pilot" / "run_clean_pipeline_05.py")] + argv
    rc = rp.main()
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


if __name__ == "__main__":
    main()
