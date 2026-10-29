#!/usr/bin/env python3
"""r8 speed lever L5 (R8_ONEPROC=on): retake -> clause_takes -> burst_align for ALL videos in ONE process, the models loaded once.
Each script still runs its own main() with its own argv (identical code paths and flags); only the loaders are cached for the process:
Qwen (run_clean_pipeline_05.load_llm), Chatterbox (ChatterboxMultilingualTTS.from_local + the speech-token guard / fast T3, installed once),
the TTS QA (Whisper + wav2vec2: TtsQA) and its CTC aligner. Measured per step and video like the per-process chain.
    python scripts/r8/r8_tts_stage.py --jobs jobs.json        jobs = [{"id": "04", "steps": [["retake", [argv...]], ["clause", [...]], ...]}]
Exit 0 even when a step fails (the step's out dir is then missing; r8_audio.sh falls back like the per-process chain)."""
import argparse, json, pathlib, sys, time, traceback
HERE = pathlib.Path(__file__).resolve().parent
for p in (HERE.parent / "pilot", HERE.parent / "quality", HERE.parent / "r7", HERE): sys.path.insert(0, str(p))


def install_caches():
    import run_clean_pipeline_05 as rp, tts_qa, tts_align, chatterbox_fp16
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    C = {}
    _llm = rp.load_llm
    def load_llm(times):
        if "llm" not in C: C["llm"] = _llm(times)
        return C["llm"]
    rp.load_llm = load_llm
    _fl = ChatterboxMultilingualTTS.from_local.__func__
    def from_local(cls, *a, **k):
        key = ("tts", str(a), str(sorted(k.items())))
        if key not in C: C[key] = _fl(cls, *a, **k)
        return C[key]
    ChatterboxMultilingualTTS.from_local = classmethod(from_local)
    _g = chatterbox_fp16.install_speech_token_guard
    def guard(m, *a, **k):
        if getattr(m, "_r8_guarded", False): return
        r = _g(m, *a, **k); m._r8_guarded = True; return r
    chatterbox_fp16.install_speech_token_guard = guard
    try:
        import t3_fast
        _f = t3_fast.install_fast_t3
        def fast(m, *a, **k):
            if getattr(m, "_r8_fast", False): return
            r = _f(m, *a, **k); m._r8_fast = True; return r
        t3_fast.install_fast_t3 = fast
    except Exception: pass
    _Q = tts_qa.TtsQA
    class CachedQA:   # TtsQA() -> one instance per process
        def __new__(cls, *a, **k):
            if "qa" not in C: C["qa"] = _Q(*a, **k)
            return C["qa"]
    tts_qa.TtsQA = CachedQA
    _al = tts_align.CtcAligner.from_qa.__func__
    def from_qa(cls, qa):
        if "al" not in C: C["al"] = _al(cls, qa)
        return C["al"]
    tts_align.CtcAligner.from_qa = classmethod(from_qa)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", required=True); a = ap.parse_args()
    install_caches()
    import retake, burst_align, clause_takes
    mains = {"retake": retake.main, "clause": clause_takes.main, "burst_align": burst_align.main}
    jobs = json.loads(pathlib.Path(a.jobs).read_text()); rep = []
    for j in jobs:
        for name, argv in j["steps"]:
            t0 = time.perf_counter(); ok = True; log = pathlib.Path(j["logs"][name]); log.parent.mkdir(parents=True, exist_ok=True)
            out0, err0 = sys.stdout, sys.stderr
            with open(log, "w") as fh:
                sys.stdout = sys.stderr = fh; sys.argv = [name] + argv
                try: mains[name]()
                except SystemExit as e: ok = e.code in (None, 0)
                except Exception: ok = False; traceback.print_exc(file=fh)
                finally: sys.stdout, sys.stderr = out0, err0
            rep.append({"id": j["id"], "step": name, "ok": ok, "seconds": round(time.perf_counter() - t0, 1)}); print(json.dumps(rep[-1]), flush=True)
    pathlib.Path(a.jobs).with_suffix(".times.json").write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
