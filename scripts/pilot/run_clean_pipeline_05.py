#!/usr/bin/env python3
"""Pilot 0.5 — REAL clean end-to-end dubbing runner (the measured subject for benchmark_05_gpu_coefficient.py).

This is a reproducible measurement runner, not a production backend. Every stage is the real component of the
frozen dabai baseline, wired in pipeline order on the ORIGINAL timeline (pauses are never removed):

   1. source probe                     ffprobe
   2. audio extraction                 ffmpeg -> mono 16 kHz PCM
   3. Silero VAD                       speech intervals -> SPEECH / PASS_THROUGH timeline (measure_stages.build_timeline)
   4. faster-whisper large-v3          real ASR, word_timestamps=True; speech units = whisper segments, slots = first/last
                                       word timestamps, units longer than --max-unit-s are split at the widest word gap
   5. pyannote speaker-diarization-3.1 real diarization (HF_TOKEN); speaker per unit = max overlap, else nearest turn
   6. Qwen2.5-7B-Instruct q4_k_m       real translation via llama.cpp, CPU only (n_gpu_layers=0, DUB_THREADS), the
                                       pilot 0.9 system prompt, temperature 0
   7. Chatterbox Multilingual v3       real TTS per unit; the speaker's own diarized speech (<=10 s, cut from the source
                                       audio) is the voice reference via the upstream `audio_prompt_path` API;
                                       fp16 cast like scripts/check_chatterbox.py, fp32 fallback
   8. duration alignment               ffmpeg atempo (scripts/check_atempo.py path): TTS longer than its slot is
                                       sped up to fit exactly; TTS shorter than its slot keeps natural speed and is
                                       padded with silence -> the global timeline never moves
   9. dubbed audio track               full source length; aligned TTS at the unit slots, silence elsewhere
  10. face-aware LatentSync 1.6        scripts/pilot/face_aware_latentsync.py functions: VALID_FACE segments ->
                                       LatentSync driven by the DUBBED audio, SMALL_FACE / NO_FACE -> pass-through,
                                       "Face not detected" never aborts the run; frame order / count preserved
  11. CodeFormer                       NOT part of the clean pipeline; --no-codeformer is mandatory
  12. final assembly                   lipsynced/pass-through frames (25 fps CFR master, as LatentSync works) + dubbed
                                       AAC track -> output MP4 on the full original timeline; no subtitles (the 0.5
                                       spec measures the clean pipeline without the 0.11 subtitle stage)

Runtime media goes to --work-dir (default /tmp/dabai_pilot_05/work_<stem>); a compact manifest JSON
(<output>.manifest.json) records every stage's inputs/outputs and wall times. Nothing is written into the repo.

    source scripts/env.sh
    python scripts/pilot/run_clean_pipeline_05.py --input test_videos/04.mp4 \
        --output /tmp/dabai_pilot_05/sanity_04.mp4 --target-lang en --no-codeformer

Exit codes: 0 ok · 2 usage · 3 blocker (missing token/model/speech) · 4 output validation failed · 5 stage failure.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")   # pyannote 3.x checkpoints, as in scripts/check_pyannote.py
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import MODELS, ROOT, THREADS, gpu_info, write_json  # noqa: E402
from measure_stages import build_timeline, ff, probe                  # noqa: E402
from benchmark_09_qwen_quant import SYSTEM_PROMPT                     # noqa: E402

DEFAULT_WORK = pathlib.Path("/tmp/dabai_pilot_05")
# Defaults of --video-backend optimized. They are the configuration verified by the optimization track
# (reports/pilot/optim/batch_sweep.md, compile_inductor.json); baseline mode ignores them.
OPTIMIZED_DEFAULTS = {"face_router": "retinaface", "window_batch_size": 2, "compile_backend": "none", "sdpa_backend": "auto"}
# batch 2: largest batch that leaves VRAM headroom (25.8 GB peak vs 30.8 GB at batch 4 for +0.7 %); inductor gave no gain over
# DeepCache eager; "auto" SDPA already dispatches the fused FLASH_ATTENTION kernel (forced flash = bit-identical, same time).
TTS_SR = 24000
REF_MAX_S = 10.0          # Chatterbox DEC_COND_LEN = 10 s @ 24 kHz
REF_MIN_S = 1.0
LANG_NAMES = {"ar": "Arabic", "da": "Danish", "de": "German", "el": "Greek", "en": "English", "es": "Spanish", "fi": "Finnish",
              "fr": "French", "he": "Hebrew", "hi": "Hindi", "it": "Italian", "ja": "Japanese", "ko": "Korean", "ms": "Malay",
              "nl": "Dutch", "no": "Norwegian", "pl": "Polish", "pt": "Portuguese", "ru": "Russian", "sv": "Swedish",
              "sw": "Swahili", "tr": "Turkish", "zh": "Chinese", "uk": "Ukrainian", "kk": "Kazakh", "cs": "Czech"}


class Blocker(RuntimeError):
    """A real component cannot run; the runner must stop instead of faking the stage."""


def lang_name(code: str) -> str:
    return LANG_NAMES.get(code.lower(), code)


def free_cuda(*objs) -> None:
    import torch
    for o in objs:
        del o
    gc.collect()
    torch.cuda.empty_cache()


def timed(times: dict, name: str):
    class _T:
        def __enter__(self):
            self.t0 = time.perf_counter(); return self

        def __exit__(self, *exc):
            times[name] = round(time.perf_counter() - self.t0, 3); return False
    return _T()


# --------------------------------------------------------------------------------------------------------------
# audio stages
# --------------------------------------------------------------------------------------------------------------
def run_vad(wav16: pathlib.Path) -> list[dict]:
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
    torch.set_num_threads(THREADS)
    model = load_silero_vad()
    audio = read_audio(str(wav16), sampling_rate=16000)
    ts = get_speech_timestamps(audio, model, sampling_rate=16000, return_seconds=True)
    return [{"start": round(float(t["start"]), 3), "end": round(float(t["end"]), 3)} for t in ts]


def run_asr(wav16: pathlib.Path, language: str | None) -> tuple[list[dict], dict]:
    import torch  # noqa: F401  maps libcublas before ctranslate2 loads (scripts/check_whisper.py)
    from faster_whisper import WhisperModel
    m = WhisperModel(str(MODELS / "whisper-large-v3"), device="cuda", compute_type="int8_float16")
    segs, info = m.transcribe(str(wav16), word_timestamps=True, beam_size=5, vad_filter=False, language=language)
    out = []
    for s in segs:
        out.append({"id": s.id, "start": round(float(s.start), 3), "end": round(float(s.end), 3), "text": s.text.strip(),
                    "words": [{"word": w.word, "start": round(float(w.start), 3), "end": round(float(w.end), 3),
                               "p": round(float(w.probability), 3)} for w in (s.words or [])]})
    meta = {"language": info.language, "language_probability": round(float(info.language_probability), 3),
            "audio_duration_s": round(float(info.duration), 3), "compute_type": m.model.compute_type}
    free_cuda(m)
    return out, meta


def build_units(segments: list[dict], duration: float, max_unit_s: float) -> list[dict]:
    """Speech units on the original timeline. Slot = first word start .. last word end (whisper segment bounds when the
    segment has no words). Segments longer than max_unit_s are split at the widest inter-word gap (closest to the middle
    on ties) so every TTS call stays inside Chatterbox's generation budget."""

    def _split(ws: list[dict]) -> list[list[dict]]:
        if ws[-1]["end"] - ws[0]["start"] <= max_unit_s or len(ws) < 4:
            return [ws]
        n = len(ws)
        i = max(range(2, n - 1), key=lambda k: (ws[k]["start"] - ws[k - 1]["end"]) - 0.01 * abs(k - n / 2))
        return _split(ws[:i]) + _split(ws[i:])

    units = []
    for s in segments:
        if not s["text"]:
            continue
        if s["words"]:
            for ws in _split(s["words"]):
                units.append({"asr_segment": s["id"], "start": max(0.0, ws[0]["start"]), "end": min(duration, ws[-1]["end"]),
                              "text": "".join(w["word"] for w in ws).strip(), "words": len(ws)})
        else:
            units.append({"asr_segment": s["id"], "start": max(0.0, s["start"]), "end": min(duration, s["end"]),
                          "text": s["text"], "words": 0})
    units = [u for u in units if u["text"] and u["end"] > u["start"]]
    units.sort(key=lambda u: u["start"])
    for i, u in enumerate(units):
        u["id"] = i; u["start"] = round(u["start"], 3); u["end"] = round(u["end"], 3)
    return units


def run_diarization(wav16: pathlib.Path) -> list[dict]:
    import torch
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not tok:
        raise Blocker("HF_TOKEN is not set (gated pyannote/speaker-diarization-3.1)")
    from pyannote.audio import Pipeline
    pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=tok)
    if pipe is None:
        raise Blocker("pyannote pipeline is None: model licenses not accepted for this token")
    pipe.to(torch.device("cuda"))
    dia = pipe(str(wav16))
    turns = [{"start": round(float(seg.start), 3), "end": round(float(seg.end), 3), "speaker": spk}
             for seg, _, spk in dia.itertracks(yield_label=True)]
    free_cuda(pipe, dia)
    return turns


def assign_speakers(units: list[dict], turns: list[dict]) -> None:
    if not turns:
        raise Blocker("diarization returned no speaker turns; cannot assign speakers")
    for u in units:
        ov: dict[str, float] = {}
        for t in turns:
            o = min(u["end"], t["end"]) - max(u["start"], t["start"])
            if o > 0:
                ov[t["speaker"]] = ov.get(t["speaker"], 0.0) + o
        if ov:
            u["speaker"] = max(ov, key=ov.get); u["speaker_method"] = "max_overlap"
            u["speaker_overlap_s"] = round(ov[u["speaker"]], 3)
        else:
            c = (u["start"] + u["end"]) / 2
            t = min(turns, key=lambda t: 0.0 if t["start"] <= c <= t["end"] else min(abs(c - t["start"]), abs(c - t["end"])))
            u["speaker"] = t["speaker"]; u["speaker_method"] = "nearest_turn_to_center"; u["speaker_overlap_s"] = 0.0


def run_translation(units: list[dict], src_lang: str, tgt_lang: str, times: dict) -> dict:
    import glob
    from llama_cpp import Llama
    files = sorted(glob.glob(str(MODELS / "qwen2.5-7b-instruct-gguf" / "*q4_k_m*.gguf")))
    if not files:
        raise Blocker("Qwen2.5-7B q4_k_m GGUF not found under models/")
    with timed(times, "translation_load"):
        llm = Llama(model_path=files[0], n_ctx=4096, n_threads=THREADS, n_threads_batch=THREADS, n_gpu_layers=0, verbose=False)
    src_name, tgt_name = lang_name(src_lang), lang_name(tgt_lang)
    usage = {"prompt_tokens": 0, "completion_tokens": 0}
    with timed(times, "translation"):
        for u in units:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Translate from {src_name} to {tgt_name}:\n{u['text']}"}]
            r = llm.create_chat_completion(msgs, max_tokens=512, temperature=0.0)
            txt = " ".join(r["choices"][0]["message"]["content"].split()).strip()
            if len(txt) >= 2 and txt[0] in "\"«“" and txt[-1] in "\"»”":
                txt = txt[1:-1].strip()
            if not txt:
                raise RuntimeError(f"empty translation for unit {u['id']}: {u['text'][:80]!r}")
            u["translation"] = txt
            for k in usage:
                usage[k] += int(r["usage"].get(k, 0))
            print(f"  [translate] u{u['id']:02d} {u['speaker']} {u['start']:.2f}-{u['end']:.2f}s | {u['text'][:60]} -> {txt[:60]}", flush=True)
    del llm
    return {"model": os.path.basename(files[0]), "n_ctx": 4096, "n_threads": THREADS, "n_gpu_layers": 0,
            "system_prompt": SYSTEM_PROMPT, "direction": f"{src_name}->{tgt_name}", "usage": usage}


def speaker_references(units: list[dict], turns: list[dict], wav16: pathlib.Path, work: pathlib.Path) -> dict:
    """Per speaker: up to REF_MAX_S seconds of that speaker's own diarized speech, longest turns first, written in
    chronological order as a 16 kHz mono wav. Speakers with < REF_MIN_S s fall back to Chatterbox's built-in voice."""
    import numpy as np
    import soundfile as sf
    audio, sr = sf.read(str(wav16), dtype="float32")
    refs = {}
    for spk in sorted({u["speaker"] for u in units}):
        picked, total = [], 0.0
        for t in sorted((t for t in turns if t["speaker"] == spk), key=lambda t: t["end"] - t["start"], reverse=True):
            if total >= REF_MAX_S:
                break
            d = min(t["end"] - t["start"], REF_MAX_S - total)
            picked.append((t["start"], t["start"] + d)); total += d
        picked.sort()
        if total < REF_MIN_S:
            refs[spk] = {"file": None, "seconds": round(total, 3), "source": "builtin_conds"}
            continue
        y = np.concatenate([audio[int(a * sr):int(b * sr)] for a, b in picked])
        p = work / f"ref_{spk}.wav"; sf.write(str(p), y, sr, subtype="PCM_16")
        refs[spk] = {"file": str(p), "seconds": round(len(y) / sr, 3), "turns_used": len(picked), "source": "diarized_speech"}
    return refs


def run_tts(units: list[dict], refs: dict, tgt_lang: str, work: pathlib.Path, seed: int, times: dict) -> dict:
    import torch
    import torchaudio
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
    if tgt_lang not in SUPPORTED_LANGUAGES:
        raise Blocker(f"target language {tgt_lang!r} not supported by Chatterbox: {sorted(SUPPORTED_LANGUAGES)}")
    with timed(times, "tts_load"):
        m = ChatterboxMultilingualTTS.from_local(MODELS / "chatterbox", "cuda", t3_model="v3")
    builtin = m.conds

    def cast(dtype):
        for mod in (m.t3, m.s3gen, m.ve):
            mod.to(dtype)
        if m.conds is not None:
            m.conds = m.conds.to(device="cuda")

    def synth_all(dtype) -> None:
        cast(dtype)
        by_spk: dict[str, list[dict]] = {}
        for u in units:
            by_spk.setdefault(u["speaker"], []).append(u)
        for spk, us in by_spk.items():
            ref = refs[spk]["file"]
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                if ref:
                    m.prepare_conditionals(ref)
                else:
                    m.conds = builtin.to(device="cuda")
            for u in us:
                torch.manual_seed(seed + u["id"])
                with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                    wav = m.generate(u["translation"], language_id=tgt_lang)
                wav = wav.detach().float().cpu()
                if not torch.isfinite(wav).all() or wav.abs().max() < 1e-4:
                    raise RuntimeError(f"non-finite or silent TTS for unit {u['id']}")
                p = work / f"tts_u{u['id']:02d}.wav"; torchaudio.save(str(p), wav, m.sr)
                u["tts"] = {"file": str(p), "seconds": round(wav.shape[-1] / m.sr, 3), "sr": m.sr,
                            "reference": refs[spk]["source"], "dtype": str(dtype).replace("torch.", "")}
                print(f"  [tts] u{u['id']:02d} {spk} {u['tts']['seconds']:.2f}s <- {u['translation'][:60]}", flush=True)

    used = None
    with timed(times, "tts"):
        for dtype in (torch.float16, torch.float32):
            try:
                synth_all(dtype); used = dtype; break
            except Exception as e:  # same fallback ladder as scripts/check_chatterbox.py
                print(f"  [tts] {dtype} failed: {type(e).__name__}: {str(e)[:200]} -> retrying next dtype", flush=True)
    if used is None:
        raise RuntimeError("Chatterbox generation failed in every dtype")
    sr = m.sr
    free_cuda(m)
    return {"model": "ChatterboxMultilingualTTS t3=v3 (models/chatterbox)", "sr": sr, "dtype": str(used).replace("torch.", ""),
            "language_id": tgt_lang, "voice_reference": "speaker's own diarized speech via audio_prompt_path/prepare_conditionals",
            "references": refs}


def align_units(units: list[dict], duration: float, work: pathlib.Path) -> None:
    """Fit every TTS clip into its slot on the original timeline. Slot end is capped at the next unit's start."""
    import numpy as np
    import soundfile as sf
    for i, u in enumerate(units):
        slot_start = u["start"]
        slot_end = min(u["end"], duration, units[i + 1]["start"] - 0.02 if i + 1 < len(units) else duration)
        slot_end = max(slot_end, slot_start + 0.2)
        slot_s = slot_end - slot_start
        tts_s = u["tts"]["seconds"]
        ratio = tts_s / slot_s
        aligned = work / f"aligned_u{u['id']:02d}.wav"
        if ratio > 1.005:
            tempo = min(ratio, 100.0)   # ffmpeg 6.1 atempo accepts 0.5..100 in one filter
            ff(["-i", u["tts"]["file"], "-af", f"atempo={tempo:.6f}", "-ar", str(TTS_SR), "-ac", "1", "-c:a", "pcm_s16le", str(aligned)])
            mode, atempo = "speed_up_to_slot", round(tempo, 4)
        else:
            ff(["-i", u["tts"]["file"], "-ar", str(TTS_SR), "-ac", "1", "-c:a", "pcm_s16le", str(aligned)])
            mode, atempo = "natural_speed_pad_silence", 1.0
        y, sr = sf.read(str(aligned), dtype="float32")
        n = int(round(slot_s * sr))
        y = y[:n] if len(y) >= n else np.pad(y, (0, n - len(y)))
        sf.write(str(aligned), y, sr, subtype="PCM_16")
        u["slot"] = {"start": round(slot_start, 3), "end": round(slot_end, 3), "seconds": round(slot_s, 3)}
        u["alignment"] = {"tts_seconds": tts_s, "ratio_tts_to_slot": round(ratio, 4), "mode": mode, "atempo": atempo,
                          "aligned_seconds": round(len(y) / sr, 3), "file": str(aligned)}


def build_dubbed_track(units: list[dict], duration: float, work: pathlib.Path) -> dict:
    import numpy as np
    import soundfile as sf
    n = int(round(duration * TTS_SR)); track = np.zeros(n, dtype=np.float32)
    for u in units:
        y, sr = sf.read(u["alignment"]["file"], dtype="float32")
        a = int(round(u["slot"]["start"] * sr)); b = min(n, a + len(y))
        track[a:b] += y[: b - a]
    peak = float(np.abs(track).max()) if n else 0.0
    if peak > 0.99:
        track *= 0.99 / peak
    wav24 = work / "dubbed_24k.wav"; sf.write(str(wav24), track, TTS_SR, subtype="PCM_16")
    wav16 = work / "dubbed_16k.wav"; ff(["-i", str(wav24), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav16)])
    m4a = work / "dubbed_aac.m4a"; ff(["-i", str(wav24), "-c:a", "aac", "-b:a", "192k", str(m4a)])
    return {"wav24k": str(wav24), "wav16k": str(wav16), "aac": str(m4a), "seconds": round(n / TTS_SR, 3), "sr": TTS_SR,
            "peak_before_norm": round(peak, 4), "speech_seconds": round(sum(u["slot"]["seconds"] for u in units), 3),
            "non_speech": "silence (original timing preserved)"}


# --------------------------------------------------------------------------------------------------------------
# video stage: face-aware LatentSync driven by the dubbed track (functions from face_aware_latentsync.py)
# --------------------------------------------------------------------------------------------------------------
def run_lipsync(src: pathlib.Path, work: pathlib.Path, dubbed16k: pathlib.Path, dubbed_aac: pathlib.Path, out: pathlib.Path,
                min_ls_frames: int, max_verify_depth: int, times: dict) -> dict:
    import face_aware_latentsync as fa
    if str(fa.LS) not in sys.path:   # latentsync.* imports (FaceDetector) resolve from the clone root, as load_latentsync() does
        sys.path.insert(0, str(fa.LS))
    with timed(times, "ls_master_25fps"):
        master, _own = fa.make_master(src, work)
    with timed(times, "face_classify"):
        frames = fa.classify_frames(master)
    segs = fa.segment(frames, min_ls_frames)
    counts = {c: sum(1 for f in frames if f["cls"] == c) for c in ("VALID_FACE", "SMALL_FACE", "NO_FACE")}
    print(f"  [lipsync] {len(frames)} frames@25fps {counts} segments={len(segs)}", flush=True)
    with timed(times, "latentsync_load"):
        runner = fa.LatentSyncRunner()
    replacements, ls_runs, verify_info = {}, [], []
    queue = [s for s in segs if s["action"] == "LATENT_SYNC"]
    with timed(times, "latentsync"):
        while queue:
            s = queue.pop(0)
            try:
                v, au = fa.cut_segment(master, dubbed16k, s, work, f"ls{s['index']:03d}")   # audio = DUBBED track slice
                if s.get("verify_depth", 0) < max_verify_depth:
                    ok = fa.verify_segment(v); s["verify_depth"] = s.get("verify_depth", 0) + 1
                    if not all(ok):
                        subs = fa.split_by_mask(s, ok, min_ls_frames, next_index=len(segs))
                        verify_info.append({"segment": s["index"], "rejected_frames": ok.count(False), "split_into": [x["index"] for x in subs]})
                        print(f"  [lipsync] seg {s['index']}: {ok.count(False)}/{len(ok)} frames rejected by LatentSync detector -> split {len(subs)}", flush=True)
                        s["action"] = "SPLIT"; s["split_into"] = [x["index"] for x in subs]
                        for x in subs:
                            x["verify_depth"] = s["verify_depth"]; segs.append(x)
                            if x["action"] == "LATENT_SYNC":
                                queue.append(x)
                        continue
                o = work / f"seg{s['index']:03d}_ls_out.mp4"
                dt = runner(v, au, o, work / f"ls_temp_{s['index']}")
                frs = fa.read_frames(o); replacements[s["index"]] = frs
                s["latentsync"] = {"seconds": dt, "frames_in": s["frames"], "frames_out": len(frs)}
                ls_runs.append(dt); print(f"  [lipsync] seg {s['index']} {s['start_s']}-{s['end_s']}s LatentSync {dt}s -> {len(frs)} frames", flush=True)
            except Exception as e:   # e.g. upstream "Face not detected": this segment is passed through, the run continues
                s["action"] = "PASS_THROUGH_LS_FAILED"; s["error"] = f"{type(e).__name__}: {str(e)[:300]}"
                print(f"  [lipsync] seg {s['index']} LatentSync FAILED -> pass-through: {s['error']}", flush=True)
    segs[:] = sorted([s for s in segs if s["action"] != "SPLIT"], key=lambda s: s["start_frame"])
    with timed(times, "final_assembly"):
        asm = fa.assemble(master, dubbed_aac, segs, replacements, out, work)   # dubbed AAC is muxed as the audio track
    free_cuda(runner)
    by_action: dict[str, dict] = {}
    for s in segs:
        d = by_action.setdefault(s["action"], {"segments": 0, "frames": 0, "seconds": 0.0})
        d["segments"] += 1; d["frames"] += s["frames"]; d["seconds"] = round(d["seconds"] + s["frames"] / fa.FPS, 3)
    lipsynced = sum(s["frames"] for s in segs if s["action"] == "LATENT_SYNC")
    return {"master_fps": fa.FPS, "master_frames": len(frames), "frame_classes": counts, "by_action": by_action,
            "latentsync_seconds_of_video": round(lipsynced / fa.FPS, 3), "pass_through_seconds_of_video": round((len(frames) - lipsynced) / fa.FPS, 3),
            "latentsync_inference_total_s": round(sum(ls_runs), 2), "latentsync_calls": len(ls_runs), "verify_splits": verify_info,
            "assembly": asm, "config": {"stage2_512": True, "steps": runner.steps, "guidance": runner.guidance, "deepcache": True,
                                        "fp16": True, "seed": runner.seed, "min_ls_frames": min_ls_frames, "max_verify_depth": max_verify_depth},
            "segments": [{k: v for k, v in s.items() if k != "face_bbox_stats"} for s in segs]}


def run_lipsync_optimized(src: pathlib.Path, work: pathlib.Path, dubbed16k: pathlib.Path, dubbed_aac: pathlib.Path, out: pathlib.Path, a, times: dict) -> dict:
    """Optimized video backend: RetinaFace routing + batched LatentSync windows (face_aware_latentsync_accel.py).
    Same audio/video semantics as run_lipsync: pass-through keeps master frames, the dubbed track covers the whole timeline."""
    import face_aware_latentsync_accel as faa
    with timed(times, "face_router_load"):
        router = faa.make_router(a.face_router, a.face_min_w, a.face_min_h, a.retina_conf)
    with timed(times, "latentsync_load"):
        runner = faa.AccelRunner(a.window_batch_size, not a.no_deepcache, a.compile_backend, a.sdpa_backend)
    rep = faa.process_video(src, work, dubbed16k, dubbed_aac, out, router=router, runner=runner, min_ls_frames=a.min_ls_frames,
                            max_verify_depth=a.max_verify_depth, times=times, log=lambda m: print(m, flush=True))
    rep["backend"] = "optimized"
    free_cuda(runner)
    return rep


# --------------------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="source video")
    ap.add_argument("--output", required=True, help="final dubbed + lip-synced MP4")
    ap.add_argument("--target-lang", required=True, help="Chatterbox language id (en, de, ru, ...)")
    ap.add_argument("--no-codeformer", action="store_true", help="REQUIRED: pilot 0.5 measures the clean pipeline without CodeFormer")
    ap.add_argument("--source-lang", default=None, help="whisper language code; default: auto-detect")
    ap.add_argument("--work-dir", default=str(DEFAULT_WORK), help="runtime media root (work_<stem>/ inside it is recreated per run)")
    ap.add_argument("--max-unit-s", type=float, default=20.0, help="split speech units longer than this at the widest word gap")
    ap.add_argument("--min-ls-frames", type=int, default=25, help="VALID_FACE runs shorter than this are passed through")
    ap.add_argument("--max-verify-depth", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1247)
    ap.add_argument("--video-backend", choices=["baseline", "optimized"], default="baseline",
                    help="baseline = face_aware_latentsync.py functions (insightface classification, sequential windows); "
                         "optimized = face_aware_latentsync_accel.py (RetinaFace router, batched windows, SDPA/compile options)")
    ap.add_argument("--face-router", choices=["retinaface", "insightface"], default=OPTIMIZED_DEFAULTS["face_router"], help="optimized backend only")
    ap.add_argument("--face-min-w", type=int, default=50); ap.add_argument("--face-min-h", type=int, default=80); ap.add_argument("--retina-conf", type=float, default=0.8)
    ap.add_argument("--window-batch-size", type=int, default=OPTIMIZED_DEFAULTS["window_batch_size"], help="optimized backend only")
    ap.add_argument("--compile-backend", choices=["none", "inductor", "tensorrt"], default=OPTIMIZED_DEFAULTS["compile_backend"], help="optimized backend only")
    ap.add_argument("--sdpa-backend", choices=["auto", "flash", "efficient", "math"], default=OPTIMIZED_DEFAULTS["sdpa_backend"], help="optimized backend only")
    ap.add_argument("--no-deepcache", action="store_true", help="optimized backend only (A/B); baseline always uses DeepCache")
    a = ap.parse_args()
    if not a.no_codeformer:
        print("pilot 0.5 requires CodeFormer disabled: pass --no-codeformer", file=sys.stderr); return 2
    tgt = a.target_lang.lower()

    src = pathlib.Path(a.input).resolve(); out = pathlib.Path(a.output).resolve()
    if not src.exists():
        print(f"no such input: {src}", file=sys.stderr); return 2
    out.parent.mkdir(parents=True, exist_ok=True)
    work = pathlib.Path(a.work_dir) / f"work_{src.stem}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    times: dict = {}; t_all = time.perf_counter()
    man: dict = {"task": "0.5-runner", "input": str(src), "output": str(out), "target_lang": tgt, "codeformer": "DISABLED (not invoked)",
                 "work_dir": str(work), "config": {"seed": a.seed, "max_unit_s": a.max_unit_s, "threads": THREADS}}
    try:
        source = probe(src); man["source"] = source; duration = source["duration_s"]
        if duration <= 0 or not source["has_audio"]:
            raise Blocker("source has no duration/audio")
        print(f"== 0.5 clean pipeline: {src.name} {duration:.3f}s -> {tgt} (CodeFormer disabled)", flush=True)

        wav16 = work / "audio16k.wav"
        with timed(times, "audio_extract"):
            ff(["-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav16)])

        with timed(times, "vad"):
            speech = run_vad(wav16)
        timeline = build_timeline(speech, duration)
        man["speech_intervals"] = speech
        man["timeline"] = {"chunks": timeline, "speech_seconds": round(sum(c["duration_s"] for c in timeline if c["type"] == "SPEECH"), 3),
                           "pass_through_seconds": round(sum(c["duration_s"] for c in timeline if c["type"] == "PASS_THROUGH"), 3)}
        print(f"  [vad] {len(speech)} speech intervals, {man['timeline']['speech_seconds']}s speech", flush=True)

        with timed(times, "asr"):
            segments, asr_meta = run_asr(wav16, a.source_lang)
        src_lang = a.source_lang or asr_meta["language"]
        units = build_units(segments, duration, a.max_unit_s)
        man["asr"] = {**asr_meta, "model": "faster-whisper large-v3", "segments": segments}
        man["source_lang"] = src_lang
        print(f"  [asr] language={asr_meta['language']} p={asr_meta['language_probability']} segments={len(segments)} "
              f"words={sum(len(s['words']) for s in segments)} -> units={len(units)}", flush=True)
        if not units:
            raise Blocker("ASR produced no speech units; nothing to dub")
        if src_lang == tgt:
            print(f"  [warn] source language equals target language ({tgt}); translation is an identity direction", flush=True)

        with timed(times, "diarization"):
            turns = run_diarization(wav16)
        assign_speakers(units, turns)
        man["diarization"] = {"model": "pyannote/speaker-diarization-3.1", "turns": turns, "speakers": sorted({t["speaker"] for t in turns})}
        print(f"  [diarization] {len(turns)} turns, speakers={man['diarization']['speakers']}", flush=True)

        man["translation"] = run_translation(units, src_lang, tgt, times)

        refs = speaker_references(units, turns, wav16, work)
        man["tts"] = run_tts(units, refs, tgt, work, a.seed, times)

        with timed(times, "alignment"):
            align_units(units, duration, work)
        with timed(times, "dubbed_track"):
            man["dubbed_track"] = build_dubbed_track(units, duration, work)
        man["units"] = units
        print(f"  [align] ratios tts/slot: " + " ".join(f"{u['alignment']['ratio_tts_to_slot']:.2f}" for u in units), flush=True)

        man["video_backend"] = a.video_backend
        if a.video_backend == "optimized":
            man["lipsync"] = run_lipsync_optimized(src, work, pathlib.Path(man["dubbed_track"]["wav16k"]), pathlib.Path(man["dubbed_track"]["aac"]), out, a, times)
        else:
            man["lipsync"] = run_lipsync(src, work, pathlib.Path(man["dubbed_track"]["wav16k"]), pathlib.Path(man["dubbed_track"]["aac"]),
                                         out, a.min_ls_frames, a.max_verify_depth, times)
    except Blocker as e:
        man["blocker"] = str(e); man["stage_seconds"] = times
        write_json(out.with_name(out.stem + ".manifest.json"), man)
        print(f"BLOCKER: {e}", file=sys.stderr); return 3
    except Exception as e:
        man["error"] = f"{type(e).__name__}: {e}"; man["stage_seconds"] = times
        write_json(out.with_name(out.stem + ".manifest.json"), man)
        raise

    times["total"] = round(time.perf_counter() - t_all, 3)
    op = probe(out) if out.exists() else {}
    fps = source["video"].get("avg_fps") or source["video"].get("fps") or 25.0
    tol = max(0.15, 2.0 / fps)
    delta = round(op.get("duration_s", 0.0) - duration, 4) if op else None
    checks = {
        "output_exists_nonempty": out.exists() and out.stat().st_size > 0,
        "output_has_video_audio": bool(op) and op["has_video"] and op["has_audio"],
        "duration_within_tolerance": delta is not None and abs(delta) <= tol,
        "resolution_preserved": bool(op) and (op["video"]["width"], op["video"]["height"]) == (source["video"]["width"], source["video"]["height"]),
        "all_units_have_tts_and_translation": all("tts" in u and u.get("translation") for u in units),
        "frame_count_preserved": man["lipsync"]["assembly"]["frames_written"] == man["lipsync"]["master_frames"],
    }
    if man["lipsync"].get("output_frame_check"):
        checks["output_frames_decodable_no_blank"] = bool(man["lipsync"]["output_frame_check"]["pass"])
    man["output_meta"] = op
    man["validation"] = {"tolerance_s": round(tol, 4), "tolerance_rule": "max(0.15 s, 2 source frames)", "duration_delta_s": delta,
                         "checks": checks, "pass": all(checks.values())}
    man["stage_seconds"] = times; man["gpu"] = gpu_info(); man["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    mp = out.with_name(out.stem + ".manifest.json"); write_json(mp, man)

    print("\n== stage wall times (s)")
    for k, v in times.items():
        print(f"  {k:<20} {v:9.3f}")
    print(f"== output {out}  duration {op.get('duration_s')}s (source {duration}s, delta {delta:+.3f}s, tol {tol:.3f}s)")
    print(f"== validation {'PASS' if man['validation']['pass'] else 'FAIL'} {checks}\n-> {mp}")
    return 0 if man["validation"]["pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
