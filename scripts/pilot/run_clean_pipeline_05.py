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
   6. Qwen2.5-7B-Instruct Q8_0         real translation via llama.cpp, CPU only (n_gpu_layers=0, DUB_THREADS), the
                                       pilot 0.9 system prompt, temperature 0. Q8_0 = pilot 0.9 decision (2026-09-18):
                                       Q5_K_M is ~1.48x faster on CPU but Q8_0 was selected after manual review for
                                       more stable translation quality; q4_k_m (the old smoke quant) is no longer used
   7. Chatterbox Multilingual v3       real TTS per unit; the speaker's own diarized speech (<=10 s, cut from the source
                                       audio) is the voice reference via the upstream `audio_prompt_path` API;
                                       fp16 cast like scripts/check_chatterbox.py, fp32 fallback
   8. duration alignment               --alignment slot (default, frozen baseline): ffmpeg atempo (scripts/check_atempo.py
                                       path): TTS longer than its slot is sped up to fit exactly; TTS shorter than its
                                       slot keeps natural speed and is padded with silence -> the global timeline never moves
                                       --alignment burst (week 3): per unit, the source speech bursts (Silero VAD in the
                                       slot) get the source words by whisper word start, Qwen cuts the unit translation
                                       into exactly N validated parts (scripts/quality/burst_aware_audio.py), Chatterbox
                                       synthesises each part, and scripts/quality/e1_burst_align.place_groups fits every
                                       chunk into its burst (extension into the following silence, atempo cap / hard cap,
                                       optional spill into the pause before the next unit) -> the dubbed speech lies on
                                       the original speech; slots and the timeline still never move
   9. dubbed audio track               full source length; aligned TTS at the unit slots, silence elsewhere
  10. face-aware LatentSync 1.6        scripts/pilot/face_aware_latentsync.py functions: VALID_FACE segments ->
                                       LatentSync driven by the DUBBED audio, SMALL_FACE / NO_FACE -> pass-through,
                                       "Face not detected" never aborts the run; frame order / count preserved.
                                       --video-backend optimized (face_aware_latentsync_accel.py) additionally gates
                                       LatentSync to speech: frames outside VAD(original) U VAD(dubbed track) (+margin,
                                       merged gaps) keep the original mouth (nobody speaks there in either track),
                                       runs RetinaFace on every --router-stride-th frame, keeps segments in memory
                                       (LatentSync's detector runs once per frame) and crossfades LS/original boundaries
  11. CodeFormer                       --codeformer off (default, = --no-codeformer): NOT invoked, the frozen pilot baseline.
                                       --codeformer optimized (week 2): scripts/optim/codeformer_accel.py in-memory stage on the
                                       LatentSync output frames of LATENT_SYNC segments only (optimized backend), before the
                                       crossfade/assembly; geometry, frame count and duration unchanged
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
OPTIMIZED_DEFAULTS = {"face_router": "retinaface", "window_batch_size": 2, "compile_backend": "none", "sdpa_backend": "auto",
                      "deepcache_interval": 5, "speech_gate": "on", "gate_margin_s": 0.24, "gate_merge_gap_s": 0.6,
                      "crossfade_frames": 4, "router_stride": 3}
# speech gate / stride / in-memory segments: quality-neutral work reduction (2026-09-18); LatentSync's own detector still
# checks every frame it processes; the dubbed track always covers the whole timeline.
# DeepCache cache_interval 5 (was 3): A/B on 04.mp4 2026-09-18 (reports/pilot/optim/deepcache_interval_ab_04.md): SyncNet
# confidence 3.23 -> 3.21 (original 3.24), AV offset 0 both, PSNR 44.2 dB between outputs, UNet -27 %, LatentSync -20 %.
# --deepcache-interval 3 restores the previous setting.
# batch 2: largest batch that leaves VRAM headroom (25.8 GB peak vs 30.8 GB at batch 4 for +0.7 %); inductor gave no gain over
# DeepCache eager; "auto" SDPA already dispatches the fused FLASH_ATTENTION kernel (forced flash = bit-identical, same time).
QWEN_QUANT = "q8_0"       # pilot 0.9 decision: Q8_0 production quant (reports/pilot/0.9_qwen_quant.md), CPU, n_gpu_layers=0
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


def burst_plan(units: list[dict], segments: list[dict], speech: list[dict], duration: float, min_burst_s: float, spill: bool) -> list[dict]:
    """--alignment burst, step 1: per unit the source bursts (Silero VAD clipped to the slot, short ones merged), the whisper
    words of the unit assigned to bursts by word start, bursts without words dropped. Slot rule identical to align_units()."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "quality"))
    from burst_aware_audio import assign_words, merge_short_bursts
    from e1_burst_align import bursts_in_slot
    vad = [(v["start"], v["end"]) for v in speech]; seg_by_id = {sg["id"]: sg for sg in segments}; plans = []
    for i, u in enumerate(units):
        slot_s = u["start"]; slot_e = min(u["end"], duration, units[i + 1]["start"] - 0.02 if i + 1 < len(units) else duration); slot_e = max(slot_e, slot_s + 0.2)
        next_start = units[i + 1]["start"] if i + 1 < len(units) else duration
        bursts = merge_short_bursts(bursts_in_slot(vad, slot_s, slot_e), min_burst_s)
        sg = seg_by_id[u["asr_segment"]]; words = [w for w in sg["words"] if w["start"] >= u["start"] - 0.011 and w["end"] <= u["end"] + 0.011]
        groups = assign_words(words, bursts); keep = [(b, g) for b, g in zip(bursts, groups) if g] or [((slot_s, slot_e), words)]
        plans.append({"slot": (slot_s, slot_e), "next_start": next_start, "bursts": [b for b, _ in keep], "groups": [" ".join(w["word"].strip() for w in g) for _, g in keep],
                      "group_counts": [len(g) for _, g in keep], "bursts_without_words": len(bursts) - len(keep)})
    if spill:
        for p, q in zip(plans, plans[1:]):
            p["next_start"] = q["bursts"][0][0]
    return plans


def run_translation(units: list[dict], src_lang: str, tgt_lang: str, times: dict, plans: list | None = None, split_retries: int = 2,
                    min_part_words: int = 3, soft_part_words: int = 4, part_max_words: int = 16, keep_llm: bool = False):
    import glob
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "quality"))
    from burst_aware_audio import missing_numbers
    from llama_cpp import Llama
    files = sorted(glob.glob(str(MODELS / "qwen2.5-7b-instruct-gguf" / f"*{QWEN_QUANT}*.gguf")))
    if not files:
        raise Blocker(f"Qwen2.5-7B {QWEN_QUANT} GGUF not found under models/ (python scripts/pilot/fetch_09_qwen_quant.py {QWEN_QUANT} --yes)")
    with timed(times, "translation_load"):
        llm = Llama(model_path=files[0], n_ctx=4096, n_threads=THREADS, n_threads_batch=THREADS, n_gpu_layers=0, verbose=False)
    src_name, tgt_name = lang_name(src_lang), lang_name(tgt_lang)
    tgt_lang_name = tgt_name
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
            miss = missing_numbers(u["text"], txt)
            if miss:   # Qwen dropped a number ("... пятница 28" -> "today Friday"): one retry that names the missing numbers
                r2 = llm.create_chat_completion(msgs[:1] + [{"role": "user", "content": f"Translate from {src_name} to {tgt_lang_name}. Keep every number exactly as written in the source ({', '.join(miss)}):\n{u['text']}"}], max_tokens=512, temperature=0.0)
                t2 = " ".join(r2["choices"][0]["message"]["content"].split()).strip()
                if len(t2) >= 2 and t2[0] in "\"«“" and t2[-1] in "\"»”":
                    t2 = t2[1:-1].strip()
                for k in usage:
                    usage[k] += int(r2["usage"].get(k, 0))
                u["translation_number_retry"] = {"missing": miss, "first": txt, "second": t2, "fixed": bool(t2) and not missing_numbers(u["text"], t2)}
                if t2 and len(missing_numbers(u["text"], t2)) < len(miss):
                    txt = t2
            u["translation"] = txt
            for k in usage:
                usage[k] += int(r["usage"].get(k, 0))
            print(f"  [translate] u{u['id']:02d} {u['speaker']} {u['start']:.2f}-{u['end']:.2f}s | {u['text'][:60]} -> {txt[:60]}", flush=True)
    split_stats = None
    if plans is not None:   # --alignment burst, step 2: cut every unit translation into one part per source burst (Qwen still loaded)
        from burst_aware_audio import consolidate_parts, norm_tokens, qwen_split
        split_stats = {"single": 0, "qwen": 0, "single_fallback": 0, "merged_parts": 0, "parts_total": 0}
        with timed(times, "translation_split"):
            for u, p in zip(units, plans):
                if len(p["bursts"]) == 1:
                    u["parts"] = [u["translation"]]; u["split"] = {"method": "single"}; split_stats["single"] += 1; split_stats["parts_total"] += 1; continue
                parts, log = qwen_split(llm, p["groups"], u["translation"], src_name, tgt_name, split_retries)
                if parts is None:
                    # no valid Qwen cut: the old lossless PROPORTIONAL cut produced 1-2 word fragments ("what", "and will") -> synthesise the unit as ONE natural phrase over the span of all its bursts
                    parts = [u["translation"]]; log["method"] = "single_fallback"
                    p["bursts"] = [(p["bursts"][0][0], p["bursts"][-1][1])]; p["groups"] = [" ".join(p["groups"])]; p["group_counts"] = [sum(p["group_counts"])]
                else:
                    parts, p["bursts"], p["groups"], p["group_counts"], merges = consolidate_parts(parts, p["bursts"], p["groups"], p["group_counts"], min_part_words, soft_part_words, part_max_words)
                    if merges:
                        log["merges"] = merges; split_stats["merged_parts"] += len(merges)
                assert norm_tokens(" ".join(parts)) == norm_tokens(u["translation"]), "split lost words"
                assert len(parts) == len(p["bursts"]) == len(p["groups"]), "parts / bursts out of sync"
                u["parts"] = parts; u["split"] = log; split_stats[log["method"]] = split_stats.get(log["method"], 0) + 1; split_stats["parts_total"] += len(parts)
                print(f"  [split] u{u['id']:02d} {len(parts)} parts [{log['method']}{'+merged' if log.get('merges') else ''}]: " + " | ".join(f"{g} -> {t}" for g, t in zip(p["groups"], parts))[:300], flush=True)
    meta = {"model": os.path.basename(files[0]), "quant": QWEN_QUANT, "n_ctx": 4096, "n_threads": THREADS, "n_gpu_layers": 0,
            "system_prompt": SYSTEM_PROMPT, "direction": f"{src_name}->{tgt_name}", "usage": usage, "burst_split": split_stats}
    if keep_llm:
        return meta, llm, src_name, tgt_name    # the burst TTS session re-translates over-long parts with the same model (tts_burst.py), caller deletes it
    del llm
    return meta


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
    from chatterbox_fp16 import install_speech_token_guard
    if tgt_lang not in SUPPORTED_LANGUAGES:
        raise Blocker(f"target language {tgt_lang!r} not supported by Chatterbox: {sorted(SUPPORTED_LANGUAGES)}")
    with timed(times, "tts_load"):
        m = ChatterboxMultilingualTTS.from_local(MODELS / "chatterbox", "cuda", t3_model="v3")
    builtin = m.conds
    guard = install_speech_token_guard(m)   # keeps the speech-token ids int64 under the fp16 cast (chatterbox_fp16.py: fp16 corrupted them -> unintelligible speech)

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
                    m.conds = builtin.to(device="cuda"); guard.use_builtin()
            for u in us:
                if "parts" in u:   # --alignment burst: one clip per part (seed per part), keeps the slot-mode path below byte-identical
                    u["tts_parts"] = []
                    for k, text in enumerate(u["parts"]):
                        torch.manual_seed(seed + 100 * u["id"] + k)
                        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                            wav = m.generate(text, language_id=tgt_lang)
                        wav = wav.detach().float().cpu()
                        if not torch.isfinite(wav).all() or wav.abs().max() < 1e-4:
                            raise RuntimeError(f"non-finite or silent TTS for unit {u['id']} part {k}")
                        p = work / f"tts_u{u['id']:02d}_p{k}.wav"; torchaudio.save(str(p), wav, m.sr)
                        u["tts_parts"].append({"file": str(p), "seconds": round(wav.shape[-1] / m.sr, 3), "text": text})
                    u["tts"] = {"file": None, "parts": len(u["parts"]), "seconds": round(sum(t["seconds"] for t in u["tts_parts"]), 3), "sr": m.sr,
                                "reference": refs[spk]["source"], "dtype": str(dtype).replace("torch.", "")}
                    print(f"  [tts] u{u['id']:02d} {spk} {len(u['parts'])} parts {u['tts']['seconds']:.2f}s", flush=True)
                    continue
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
            "speech_token_guard": {"enabled": True, "flow_calls": guard.calls, "violations": guard.violations}, "references": refs}


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


def align_units_burst(units: list[dict], plans: list[dict], work: pathlib.Path, opts) -> dict:
    """--alignment burst, step 3: trim every part to its Silero speech span and fit the chunks into their bursts with
    e1_burst_align.place_groups (shared with the E1 / burst_aware_audio candidates). Slot fields as in align_units()."""
    import numpy as np
    import soundfile as sf
    from e1_burst_align import SR16, TTS_SR as E1_SR, burst_fit_ratios, place_groups, place_slot_fallback, placement_summary, vad_intervals
    report = []; n_fallback = 0; prev_end = None
    for u, p in zip(units, plans):
        chunks = []
        for k, t in enumerate(u["tts_parts"]):
            y, sr = sf.read(t["file"], dtype="float32"); assert sr == E1_SR
            if y.ndim > 1: y = y.mean(axis=1)
            if t.get("speech_span"):   # already computed (and possibly re-synthesised) by the burst TTS session: same rule, same clip
                s_, e_ = t["speech_span"]
            else:
                from tts_burst import speech_span
                s_, e_ = speech_span(y, sr, work, f"u{u['id']:02d}_p{k}"); t["speech_span"] = [round(s_, 3), round(e_, 3)]
            chunks.append(y[int(s_ * sr):int(e_ * sr)])
        slot_s, slot_e = p["slot"]
        need = burst_fit_ratios(chunks, p["bursts"], p["slot"], p["next_start"], opts, prev_end=prev_end); over = [r for r in need if r is not None and r > opts.atempo_cap]
        if len(need) > 1 and len(over) >= getattr(opts, "fallback_frac", 0.5) * len([r for r in need if r is not None]):
            track, placements = place_slot_fallback(chunks, p["slot"], work, f"u{u['id']:02d}"); n_fallback += 1
            placements[0]["text"] = " ".join(t["text"] for t in u["tts_parts"]); placements[0]["source_words"] = " | ".join(p["groups"]); placements[0]["burst_fit_ratios"] = need
            print(f"  [align] u{u['id']:02d} slot fallback ({len(over)}/{len(need)} groups would need > {opts.atempo_cap}x)", flush=True)
        else:
            track, placements = place_groups(chunks, p["bursts"], p["slot"], p["next_start"], opts, work, f"u{u['id']:02d}", sr=E1_SR, labels=[[k] for k in range(len(chunks))], prev_end=prev_end)
            for pl, t, g in zip(placements, u["tts_parts"], p["groups"]):
                pl["text"] = t["text"]; pl["source_words"] = g
        aligned = work / f"aligned_u{u['id']:02d}.wav"; sf.write(str(aligned), track, E1_SR, subtype="PCM_16")
        u["slot"] = {"start": round(slot_s, 3), "end": round(slot_e, 3), "seconds": round(slot_e - slot_s, 3)}
        u["alignment"] = {"tts_seconds": u["tts"]["seconds"], "ratio_tts_to_slot": round(u["tts"]["seconds"] / max(slot_e - slot_s, 1e-6), 4), "mode": "burst",
                          "atempo": max(pl.get("atempo", 1.0) for pl in placements), "bursts": len(p["bursts"]), "bursts_without_words": p["bursts_without_words"],
                          "placements": placements, "aligned_seconds": round(len(track) / E1_SR, 3), "file": str(aligned),
                          "track_start": next((pl["track_start"] for pl in placements if "track_start" in pl), round(slot_s, 3))}
        placed_ends = [pl["placed"][1] for pl in placements if pl.get("placed")]
        prev_end = max(placed_ends) if placed_ends else prev_end
        report.append({"id": u["id"], "placements": placements})
    return {"mode": "burst", "min_burst_s": getattr(opts, "min_burst_s", None), "spill": opts.spill, "atempo_cap": opts.atempo_cap, "hard_cap": opts.hard_cap,
            "fill_slowdown": opts.fill_slowdown, "fallback_frac": getattr(opts, "fallback_frac", 0.5), "units_slot_fallback": n_fallback, "summary": placement_summary(report)}


def build_dubbed_track(units: list[dict], duration: float, work: pathlib.Path) -> dict:
    import numpy as np
    import soundfile as sf
    n = int(round(duration * TTS_SR)); track = np.zeros(n, dtype=np.float32)
    for u in units:
        y, sr = sf.read(u["alignment"]["file"], dtype="float32")
        a = max(0, int(round(u["alignment"].get("track_start", u["slot"]["start"]) * sr))); b = min(n, a + len(y))   # burst mode may start a short phrase up to --burst-max-lead before its burst
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


def run_lipsync_optimized(src: pathlib.Path, work: pathlib.Path, dubbed16k: pathlib.Path, dubbed_aac: pathlib.Path, out: pathlib.Path, a, times: dict,
                          speech_gate: list | None = None) -> dict:
    """Optimized video backend: RetinaFace routing + speech gate + batched LatentSync windows (face_aware_latentsync_accel.py).
    Same audio/video semantics as run_lipsync: pass-through keeps master frames, the dubbed track covers the whole timeline."""
    import face_aware_latentsync_accel as faa
    with timed(times, "face_router_load"):
        router = faa.make_router(a.face_router, a.face_min_w, a.face_min_h, a.retina_conf)
    with timed(times, "latentsync_load"):
        runner = faa.AccelRunner(a.window_batch_size, not a.no_deepcache, a.compile_backend, a.sdpa_backend, deepcache_interval=a.deepcache_interval)
    cf_runner = None
    if a.codeformer == "optimized":
        from codeformer_accel import CodeFormerAccel
        with timed(times, "codeformer_load"):
            cf_runner = CodeFormerAccel(fidelity_weight=a.codeformer_w, batch_size=a.codeformer_batch, landmark_source=a.codeformer_landmarks,
                                        autocast=a.codeformer_precision == "fp16", parse_autocast=a.codeformer_precision != "fp32")
    rep = faa.process_video(src, work, dubbed16k, dubbed_aac, out, router=router, runner=runner, min_ls_frames=a.min_ls_frames,
                            max_verify_depth=a.max_verify_depth, codeformer=a.codeformer, times=times, log=lambda m: print(m, flush=True),
                            speech_gate=speech_gate, crossfade_frames=a.crossfade_frames, router_stride=a.router_stride,
                            in_memory=not a.segment_files, cf_runner=cf_runner)
    rep["backend"] = "optimized"
    if cf_runner is not None:
        cf_runner.close()
    free_cuda(runner, cf_runner)
    return rep


# --------------------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="source video")
    ap.add_argument("--output", required=True, help="final dubbed + lip-synced MP4")
    ap.add_argument("--target-lang", required=True, help="Chatterbox language id (en, de, ru, ...)")
    ap.add_argument("--no-codeformer", action="store_true", help="pilot 0.5 contract: the clean pipeline without CodeFormer (= --codeformer off)")
    ap.add_argument("--codeformer", choices=["off", "optimized"], default=None,
                    help="off = frozen baseline (never invoked); optimized = week-2 in-memory CodeFormer on LATENT_SYNC frames (optimized backend only)")
    ap.add_argument("--codeformer-w", type=float, default=1.0, help="CodeFormer fidelity weight w (optimized mode); 1.0 = week-2 A/B choice on 04 (smallest SyncNet drop, AV offset 0)")
    ap.add_argument("--codeformer-batch", type=int, default=8, help="faces per CodeFormer batch (optimized mode); 8 = week-2 choice (5 GB alone, no E2E peak increase)")
    ap.add_argument("--codeformer-landmarks", choices=["insightface", "retinaface"], default="insightface",
                    help="optimized mode: align with LatentSync's insightface landmarks (no extra detector) or a fresh RetinaFace pass per frame (A/B)")
    ap.add_argument("--codeformer-precision", choices=["fp32", "parse16", "fp16"], default="fp16",
                    help="optimized mode: fp32 = upstream numerics; parse16 = ParseNet under fp16 autocast; fp16 = CodeFormer net + ParseNet under autocast")
    ap.add_argument("--source-lang", default=None, help="whisper language code; default: auto-detect")
    ap.add_argument("--work-dir", default=str(DEFAULT_WORK), help="runtime media root (work_<stem>/ inside it is recreated per run)")
    ap.add_argument("--max-unit-s", type=float, default=20.0, help="split speech units longer than this at the widest word gap")
    ap.add_argument("--alignment", choices=["slot", "burst"], default="slot", help="slot = frozen baseline aligner; burst = week-3 burst-aware translation split / TTS / placement")
    ap.add_argument("--burst-min-s", type=float, default=0.0, help="burst mode: merge source bursts shorter than this into a neighbour")
    ap.add_argument("--burst-spill", choices=["on", "off"], default="on", help="burst mode: last chunk may run into the pause before the next unit (hard cap before any cut)")
    ap.add_argument("--atempo-cap", type=float, default=1.15, help="burst mode: preferred max speed-up of a part (intelligibility degrades above ~1.15-1.2, first_res review)")
    ap.add_argument("--hard-cap", type=float, default=1.2, help="burst mode: absolute max speed-up (used only when a part would collide with the next burst / be cut at the track end)")
    ap.add_argument("--fill-slowdown", type=float, default=1.0)
    ap.add_argument("--split-retries", type=int, default=2, help="burst mode: Qwen JSON split retries before falling back to ONE natural phrase per unit")
    ap.add_argument("--min-part-words", type=int, default=3, help="burst mode: hard minimum words of a TTS part (shorter parts are merged into a neighbour)")
    ap.add_argument("--soft-part-words", type=int, default=4, help="burst mode: preferred minimum words; parts below it are merged while the merged part stays <= --part-max-words")
    ap.add_argument("--part-max-words", type=int, default=16)
    ap.add_argument("--burst-fallback-frac", type=float, default=1.01, help="burst mode: a unit whose fit needs > atempo-cap on >= this fraction of its groups uses the baseline slot rule, which speeds up WITHOUT a cap (1.01 = never; default since the first_res review)")
    ap.add_argument("--fit-retranslate", choices=["on", "off"], default="on", help="burst mode: a part that would need more than --atempo-cap is re-translated concisely (Qwen) and re-synthesised before any speed-up")
    ap.add_argument("--fit-rounds", type=int, default=4); ap.add_argument("--fit-target-ratio", type=float, default=1.05)
    ap.add_argument("--burst-max-lead", type=float, default=0.4, help="burst mode: a part that would need more than --atempo-cap may start up to this many seconds before its burst, into free silence, instead of being sped up")
    ap.add_argument("--tts-qa", choices=["on", "off"], default="off", help="burst mode: independent-ASR QA of every TTS part with per-part retry (English targets only)")
    ap.add_argument("--tts-retries", type=int, default=2, help="QA retry ladder length after the first attempt: new seed, then new seed + --qa-temp")
    ap.add_argument("--qa-temp", type=float, default=0.6)
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
    ap.add_argument("--deepcache-interval", type=int, default=OPTIMIZED_DEFAULTS["deepcache_interval"], help="optimized backend only; 5 = A/B-validated default, 3 = previous setting")
    ap.add_argument("--speech-gate", choices=["on", "off"], default=OPTIMIZED_DEFAULTS["speech_gate"],
                    help="optimized backend only: LatentSync only inside VAD(original) U VAD(dubbed) (+margin); elsewhere the original mouth is kept")
    ap.add_argument("--gate-margin-s", type=float, default=OPTIMIZED_DEFAULTS["gate_margin_s"])
    ap.add_argument("--gate-merge-gap-s", type=float, default=OPTIMIZED_DEFAULTS["gate_merge_gap_s"])
    ap.add_argument("--crossfade-frames", type=int, default=OPTIMIZED_DEFAULTS["crossfade_frames"], help="LS/original boundary blend length (0 = off)")
    ap.add_argument("--router-stride", type=int, default=OPTIMIZED_DEFAULTS["router_stride"], help="RetinaFace on every N-th gated frame")
    ap.add_argument("--segment-files", action="store_true", help="optimized backend: legacy per-segment mp4 path instead of in-memory segments")
    a = ap.parse_args()
    if a.codeformer is None:
        a.codeformer = "off" if a.no_codeformer else None
    if a.codeformer is None:
        print("CodeFormer mode not chosen: pass --no-codeformer (= --codeformer off, pilot baseline) or --codeformer optimized", file=sys.stderr); return 2
    if a.no_codeformer and a.codeformer != "off":
        print("--no-codeformer contradicts --codeformer optimized", file=sys.stderr); return 2
    if a.codeformer != "off" and a.video_backend != "optimized":
        print("--codeformer optimized needs --video-backend optimized", file=sys.stderr); return 2
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
    man: dict = {"task": "0.5-runner", "input": str(src), "output": str(out), "target_lang": tgt, "alignment_mode": a.alignment,
                 "codeformer": "DISABLED (not invoked)" if a.codeformer == "off" else {"mode": a.codeformer, "w": a.codeformer_w, "batch": a.codeformer_batch, "landmarks": a.codeformer_landmarks, "precision": a.codeformer_precision},
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

        plans = None
        if a.alignment == "burst":
            plans = burst_plan(units, segments, speech, duration, a.burst_min_s, a.burst_spill == "on")
            print(f"  [burst] {sum(len(p['bursts']) for p in plans)} bursts with words in {len(units)} units, {sum(p['bursts_without_words'] for p in plans)} bursts without words", flush=True)
        burst_opts = None
        if a.alignment == "burst":
            sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "quality"))
            from e1_burst_align import PlaceOpts
            burst_opts = PlaceOpts(a.atempo_cap, a.hard_cap, True, 0.06, a.fill_slowdown, spill=a.burst_spill == "on", max_lead=a.burst_max_lead); burst_opts.min_burst_s = a.burst_min_s; burst_opts.fallback_frac = a.burst_fallback_frac
        tr = run_translation(units, src_lang, tgt, times, plans=plans, split_retries=a.split_retries,
                             min_part_words=a.min_part_words, soft_part_words=a.soft_part_words, part_max_words=a.part_max_words, keep_llm=burst_opts is not None and a.fit_retranslate == "on")
        llm = None
        if isinstance(tr, tuple):
            man["translation"], llm, src_name, tgt_name = tr
        else:
            man["translation"] = tr

        refs = speaker_references(units, turns, wav16, work)
        if a.alignment == "burst":
            from tts_burst import run_tts_burst
            qa_on = a.tts_qa == "on"
            if qa_on and tgt != "en":
                print(f"  [warn] --tts-qa uses English ASR models; disabled for target language {tgt}", flush=True); qa_on = False
            man["tts"] = run_tts_burst(units, plans, refs, tgt, work, a.seed, times, opts=burst_opts, llm=llm, src_name=lang_name(src_lang), tgt_name=lang_name(tgt),
                                       fit={"enabled": llm is not None, "rounds": a.fit_rounds, "target_ratio": a.fit_target_ratio},
                                       qa_cfg={"enabled": qa_on, "retries": a.tts_retries, "low_temp": a.qa_temp}, min_part_words=a.min_part_words,
                                       log=lambda m_: print(m_, flush=True))
            llm = tr = None    # release Qwen (the translation tuple also holds a reference)
        else:
            man["tts"] = run_tts(units, refs, tgt, work, a.seed, times)

        with timed(times, "alignment"):
            if a.alignment == "burst":
                man["alignment"] = align_units_burst(units, plans, work, burst_opts)
            else:
                align_units(units, duration, work); man["alignment"] = {"mode": "slot"}
        with timed(times, "dubbed_track"):
            man["dubbed_track"] = build_dubbed_track(units, duration, work)
        man["units"] = units
        print(f"  [align] ratios tts/slot: " + " ".join(f"{u['alignment']['ratio_tts_to_slot']:.2f}" for u in units), flush=True)

        man["video_backend"] = a.video_backend
        if a.video_backend == "optimized":
            speech_gate = None
            if a.speech_gate == "on":
                import face_aware_latentsync_accel as faa
                with timed(times, "speech_gate"):
                    dub_speech = run_vad(pathlib.Path(man["dubbed_track"]["wav16k"]))
                    speech_gate = faa.build_speech_gate([speech, dub_speech], duration, a.gate_margin_s, a.gate_merge_gap_s)
                gated = sum(e - s for s, e in speech_gate)
                man["speech_gate"] = {"enabled": True, "margin_s": a.gate_margin_s, "merge_gap_s": a.gate_merge_gap_s, "crossfade_frames": a.crossfade_frames,
                                      "source_speech_seconds": round(sum(i["end"] - i["start"] for i in speech), 3),
                                      "dubbed_speech_seconds": round(sum(i["end"] - i["start"] for i in dub_speech), 3),
                                      "gate_seconds": round(gated, 3), "gate_fraction": round(gated / duration, 4), "intervals": speech_gate}
                print(f"  [gate] {len(speech_gate)} intervals, {gated:.1f}s of {duration:.1f}s ({gated / duration:.0%}) eligible for LatentSync", flush=True)
            else:
                man["speech_gate"] = {"enabled": False}
            man["lipsync"] = run_lipsync_optimized(src, work, pathlib.Path(man["dubbed_track"]["wav16k"]), pathlib.Path(man["dubbed_track"]["aac"]), out, a, times,
                                                   speech_gate=speech_gate)
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
