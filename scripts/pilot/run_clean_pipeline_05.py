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
                                       AAC track (--background on: dubbed voice + the source background bed, background_bed.py) -> output MP4 on the full original timeline; no subtitles (the 0.5
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
import re
import json
import os
import pathlib
import shutil
import subprocess
import sys
import time

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")   # pyannote 3.x checkpoints, as in scripts/check_pyannote.py
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import MODELS, QWEN_THREADS, ROOT, THREADS, dev, empty_cache, gpu_info, tts_dtypes, whisper_compute, write_json  # noqa: E402
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


# --audio-policy (2026-10-17, Agent C r2): flag bundles of the audio stage. A flag passed explicitly on the command line always wins over the bundle.
#   legacy = the argparse defaults below (pre-cand5 behaviour); cand5 = the 2026-10-13 production-candidate audio flags;
#   r2     = cand5 + layer-1 timing (flow 0.2 + spread pauses) + slow-down floor 0.875 (reports/quality/audio_timing/p1_timing_ab.md, r2/CHANGES.md)
AUDIO_POLICY_CAND5 = {"phrase_cut": "acoustic", "long_pause_func": 1.2, "cut_fade": "adaptive", "tts_gap_qa": "on"}
AUDIO_POLICIES = {"legacy": {}, "cand5": dict(AUDIO_POLICY_CAND5),
                  "r2": {**AUDIO_POLICY_CAND5, "sentence_flow_gap": 0.2, "spread": "pauses", "fill_slowdown": 0.875}}


def apply_audio_policy(a, argv: list[str]) -> dict:
    """Set the policy's values on every option NOT given explicitly in argv; returns the record for the manifest."""
    explicit = {}; applied = {}
    for k, v in AUDIO_POLICIES[a.audio_policy].items():
        flag = "--" + k.replace("_", "-")
        if any(x == flag or x.startswith(flag + "=") for x in argv):
            explicit[k] = getattr(a, k)
        else:
            setattr(a, k, v); applied[k] = v
    return {"name": a.audio_policy, "applied": applied, "explicit_overrides": explicit}


def lang_name(code: str) -> str:
    return LANG_NAMES.get(code.lower(), code)


def free_cuda(*objs) -> None:
    import torch
    for o in objs:
        del o
    gc.collect()
    empty_cache()


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
    import model_cache
    model = model_cache.get("silero_vad", load_silero_vad)
    audio = read_audio(str(wav16), sampling_rate=16000)
    ts = get_speech_timestamps(audio, model, sampling_rate=16000, return_seconds=True)
    return [{"start": round(float(t["start"]), 3), "end": round(float(t["end"]), 3)} for t in ts]


def run_asr(wav16: pathlib.Path, language: str | None) -> tuple[list[dict], dict]:
    import torch  # noqa: F401  maps libcublas before ctranslate2 loads (scripts/check_whisper.py)
    from faster_whisper import WhisperModel
    import model_cache
    m = model_cache.get("whisper_asr", lambda: WhisperModel(str(MODELS / "whisper-large-v3"), device=dev(), compute_type=whisper_compute("int8_float16")))
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
    import model_cache
    def _load():
        p_ = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=tok)
        if p_ is not None:
            p_.to(torch.device(dev()))
        return p_
    pipe = model_cache.get("pyannote", _load)
    if pipe is None:
        raise Blocker("pyannote pipeline is None: model licenses not accepted for this token")
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
        words = sorted((w for sg in segments for w in sg["words"] if w["start"] >= u["start"] - 0.011 and w["end"] <= u["end"] + 0.011 and re.findall(r"[\w']+", w["word"])), key=lambda w: (w["start"], w["end"]))
        groups = assign_words(words, bursts); keep = [(b, g) for b, g in zip(bursts, groups) if g] or [((slot_s, slot_e), words)]
        plans.append({"slot": (slot_s, slot_e), "next_start": next_start, "bursts": [b for b, _ in keep], "groups": [" ".join(w["word"].strip() for w in g) for _, g in keep],
                      "group_counts": [len(g) for _, g in keep], "bursts_without_words": len(bursts) - len(keep)})
    if spill:
        for p, q in zip(plans, plans[1:]):
            p["next_start"] = q["bursts"][0][0]
    return plans


QWEN_GPU_LAYERS = 0       # --qwen-gpu-layers (A/B 2026-10-12): needs a CUDA build of llama-cpp-python first on PYTHONPATH; 0 = frozen CPU baseline


def load_llm(times: dict):
    """Qwen2.5-7B Q8_0 via llama.cpp (CPU), loaded once and shared by sentence segmentation, translation, burst split and the fit rewrites."""
    import glob
    from llama_cpp import Llama
    files = sorted(glob.glob(str(MODELS / "qwen2.5-7b-instruct-gguf" / f"*{QWEN_QUANT}*.gguf")))
    if not files:
        raise Blocker(f"Qwen2.5-7B {QWEN_QUANT} GGUF not found under models/ (python scripts/pilot/fetch_09_qwen_quant.py {QWEN_QUANT} --yes)")
    with timed(times, "translation_load"):
        llm = Llama(model_path=files[0], n_ctx=4096, n_threads=QWEN_THREADS, n_threads_batch=QWEN_THREADS, n_gpu_layers=QWEN_GPU_LAYERS, verbose=False)
    return llm, os.path.basename(files[0])


ORDER_SHOTS = [   # generic examples, deliberately NOT taken from the five test videos (no leakage into the evaluation)
    ("в прошлом году | мы с братом поехали | на море.", "Last year | my brother and I went | to the sea."),
    ("книгу эту | я прочитал | за два дня.", "This book | I read | in just two days."),
    ("ну что | погнали | дальше.", "Well then, | let's move | on."),
    ("мы приедем 5 | июня в пятницу.", "We will arrive on the 5th | of June, on Friday."),
    ("короче | всё прошло | нормально.", "Anyway, | everything went | fine."),
    ("большая | светлая | комната и балкончик.", "A big, | bright | room and a small balcony.")]


def _order_instr(src_name: str, tgt_name: str, k: int) -> str:
    return (f"Translate from {src_name} to {tgt_name} for dubbing. The speaker pauses at every \" | \". Write a natural, fluent {tgt_name} translation "
            f"of the whole sentence with exactly {k - 1} \" | \" mark{'s' if k > 2 else ''}, placed so that each {tgt_name} piece says what the "
            f"{src_name} piece at the same position says (keep the order of the pieces). Translate the meaning, not word for word; keep the punctuation. "
            f"Answer with the marked translation only.")


def ordered_translation(llm, src_name: str, tgt_name: str, pieces: list[str]) -> tuple[list[str] | None, dict]:
    """--ordered-translation (2026-10-13, 04 11-12 s "какой-то бред"): the source sentence is sent with " | " at its long pauses and Qwen keeps
    the pieces in order, so the dub says at every source burst what the speaker says there. Plain translation of 04 u2 "город | сочи 28 |
    августа сегодня и | мы начинаем ..." was "Sochi city today, August 28, and we start ...": the cut DP then put "today" on "сочи 28" and
    "August 28" on "августа сегодня". One repair turn when the answer has another number of pieces. Returns (pieces | None, usage + raw)."""
    k = len(pieces); msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for a_, b_ in ORDER_SHOTS:
        msgs += [{"role": "user", "content": f"{_order_instr(src_name, tgt_name, a_.count('|') + 1)}\n{a_}"}, {"role": "assistant", "content": b_}]
    msgs.append({"role": "user", "content": f"{_order_instr(src_name, tgt_name, k)}\n{' | '.join(pieces)}"})
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "raw": []}
    for attempt in range(2):
        r = llm.create_chat_completion(msgs, max_tokens=512, temperature=0.0)
        out = " ".join(r["choices"][0]["message"]["content"].split()).strip().strip('"«»“”')
        for key in ("prompt_tokens", "completion_tokens"):
            usage[key] += int(r["usage"].get(key, 0))
        usage["raw"].append(out); got = [x.strip() for x in out.split("|")]
        if len(got) == k and all(re.findall(r"[\w']+", x) for x in got):
            return got, usage
        msgs += [{"role": "assistant", "content": out},
                 {"role": "user", "content": f"That has {len(got)} pieces; it must have exactly {k} pieces ({k - 1} \" | \" marks), one per source piece. Answer again."}]
    return None, usage


def source_pieces(u: dict, p: dict, gap: float) -> list[str]:
    """Source words of unit u grouped by burst, bursts closer than `gap` s joined (the same rule as tts_burst.flow_merge_bursts)."""
    toks = u["text"].split(); counts = list(p["group_counts"])
    groups = [" ".join(toks[sum(counts[:i]):sum(counts[:i + 1])]) for i in range(len(counts))] if len(toks) == sum(counts) else list(p["groups"])
    out = [groups[0]]
    for prev, cur, g in zip(p["bursts"], p["bursts"][1:], groups[1:]):
        if cur[0] - prev[1] < gap: out[-1] += " " + g
        else: out.append(g)
    return out


def run_translation(units: list[dict], src_lang: str, tgt_lang: str, times: dict, plans: list | None = None, split_retries: int = 2,
                    min_part_words: int = 3, soft_part_words: int = 4, part_max_words: int = 16, keep_llm: bool = False, llm=None, split: bool = True, ordered_gap: float = 0.0):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "quality"))
    from burst_aware_audio import missing_numbers
    if llm is None:
        llm, model_name = load_llm(times)
    else:
        model_name = getattr(llm, "model_path", "qwen2.5-7b-instruct") if isinstance(getattr(llm, "model_path", None), str) else "qwen2.5-7b-instruct"
        model_name = os.path.basename(model_name)
    files = [model_name]
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
            if tgt_lang == "en":   # r3: Qwen drift guard (foreign script / bracketed asides) -> one strict retry, then strip as the last resort
                from tts_text import script_defects, strip_defects
                bad = script_defects(txt)
                if bad:
                    r3 = llm.create_chat_completion(msgs[:1] + [{"role": "user", "content": f"Translate from {src_name} to {tgt_name}. Write only {tgt_name} words in the Latin alphabet, "
                                                                 f"no brackets, no notes, no other languages:\n{u['text']}"}], max_tokens=512, temperature=0.0)
                    t3 = " ".join(r3["choices"][0]["message"]["content"].split()).strip().strip("\"«“»”")
                    for k in usage:
                        usage[k] += int(r3["usage"].get(k, 0))
                    ok = bool(t3) and not script_defects(t3) and len(missing_numbers(u["text"], t3)) <= len(missing_numbers(u["text"], txt))
                    u["translation_script_retry"] = {"defects": bad, "first": txt, "second": t3, "used": "second" if ok else "stripped"}
                    txt = t3 if ok else strip_defects(txt)
                    print(f"  [translate] u{u['id']:02d} drift guard {bad} -> {u['translation_script_retry']['used']}", flush=True)
            u["translation"] = txt
            for k in usage:
                usage[k] += int(r["usage"].get(k, 0))
            if ordered_gap > 0 and plans is not None:
                pcs = source_pieces(u, plans[units.index(u)], ordered_gap)
                if len(pcs) >= 2:
                    got, us = ordered_translation(llm, src_name, tgt_name, pcs); joined = " ".join(got) if got else ""
                    for k in usage:
                        usage[k] += int(us.get(k, 0))
                    why = ("malformed" if got is None else "lost a number" if len(missing_numbers(u["text"], joined)) > len(missing_numbers(u["text"], txt))
                           else "length" if not 0.6 <= len(joined.split()) / max(len(txt.split()), 1) <= 1.6 else None)
                    u["ordered_translation"] = {"source": pcs, "pieces": got, "plain": txt, "used": why is None, "reject": why, "raw": us.get("raw")}
                    if why is None:
                        u["translation"] = txt = joined; u["ordered_pieces"] = got
                    print(f"  [ordered] u{u['id']:02d} {' | '.join(pcs)[:70]} -> {' | '.join(got or [])[:70]} ({why or 'used'})", flush=True)
            print(f"  [translate] u{u['id']:02d} {u['speaker']} {u['start']:.2f}-{u['end']:.2f}s | {u['text'][:60]} -> {txt[:60]}", flush=True)
    split_stats = None
    if plans is not None and split:   # --alignment burst, step 2: cut every unit translation into one part per source burst (Qwen still loaded)
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
    meta = {"model": os.path.basename(files[0]), "quant": QWEN_QUANT, "n_ctx": 4096, "n_threads": QWEN_THREADS, "n_gpu_layers": QWEN_GPU_LAYERS,
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
        m = ChatterboxMultilingualTTS.from_local(MODELS / "chatterbox", dev(), t3_model="v3")
    builtin = m.conds
    guard = install_speech_token_guard(m)   # keeps the speech-token ids int64 under the fp16 cast (chatterbox_fp16.py: fp16 corrupted them -> unintelligible speech)

    def cast(dtype):
        for mod in (m.t3, m.s3gen, m.ve):
            mod.to(dtype)
        if m.conds is not None:
            m.conds = m.conds.to(device=dev())

    def synth_all(dtype) -> None:
        cast(dtype)
        by_spk: dict[str, list[dict]] = {}
        for u in units:
            by_spk.setdefault(u["speaker"], []).append(u)
        for spk, us in by_spk.items():
            ref = refs[spk]["file"]
            with torch.autocast(dev(), dtype=dtype, enabled=dtype != torch.float32):
                if ref:
                    m.prepare_conditionals(ref)
                else:
                    m.conds = builtin.to(device=dev()); guard.use_builtin()
            for u in us:
                if "parts" in u:   # --alignment burst: one clip per part (seed per part), keeps the slot-mode path below byte-identical
                    u["tts_parts"] = []
                    for k, text in enumerate(u["parts"]):
                        torch.manual_seed(seed + 100 * u["id"] + k)
                        with torch.autocast(dev(), dtype=dtype, enabled=dtype != torch.float32):
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
                with torch.autocast(dev(), dtype=dtype, enabled=dtype != torch.float32):
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
        for dtype in tts_dtypes(torch):
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
    spread_cfg = getattr(opts, "spread", None); spread_cfg = spread_cfg if spread_cfg and spread_cfg.get("mode", "off") != "off" else None; spread_log = []
    if spread_cfg:   # --spread (2026-10-16, Agent C): leftover silence of a part shorter than its source burst -> late start + pauses at phrase boundaries
        import timing_spread
        from tts_burst import CUT_FUNC_WORDS, FIXED_PHRASES
        from tts_qa import norm as tts_norm
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
        prev_end = max(placed_ends) if placed_ends else prev_end      # before --spread: the next unit's lead room is that of the plain placement
        if spread_cfg and u.get("tts_raw_file") and (u.get("tts") or {}).get("word_spans"):
            y_raw, _ = sf.read(u["tts_raw_file"], dtype="float32")
            track2, lg = timing_spread.spread_unit(u, track, E1_SR, y_raw, spread_cfg, tts_norm, CUT_FUNC_WORDS, FIXED_PHRASES)
            if lg:
                sf.write(str(aligned), track2, E1_SR, subtype="PCM_16"); spread_log += lg
        report.append({"id": u["id"], "placements": placements})
    out = {"mode": "burst", "min_burst_s": getattr(opts, "min_burst_s", None), "spill": opts.spill, "atempo_cap": opts.atempo_cap, "hard_cap": opts.hard_cap,
           "fill_slowdown": opts.fill_slowdown, "fallback_frac": getattr(opts, "fallback_frac", 0.5), "units_slot_fallback": n_fallback, "summary": placement_summary(report)}
    if spread_cfg:
        out["spread"] = {"cfg": {k: v for k, v in spread_cfg.items() if k != "src_ivs"}, "parts_spread": len(spread_log), "pauses": sum(len(x["pauses"]) for x in spread_log),
                         "pause_s": round(sum(q["s"] for x in spread_log for q in x["pauses"]), 2), "head_s": round(sum(x["head_s"] for x in spread_log), 2), "log": spread_log}
        print(f"  [spread] {out['spread']['parts_spread']} parts, {out['spread']['pauses']} inner pauses ({out['spread']['pause_s']} s), late start {out['spread']['head_s']} s", flush=True)
    return out


def build_dubbed_track(units: list[dict], duration: float, work: pathlib.Path, declick: bool = False) -> dict:
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
    dc = None
    if declick:   # r2 2026-10-17: isolated vocoder / atempo clicks in the dub voice, repaired in +-1.5 ms (declick.py)
        from declick import declick_file
        dc = declick_file(str(wav24))
    wav16 = work / "dubbed_16k.wav"; ff(["-i", str(wav24), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(wav16)])
    m4a = work / "dubbed_aac.m4a"; ff(["-i", str(wav24), "-c:a", "aac", "-b:a", "192k", str(m4a)])
    return {"wav24k": str(wav24), "wav16k": str(wav16), "aac": str(m4a), "seconds": round(n / TTS_SR, 3), "sr": TTS_SR,
            "peak_before_norm": round(peak, 4), "speech_seconds": round(sum(u["slot"]["seconds"] for u in units), 3),
            "non_speech": "silence (original timing preserved)", "declick": dc}


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
                          speech_gate: list | None = None, prep=None) -> dict:
    """Optimized video backend: RetinaFace routing + speech gate + batched LatentSync windows (face_aware_latentsync_accel.py).
    Same audio/video semantics as run_lipsync: pass-through keeps master frames, the dubbed track covers the whole timeline."""
    import face_aware_latentsync_accel as faa
    with timed(times, "face_router_load"):
        import model_cache
        router = prep.router if prep is not None else model_cache.get(("router", a.face_router, a.face_min_w, a.face_min_h, a.retina_conf),
                                                                      lambda: faa.make_router(a.face_router, a.face_min_w, a.face_min_h, a.retina_conf))
    with timed(times, "latentsync_load"):
        runner = model_cache.get(("latentsync", a.window_batch_size, not a.no_deepcache, a.compile_backend, a.sdpa_backend, a.deepcache_interval, a.ls_steps),
                                 lambda: faa.AccelRunner(a.window_batch_size, not a.no_deepcache, a.compile_backend, a.sdpa_backend, steps=a.ls_steps, deepcache_interval=a.deepcache_interval))
    if a.mouth_only_pasteback == "on":   # E4: LatentSync pixels only inside a feathered lower-face ellipse, the rest of the frame = master (scripts/quality/candidate_video_stage.py)
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "quality"))
        from candidate_video_stage import MouthOnlyRunner
        runner = MouthOnlyRunner(runner, a.mask_top, a.mask_feather, a.mask_smooth, defer=a.codeformer_scope == "mouth")
    cf_runner = None
    if a.codeformer == "optimized":
        from codeformer_accel import CodeFormerAccel
        with timed(times, "codeformer_load"):
            cf_runner = model_cache.get(("codeformer", a.codeformer_w, a.codeformer_batch, a.codeformer_landmarks, a.codeformer_precision),
                                        lambda: CodeFormerAccel(fidelity_weight=a.codeformer_w, batch_size=a.codeformer_batch, landmark_source=a.codeformer_landmarks,
                                                                autocast=a.codeformer_precision == "fp16", parse_autocast=a.codeformer_precision != "fp32"))
    rep = faa.process_video(src, work, dubbed16k, dubbed_aac, out, router=router, runner=runner, min_ls_frames=a.min_ls_frames,
                            max_verify_depth=a.max_verify_depth, codeformer=a.codeformer, times=times, log=lambda m: print(m, flush=True),
                            speech_gate=speech_gate, crossfade_frames=a.crossfade_frames, router_stride=a.router_stride,
                            in_memory=not a.segment_files, cf_runner=cf_runner, lm_smooth=a.ls_landmark_smooth, prep=prep)
    rep["backend"] = "optimized"
    if cf_runner is not None and not model_cache.cached("codeformer"):
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
    ap.add_argument("--fill-slowdown", type=float, default=0.9, help="burst mode: a span shorter than its source burst may be slowed down to this tempo to fill it (2026-10-02: 0.9 after the fit-up re-translation; 1.0 = never slow down, previous default)")
    ap.add_argument("--split-retries", type=int, default=2, help="burst mode: Qwen JSON split retries before falling back to ONE natural phrase per unit")
    ap.add_argument("--min-part-words", type=int, default=3, help="burst mode: hard minimum words of a TTS part (shorter parts are merged into a neighbour)")
    ap.add_argument("--soft-part-words", type=int, default=4, help="burst mode: preferred minimum words; parts below it are merged while the merged part stays <= --part-max-words")
    ap.add_argument("--part-max-words", type=int, default=16)
    ap.add_argument("--burst-fallback-frac", type=float, default=1.01, help="burst mode: a unit whose fit needs > atempo-cap on >= this fraction of its groups uses the baseline slot rule, which speeds up WITHOUT a cap (1.01 = never; default since the first_res review)")
    ap.add_argument("--fit-retranslate", choices=["on", "off"], default="on", help="burst mode: a part that would need more than --atempo-cap is re-translated concisely (Qwen) and re-synthesised before any speed-up")
    ap.add_argument("--fit-rounds", type=int, default=4); ap.add_argument("--fit-target-ratio", type=float, default=1.05)
    ap.add_argument("--burst-max-lead", type=float, default=0.4, help="burst mode: a part that would need more than --atempo-cap may start up to this many seconds before its burst, into free silence, instead of being sped up")
    ap.add_argument("--tts-qa", choices=["on", "off"], default="on", help="burst mode: independent-ASR QA of every TTS part with per-part retry (English targets only)")
    ap.add_argument("--tts-retries", type=int, default=2, help="QA retry ladder length after the first attempt: new seed, then new seed + --qa-temp")
    ap.add_argument("--qa-temp", type=float, default=0.6)
    ap.add_argument("--segmentation", choices=["whisper", "sentence"], default="sentence",
                    help="burst mode (2026-10-02): sentence = Qwen punctuation pass -> units are sentences (lossless, whisper fallback); whisper = previous behaviour (whisper segments)")
    ap.add_argument("--tiny-unit-merge", choices=["on", "off"], default="on", help="burst mode: merge units with < --tiny-min-words words or < --tiny-min-s seconds into the neighbour (same speaker, pause <= --tiny-max-gap-s)")
    ap.add_argument("--tiny-min-words", type=int, default=3); ap.add_argument("--tiny-min-s", type=float, default=1.0); ap.add_argument("--tiny-max-gap-s", type=float, default=2.0)
    ap.add_argument("--tts-mode", choices=["parts", "phrase"], default="phrase",
                    help="burst mode: phrase = ONE Chatterbox generation per unit, CTC word alignment, cuts between words distributed over the source bursts (2026-10-02); parts = one generation per burst part (previous behaviour)")
    ap.add_argument("--fit-up", choices=["on", "off"], default="on", help="phrase mode: fuller Qwen re-translation when the synthesized speech is shorter than --fit-up-ratio x the unit's source speech (accepted only if closer and not less intelligible)")
    ap.add_argument("--fit-up-ratio", type=float, default=0.8, help="phrase mode: fit-up when synthesized speech / source speech of the unit is below this"); ap.add_argument("--fit-up-rounds", type=int, default=3)
    ap.add_argument("--fit-up-aim", type=float, default=1.0, help="phrase mode: the fuller rewrite targets aim x the source speech duration")
    ap.add_argument("--phrase-split", choices=["timing", "qwen"], default="timing", help="phrase mode: timing = distribute the aligned TTS words over the source bursts by duration (cuts between words, at punctuation / pauses when near); qwen = semantic JSON split with lossless fallbacks")
    ap.add_argument("--phrase-cut", choices=["syntax", "acoustic", "timing"], default="timing",
                    help="phrase mode, timing split: timing = distribute_words (cumulative-duration point, small punctuation / pause bonus; default); syntax = cuts chosen "
                         "jointly by timing (steep beyond 0.4 s) + punctuation / pauses + acoustic silence, not after a function word or inside number+unit, 5/20 ms edge "
                         "fades (2026-10-12 candidate: hot cut edges 39/43 -> 20/38, function-word cuts 37 -> 4, but coverage / CTC CER not yet at baseline -> off); "
                         "acoustic = timing split that prefers acoustically quiet word boundaries (no function-word rule), 5 ms declick fades")
    ap.add_argument("--long-pause-func", type=float, default=0.0, help="--phrase-cut acoustic: extra cost of a cut after a function word / inside a fixed phrase, scaled by the "
                    "source pause there (0 at 0.3 s, full at 0.8 s); 0 = off. 2026-10-13 candidate (03 offline: weak-boundary pauses > 0.3 s 9 -> 2 at 1.2)")
    ap.add_argument("--tts-gap-qa", choices=["on", "off"], default="off", help="phrase mode: re-generate (new seed, <= 2x) a sentence whose CTC alignment shows speech-level sound "
                    "BETWEEN words (inserted syllable / drawl / filler that the Whisper QA normalises away; 04 u03 2026-10-13)")
    ap.add_argument("--long-pause-grow", type=float, default=0.0, help="--long-pause-func weight grows to (1 + grow)x for source pauses of 0.8 -> 2.8 s (2026-10-13: with --pause-marks qwen, "
                    "weak-boundary pause seconds 46.8 -> 21.9 on the five videos offline, +1.1 s source without dub)")
    ap.add_argument("--sentence-flow-gap", type=float, default=0.0, help="phrase mode: inside one sentence, source pauses shorter than this (s) are not reproduced: "
                    "the sentence is spoken as one connected phrase (0 = off: one part per source burst, previous behaviour)")
    ap.add_argument("--ordered-translation", choices=["off", "on"], default="off", help="phrase mode + --sentence-flow-gap: a sentence with source pauses >= the flow gap is "
                    "translated piece by piece in source order (\" | \" at the pauses) and those pieces become the parts (04 u2 2026-10-13: \"Sochi city | today | August 28\" on "
                    "\"город | сочи 28 | августа сегодня\"); plain translation on a malformed answer / a fit rewrite")
    ap.add_argument("--tts-word-qa", choices=["off", "on"], default="off", help="phrase mode: an attempt where Whisper AND CTC both miss a content word of the text (>= 4 letters) "
                    "is QA-bad -> retry ladder (04 \"Sochi\" -> \"Salkis\", 05 \"Just here\" -> \"This year\"; the sentence CER hid them)")
    ap.add_argument("--pause-marks", choices=["off", "qwen"], default="off", help="with --long-pause-func: qwen = Qwen marks the natural pause points of every multi-burst "
                    "sentence; a long source pause is free only there (function-word list otherwise / on a malformed answer)")
    ap.add_argument("--spread", choices=["off", "centre", "pauses"], default="off", help="burst + phrase mode (2026-10-16, Agent C P1): where the leftover silence of a part "
                    "shorter than its source burst goes: off = one tail gap (dub silent while the speaker still talks); centre = late start <= --spread-head-max + tail; pauses = + "
                    "pauses <= --spread-pause-max at phrase boundaries inside the part (offline 01/03/04/05: LatentSync frames in gaps >= 0.35 s 525 -> 382, gaps > 1 s 6 -> 3)")
    ap.add_argument("--spread-ok", type=float, default=0.3, help="--spread: tail gaps up to this many s are left alone / the target")
    ap.add_argument("--spread-head-max", type=float, default=0.2); ap.add_argument("--spread-pause-max", type=float, default=0.25)
    ap.add_argument("--spread-pause-total-max", type=float, default=0.35, help="--spread: max silence at one boundary = the TTS's own pause + the inserted one")
    ap.add_argument("--spread-min-seg", type=float, default=0.5, help="--spread: min s of speech between two inserted pauses / a pause and the part edge")
    ap.add_argument("--cut-fade", choices=["fixed", "adaptive"], default="fixed", help="--phrase-cut acoustic: fixed = 5 ms declick fades; adaptive = 25 ms fade-out / 8 ms fade-in on inner cuts that are still voiced")
    ap.add_argument("--fit-up-guard", choices=["on", "off"], default="off", help="phrase mode: reject fit-up rewrites that append a clause / end on a dangling auxiliary before the Qwen judge "
                                                                                  "(deterministic; 2026-10-12 candidate, fewer judge calls)")
    ap.add_argument("--fit-down-short-words", type=int, default=4, help="phrase mode: concise rewrite also for 4-5-word sentences when they overflow their window by > --fit-down-short-overflow-s at the hard cap (previous rule: >= 6 words only)")
    ap.add_argument("--fit-down-short-overflow-s", type=float, default=0.3)
    ap.add_argument("--phrase-min-part-words", type=int, default=1); ap.add_argument("--phrase-soft-part-words", type=int, default=2)
    ap.add_argument("--qwen-resident", choices=["on", "off"], default="off", help="keep Qwen loaded until the end of the run (with --qwen-gpu-layers -1: "
                    "all models + Qwen resident on the GPU at the LatentSync peak = the pilot 0.4 question); off = released before LatentSync")
    ap.add_argument("--qwen-quant", choices=["q8_0", "q5_k_m"], default="q8_0", help="Qwen2.5-7B-Instruct GGUF quant (same Apache-2.0 repo). r3 2026-10-19: q5_k_m (5.1 GiB) fits fully on a shared GPU next to Chatterbox + QA within ~10 GB; q8_0 = pilot 0.9 production quant")
    ap.add_argument("--tts-text-norm", choices=["on", "off"], default="off", help="phrase mode: numbers / units / '..' -> speakable English words before Chatterbox, QA and alignment (tts_text.py; r3 2026-10-19: 01 '600x600 ... 9 mm' was heard as '600 x 67')")
    ap.add_argument("--qa-whisper-compute", choices=["float16", "int8_float16"], default="float16", help="faster-whisper compute type of the TTS QA (r3: int8_float16 saves 1.8 GB VRAM on a shared GPU)")
    ap.add_argument("--qwen-tts-gpu-layers", type=int, default=None, help="r3: reload Qwen with this many GPU layers for the TTS session (fit rewrites run next to Chatterbox + QA); default = --qwen-gpu-layers. 14 keeps Q8_0 at 4.7 GB")
    ap.add_argument("--qwen-gpu-layers", type=int, default=0, help="Qwen layers offloaded to the GPU (-1 = all); 0 = frozen CPU baseline. Needs a CUDA llama-cpp-python build "
                                                                   "(A/B 2026-10-12: translation 18x faster, outputs not identical to the CPU)")
    ap.add_argument("--audio-policy", choices=sorted(AUDIO_POLICIES), default="r2", help="audio-stage flag bundle (explicit flags win): r2 (default since 2026-10-17) = cand5 + "
                    "--sentence-flow-gap 0.2 --spread pauses --fill-slowdown 0.875; cand5 = --phrase-cut acoustic --long-pause-func 1.2 --cut-fade adaptive --tts-gap-qa on; legacy = the plain flag defaults")
    ap.add_argument("--declick", choices=["on", "off"], default="on", help="repair isolated impulsive clicks of the assembled dub voice in +-1.5 ms (declick.py; r2 2026-10-17: "
                    "Chatterbox vocoder clicks 01 / 05, atempo splice clicks 01)")
    ap.add_argument("--bed-separator", choices=["spleeter", "hdemucs"], default="spleeter", help="--background on: stem separator of the bed. spleeter = Deezer Spleeter 2stems (MIT code, "
                    "weights = MIT repo release asset; own TF venv, scripts/setup_spleeter.sh; default since 2026-10-17); hdemucs = torchaudio HDEMUCS_HIGH_MUSDB_PLUS, weights WITHOUT a licence: "
                    "reference only, refused unless DUB_ALLOW_NC_BED=1")
    ap.add_argument("--background", choices=["on", "off"], default="on",
                    help="final mix: on = source background bed (--bed-separator accompaniment stem, original voice/bed balance, ducked under the dub) under the dubbed voice; "
                         "off = dubbed voice over digital silence (previous behaviour). LatentSync / speech gate always use the clean dubbed voice. Default on since 2026-10-12 (dead air 16-54 s -> 0-6 s, WER unchanged)")
    ap.add_argument("--bg-duck-db", type=float, default=-4.0, help="--background on: bed attenuation while the dub speaks (dB). r2 2026-10-17: -4 (was -6): with the clean Spleeter bed "
                    "the duck only keeps the dub in front; -6 made the bed 'breathe' 7-8 dB between dub phrases (Whisper WER on the mix unchanged at -4)")
    ap.add_argument("--bg-duck-attack-s", type=float, default=0.25, help="--background on: duck ramp before a dub onset (r2: 0.25, was 0.08: the fast ramp made 8-10 dB bed steps "
                    "in the pause right before each phrase, 03: 9 -> 1)")
    ap.add_argument("--bg-duck-release-s", type=float, default=0.6, help="--background on: duck release after a dub offset (r2: 0.6, was 0.35)")
    ap.add_argument("--bg-src-duck-db", type=float, default=0.0, help="--background on: extra bed attenuation where the source speaks and the dub is silent (Demucs voice residue); 0 = off")
    ap.add_argument("--bg-residue-duck-db", type=float, default=-15.0, help="--background on: bed attenuation where the BED itself sounds like speech (Silero on the "
                    "separated stem) while the source speaks and the dub is silent = voice leak of the separator (r2 2026-10-17: Spleeter 02 39.0-39.4 s); 0 = off")
    ap.add_argument("--bg-offset-db", type=float, default=0.0, help="--background on: bed level relative to the original voice/background balance (dB)")
    ap.add_argument("--stop-after", choices=["none", "audio"], default="none", help="audio = stop after the dubbed track (audio-only candidate: manifest + dubbed_24k/16k.wav, no LatentSync)")
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
    ap.add_argument("--ls-steps", type=int, default=20, help="optimized backend only: LatentSync DDIM steps (A/B 2026-10-13 on 04/05: 16 with --deepcache-interval 7 = -13 %% "
                                                              "LatentSync, SyncNet / lip jitter unchanged; 12 = -26 %%, a human sees occasional lip twitches)")
    ap.add_argument("--speech-gate", choices=["on", "off"], default=OPTIMIZED_DEFAULTS["speech_gate"],
                    help="optimized backend only: LatentSync only inside VAD(original) U VAD(dubbed) (+margin); elsewhere the original mouth is kept")
    ap.add_argument("--gate-margin-s", type=float, default=OPTIMIZED_DEFAULTS["gate_margin_s"])
    ap.add_argument("--gate-merge-gap-s", type=float, default=OPTIMIZED_DEFAULTS["gate_merge_gap_s"])
    ap.add_argument("--crossfade-frames", type=int, default=OPTIMIZED_DEFAULTS["crossfade_frames"], help="LS/original boundary blend length (0 = off)")
    ap.add_argument("--router-stride", type=int, default=OPTIMIZED_DEFAULTS["router_stride"], help="RetinaFace on every N-th gated frame")
    ap.add_argument("--mouth-only-pasteback", choices=["on", "off"], default="off",
                    help="optimized backend: keep the master pixels outside a feathered lower-face mask (E4; 03 A/B 2026-10-12: upper-face sharpness 0.818 -> 0.858, SyncNet 5.81 -> 5.84)")
    ap.add_argument("--mask-top", type=float, default=0.52); ap.add_argument("--mask-feather", type=float, default=0.08)
    ap.add_argument("--mask-smooth", type=int, default=0, help="--mouth-only-pasteback: mask bbox = centred moving average over +-N frames (0 = raw per-frame detection)")
    ap.add_argument("--codeformer-scope", choices=["face", "mouth"], default="face",
                    help="with --mouth-only-pasteback: face = CodeFormer on the composited frame (re-renders eyes / brows, previous behaviour); mouth = CodeFormer on the raw "
                         "LatentSync frame, then the mouth-only composite (eyes / brows stay the source; 2026-10-13 'мимика дергается')")
    ap.add_argument("--ls-landmark-smooth", type=int, default=0, help="optimized backend: centred moving average (+-N frames) of the landmarks replayed into LatentSync's affine crop")
    ap.add_argument("--face-stack", choices=["insightface", "mediapipe"], default="insightface",
                    help="optimized backend: mediapipe = clean face stack (scripts/pilot/clean_face_stack.py, Apache-2.0 MediaPipe Face Landmarker in /venv/mediapipe) "
                         "for the router, LatentSync's crop / verify detections and the mouth-mask landmarks; InsightFace / RetinaFace loaders are blocked. "
                         "insightface = previous stack (non-commercial weights). Use with --codeformer off for the clean pipeline (clean0)")
    ap.add_argument("--mp-anchor-calib", choices=["on", "off"], default="on", help="--face-stack mediapipe: constant anchor correction to LatentSync's training alignment")
    ap.add_argument("--video-prep", choices=["sequential", "parallel"], default="sequential",
                    help="optimized backend: parallel = master decode + router / LatentSync face detection of every frame in a thread during the audio stage, "
                         "replayed by the video stage (same calls, same results; 2026-10-13 speed lever)")
    ap.add_argument("--segment-files", action="store_true", help="optimized backend: legacy per-segment mp4 path instead of in-memory segments")
    a = ap.parse_args()
    audio_policy = apply_audio_policy(a, sys.argv[1:])
    if a.background == "on" and a.bed_separator == "hdemucs" and os.environ.get("DUB_ALLOW_NC_BED") != "1":
        print("--bed-separator hdemucs: the HDEMUCS weights have no licence (non-commercial reference only); set DUB_ALLOW_NC_BED=1 for a reference run", file=sys.stderr); return 2
    global QWEN_GPU_LAYERS, QWEN_QUANT
    QWEN_GPU_LAYERS = a.qwen_gpu_layers; QWEN_QUANT = a.qwen_quant
    os.environ["DUB_QA_WHISPER_COMPUTE"] = a.qa_whisper_compute
    os.environ["DUB_TTS_TEXT_NORM"] = a.tts_text_norm
    if a.qwen_gpu_layers:
        import llama_cpp
        if not llama_cpp.llama_supports_gpu_offload():
            print(f"--qwen-gpu-layers {a.qwen_gpu_layers}: the imported llama_cpp ({llama_cpp.__file__}) is a CPU build", file=sys.stderr); return 2
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
                 "tts_mode": a.tts_mode if a.alignment == "burst" else "slot", "segmentation_mode": a.segmentation if a.alignment == "burst" else "whisper",
                 "codeformer": "DISABLED (not invoked)" if a.codeformer == "off" else {"mode": a.codeformer, "w": a.codeformer_w, "batch": a.codeformer_batch, "landmarks": a.codeformer_landmarks, "precision": a.codeformer_precision},
                 "work_dir": str(work), "config": {"seed": a.seed, "max_unit_s": a.max_unit_s, "threads": THREADS, "qwen_quant": a.qwen_quant, "qwen_gpu_layers": a.qwen_gpu_layers, "qwen_tts_gpu_layers": a.qwen_tts_gpu_layers, "qa_whisper_compute": a.qa_whisper_compute, "tts_text_norm": a.tts_text_norm},
                 "audio_policy": {**audio_policy, "resolved": {k: getattr(a, k) for k in AUDIO_POLICIES["r2"]}}}
    try:
        source = probe(src); man["source"] = source; duration = source["duration_s"]
        if duration <= 0 or not source["has_audio"]:
            raise Blocker("source has no duration/audio")
        print(f"== 0.5 clean pipeline: {src.name} {duration:.3f}s -> {tgt} (CodeFormer {a.codeformer}; alignment {a.alignment}" + (f", segmentation {a.segmentation}, tts {a.tts_mode}" if a.alignment == "burst" else "") + ")", flush=True)

        wav16 = work / "audio16k.wav"
        with timed(times, "audio_extract"):
            ff(["-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav16)])

        video_prep = None
        if a.video_prep == "parallel" and a.video_backend == "optimized" and a.face_router == "retinaface" and a.face_stack == "insightface" and not a.segment_files and a.stop_after != "audio":
            import face_aware_latentsync_accel as faa
            video_prep = faa.VideoPrep(src, work, a.face_router, a.face_min_w, a.face_min_h, a.retina_conf)   # runs while the audio stage runs
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
        llm = None; man["segmentation"] = {"mode": "whisper"}
        if a.alignment == "burst" and a.segmentation == "sentence":
            from sentence_units import sentence_units
            llm, _model_name = load_llm(times)
            with timed(times, "segmentation"):
                units, seg_rep = sentence_units(segments, llm, lang_name(src_lang), duration, a.max_unit_s)
            man["segmentation"] = {"mode": "sentence", **seg_rep}
            print(f"  [segmentation] sentence units={len(units)} (sentences {seg_rep['sentences']}, whisper-fallback chunks {seg_rep['fallback_chunks']}/{len(seg_rep['chunks'])}, long split {seg_rep['split_long']})", flush=True)
        else:
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
        man["tiny_unit_merge"] = {"enabled": False}
        if a.alignment == "burst" and a.tiny_unit_merge == "on":
            from sentence_units import merge_tiny_units
            n_before = len(units); units, merges = merge_tiny_units(units, a.tiny_min_words, a.tiny_min_s, a.tiny_max_gap_s, a.max_unit_s * 1.25)
            man["tiny_unit_merge"] = {"enabled": True, "min_words": a.tiny_min_words, "min_s": a.tiny_min_s, "max_gap_s": a.tiny_max_gap_s, "units_before": n_before, "units_after": len(units), "merges": merges,
                                      "unmerged_tiny": [u["tiny_unmerged"] | {"text": u["text"]} for u in units if u.get("tiny_unmerged")]}
            if merges:
                print(f"  [units] tiny-unit merge: {len(merges)} merged -> {len(units)} units: " + "; ".join(f"'{m_['tiny']['text']}' -> {m_['merged']}" for m_ in merges), flush=True)

        plans = None
        if a.alignment == "burst":
            plans = burst_plan(units, segments, speech, duration, a.burst_min_s, a.burst_spill == "on")
            print(f"  [burst] {sum(len(p['bursts']) for p in plans)} bursts with words in {len(units)} units, {sum(p['bursts_without_words'] for p in plans)} bursts without words", flush=True)
        burst_opts = None
        if a.alignment == "burst":
            sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "quality"))
            from e1_burst_align import PlaceOpts
            burst_opts = PlaceOpts(a.atempo_cap, a.hard_cap, True, 0.06, a.fill_slowdown, spill=a.burst_spill == "on", max_lead=a.burst_max_lead); burst_opts.min_burst_s = a.burst_min_s; burst_opts.fallback_frac = a.burst_fallback_frac
            burst_opts.spread = {"mode": a.spread, "ok": a.spread_ok, "head_max": a.spread_head_max, "pause_max": a.spread_pause_max, "pause_total_max": a.spread_pause_total_max,
                                 "min_seg": a.spread_min_seg, "src_ivs": [(v["start"], v["end"]) for v in speech]}
        phrase_mode = a.alignment == "burst" and a.tts_mode == "phrase"
        tr = run_translation(units, src_lang, tgt, times, plans=plans, split_retries=a.split_retries,
                             min_part_words=a.min_part_words, soft_part_words=a.soft_part_words, part_max_words=a.part_max_words,
                             keep_llm=burst_opts is not None and (a.fit_retranslate == "on" or phrase_mode), llm=llm, split=not phrase_mode,
                             ordered_gap=a.sentence_flow_gap if (a.ordered_translation == "on" and phrase_mode) else 0.0)
        llm = None
        if isinstance(tr, tuple):
            man["translation"], llm, src_name, tgt_name = tr
            if a.qwen_tts_gpu_layers is not None and a.qwen_tts_gpu_layers != QWEN_GPU_LAYERS:
                # r3 shared GPU: Qwen ran fully on the GPU for segmentation + translation (nothing else loaded); for the TTS session (fit rewrites next to
                # Chatterbox + QA) it is reloaded with fewer GPU layers so the stage stays <= ~10 GB (Q8_0: 14 layers = 4.7 GB)
                del llm; gc.collect(); g = QWEN_GPU_LAYERS; QWEN_GPU_LAYERS = a.qwen_tts_gpu_layers
                tl = {}; llm, _ = load_llm(tl); QWEN_GPU_LAYERS = g
                man["translation"]["tts_session_gpu_layers"] = a.qwen_tts_gpu_layers; times["qwen_reload"] = tl.get("translation_load")
        else:
            man["translation"] = tr

        refs = speaker_references(units, turns, wav16, work)
        if a.alignment == "burst":
            from tts_burst import run_tts_burst
            qa_on = a.tts_qa == "on"
            if qa_on and tgt != "en":
                print(f"  [warn] --tts-qa uses English ASR models; disabled for target language {tgt}", flush=True); qa_on = False
            if phrase_mode:
                from tts_burst import run_tts_phrase
                import tts_qa; tts_qa.WORD_QA = a.tts_word_qa == "on"
                man["tts"] = run_tts_phrase(units, plans, refs, tgt, work, a.seed, times, opts=burst_opts, llm=llm, src_name=lang_name(src_lang), tgt_name=lang_name(tgt),
                                            fit={"enabled": llm is not None and a.fit_retranslate == "on", "rounds": a.fit_rounds, "target_ratio": a.fit_target_ratio,
                                                 "short_min_words": a.fit_down_short_words, "short_overflow_s": a.fit_down_short_overflow_s},
                                            qa_cfg={"enabled": qa_on, "retries": a.tts_retries, "low_temp": a.qa_temp},
                                            fit_up={"enabled": a.fit_up == "on" and llm is not None, "ratio": a.fit_up_ratio, "rounds": a.fit_up_rounds, "aim": a.fit_up_aim, "guard": a.fit_up_guard == "on"},
                                            split_retries=a.split_retries, min_part_words=a.phrase_min_part_words, soft_part_words=a.phrase_soft_part_words, part_max_words=a.part_max_words,
                                            phrase_split=a.phrase_split, phrase_cut=a.phrase_cut, long_pause_func=a.long_pause_func, long_pause_grow=a.long_pause_grow, cut_fade=a.cut_fade, gap_qa=a.tts_gap_qa == "on", pause_mark_mode=a.pause_marks, sentence_flow_gap=a.sentence_flow_gap, log=lambda m_: print(m_, flush=True))
            else:
                man["tts"] = run_tts_burst(units, plans, refs, tgt, work, a.seed, times, opts=burst_opts, llm=llm, src_name=lang_name(src_lang), tgt_name=lang_name(tgt),
                                           fit={"enabled": llm is not None, "rounds": a.fit_rounds, "target_ratio": a.fit_target_ratio},
                                           qa_cfg={"enabled": qa_on, "retries": a.tts_retries, "low_temp": a.qa_temp}, min_part_words=a.min_part_words,
                                           log=lambda m_: print(m_, flush=True))
            if a.qwen_resident == "on":   # residency measurement (pilot 0.4 with Qwen on the GPU): keep Qwen loaded through the video stage
                resident_qwen = llm
            llm = tr = None    # release Qwen (the translation tuple also holds a reference)
            gc.collect()       # --qwen-gpu-layers: frees the llama.cpp CUDA buffers before LatentSync
        else:
            man["tts"] = run_tts(units, refs, tgt, work, a.seed, times)

        with timed(times, "alignment"):
            if a.alignment == "burst":
                man["alignment"] = align_units_burst(units, plans, work, burst_opts)
            else:
                align_units(units, duration, work); man["alignment"] = {"mode": "slot"}
        with timed(times, "dubbed_track"):
            man["dubbed_track"] = build_dubbed_track(units, duration, work, declick=a.declick == "on")
        man["units"] = units
        mux_aac = pathlib.Path(man["dubbed_track"]["aac"]); man["background"] = {"enabled": False}
        if a.background == "on":   # only the MUXED audio changes; dubbed_16k (LatentSync, speech gate) stays the clean voice
            from background_bed import mix_background, separate_background
            with timed(times, "background_sep"):
                sep = separate_background(src, work, separator=a.bed_separator)
            with timed(times, "background_mix"):
                dub_vad = run_vad(pathlib.Path(man["dubbed_track"]["wav16k"]))
                mix = mix_background(sep, pathlib.Path(man["dubbed_track"]["wav24k"]), [(v["start"], v["end"]) for v in speech], [(v["start"], v["end"]) for v in dub_vad],
                                     duration, work, bg_offset_db=a.bg_offset_db, duck_db=a.bg_duck_db, src_duck_db=a.bg_src_duck_db,
                                     residue_duck_db=a.bg_residue_duck_db, attack_s=a.bg_duck_attack_s, release_s=a.bg_duck_release_s)
            man["background"] = {"enabled": True, "separation": sep, "mix": mix}; mux_aac = pathlib.Path(mix["aac"])
            print(f"  [background] bed gain {mix['bg_gain_db']:+.1f} dB, duck {mix['duck_db']} dB, separation {sep['separate_s']}s (+load {sep['load_s']}s), bed in dub pauses {mix['bed_db_in_dub_pauses']} dBFS", flush=True)
        print(f"  [align] ratios tts/slot: " + " ".join(f"{u['alignment']['ratio_tts_to_slot']:.2f}" for u in units), flush=True)
        if a.stop_after == "audio":
            times["total"] = round(time.perf_counter() - t_all, 3); man["stage_seconds"] = times; man["gpu"] = gpu_info(); man["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            man["audio_only"] = True
            checks = {"all_units_have_tts_and_translation": all("tts" in u and u.get("translation") for u in units),
                      "dubbed_track_exists": pathlib.Path(man["dubbed_track"]["wav24k"]).exists(), "dubbed_track_full_length": abs(man["dubbed_track"]["seconds"] - duration) < 0.05}
            man["validation"] = {"audio_only": True, "checks": checks, "pass": all(checks.values())}
            mp = out.with_name(out.stem + ".manifest.json"); write_json(mp, man)
            print("\n== stage wall times (s)")
            for k, v in times.items():
                print(f"  {k:<20} {v:9.3f}")
            print(f"== audio-only candidate: dubbed track {man['dubbed_track']['wav24k']} ({man['dubbed_track']['seconds']}s)\n== validation {'PASS' if man['validation']['pass'] else 'FAIL'} {checks}\n-> {mp}")
            return 0 if man["validation"]["pass"] else 4

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
            prep = None
            if video_prep is not None:
                prep = video_prep.result(); man["video_prep"] = {"mode": "parallel", "times": video_prep.times, "error": video_prep.error}
                print(f"  [video-prep] parallel {video_prep.times}" + (f" FAILED -> sequential: {video_prep.error}" if video_prep.error else ""), flush=True)
            if a.face_stack == "mediapipe":   # 2026-10-16 CLEAN-ONLY: MediaPipe router + LatentSync crop / verify / mask landmarks, NC loaders blocked
                import clean_face_stack as cfs
                cfs.guard_nc()
                with timed(times, "face_stack_mediapipe"):
                    prep = cfs.CleanPrep(src, work, a.face_min_w, a.face_min_h, calib=a.mp_anchor_calib == "on")
                man["face_stack"] = {**prep.info, "times": prep.times}
                print(f"  [face-stack] mediapipe {prep.info['face_frames']}/{prep.info['frames']} face frames, {prep.info['ls_detections']} LatentSync detections {prep.times}", flush=True)
            man["lipsync"] = run_lipsync_optimized(src, work, pathlib.Path(man["dubbed_track"]["wav16k"]), mux_aac, out, a, times,
                                                   speech_gate=speech_gate, prep=prep)
        else:
            man["lipsync"] = run_lipsync(src, work, pathlib.Path(man["dubbed_track"]["wav16k"]), mux_aac,
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
        # 2026-10-13: a LatentSync segment that failed (OOM, ...) is passed through with the ORIGINAL mouth: the output is valid video but not lip-synced
        # ("Face not detected" pass-through stays a valid design outcome; a resource failure is not)
        checks["no_latentsync_oom"] = not any(sg.get("action") == "PASS_THROUGH_LS_FAILED" and str(sg.get("error", "")).startswith("OutOfMemoryError")
                                              for sg in man["lipsync"].get("segments", []))
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
