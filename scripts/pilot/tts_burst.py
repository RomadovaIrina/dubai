"""Burst-mode TTS session: synthesis of every part with duration-aware fitting (Fix 3) and per-part intelligibility QA + retry (Fix 4).

Order of preference when a part does not fit its burst window (first_res review: atempo above ~1.15-1.2 measurably hurts intelligibility, 1.3-1.6 was the
normal path before):  concise duration-aware re-translation of THAT part  ->  regenerate TTS for that part  ->  only then a moderate atempo (caps in PlaceOpts:
soft 1.15, hard 1.2).  Every TTS part goes through  expected text -> TTS -> independent ASR -> score  and is retried alone (new seed, then new seed + lower
temperature) when it is not intelligible; the whole video is never re-synthesised.

The caller (run_clean_pipeline_05.py) owns the translation stage; this module receives the live Qwen `llm` for the re-translation and the plans/units of the burst path.
"""
from __future__ import annotations

import pathlib
import re
import sys
import time

import numpy as np
import soundfile as sf

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "quality"))
from e1_burst_align import SR16, burst_fit_ratios, ff, vad_intervals  # noqa: E402

TTS_SR = 24000
REWRITE_SYSTEM = "You are a professional dubbing script adapter. You answer with the rewritten phrase only, without quotes or explanations."


def words_of(text: str) -> int:
    return len(re.findall(r"[\w']+", text))


def speech_span(y: np.ndarray, sr: int, work: pathlib.Path, tag: str) -> tuple[float, float]:
    """Silero VAD speech span of one TTS part: first start - 40 ms .. last end + 60 ms (clamped to the clip)."""
    t16 = work / f"tts16_{tag}.wav"
    sf.write(str(work / f"_span_{tag}.wav"), y, sr, subtype="PCM_16")
    ff(["-i", str(work / f"_span_{tag}.wav"), "-ar", str(SR16), "-ac", "1", "-c:a", "pcm_s16le", str(t16)])
    raw = vad_intervals(t16) or [(0.0, len(y) / sr)]
    t16.unlink(); (work / f"_span_{tag}.wav").unlink()
    return max(0.0, raw[0][0] - 0.04), min(len(y) / sr, raw[-1][1] + 0.06)


def concise_rewrite(llm, src_name: str, tgt_name: str, source_group: str, text: str, max_words: int, seconds: float) -> str | None:
    """Ask Qwen for a shorter natural spoken version of ONE part. Returns None when the answer is unusable (empty, not shorter, loses a number)."""
    from burst_aware_audio import missing_numbers
    prompt = (f"The {src_name} phrase \"{source_group}\" is spoken in {seconds:.1f} seconds. Its current {tgt_name} dubbing text is:\n\"{text}\"\n\n"
              f"Rewrite the {tgt_name} text as a concise, natural, easy-to-pronounce spoken phrase of at most {max_words} words that keeps the meaning and every number. "
              f"Do not add information. Answer with the phrase only.")
    r = llm.create_chat_completion([{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": prompt}], max_tokens=96, temperature=0.0)
    new = " ".join(r["choices"][0]["message"]["content"].split()).strip().strip('"«»“”')
    if not new or "\n" in new or words_of(new) > max_words or words_of(new) >= words_of(text) or missing_numbers(text, new):
        return None
    return new


def run_tts_burst(units: list, plans: list, refs: dict, tgt_lang: str, work: pathlib.Path, seed: int, times: dict, *, opts, llm=None, src_name: str = "", tgt_name: str = "",
                  fit: dict | None = None, qa_cfg: dict | None = None, min_part_words: int = 3, log=print) -> dict:
    """Synthesise every part of every unit (speaker-grouped like the slot path), fit / retry per part, write tts_u<id>_p<k>.wav and u['tts_parts'] / u['tts'].
    fit = {"enabled": bool, "rounds": int, "target_ratio": float}; qa_cfg = {"enabled": bool, "retries": int, "low_temp": float} (None / disabled = single plain attempt)."""
    import torch
    import torchaudio
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
    from chatterbox_fp16 import install_speech_token_guard
    if tgt_lang not in SUPPORTED_LANGUAGES:
        raise RuntimeError(f"target language {tgt_lang!r} not supported by Chatterbox")
    fit = fit or {"enabled": False}; qa_cfg = qa_cfg or {"enabled": False}
    t0 = time.perf_counter(); m = ChatterboxMultilingualTTS.from_local(pathlib.Path(__file__).resolve().parents[2] / "models" / "chatterbox", "cuda", t3_model="v3")
    builtin = m.conds; guard = install_speech_token_guard(m); times["tts_load"] = round(time.perf_counter() - t0, 3)
    prev_end_of = {u["id"]: plans[i - 1]["slot"][1] for i, u in enumerate(units) if i > 0}    # dry-run estimate of the previous speech end (the aligner uses the real one)
    qa = None
    if qa_cfg.get("enabled"):
        from tts_qa import TtsQA
        t0 = time.perf_counter(); qa = TtsQA(); times["qa_load"] = round(time.perf_counter() - t0, 3)
    stats = {"parts": 0, "attempts": 0, "retried_parts": 0, "retry_success": 0, "final_bad": 0, "fit_parts_rewritten": 0, "fit_parts_merged": 0, "fit_rounds_used": 0, "fit_rewrite_rejected": 0, "gen_seconds": 0.0, "qa_seconds": 0.0}

    def cast(dtype):
        for mod in (m.t3, m.s3gen, m.ve):
            mod.to(dtype)
        if m.conds is not None:
            m.conds = m.conds.to(device="cuda")

    def gen_part(u, k: int, text: str, dtype, salt: int = 0) -> dict:
        """One TTS part with the QA retry ladder: attempt 1 production settings, 2 new seed, 3 new seed + lower temperature. Keeps the best attempt."""
        base = seed + 100 * u["id"] + k + 100003 * salt
        ladder = [(0, 0.8)] + ([(1, 0.8), (2, qa_cfg.get("low_temp", 0.6))][: qa_cfg.get("retries", 2)] if qa is not None else [])
        attempts, best = [], None
        for ai, (soff, temp) in enumerate(ladder):
            s = base + 7919 * soff
            torch.manual_seed(s)
            tg = time.perf_counter()
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                wav = m.generate(text, language_id=tgt_lang, temperature=temp)
            wav = wav.detach().float().cpu()
            if not torch.isfinite(wav).all() or wav.abs().max() < 1e-4:
                raise RuntimeError(f"non-finite or silent TTS for unit {u['id']} part {k}")
            stats["gen_seconds"] += time.perf_counter() - tg; stats["attempts"] += 1
            rec = {"attempt": ai + 1, "seed": s, "temperature": temp, "seconds": round(wav.shape[-1] / m.sr, 3)}
            if qa is not None:
                tq = time.perf_counter(); r = qa.assess(text, wav[0].numpy(), m.sr, tgt_lang); stats["qa_seconds"] += time.perf_counter() - tq
                rec.update({"score": r["score"], "bad": r["bad"], "reasons": r["reasons"], "ctc": r["ctc"][:120], "whisper": r["whisper"][:120]})
            attempts.append(rec)
            if best is None or (qa is not None and (rec["bad"], rec["score"]) < (best[1]["bad"], best[1]["score"])):
                best = (wav, rec)
            if qa is None or not rec["bad"]:
                break
        wav, rec = best
        f = work / (f"tts_u{u['id']:02d}_p{k}.wav" if not salt else f"tts_u{u['id']:02d}_p{k}_r{salt}.wav"); torchaudio.save(str(f), wav, m.sr)
        bad_final = bool(rec.get("bad", False))
        stats["parts"] += 0 if salt else 1
        if len(attempts) > 1 and not salt:
            stats["retried_parts"] += 1; stats["retry_success"] += 0 if bad_final else 1
        return {"file": str(f), "seconds": round(wav.shape[-1] / m.sr, 3), "text": text, "attempts": attempts, "qa_bad_final": bad_final, "qa_score": rec.get("score"), "seed_used": rec["seed"], "temperature_used": rec["temperature"]}

    def chunk_of(tp: dict, tag: str):
        y, sr = sf.read(tp["file"], dtype="float32"); assert sr == TTS_SR
        if y.ndim > 1: y = y.mean(axis=1)
        s_, e_ = speech_span(y, sr, work, tag); tp["speech_span"] = [round(s_, 3), round(e_, 3)]
        return y[int(s_ * sr):int(e_ * sr)]

    def synth_all(dtype) -> None:
        cast(dtype); by_spk: dict = {}
        for u, p in zip(units, plans):
            by_spk.setdefault(u["speaker"], []).append((u, p))
        for spk, ups in by_spk.items():
            ref = refs[spk]["file"]
            with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                if ref: m.prepare_conditionals(ref)
                else: m.conds = builtin.to(device="cuda"); guard.use_builtin()
            for u, p in ups:
                u["parts_original"] = list(u["parts"]); u["tts_parts"] = []; u["fit"] = []
                for k, text in enumerate(u["parts"]):
                    u["tts_parts"].append(gen_part(u, k, text, dtype))
                chunks = [chunk_of(tp, f"u{u['id']:02d}_p{k}") for k, tp in enumerate(u["tts_parts"])]
                if fit.get("enabled"):
                    prev_end = prev_end_of.get(u["id"])
                    for rnd in range(fit.get("rounds", 4)):
                        need = burst_fit_ratios(chunks, p["bursts"], p["slot"], p["next_start"], opts, prev_end=prev_end)
                        over = [(k, r) for k, r in enumerate(need) if r is not None and r > opts.atempo_cap + 1e-3]
                        if not over:
                            break
                        k, r = max(over, key=lambda kr: kr[1])
                        rec = {"round": rnd + 1, "part": k, "ratio_before": r, "old": u["parts"][k], "parts_in_unit": len(u["parts"])}
                        stats["fit_rounds_used"] += 1
                        if len(u["parts"]) > 1:
                            # 1) meaning-preserving: merge the over-long part with the neighbour that has more slack -> ONE natural phrase in the union of both bursts
                            nb = [j for j in (k - 1, k + 1) if 0 <= j < len(u["parts"])]
                            j = min(nb, key=lambda j_: need[j_] if need[j_] is not None else 0.0)
                            a_, b_ = sorted((k, j)); text = u["parts"][a_] + " " + u["parts"][b_]
                            rec.update(action="merge_parts", with_part=j, new=text)
                            u["parts"][a_:b_ + 1] = [text]; p["bursts"][a_:b_ + 1] = [(p["bursts"][a_][0], p["bursts"][b_][1])]
                            p["groups"][a_:b_ + 1] = [p["groups"][a_] + " " + p["groups"][b_]]; p["group_counts"][a_:b_ + 1] = [p["group_counts"][a_] + p["group_counts"][b_]]
                            u["tts_parts"][a_:b_ + 1] = [gen_part(u, a_, text, dtype, salt=rnd + 1)]; chunks[a_:b_ + 1] = [chunk_of(u["tts_parts"][a_], f"u{u['id']:02d}_p{a_}r{rnd + 1}")]
                            stats["fit_parts_merged"] += 1
                        else:
                            # 2) a whole sentence that is still far too long for its window: concise re-translation of the sentence (never of a fragment), then regenerate
                            nw = words_of(u["parts"][0]); target = max(int(nw * fit.get("target_ratio", 1.05) / r), min_part_words + 1)
                            if llm is None or not fit.get("rewrite", True) or r <= opts.hard_cap + 0.05 or nw < 6 or target >= nw:
                                rec["action"] = "kept"; rec["result"] = "short phrase or marginal overflow: lead / spill / moderate atempo instead"; u["fit"].append(rec); break
                            new = concise_rewrite(llm, src_name, tgt_name, p["groups"][0], u["parts"][0], target, u["tts_parts"][0]["seconds"] / r)
                            rec.update(action="concise_rewrite", max_words=target)
                            if new is None:
                                rec["result"] = "rewrite_rejected"; stats["fit_rewrite_rejected"] += 1; u["fit"].append(rec); break
                            u["parts"][0] = new; rec.update(new=new, result="rewritten"); stats["fit_parts_rewritten"] += 1
                            u["tts_parts"][0] = gen_part(u, 0, new, dtype, salt=rnd + 1); chunks[0] = chunk_of(u["tts_parts"][0], f"u{u['id']:02d}_p0r{rnd + 1}")
                        u["fit"].append(rec)
                    need = burst_fit_ratios(chunks, p["bursts"], p["slot"], p["next_start"], opts, prev_end=prev_end)
                    u["fit_final_ratios"] = need
                if u["parts"] != u["parts_original"]:
                    u["translation_fitted"] = " ".join(u["parts"])
                stats["final_bad"] += sum(tp["qa_bad_final"] for tp in u["tts_parts"])
                u["tts"] = {"file": None, "parts": len(u["parts"]), "seconds": round(sum(t["seconds"] for t in u["tts_parts"]), 3), "sr": m.sr, "reference": refs[spk]["source"], "dtype": str(dtype).replace("torch.", "")}
                log(f"  [tts] u{u['id']:02d} {spk} {len(u['parts'])} parts {u['tts']['seconds']:.2f}s" + (f" fit-ops {len(u['fit'])}" if u['fit'] else "") +
                    (f" retried {sum(len(t['attempts']) > 1 for t in u['tts_parts'])}" if qa is not None else ""))

    used = None
    t0 = time.perf_counter()
    for dtype in (torch.float16, torch.float32):
        try:
            for k_ in stats:
                stats[k_] = 0 if not isinstance(stats[k_], float) else 0.0
            synth_all(dtype); used = dtype; break
        except Exception as e:
            log(f"  [tts] {dtype} failed: {type(e).__name__}: {str(e)[:200]} -> retrying next dtype")
    if used is None:
        raise RuntimeError("Chatterbox generation failed in every dtype")
    times["tts"] = round(time.perf_counter() - t0, 3)
    sr = m.sr
    del m
    if qa is not None:
        qa.close()
    torch.cuda.empty_cache()
    stats["gen_seconds"] = round(stats["gen_seconds"], 2); stats["qa_seconds"] = round(stats["qa_seconds"], 2)
    stats["retry_success_rate"] = round(stats["retry_success"] / stats["retried_parts"], 3) if stats["retried_parts"] else None
    return {"model": "ChatterboxMultilingualTTS t3=v3 (models/chatterbox)", "sr": sr, "dtype": str(used).replace("torch.", ""), "language_id": tgt_lang,
            "voice_reference": "speaker's own diarized speech via audio_prompt_path/prepare_conditionals",
            "speech_token_guard": {"enabled": True, "flow_calls": guard.calls, "violations": guard.violations}, "qa": {"enabled": qa is not None, **qa_cfg}, "fit": {**fit}, "stats": stats, "references": refs}
