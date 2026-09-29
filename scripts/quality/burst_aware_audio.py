#!/usr/bin/env python3
"""Phase B — burst-aware translation / TTS / placement (candidate audio path; the production aligner is untouched).

Per whisper unit of a baseline run (manifest + work dir of run_clean_pipeline_05.py):
  1. bursts   = Silero VAD intervals of the SOURCE inside the unit slot (e1_burst_align.bursts_in_slot);
  2. words    = faster-whisper word timestamps of the unit, assigned to bursts by word START (whisper word ends absorb the following
                pause, starts do not); bursts without words get no chunk;
  3. context  = the unit's existing full translation (manifest, Qwen pilot-0.9 path) — nothing is re-translated;
  4. split    = Qwen2.5-7B Q8_0 (CPU, same loader as the runner) is asked for JSON {"parts": [N strings]} cutting THAT translation
                into N consecutive parts matching the N source word groups. Validation is strict: JSON parses, len == N, no
                empty part, and the normalized concatenation of the parts equals the normalized full translation (no word lost,
                added or reworded). retry (up to --retries, with the validation error fed back) -> fallback = proportional split
                of the translation words by source-group word counts (lossless, deterministic). Every unit records which path was used;
  5. TTS      = Chatterbox Multilingual v3 per part (speaker reference, fp16 cast + autocast like the runner, fp32 fallback), each
                chunk trimmed to its own Silero VAD speech span;
  6. placement= e1_burst_align.place_groups (window = burst start .. next burst start, extension into silence, atempo cap /
                hard cap, optional fill-slowdown), slot and timeline never move; e1_burst_align.build_dubbed_track builds the track.
Writes <out>/work_<id>/{tts_u*_p*.wav, aligned_u*.wav, dubbed_24k.wav, dubbed_16k.wav, dubbed_aac.m4a, burst_alignment.json (+ e1_alignment.json
in the E1 schema so candidate_video_stage.py / dub_coverage.py read it unchanged)}.
    source scripts/env.sh
    python scripts/quality/burst_aware_audio.py --id 04 --out /tmp/dabai_quality/candidates/burst [--fill-slowdown 0.85] [--split proportional]
"""
from __future__ import annotations
import argparse, json, pathlib, re, sys, time
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pilot"))
from e1_burst_align import PlaceOpts, TTS_SR, SR16, bursts_in_slot, build_dubbed_track, ff, place_groups, placement_summary, vad_intervals  # noqa: E402
from pilot_common import MODELS, THREADS  # noqa: E402

Q = pathlib.Path("/tmp/dabai_quality")
SPLIT_SYSTEM = "You are a professional translator. You answer with JSON only."


def norm_tokens(s: str) -> list[str]:
    return re.findall(r"[\w']+", s.lower().replace("’", "'"))


def unit_words(man: dict, u: dict) -> list[dict]:
    seg = next(s for s in man["asr"]["segments"] if s["id"] == u["asr_segment"])
    return [w for w in seg["words"] if w["start"] >= u["start"] - 0.011 and w["end"] <= u["end"] + 0.011]


def assign_words(words: list[dict], bursts: list) -> list[list[dict]]:
    """Word -> burst by word start: inside a burst -> that burst; in a pause -> the next burst; after the last burst -> the last."""
    groups: list[list[dict]] = [[] for _ in bursts]
    for w in words:
        t = w["start"]; k = None
        for j, (s, e) in enumerate(bursts):
            if s <= t <= e:
                k = j; break
            if t < s:
                k = j; break
        groups[k if k is not None else len(bursts) - 1].append(w)
    return groups


def merge_short_bursts(bursts: list, min_s: float) -> list:
    """Merge bursts shorter than min_s into the neighbour with the smaller gap (the merged burst spans both, the pause included)."""
    b = [list(x) for x in bursts]
    while len(b) > 1:
        short = [j for j, (s, e) in enumerate(b) if e - s < min_s]
        if not short:
            break
        j = min(short, key=lambda k: b[k][1] - b[k][0])
        left = b[j][0] - b[j - 1][1] if j > 0 else None; right = b[j + 1][0] - b[j][1] if j + 1 < len(b) else None
        if right is not None and (left is None or right <= left):
            b[j + 1] = [b[j][0], b[j + 1][1]]; del b[j]
        else:
            b[j - 1] = [b[j - 1][0], b[j][1]]; del b[j]
    return [tuple(x) for x in b]


def proportional_split(translation: str, counts: list[int]) -> list[str]:
    toks = translation.split(); n = len(toks); tot = max(sum(counts), 1); parts = []; pos = 0
    for j, c in enumerate(counts):
        take = n - pos if j == len(counts) - 1 else max(1, int(round(n * c / tot)))
        take = min(take, n - pos - (len(counts) - 1 - j)) if j < len(counts) - 1 else take
        parts.append(" ".join(toks[pos:pos + max(take, 0)])); pos += max(take, 0)
    return parts


def validate_parts(obj, n: int, translation: str) -> tuple[list[str] | None, str | None]:
    if isinstance(obj, list):                       # a bare JSON list of strings is accepted as the parts
        obj = {"parts": obj}
    if not isinstance(obj, dict) or "parts" not in obj or not isinstance(obj["parts"], list):
        return None, "no 'parts' list"
    parts = [str(p).strip().strip('"') for p in obj["parts"]]
    if len(parts) != n:
        return None, f"expected exactly {n} parts, got {len(parts)}"
    if any(not norm_tokens(p) for p in parts):
        return None, "a part is empty"
    if norm_tokens(" ".join(parts)) != norm_tokens(translation):
        return None, "the parts joined do not reproduce the translation word for word (words missing, added or reordered)"
    return parts, None


def qwen_split(llm, src_groups: list[str], translation: str, src_name: str, tgt_name: str, retries: int) -> tuple[list[str] | None, dict]:
    n = len(src_groups); log = {"attempts": [], "method": None}
    groups_txt = "\n".join(f"{j + 1}: {g}" for j, g in enumerate(src_groups))
    prompt = (f"The {src_name} sentence below was spoken in {n} consecutive groups (speech bursts):\n{groups_txt}\n\n"
              f"Its full {tgt_name} translation is:\n\"{translation}\"\n\n"
              f"Cut this {tgt_name} translation into exactly {n} consecutive parts so that part k is the translation of group k. "
              f"Keep every word of the translation in the same order; do not add, remove, reorder or reword anything, only choose where to cut. "
              f"A part may be short but must not be empty. Answer with JSON only, in the form {{\"parts\": [\"...\", \"...\"]}} with exactly {n} strings.")
    msgs = [{"role": "system", "content": SPLIT_SYSTEM}, {"role": "user", "content": prompt}]
    for attempt in range(retries + 1):
        r = llm.create_chat_completion(msgs, max_tokens=512, temperature=0.0)
        txt = r["choices"][0]["message"]["content"].strip(); raw = txt
        txt = re.sub(r"^```(?:json)?\s*|\s*```$", "", txt.strip())
        m = re.search(r"\{.*\}", txt, re.S) or re.search(r"\[.*\]", txt, re.S); obj = None; err = None
        try:
            obj = json.loads(m.group(0) if m else txt)
        except Exception as e:
            err = f"invalid JSON: {e}"
        parts = None
        if obj is not None:
            parts, err = validate_parts(obj, n, translation)
        log["attempts"].append({"attempt": attempt, "raw": raw[:400], "error": err})
        if parts is not None:
            log["method"] = "qwen"; return parts, log
        msgs = msgs + [{"role": "assistant", "content": raw}, {"role": "user", "content": f"That is invalid: {err}. Return JSON only with exactly {n} non-empty parts whose concatenation is the translation word for word."}]
    return None, log


def main() -> int:
    import soundfile as sf
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True); ap.add_argument("--run-dir", default=str(Q / "baseline")); ap.add_argument("--work-dir", default=str(Q / "work")); ap.add_argument("--out", required=True)
    ap.add_argument("--split", choices=["qwen", "proportional"], default="qwen"); ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--atempo-cap", type=float, default=1.3); ap.add_argument("--hard-cap", type=float, default=1.6); ap.add_argument("--extend", choices=["on", "off"], default="on")
    ap.add_argument("--gap", type=float, default=0.06); ap.add_argument("--fill-slowdown", type=float, default=1.0); ap.add_argument("--seed", type=int, default=1247)
    ap.add_argument("--min-burst-words", type=int, default=1)
    ap.add_argument("--spill", choices=["on", "off"], default="off", help="last chunk of a unit may run into the pause before the next unit's first burst (hard cap before any cut)")
    ap.add_argument("--min-burst-s", type=float, default=0.0, help="merge a source burst shorter than this into its neighbour (closer gap) before grouping words; 0 = keep every VAD burst")
    a = ap.parse_args()
    man = json.loads((pathlib.Path(a.run_dir) / f"{a.id}_baseline.manifest.json").read_text())
    out = pathlib.Path(a.out) / f"work_{a.id}"; out.mkdir(parents=True, exist_ok=True)
    duration = man["source"]["duration_s"]; units = man["units"]; vad = [(v["start"], v["end"]) for v in man["speech_intervals"]]
    src_name = {"ru": "Russian", "en": "English"}.get(man["source_lang"], man["source_lang"]); tgt_name = {"en": "English", "ru": "Russian"}.get(man["target_lang"], man["target_lang"])
    tgt = man["target_lang"]; refs = man["tts"]["references"]; times = {}
    # ---- 1-3: bursts, word groups, contexts
    plan = []
    for i, u in enumerate(units):
        slot_s, slot_e = u["slot"]["start"], u["slot"]["end"]; next_start = units[i + 1]["slot"]["start"] if i + 1 < len(units) else duration
        bursts = merge_short_bursts(bursts_in_slot(vad, slot_s, slot_e), a.min_burst_s); words = unit_words(man, u); groups = assign_words(words, bursts)
        keep = [(b, g) for b, g in zip(bursts, groups) if len(g) >= a.min_burst_words]
        if not keep:
            keep = [((slot_s, slot_e), words)]
        plan.append({"unit": u, "slot": (slot_s, slot_e), "next_start": next_start, "next_slot_start": next_start, "bursts_all": bursts, "bursts": [b for b, _ in keep],
                     "groups": [" ".join(w["word"].strip() for w in g) for _, g in keep], "group_counts": [len(g) for _, g in keep],
                     "bursts_without_words": len(bursts) - len(keep)})
    if a.spill == "on":   # the last chunk may spill up to the next unit's FIRST BURST (not its slot start): the pause between units is usable
        for p, q in zip(plan, plan[1:]):
            p["next_start"] = q["bursts"][0][0]
    # ---- 4: split the existing translations
    t0 = time.perf_counter(); llm = None
    if a.split == "qwen" and any(len(p["bursts"]) > 1 for p in plan):
        import glob
        from llama_cpp import Llama
        files = sorted(glob.glob(str(MODELS / "qwen2.5-7b-instruct-gguf" / "*q8_0*.gguf")))
        llm = Llama(model_path=files[0], n_ctx=4096, n_threads=THREADS, n_threads_batch=THREADS, n_gpu_layers=0, verbose=False)
    times["qwen_load"] = round(time.perf_counter() - t0, 3); t0 = time.perf_counter(); n_qwen = n_fb = n_single = 0
    for p in plan:
        u = p["unit"]; n = len(p["bursts"])
        if n == 1:
            p["parts"] = [u["translation"]]; p["split"] = {"method": "single"}; n_single += 1; continue
        parts, log = (None, {"method": None, "attempts": []})
        if llm is not None:
            parts, log = qwen_split(llm, p["groups"], u["translation"], src_name, tgt_name, a.retries)
        if parts is None:
            parts = proportional_split(u["translation"], p["group_counts"]); log["method"] = "proportional_fallback"; n_fb += 1
        else:
            n_qwen += 1
        assert norm_tokens(" ".join(parts)) == norm_tokens(u["translation"]), "split lost words"   # never lose or add text
        p["parts"] = parts; p["split"] = log
        print(f"u{u['id']:02d} {n} bursts [{log['method']}]: " + " | ".join(f"{g} -> {t}" for g, t in zip(p["groups"], parts)), flush=True)
    times["split"] = round(time.perf_counter() - t0, 3); del llm
    # ---- 5: TTS per part
    import torch, torchaudio
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    t0 = time.perf_counter(); m = ChatterboxMultilingualTTS.from_local(MODELS / "chatterbox", "cuda", t3_model="v3"); builtin = m.conds; times["tts_load"] = round(time.perf_counter() - t0, 3)

    def cast(dtype):
        for mod in (m.t3, m.s3gen, m.ve):
            mod.to(dtype)
        if m.conds is not None:
            m.conds = m.conds.to(device="cuda")

    def synth_all(dtype):
        cast(dtype); cur = None
        for p in plan:
            u = p["unit"]; spk = u["speaker"]; ref = refs[spk]["file"]
            if cur != spk:
                with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                    if ref: m.prepare_conditionals(ref)
                    else: m.conds = builtin.to(device="cuda")
                cur = spk
            p["tts"] = []
            for k, text in enumerate(p["parts"]):
                torch.manual_seed(a.seed + 100 * u["id"] + k)
                with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                    wav = m.generate(text, language_id=tgt)
                wav = wav.detach().float().cpu()
                if not torch.isfinite(wav).all() or wav.abs().max() < 1e-4:
                    raise RuntimeError(f"non-finite or silent TTS u{u['id']} part {k}")
                f = out / f"tts_u{u['id']:02d}_p{k}.wav"; torchaudio.save(str(f), wav, m.sr); p["tts"].append({"file": str(f), "seconds": round(wav.shape[-1] / m.sr, 3), "text": text})
    t0 = time.perf_counter(); used = None
    for dtype in (torch.float16, torch.float32):
        try:
            synth_all(dtype); used = dtype; break
        except Exception as e:
            print(f"  [tts] {dtype} failed: {type(e).__name__}: {str(e)[:200]} -> next dtype", flush=True)
    if used is None:
        raise RuntimeError("TTS failed in every dtype")
    times["tts"] = round(time.perf_counter() - t0, 3); del m; torch.cuda.empty_cache()
    # ---- 6: placement (shared E1 functions)
    opts = PlaceOpts(a.atempo_cap, a.hard_cap, a.extend == "on", a.gap, a.fill_slowdown, spill=a.spill == "on"); report_units = []
    for p in plan:
        u = p["unit"]; chunks = []
        for k, t in enumerate(p["tts"]):
            y, sr = sf.read(t["file"], dtype="float32"); assert sr == TTS_SR
            if y.ndim > 1: y = y.mean(axis=1)
            t16 = out / f"tts16_u{u['id']:02d}_p{k}.wav"; ff(["-i", t["file"], "-ar", str(SR16), "-ac", "1", "-c:a", "pcm_s16le", str(t16)])
            raw = vad_intervals(t16) or [(0.0, len(y) / sr)]; t16.unlink()
            s_, e_ = max(0.0, raw[0][0] - 0.04), min(len(y) / sr, raw[-1][1] + 0.06); t["speech_span"] = [round(s_, 3), round(e_, 3)]
            chunks.append(y[int(s_ * sr):int(e_ * sr)])
        track, placements = place_groups(chunks, p["bursts"], p["slot"], p["next_start"], opts, out, f"u{u['id']:02d}", sr=TTS_SR, labels=[[k] for k in range(len(chunks))])
        for pl, t, g in zip(placements, p["tts"], p["groups"]):
            pl["text"] = t["text"]; pl["source_words"] = g
        aligned = out / f"aligned_u{u['id']:02d}.wav"; sf.write(str(aligned), track, TTS_SR, subtype="PCM_16")
        report_units.append({"id": u["id"], "slot": list(p["slot"]), "tts_s": round(sum(t["seconds"] for t in p["tts"]), 3), "tts_speech_s": round(sum(len(c) / TTS_SR for c in chunks), 3),
                             "bursts": len(p["bursts"]), "bursts_without_words": p["bursts_without_words"], "phrases": len(chunks), "baseline_ratio": u["alignment"]["ratio_tts_to_slot"], "baseline_atempo": u["alignment"]["atempo"],
                             "translation": u["translation"], "parts": p["parts"], "source_groups": p["groups"], "split": p["split"], "tts": p["tts"],
                             "placements": placements, "max_atempo": max(pl.get("atempo", 1.0) for pl in placements), "file": str(aligned)})
        print(f"u{u['id']:02d} slot {p['slot'][0]:.2f}-{p['slot'][1]:.2f} bursts {len(p['bursts'])} -> " + " ".join(f"[{pl['placed'][0]:.2f}-{pl['placed'][1]:.2f} x{pl['atempo']}]" for pl in placements if pl.get("placed")), flush=True)
    tr = build_dubbed_track([(r["file"], r["slot"][0]) for r in report_units], duration, out)
    report = {"id": a.id, "mode": "burst_aware", "split": a.split, "min_burst_s": a.min_burst_s, "spill": a.spill, "atempo_cap": a.atempo_cap, "hard_cap": a.hard_cap, "extend": a.extend, "fill_slowdown": a.fill_slowdown, "seed": a.seed,
              "tts_dtype": str(used).replace("torch.", ""), "units": report_units, "stage_seconds": times,
              "summary": {**placement_summary(report_units), **tr, "units": len(units), "units_single_burst": n_single, "units_qwen_split": n_qwen, "units_proportional_fallback": n_fb,
                          "chunks": sum(len(r["tts"]) for r in report_units), "bursts_without_words": sum(r["bursts_without_words"] for r in report_units)}}
    (out / "burst_alignment.json").write_text(json.dumps(report, indent=1, ensure_ascii=False)); (out / "e1_alignment.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print("summary", json.dumps(report["summary"]), "\ntimes", json.dumps(times)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
