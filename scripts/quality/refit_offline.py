#!/usr/bin/env python3
"""Layer 3 (Agent C, 2026-10-16): per-burst literal refit of the sentences that still leave the speaker talking while the dub is silent.

Design: reports/quality/audio_timing/p2_proposal.md section 3. On a finished L1 run (recut manifest + residual list of residual_for_video.py):
  plan     residual gaps with layer3_could_fill -> per unit the burst pieces (Russian words / English part), the SHORT parts and their word budgets
  rewrite  Qwen2.5-7B (CPU) rewrites ONLY the short English pieces, rendering their own Russian words more fully / literally; guards + the
           existing faithful_check judge; other pieces byte-identical                                          [CPU]
  tts      Chatterbox takes of every accepted rewrite (the pipeline's QA-ladder seeds + one gap-QA alternative), nothing else on the GPU [GPU, flock]
  qa       Whisper + CTC QA, CTC word spans and gap-voice of every take; the pipeline's choice rule -> tts_u<id>_refit_raw.wav  [GPU, flock]
  build    manifest with the refit units (pieces = parts, no cut DP) -> recut_offline (same L1 flags) -> per-unit acceptance vs L1 -> final [CPU]

    python scripts/quality/refit_offline.py plan    --vid 03 --l1 AB/03_x/recut.manifest.json --residual RES/03_L1_0.875.residual.json --out R/03
    python scripts/quality/refit_offline.py rewrite --out R/03
    flock /workspace/dub/gpu.lock python scripts/quality/refit_offline.py tts --out R/03
    flock /workspace/dub/gpu.lock python scripts/quality/refit_offline.py qa  --out R/03
    python scripts/quality/refit_offline.py build   --out R/03 --base BASE/03_final.manifest.json --l1-flags "..."
"""
from __future__ import annotations
import argparse, copy, json, os, pathlib, re, subprocess, sys, time
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
for d in ("pilot", "quality"): sys.path.insert(0, str(HERE.parent / d))
SEED = 1247; SALT = 60; VRAM_GB = 6.0
VOCATIVE = re.compile(r"\b(folks|guys|everyone|everybody|team|friends|partners|people|y'all|my friends)\b[.!?]?$|,\s*(folks|guys|everyone|team|friends|partners)\b", re.I)
DANGLING_END = frozenset("and or but so to of for in on at by with from as that which who the a an my our your their is are was were will would can could should must let".split())


def jload(p): return json.loads(pathlib.Path(p).read_text())
def jsave(p, o): pathlib.Path(p).write_text(json.dumps(o, indent=1, ensure_ascii=False))
def nwords(t): return len(t.split())


# ----------------------------------------------------------------------------------------------------------------------------------- plan
def plan(a) -> int:
    man = jload(a.l1); res = jload(a.residual); units = {u["id"]: u for u in man["units"]}; out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    targets = {}
    for g in res["gaps"]:
        if not g.get("layer3_could_fill") or g["unit"] is None or not g.get("prev_dub") or g["prev_dub"]["unit"] != g["unit"]: continue
        u = units[g["unit"]]; pls = u["alignment"]["placements"]
        if any(pl.get("mode") == "slot_fallback" for pl in pls) or len(pls) != len(u["tts_parts"]): continue
        k = min(range(len(pls)), key=lambda i: abs((pls[i].get("placed") or [0, 1e9])[1] - g["prev_dub"]["end"]))
        t = targets.setdefault(u["id"], {"unit": u["id"], "short": {}, "gaps": []}); t["short"][k] = round(t["short"].get(k, 0.0) + g["seconds"], 2); t["gaps"].append(g["id"])
    rows = []
    for uid, t in sorted(targets.items()):
        u = units[uid]; text = u.get("translation_fitted") or u["translation"]; sp = u["tts"]["word_spans"]
        rate = nwords(text) / max(0.5, sp[-1]["end"] - sp[0]["start"]) if sp else 2.6
        pieces = [tp["text"] for tp in u["tts_parts"]]; pls = u["alignment"]["placements"]
        short = {str(k): {"words_now": nwords(pieces[k]), "shortfall_s": s, "budget": nwords(pieces[k]) + int(min(6, max(1, round(s * rate))))} for k, s in sorted(t["short"].items())}
        rows.append({"unit": uid, "source": u["text"], "text": text, "rate_wps": round(rate, 2), "pieces": pieces, "ru": [pl.get("source_words", "") for pl in pls],
                     "bursts": [pl["burst"] for pl in pls], "short": short, "gaps": t["gaps"], "speaker": u["speaker"]})
        print(f"u{uid:02d} rate {rate:.2f} w/s short {short}")
    jsave(out / "plan.json", {"vid": a.vid, "l1": a.l1, "residual": a.residual, "units": rows}); return 0


# ----------------------------------------------------------------------------------------------------------------------------------- rewrite (CPU Qwen)
def rewrite_prompt(r: dict) -> str:
    """v2: Qwen returns ONLY the SHORT pieces (v1 asked for all pieces back: it edited fixed pieces / ignored the length on 3 of 3 units)."""
    lines = [f'{k + 1}. RU: "{ru}" | EN: "{en}"' for k, (ru, en) in enumerate(zip(r["ru"], r["pieces"]))]
    asks = []
    for k_, s in sorted(r["short"].items(), key=lambda kv: int(kv[0])):
        k = int(k_); lo, hi = s["words_now"] + 1, s["budget"] + 2
        asks.append(f'- piece {k + 1}: now "{r["pieces"][k]}" ({s["words_now"]} words) -> write {lo} to {hi} words that translate ITS Russian words "{r["ru"][k]}" fully')
    keys = ", ".join(f'"{int(k) + 1}": "..."' for k in sorted(r["short"], key=int))
    return (f'A Russian speaker says this sentence in a video:\n"{r["source"]}"\n\nThe sentence is split at the speaker\'s pauses; each piece has its current English dubbing:\n'
            + "\n".join(lines) +
            "\n\nSome English pieces end too early (the Russian speaker is still talking). Write a longer English version of ONLY these pieces:\n" + "\n".join(asks) +
            "\nTranslate the Russian words of that piece more fully and literally: render every Russian word, use full verb forms and longer exact equivalents, "
            "spell out subjects and pronouns. Do NOT add any fact, quality, outcome or idea the Russian does not say, no filler words (um, you know, basically, actually). "
            "Each new piece must connect grammatically to the unchanged pieces before and after it, so that the whole sentence stays fluent. "
            f"Answer with JSON only: {{{keys}}}")


def check_pieces(r: dict, new: list) -> str | None:
    from burst_aware_audio import missing_numbers
    from tts_burst import FILLER, appended_clause
    if not isinstance(new, list) or len(new) != len(r["pieces"]) or not all(isinstance(x, str) and x.strip() for x in new): return f"need exactly {len(r['pieces'])} non-empty pieces"
    changed = 0
    for k, (old, nw) in enumerate(zip(r["pieces"], new)):
        nw = " ".join(nw.split()); s = r["short"].get(str(k))
        if not s:
            if nw != " ".join(old.split()): return f"piece {k + 1} must stay exactly \"{old}\""
            continue
        w = nwords(nw)
        if w <= s["words_now"]: return f"piece {k + 1} is not longer ({w} words); it needs about {s['budget']} words"
        if w > s["budget"] + 2: return f"piece {k + 1} is too long ({w} words); it needs about {s['budget']} words"
        if missing_numbers(old, nw): return f"piece {k + 1} lost a number"
        if FILLER.search(nw): return f"piece {k + 1} has a filler word '{FILLER.search(nw).group(0)}'"
        voc = VOCATIVE.search(nw)   # r2: unconstrained v4 padded with "..., folks." / "guys" / "team" where the Russian addresses nobody
        if voc and not VOCATIVE.search(old) and not re.search(r"ребят|друз|народ|всем привет|коллег", r["ru"][k].lower()): return f"piece {k + 1} adds an address '{voc.group(0)}' that the Russian does not have"
        changed += 1
    tok = lambda t: re.findall(r"[a-z0-9']+", t.lower())
    for k in range(len(new) - 1):   # a rewritten piece must not repeat the start / end of its neighbour (03 u05: "... in general with | with you and in general with")
        if not (r["short"].get(str(k)) or r["short"].get(str(k + 1))): continue
        a_, b_ = tok(new[k]), tok(new[k + 1])
        for n in (4, 3):
            if len(a_) >= n and len(b_) >= n and any(a_[i:i + n] == b_[:n] for i in range(len(a_) - n + 1)): return f"pieces {k + 1} and {k + 2} repeat the words '{' '.join(b_[:n])}'"
        from tts_burst import CUT_FUNC_WORDS   # r2: a content bigram on both sides of the boundary (02 u08 "such a nice | what a nice smell")
        bg = lambda t: {(t[i], t[i + 1]) for i in range(len(t) - 1) if t[i] not in CUT_FUNC_WORDS or t[i + 1] not in CUT_FUNC_WORDS}
        rep = bg(a_[-4:]) & bg(b_[:4])
        if rep: return f"pieces {k + 1} and {k + 2} repeat '{' '.join(sorted(rep)[0])}' across their boundary"
    for k, (old, nw) in enumerate(zip(r["pieces"], new)):   # r2 (human review 10-16: 03 u08 "... and I wish"): a piece keeps its sentence-end role
        if not r["short"].get(str(k)): continue
        nw = " ".join(nw.split()); end_old = old.rstrip()[-1:] in ".?!"; last = k == len(new) - 1
        if (end_old or last) and nw.rstrip()[-1:] not in ".?!": return f"piece {k + 1} must end the sentence like \"{old}\" does (with a full stop / ? / !)"
        if not end_old and not last and nw.rstrip()[-1:] in ".?!": return f"piece {k + 1} must not end the sentence: the sentence continues with piece {k + 2}"
        fw = nw.split()[0]
        if k > 0 and old[:1].islower() and fw[:1].isupper() and fw not in ("I", "I'm", "I'll", "I've", "I'd") and not old.split()[0][:1].isupper():
            return f"piece {k + 1} continues the sentence: it must not start with a capital letter ('{fw}')"
        lw = tok(nw)[-1:] or [""]
        if (end_old or last) and (lw[0] in DANGLING_END or re.search(r"\b(i|we|you) (wish|want|hope|think)[.?!]?$", nw.lower())): return f"piece {k + 1} ends on a dangling word '{lw[0]}'"
    old_t, new_t = " ".join(r["pieces"]), " ".join(" ".join(x.split()) for x in new)
    ac = appended_clause(old_t, new_t)
    if ac: return ac + " (translate the Russian words of the SHORT piece instead of adding a clause)"
    return None if changed else "nothing changed"


def fluent_check(llm, sentence: str) -> tuple[bool, str]:
    """Qwen as a strict copy editor: is the sentence fluent, grammatical spoken English (no missing / doubled words)? An unparsable answer = not fluent."""
    from tts_burst import JUDGE_SYSTEM
    prompt = (f'English dubbing line: "{sentence}"\n\nIs this one fluent, grammatical spoken English sentence, with no missing, doubled or misplaced words? '
              'Answer with JSON only: {"fluent": true} or {"fluent": false, "problem": "..."}')
    r = llm.create_chat_completion([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}], max_tokens=120, temperature=0.0)
    txt = r["choices"][0]["message"]["content"].strip(); m_ = re.search(r"\{.*\}", txt, re.S)
    try:
        o = json.loads(m_.group(0) if m_ else txt); return bool(o.get("fluent")), str(o.get("problem", ""))
    except Exception:
        return False, f"judge answer unparsable: {txt[:100]}"


def rewrite_prompt_v4(r: dict, n_alt: int) -> str:
    """v4 (r2, 2026-10-17): ONE call returns n_alt alternatives; natural spoken English instead of "literal" (v3's literal wording gave stiff
    English: "we simply wish to observe how", "the result we envisioned to achieve")."""
    lines = [f'{k + 1}. RU: "{ru}" | EN: "{en}"' for k, (ru, en) in enumerate(zip(r["ru"], r["pieces"]))]
    asks = []
    for k_, s in sorted(r["short"].items(), key=lambda kv: int(kv[0])):
        k = int(k_); lo, hi = s["words_now"] + 1, s["budget"] + 2
        asks.append(f'- piece {k + 1}: now "{r["pieces"][k]}" ({s["words_now"]} words) -> {lo} to {hi} words, saying what ITS Russian words "{r["ru"][k]}" say')
    keys = ", ".join(f'"{int(k) + 1}": "..."' for k in sorted(r["short"], key=int))
    return (f'A Russian video blogger says this sentence:\n"{r["source"]}"\n\nThe sentence is split at the speaker\'s pauses; each piece has its current English dubbing:\n'
            + "\n".join(lines) +
            "\n\nThe English of some pieces ends too early: the speaker is still talking. Make ONLY these pieces longer:\n" + "\n".join(asks) +
            "\nWrite natural, relaxed spoken English, the way a native-speaking blogger would say it on camera, never stiff or formal. Lengthen by saying the "
            "SAME Russian words more fully: every Russian word rendered, a natural fuller phrasing (\"we want to see\" -> \"we'd really like to see\", "
            "\"modern phones\" -> \"modern smartphones\"), the subject or object said out loud. Do NOT add any fact, quality, outcome or idea that the Russian does not say, "
            "no filler words (um, you know, basically, actually), no word that repeats the meaning of a word next to it. Each new piece must connect grammatically to the "
            "unchanged pieces before and after it, so that the whole sentence reads as one fluent sentence; keep the sentence-final punctuation where the sentence ends. "
            f"Give {n_alt} clearly different alternatives. Answer with JSON only: {{\"alternatives\": [{n_alt} objects, each {{{keys}}}]}}")


def natural_check(llm, sentence: str) -> tuple[bool, str]:
    """v4 judge: fluent AND natural spoken English (not stiff / formal / redundant). An unparsable answer = not natural."""
    from tts_burst import JUDGE_SYSTEM
    prompt = (f'English dubbing line of a video blogger: "{sentence}"\n\nIs this one fluent, grammatical sentence of NATURAL spoken English, with no missing, doubled or '
              'misplaced words, no two neighbouring words with the same meaning, and nothing stiff, bookish or overly formal that a native speaker would not say on camera? '
              'Answer with JSON only: {"natural": true} or {"natural": false, "problem": "..."}')
    r = llm.create_chat_completion([{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": prompt}], max_tokens=120, temperature=0.0)
    txt = r["choices"][0]["message"]["content"].strip(); m_ = re.search(r"\{.*\}", txt, re.S)
    try:
        o = json.loads(m_.group(0) if m_ else txt); return bool(o.get("natural")), str(o.get("problem", ""))
    except Exception:
        return False, f"judge answer unparsable: {txt[:100]}"


def _alts_schema(r: dict, n_alt: int) -> dict:
    keys = [str(int(k) + 1) for k in sorted(r["short"], key=int)]
    item = {"type": "object", "properties": {k: {"type": "string"} for k in keys}, "required": keys}
    return {"type": "object", "properties": {"alternatives": {"type": "array", "items": item, "minItems": n_alt, "maxItems": n_alt}}, "required": ["alternatives"]}


def _parse_alts(txt: str, r: dict) -> list:
    """Schema-constrained answer {"alternatives": [{...}, ...]}; fallback: every "<key>": "<text>" pair in order, a new alternative whenever a key repeats
    (unconstrained Qwen wrote ["1": "..."] lists, 2026-10-17)."""
    keys = [str(int(k) + 1) for k in sorted(r["short"], key=int)]; objs = []
    try:
        o = json.loads(txt); objs = o["alternatives"] if isinstance(o, dict) else o
    except Exception:
        cur = {}
        for k, v in re.findall(r'"(\d+)"\s*:\s*"([^"]*)"', txt):
            if k in cur: objs.append(cur); cur = {}
            cur[k] = v
        if cur: objs.append(cur)
    out = []
    for o in objs:
        if not isinstance(o, dict) or not all(k in o for k in keys): continue
        new = list(r["pieces"])
        for k_ in r["short"]: new[int(k_)] = str(o[str(int(k_) + 1)]).strip().rstrip("}").strip()
        out.append(new)
    return out


def rewrite(a) -> int:
    out = pathlib.Path(a.out); P = jload(out / "plan.json")
    from llama_cpp import Llama
    import glob
    from pilot_common import MODELS
    from tts_burst import REWRITE_SYSTEM, faithful_check
    f = sorted(glob.glob(str(MODELS / "qwen2.5-7b-instruct-gguf" / "*q8_0*.gguf")))[0]; th = int(os.environ.get("DUB_THREADS", "6"))
    t0 = time.perf_counter(); llm = Llama(model_path=f, n_ctx=4096, n_threads=th, n_threads_batch=th, n_gpu_layers=a.qwen_gpu_layers, verbose=False, seed=SEED); load_s = time.perf_counter() - t0
    res = []; calls = 0
    for r in P["units"]:
        msgs = [{"role": "system", "content": REWRITE_SYSTEM}, {"role": "user", "content": rewrite_prompt_v4(r, a.alts)}]; log = []; accepted = None; t1 = time.perf_counter()
        for rnd in range(a.rounds):
            raw = llm.create_chat_completion(msgs, max_tokens=900, temperature=0.0 if rnd == 0 else 0.7, seed=SEED + rnd,
                                             response_format={"type": "json_object", "schema": _alts_schema(r, a.alts)})["choices"][0]["message"]["content"]; calls += 1
            alts = _parse_alts(raw, r); why_all = []
            cands = []
            for new in alts:   # deterministic guards first (cheap), then the judges on the survivors, longest first (= most fill)
                why = check_pieces(r, new)
                if why: why_all.append(why); log.append({"round": rnd + 1, "pieces": new, "result": why}); continue
                cands.append([" ".join(x.split()) for x in new])
            cands.sort(key=lambda nw: -sum(nwords(nw[int(k)]) for k in r["short"]))
            for new in cands[:a.judge_max]:
                ok, added = faithful_check(llm, "Russian", "English", r["source"], r["text"], " ".join(new)); calls += 1
                if not ok: why_all.append("added information: " + "; ".join(added)[:160]); log.append({"round": rnd + 1, "pieces": new, "result": why_all[-1]}); continue
                nat, prob = natural_check(llm, " ".join(new)); calls += 1
                if not nat: why_all.append("not natural: " + prob[:160]); log.append({"round": rnd + 1, "pieces": new, "result": why_all[-1]}); continue
                accepted = new; log.append({"round": rnd + 1, "pieces": new, "result": "accepted"}); break
            if accepted: break
            if not alts: why_all = ["answer is not a JSON list of objects with one key per requested piece number"]
            fb = ("None of these can be used: " + " | ".join(sorted(set(why_all)))[:600] + f". Give {a.alts} new, different alternatives that fix this. Answer with the JSON only.")
            msgs = msgs + [{"role": "assistant", "content": raw}, {"role": "user", "content": fb}]
        res.append({"unit": r["unit"], "accepted": accepted is not None, "pieces": accepted, "text": " ".join(accepted) if accepted else None, "log": log, "seconds": round(time.perf_counter() - t1, 1)})
        print(f"u{r['unit']:02d} {'ACCEPTED' if accepted else 'rejected'} {res[-1]['seconds']}s: {' | '.join(accepted) if accepted else (log[-1]['result'] if log else 'no candidates')}", flush=True)
    jsave(out / "rewrite.json", {"version": "v4", "qwen": os.path.basename(f), "gpu_layers": a.qwen_gpu_layers, "load_s": round(load_s, 1), "calls": calls, "units": res}); return 0


# ----------------------------------------------------------------------------------------------------------------------------------- GPU helpers
def _vram_cap():
    import torch
    tot = torch.cuda.get_device_properties(0).total_memory; torch.cuda.set_per_process_memory_fraction(min(1.0, VRAM_GB * 2 ** 30 / tot)); return tot


def _ladder():   # run_tts_phrase.gen_sentence: (seed offset, temperature) with QA retries 2, low temp 0.6; + one gap-QA alternative (salt + 1)
    return [(0, 0, 0.8), (0, 1, 0.8), (0, 2, 0.6), (1, 0, 0.8)]


def tts(a) -> int:
    import torch, soundfile as sf
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from chatterbox_fp16 import install_speech_token_guard
    from pilot_common import MODELS
    out = pathlib.Path(a.out); P = jload(out / "plan.json"); R = {r["unit"]: r for r in jload(out / "rewrite.json")["units"]}
    base = jload(jload(out / "plan.json")["l1"]); refs = base["tts"]["references"]
    t0 = time.perf_counter(); _vram_cap(); torch.cuda.reset_peak_memory_stats()
    m = ChatterboxMultilingualTTS.from_local(MODELS / "chatterbox", "cuda", t3_model="v3"); guard = install_speech_token_guard(m)
    for mod in (m.t3, m.s3gen, m.ve): mod.to(torch.float16)
    if m.conds is not None: m.conds = m.conds.to(device="cuda")
    load_s = time.perf_counter() - t0; takes = []; t1 = time.perf_counter()
    for r in P["units"]:
        rw = R.get(r["unit"])
        if not rw or not rw["accepted"]: continue
        with torch.autocast("cuda", dtype=torch.float16): m.prepare_conditionals(refs[r["speaker"]]["file"])
        for i, (salt, soff, temp) in enumerate(_ladder()):
            s = SEED + 100 * r["unit"] + 100003 * (SALT + salt) + 7919 * soff; torch.manual_seed(s)
            with torch.autocast("cuda", dtype=torch.float16): wav = m.generate(rw["text"], language_id="en", temperature=temp)
            y = wav.detach().float().cpu()[0].numpy(); f = out / f"take_u{r['unit']:02d}_{i}.wav"; sf.write(str(f), y, m.sr, subtype="FLOAT")
            takes.append({"unit": r["unit"], "i": i, "seed": s, "temperature": temp, "file": str(f), "seconds": round(len(y) / m.sr, 3), "gap_alt": salt == 1})
        print(f"u{r['unit']:02d} {len(_ladder())} takes", flush=True)
    gen_s = time.perf_counter() - t1
    jsave(out / "takes.json", {"sr": m.sr, "load_s": round(load_s, 1), "gen_s": round(gen_s, 1), "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2),
                              "guard_violations": guard.violations, "takes": takes})
    print(f"tts: load {load_s:.1f}s gen {gen_s:.1f}s peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB"); return 0


def qa(a) -> int:
    import torch, soundfile as sf
    from tts_qa import TtsQA
    from tts_align import CtcAligner
    from tts_burst import gap_voice
    from burst_aware_audio import missing_numbers
    out = pathlib.Path(a.out); T = jload(out / "takes.json"); R = {r["unit"]: r for r in jload(out / "rewrite.json")["units"]}
    t0 = time.perf_counter(); _vram_cap(); torch.cuda.reset_peak_memory_stats(); q = TtsQA(device="cuda"); al = CtcAligner.from_qa(q); load_s = time.perf_counter() - t0
    by = {}
    for t in T["takes"]: by.setdefault(t["unit"], []).append(t)
    chosen = {}; t1 = time.perf_counter()
    for uid, ts in by.items():
        text = R[uid]["text"]
        for t in ts:
            y, sr = sf.read(t["file"], dtype="float32"); r = q.assess(text, y, sr, "en"); sp = al.align(y, sr, text); gv = gap_voice(y, sr, sp) if sp else []
            t.update({"bad": bool(r["bad"]), "score": r["score"], "reasons": r["reasons"], "whisper": r["whisper"][:160], "spans": sp, "gap_voice_s": round(sum(b - a_ for a_, b, _ in gv), 3)})
            if missing_numbers(text, r["whisper"]):   # r2: 01 u14 take 0 said "0.0 %" for "0.5 percent" at CER 0.078 (< the QA threshold) and was chosen
                t["bad"] = True; t["reasons"] = list(t["reasons"] or []) + [f"number missing in the ASR: {missing_numbers(text, r['whisper'])}"]
        ladder = [t for t in ts if not t["gap_alt"]]; alt = [t for t in ts if t["gap_alt"]]
        best = next((t for t in ladder if not t["bad"]), min(ladder, key=lambda t: (t["bad"], t["score"])))
        if best["gap_voice_s"] > 0 and alt and not alt[0]["bad"] and alt[0]["spans"] and alt[0]["gap_voice_s"] < best["gap_voice_s"]: best = alt[0]
        if not best["spans"]: chosen[uid] = None; continue
        y, sr = sf.read(best["file"], dtype="float32"); f = out / f"tts_u{uid:02d}_refit_raw.wav"; sf.write(str(f), y, sr, subtype="PCM_16")
        chosen[uid] = {"file": str(f), "take": best["i"], "seed": best["seed"], "temperature": best["temperature"], "bad": best["bad"], "score": best["score"], "spans": best["spans"],
                       "gap_voice_s": best["gap_voice_s"], "whisper": best["whisper"]}
        print(f"u{uid:02d} take {best['i']} bad {best['bad']} score {best['score']} gap {best['gap_voice_s']} | {best['whisper'][:90]}", flush=True)
    qa_s = time.perf_counter() - t1
    jsave(out / "takes.json", T); jsave(out / "chosen.json", {"load_s": round(load_s, 1), "qa_s": round(qa_s, 1), "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 2 ** 30, 2), "units": chosen})
    print(f"qa: load {load_s:.1f}s qa {qa_s:.1f}s peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB"); return 0


# ----------------------------------------------------------------------------------------------------------------------------------- build (CPU)
def unit_stats(gaps: dict, man: dict, uid: int) -> dict:
    u = next(x for x in man["units"] if x["id"] == uid); s, e = u["slot"]["start"] - 0.05, u["slot"]["end"] + 0.6
    G = [g for g in gaps["gaps"] if g["unit"] == uid or (s <= 0.5 * (g["t0"] + g["t1"]) <= e)]
    pls = [pl for pl in u["alignment"]["placements"] if pl.get("placed")]
    return {"residual_s": round(sum(g["seconds"] for g in G if g["seconds"] >= 0.35), 2), "max_gap": max([g["seconds"] for g in G] + [0]),
            "atempo_max": max([pl.get("atempo", 1.0) for pl in pls] + [1.0]), "ls_frames": sum(g["ls_frames_fixed"] for g in G if g["seconds"] >= 0.35)}


def build(a) -> int:
    out = pathlib.Path(a.out); P = jload(out / "plan.json"); R = {r["unit"]: r for r in jload(out / "rewrite.json")["units"]}; C = jload(out / "chosen.json")["units"]
    base = jload(a.base); py = sys.executable; flags = a.l1_flags.split()
    def run(units_on: list, tag: str) -> tuple:
        m2 = copy.deepcopy(base)
        for u in m2["units"]:
            if u["id"] not in units_on: continue
            r = next(x for x in P["units"] if x["unit"] == u["id"]); c = C[str(u["id"])]; rw = R[u["id"]]
            u["translation_fitted"] = rw["text"]; u["tts_raw_file"] = c["file"]; u["tts"] = {**u["tts"], "word_spans": c["spans"], "refit": {"seed": c["seed"], "take": c["take"], "qa_bad": c["bad"]}}
            u["refit_pieces"] = rw["pieces"]; u["refit_bursts"] = r["bursts"]; u["refit_groups"] = r["ru"]
        mp = out / f"manifest_{tag}.json"; jsave(mp, m2); o = out / f"recut_{tag}"
        subprocess.run([py, str(HERE / "recut_offline.py"), "--manifest", str(mp), "--out", str(o), *flags], check=True, capture_output=True)
        subprocess.run([py, str(HERE / "dub_gaps.py"), "--manifest", str(o / "recut.manifest.json"), "--ref-manifest", a.ref_manifest or a.base, "--tag", tag, "--out", str(out / "gaps")], check=True, capture_output=True)
        return jload(o / "recut.manifest.json"), jload(out / "gaps" / f"{P['vid']}_{tag}.gaps.json")
    cand = [int(k) for k, v in C.items() if v and not v["bad"]]
    l1_man = jload(P["l1"]); l1_g = jload(pathlib.Path(a.l1_gaps))
    m_all, g_all = run(cand, "all")
    dec = {}
    for uid in cand:
        b, n = unit_stats(l1_g, l1_man, uid), unit_stats(g_all, m_all, uid)
        ok = n["residual_s"] <= 0.7 * b["residual_s"] and n["max_gap"] <= max(0.5, b["max_gap"]) and n["atempo_max"] <= 1.15 + 1e-3
        dec[uid] = {"before": b, "after": n, "accepted": ok}; print(f"u{uid:02d} {'ACCEPT' if ok else 'reject'} residual {b['residual_s']} -> {n['residual_s']} s, max gap {b['max_gap']} -> {n['max_gap']}, atempo {n['atempo_max']}", flush=True)
    acc = [u for u, d in dec.items() if d["accepted"]]
    m_fin, g_fin = run(acc, "final")
    jsave(out / "build.json", {"candidates": cand, "accepted": acc, "decisions": dec, "final_manifest": str(out / "recut_final" / "recut.manifest.json")}); return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); ap.add_argument("stage", choices=["plan", "rewrite", "tts", "qa", "build"])
    ap.add_argument("--out", required=True); ap.add_argument("--vid"); ap.add_argument("--l1"); ap.add_argument("--residual"); ap.add_argument("--base"); ap.add_argument("--l1-flags", default="")
    ap.add_argument("--l1-gaps"); ap.add_argument("--ref-manifest", default=None, help="build: dub_gaps LS reference (default --base; face_ref.py output when the base has no video run)"); ap.add_argument("--qwen-gpu-layers", type=int, default=0, help="rewrite: Qwen layers on the GPU (-1 = all; CUDA llama.cpp build on PYTHONPATH)")
    ap.add_argument("--alts", type=int, default=5, help="rewrite v4: alternatives per Qwen call"); ap.add_argument("--rounds", type=int, default=3, help="rewrite v4: calls per unit (feedback between)")
    ap.add_argument("--judge-max", type=int, default=3, help="rewrite v4: guard-passing alternatives judged per round")
    a = ap.parse_args()
    return {"plan": plan, "rewrite": rewrite, "tts": tts, "qa": qa, "build": build}[a.stage](a)


if __name__ == "__main__":
    raise SystemExit(main())
