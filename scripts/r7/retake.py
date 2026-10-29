#!/usr/bin/env python3
"""r7 v3: longer dub takes where the speaker is on screen (user: "longer dub takes"). On the r6 timeline (before lip_place), per video:
  1. select units: face-visible source speech of the unit's bursts exceeds the take's speech by >= --short-s (the dub ends early while the
     face still talks), or the r7v2 faithfulness judge scored the unit < 4;
  2. re-translate with the duration-aware Qwen 14B (dur_translate.piece_candidates: one English piece per source burst, word budgets = this voice's
     words / s x the burst seconds x --scales, so the English is fuller wording, never filler), deterministic guards (numbers, names, filler),
     the SAME judge (back-translation + meaning 1-5, missing / added; a unit whose r7v2 row lists added content is selected too): kept only with score >= max(4, the old sentence's score), nothing added,
     natural >= min(4, the old sentence's);
  3. Chatterbox: ONE whole-sentence take per kept candidate (same speaker reference as r6, QA retry ladder: 2 seeds + a lower temperature; the
     QA's Whisper must hear the text), CTC word alignment, cut between words into the pieces;
  4. each piece placed at its source burst onset (or after the previous piece + 0.12 s), tempo 1.0, inside the unit's window (up to the next
     unit's take - 0.15 s); cost = face-visible source speech left uncovered + 2 x overflow past the window + 0.3 x dub over source silence;
     the cheapest candidate replaces the old take when it beats it (short units: by >= --min-gain s; faith units: a higher judge score and
     no more than +0.5 s cost), level-matched to the old take (speech p95);
  5. the old take's span is faded out (20-ms sin ramps), the new pieces added (10-ms edge fades); manifest: translation / tts_parts / placements
     updated, u["r7_retake"] = the record. Every other unit's audio is bit-identical.
    python scripts/r7/retake.py --work <r6 work dir> --out <new work dir> --face <mediapipe npz> --faith <r7v2 metrics json> --src16 <source 16k wav>"""
import argparse, copy, json, os, pathlib, re, sys, time
import numpy as np, soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent
for p in (HERE.parent / "pilot", HERE.parent / "quality", HERE): sys.path.insert(0, str(p))
FR = 0.02


def ramp(n):
    return np.sin(np.linspace(0, np.pi / 2, n)) ** 2


NOCUT = {"the", "a", "an", "to", "of", "in", "on", "at", "for", "with", "from", "by", "and", "or", "but", "our", "my", "their", "your", "his", "her", "its",
         "we", "i", "they", "he", "she", "is", "are", "was", "were", "be", "will", "would", "can", "have", "has", "very", "just", "really", "that", "which", "who", "this", "these"}
FUNC = {"the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for", "with", "we", "i", "it", "that", "this", "so", "is", "are", "was", "be", "our", "my", "you", "they", "he", "she", "as", "by", "from", "which", "who"}
DET_KIND = re.compile(r"\b(this|that|what|a|the|any|some|every|which|these|those)\s+(kind|sort)\s+of\b", re.I)


RU_MARKERS = re.compile(r"(?<!\w)(вот|ну|короче|так сказать|то есть|значит|как бы|просто|вообще|типа|собственно|в общем)(?!\w)", re.I)
FILLER_SHOTS = [   # generic (not from the test videos): the speaker's own discourse markers kept as spoken English, nothing else added
    ("Ну, короче, мы туда приехали уже вечером.", "Well, anyway, we got there in the evening already."),
    ("Вот, значит, это наша кухня, так сказать.", "So, here, this is our kitchen, so to speak."),
    ("Просто хочу показать, как это работает.", "I just want to show how this works.")]


def filler_candidates(dt, llm, ru, old_en, budgets):
    """r8 v4: when the English is much shorter than the Russian speech, a version that keeps the speaker's OWN discourse markers / hesitation
    words (вот, ну, короче, так сказать, то есть, значит, просто ...) as natural spoken English (well, so, anyway, so to speak, that is, just),
    adding nothing else; the guard and the judge still decide"""
    if not RU_MARKERS.search(ru): return []
    sys_ = ("You are a professional dubbing translator. Translate the Russian line into natural spoken English for dubbing. Keep every discourse "
            "marker and hesitation word the speaker uses (вот, ну, короче, так сказать, то есть, значит, просто, вообще, в общем) as a natural spoken "
            "English equivalent (so, well, anyway, so to speak, that is, just), in the same place. Do not add any content, "
            "greeting, question or filler the speaker did not say. Answer with the translation only.")
    out = []
    for n in budgets[:2]:
        msgs = [{"role": "system", "content": sys_}]
        for r_, e_ in FILLER_SHOTS: msgs += [{"role": "user", "content": r_}, {"role": "assistant", "content": e_}]
        msgs.append({"role": "user", "content": f"{ru}\n(current version, too short: \"{old_en}\"; about {n} words)"})
        v = " ".join(dt._chat(llm, msgs, max_tokens=24 + 4 * n).split()).strip().strip('"«»“”')
        if v and v.lower() != old_en.lower() and v not in out: out.append(v)
    return out


def guard(dt, source, plain, cand, ents):
    """dur_translate.guard with two rules narrowed for retakes (the judge then decides): 'kind of' / 'sort of' after a determiner is not filler
    ("this kind of feature"); a dropped lower-case common word is a synonym choice, not a lost name ("variants" -> "options"): only a dropped
    capitalised / digit-bearing word (a name, a brand, a number) still blocks."""
    from burst_aware_audio import missing_numbers
    from tts_burst import dropped_words
    from tts_text import script_defects
    if not cand or "\n" in cand: return "empty"
    if cand.lower().strip(" .") == plain.lower().strip(" ."): return "identical"
    if script_defects(cand): return f"script {script_defects(cand)}"
    if missing_numbers(source, cand) and not missing_numbers(source, plain): return "number lost"
    dw = [w for w in dropped_words(plain, cand, source) if w[:1].isupper() or re.search(r"\d", w)]
    if dw: return f"name dropped {dw}"
    for e in ents or []:
        if re.search(r"\b" + re.escape(e["english"]) + r"\b", plain, re.I) and not re.search(r"\b" + re.escape(e["english"]) + r"\b", cand, re.I):
            return f"named entity dropped: {e['english']}"
    c2, p2 = DET_KIND.sub("", cand), DET_KIND.sub("", plain)
    if dt._FILLER.search(c2) and not dt._FILLER.search(p2): return f"filler '{dt._FILLER.search(c2).group(0)}'"
    if dt.words_of(cand) < 2: return "too short"
    return None


def sub_bursts(ps, man, words_all, unit_text, min_pause=0.4, min_piece=0.3):
    """the unit's source bursts cut again at every Silero pause >= min_pause inside them (04 u2: one 12-s burst holding two 2-s pauses), each
    piece with the Russian words whose midpoint falls in it; pieces without words (or < min_piece s) join their neighbour"""
    out = []
    for p in ps:
        b0, b1 = p["burst"]; iv = [(max(b0, v["start"]), min(b1, v["end"])) for v in man["speech_intervals"] if v["end"] > b0 and v["start"] < b1]
        mer = []
        for a, b in iv:
            if mer and a - mer[-1][1] < min_pause: mer[-1] = (mer[-1][0], b)
            else: mer.append((a, b))
        out += [list(x) for x in (mer or [(b0, b1)])]
    W = [[] for _ in out]; lo_t, hi_t = out[0][0] - 0.5, out[-1][1] + 0.5
    import difflib
    nrm = lambda x: re.sub(r"[^\w]", "", x.lower())
    win = [w for w in words_all if lo_t <= (w["start"] + w["end"]) / 2 < hi_t]
    mb = [blk for blk in difflib.SequenceMatcher(None, [nrm(w["word"]) for w in win], [nrm(t) for t in unit_text.split()], autojunk=False).get_matching_blocks() if blk.size]
    if mb: win = win[mb[0].a:mb[-1].a + mb[-1].size]   # only this unit's words (the window reaches into the neighbours)
    for w in win:   # every word of the unit's span goes to the nearest piece (a word timed into a pause is not lost)
        c = (w["start"] + w["end"]) / 2
        if lo_t <= c < hi_t: W[min(range(len(out)), key=lambda k: 0 if out[k][0] <= c < out[k][1] else min(abs(c - out[k][0]), abs(c - out[k][1])))].append(w["word"].strip())
    k = 0
    while k < len(out):   # merge empty / tiny pieces into the closer neighbour
        if (not W[k] or out[k][1] - out[k][0] < min_piece) and len(out) > 1:
            j = k - 1 if k > 0 and (k + 1 >= len(out) or out[k][0] - out[k - 1][1] <= out[k + 1][0] - out[k][1]) else k + 1
            lo_, hi_ = min(j, k), max(j, k); out[lo_] = [out[lo_][0], out[hi_][1]]; W[lo_] = W[lo_] + W[hi_] if lo_ == j else W[lo_] + W[hi_]
            del out[hi_]; del W[hi_]; k = max(0, lo_ - 1) if lo_ < k else k; continue
        k += 1
    return out, [" ".join(w) for w in W]


def split_by_time(spans, secs, text):
    """word counts per burst for a whole-sentence take: the cut after word i lands nearest the burst's share of the take's speech time, preferring
    a comma / a wider inter-word gap (each piece >= 1 word)"""
    n, K = len(spans), len(secs)
    if K == 1 or n < K: return [n]   # one piece at the first burst onset
    words = text.split(); t0, t1 = spans[0]["start"], spans[-1]["end"]; cum = np.cumsum(secs) / sum(secs); cuts = []; last = 0
    for k in range(K - 1):
        T = t0 + (t1 - t0) * cum[k]; best = None
        for i in range(last + 1, n - (K - 1 - k)):
            gap = spans[i]["start"] - spans[i - 1]["end"]; pun = 1 if i - 1 < len(words) and re.search(r"[,.;:!?]$", words[i - 1]) else 0
            sc = abs(spans[i]["start"] - T) - 0.5 * gap - 0.3 * pun
            if best is None or sc < best[0]: best = (sc, i)
        cuts.append(best[1]); last = best[1]
    b = [0] + cuts + [n]; return [b[k + 1] - b[k] for k in range(K)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); ap.add_argument("--out", required=True); ap.add_argument("--face"); ap.add_argument("--faith")
    ap.add_argument("--src16", required=True); ap.add_argument("--short-s", type=float, default=0.6); ap.add_argument("--min-gain", type=float, default=0.3)
    ap.add_argument("--scales", default="1.0,1.15,1.3"); ap.add_argument("--max-tts", type=int, default=3); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--select-only", action="store_true")
    ap.add_argument("--budget-scales", default="1.15,1.3,1.5", help="r8 v2: whole-sentence word budgets x the old sentence's words")
    ap.add_argument("--cap-scale", type=float, default=1.0, help="r8 v2: word cap = this x the source time at this voice's rate")
    ap.add_argument("--min-piece", type=float, default=0.3, help="r8 v2: source pieces shorter than this join a neighbour")
    ap.add_argument("--fillers", choices=["on", "off"], default="off", help="r8 v4: also a candidate keeping the speaker's own discourse markers (вот / ну / короче ...) as spoken English")
    ap.add_argument("--natural-cuts", choices=["on", "off"], default="off", help="r8 v3 (user on r8v1: 'I'm saying ..... the phrase', words cut off): cuts only at punctuation / clause starts or a real quiet take pause >= 0.15 s, never after a function word; an inserted gap <= the source pause + 0.35 s")
    ap.add_argument("--cover-all", choices=["on", "off"], default="off", help="r8 --cover-all on (management on final_r7 02 / 05: the dub keeps disappearing): off-screen source speech counts like face-visible speech"); a = ap.parse_args()
    import lip_place, dur_translate as dt
    W, O = pathlib.Path(a.work).resolve(), pathlib.Path(a.out); O.mkdir(parents=True, exist_ok=True)
    mp = next(W.glob("*_r4a.manifest.json")); man = json.loads(mp.read_text()); D = man["source"]["duration_s"]
    y, sr = sf.read(str(W / "dubbed_24k.wav"), dtype="float32"); y = y if y.ndim == 1 else y.mean(1); n = int(D / FR) + 1
    fv, _ = lip_place.face_mask(a.face, n, D)
    if a.cover_all == "on": fv[:] = True   # r8: the dub speaks whenever the source speaks, on screen or not (02 has no face at all)
    S = np.zeros(n, bool)
    for s in man["speech_intervals"]: S[int(s["start"] / FR):int(np.ceil(s["end"] / FR))] = True
    llm = None
    if a.faith == "judge":   # speed track R7_LEAN: the r5_faith judge here (same prompt, budget, retry, 14B) on the pipeline's own text; no separate pass
        os.environ.setdefault("DUB_QWEN_SIZE", "14b")
        import run_clean_pipeline_05 as rp
        rp.QWEN_GPU_LAYERS = -1; llm, qname = rp.load_llm({}); faith = {}
        for u in man["units"]:
            en = " ".join(t["text"] for t in u.get("tts_parts") or []) or u.get("translation_fitted") or u.get("translation", "")
            mt = 200 + 14 * (len(u["text"].split()) + len(en.split())); j = dt.judge(llm, "Russian", "English", u["text"], en, max_tokens=mt)
            if j.get("unparsable") is not None: j = dt.judge(llm, "Russian", "English", u["text"], en, max_tokens=2 * mt); j["retried"] = True
            faith[u["id"]] = {"id": u["id"], "source": u["text"], "english": en, **j}
        (O / "faith.json").write_text(json.dumps(list(faith.values()), indent=1, ensure_ascii=False))
    else:
        faith = {r["id"]: r for r in (json.loads(pathlib.Path(a.faith).read_text()).get("faith") or {}).get("rows", [])} if a.faith else {}
    words_all = [w for sg in man["asr"]["segments"] for w in sg.get("words") or []]
    units = man["units"]; pls_all = [(u, p) for u in units for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed")]
    syl = sum(dt.syllables(p["text"]) for _, p in pls_all); spk_s = sum(p["group_s"] for _, p in pls_all); wrd = sum(dt.words_of(p["text"]) for _, p in pls_all)
    sps, wps = syl / spk_s, wrd / spk_s   # this voice's measured rate on the r6 takes
    rows = []

    def span_of(u):
        ps = [p for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed")]
        return (min(p["placed"][0] for p in ps), max(p["placed"][1] for p in ps)) if ps else None

    def cost_of(placed, bursts, win):   # placed: [(t0, t1)] dub speech of the unit; bursts: source bursts
        Dm = np.zeros(n, bool)
        for t0, t1 in placed: Dm[int(t0 / FR):int(t1 / FR)] = True
        lo, hi = int(min(b[0] for b in bursts) / FR), int(min(D, max(max(b[1] for b in bursts), max(t1 for _, t1 in placed))) / FR)
        unc = float((S & fv & ~Dm)[lo:hi].sum() * FR); over = max(0.0, max(t1 for _, t1 in placed) - win); sil = float((Dm & ~S)[lo:hi].sum() * FR)
        return unc + 2 * over + 0.3 * sil, {"uncovered_s": round(unc, 2), "overflow_s": round(over, 2), "over_silence_s": round(sil, 2)}

    for i, u in enumerate(units):
        ps = [p for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed")]
        if not ps: continue
        bursts, srcw = sub_bursts(ps, man, words_all, u["text"], min_piece=a.min_piece); lo, hi = int(bursts[0][0] / FR), int(bursts[-1][1] / FR)
        src_face = float((S & fv)[lo:hi].sum() * FR); dub_s = sum(p["group_s"] for p in ps); fvf = float(fv[lo:hi].mean()) if hi > lo else 0
        f = (faith.get(u["id"]) or {}).get("score"); why = []
        if src_face - dub_s >= a.short_s and fvf >= 0.5: why.append("short")
        if f is not None and f < 4: why.append("faith")
        if why and (faith.get(u["id"]) or {}).get("added"): why.append("added")   # padding the judge flagged ("shall we?"); never alone (01: minor "us" / "really" retakes cost pauses)
        nxt = next((span_of(v)[0] for v in units[i + 1:] if span_of(v)), D); win = nxt - 0.15
        if why: rows.append({"u": u, "i": i, "why": why, "bursts": bursts, "srcw": srcw, "coarse": [p["burst"] for p in ps], "coarse_w": [p["source_words"] for p in ps], "win": win, "src_face_s": round(src_face, 2), "dub_s": round(dub_s, 2), "faith_old": f})
    print(f"rate {sps:.2f} syl/s {wps:.2f} words/s; selected", [(r["u"]["id"], r["why"], r["src_face_s"], r["dub_s"], r["faith_old"]) for r in rows], flush=True)
    rep = {"rate_sps": round(sps, 2), "rate_wps": round(wps, 2), "units": []}
    if a.select_only or not rows:
        (O / "retake.json").write_text(json.dumps({**rep, "selected": [{k: v for k, v in r.items() if k != "u"} | {"id": r["u"]["id"]} for r in rows]}, indent=1, ensure_ascii=False)); return
    # models: Qwen 14B (GPU), Chatterbox, TTS QA (Whisper + wav2vec2 CTC)
    import torch   # os: module level (the --faith judge path uses it earlier in main)
    os.environ.setdefault("DUB_QWEN_SIZE", "14b")
    import run_clean_pipeline_05 as rp
    import llama_cpp
    if not llama_cpp.llama_supports_gpu_offload(): sys.exit(f"CPU llama_cpp ({llama_cpp.__file__}): PYTHONPATH=third_party/llama_cuda")
    if llm is None: rp.QWEN_GPU_LAYERS = -1; llm, qname = rp.load_llm({})
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from chatterbox_fp16 import install_speech_token_guard
    from tts_qa import TtsQA, norm
    from tts_align import CtcAligner, cut_parts, hush_gaps
    import ne_guard
    import r4_pause_logic as pl
    m = ChatterboxMultilingualTTS.from_local(HERE.parents[1] / "models" / "chatterbox", "cuda", t3_model="v3"); install_speech_token_guard(m)
    if os.environ.get("DUB_TTS_FAST_T3", "0") == "1":   # speed track: same switch as the pipeline's --tts-fast-t3
        from t3_fast import install_fast_t3
        install_fast_t3(m)
    qa = TtsQA(); aligner = CtcAligner.from_qa(qa)
    refs = rp.speaker_references(units, man["diarization"]["turns"], pathlib.Path(a.src16), O); cur_spk = None
    z = y.copy(); h = int(FR * sr)
    for r in rows:
        u = r["u"]; ps = [p for p in u["alignment"]["placements"] if p.get("placed")]; t_start = time.perf_counter()
        old_text = " ".join(p["text"] for p in ps); old_j = dt.judge(llm, "Russian", "English", u["text"], old_text)
        base_cost, base_det = cost_of([tuple(p["placed"]) for p in ps], r["bursts"], r["win"])
        pieces = r["coarse_w"]; secs = [b[1] - b[0] for b in r["coarse"]]   # Qwen sees the r6 bursts (every fine pause marked made it pad: "..., comma.")
        scl = tuple(float(s) for s in a.scales.split(","))
        cands = [(" ".join(g)[0].upper() + " ".join(g)[1:], g) for _, g in dt.piece_candidates(llm, "Russian", "English", pieces, secs, wps, scales=scl[:1])]   # spoken without the joiner's commas
        cap = max(2, int(round(a.cap_scale * wps * sum(secs))))   # the source time at this voice's rate: an upper bound (slow speakers: filling it all = padding)
        budgets = sorted({min(cap, max(2, int(round(dt.words_of(old_text) * k)))) for k in (float(x) for x in a.budget_scales.split(","))})   # whole sentence, budget enforced
        cands += [(t, None) for t in dt.candidates(llm, "Russian", "English", f" {dt.SHOTS_SRC_PAUSE} ".join(pieces), secs, old_text, budgets, sum(secs))]
        if a.fillers == "on": cands += [(t, None) for t in filler_candidates(dt, llm, u["text"], old_text, budgets)]   # r8 v4: the speaker's own fillers
        seen = set(); cands = [c for c in cands if not (c[0].lower() in seen or seen.add(c[0].lower()))]
        rec = {"id": u["id"], "why": r["why"], "ru": u["text"], "old_en": old_text, "old_judge": {k: old_j.get(k) for k in ("score", "natural", "missing", "added")},
               "old_cost": round(base_cost, 2), "old_detail": base_det, "candidates": []}
        need = max(4, old_j.get("score") or 0); kept = []
        for txt, pcs in cands:
            g = guard(dt, u["text"], old_text, txt, u.get("ne") or [])
            if g: rec["candidates"].append({"en": txt, "rejected": g}); continue
            j = dt.judge(llm, "Russian", "English", u["text"], txt)
            c = {"en": txt, "pieces": pcs, "judge": {k: j.get(k) for k in ("score", "natural", "missing", "added")}, "pred_s": round(dt.syllables(txt) / sps, 2)}
            if (j.get("score") or 0) < need or j.get("added") or (j.get("natural") or 5) < min(4, old_j.get("natural") or 5): c["rejected"] = "judge"; rec["candidates"].append(c); continue
            kept.append(c); rec["candidates"].append(c)
        target = sum(secs); kept.sort(key=lambda c: abs(c["pred_s"] - target))
        best = None
        if kept and u["speaker"] != cur_spk:
            ref = refs[u["speaker"]]["file"]
            if ref: m.prepare_conditionals(ref)
            cur_spk = u["speaker"]
        kept = [dict(c, cfg=w) for c in kept[:a.max_tts] for w in (0.5, 0.3)]   # 0.3 = Chatterbox's slower, more deliberate delivery (no atempo)
        rec["candidates"] += [c for c in kept if c["cfg"] == 0.3]
        for c in kept:
            ents = [e for e in (u.get("ne") or []) if re.search(r"\b" + re.escape(e["english"]) + r"\b", c["en"], re.I)]
            toks = c["en"].split(); names = [re.sub(r"[^\w'-]", "", t) for k_, t in enumerate(toks) if k_ and t[:1].isupper() and not re.search(r"[.!?]$", toks[k_ - 1])]
            unheard_of = lambda wh: [w for w in names if w.lower() not in FUNC and len(w) >= 3 and not ne_guard.heard(w, wh) and not any(ne_guard.heard(e, wh) for e in ents if w.lower() in e["english"].lower())] \
                + [e["english"] for e in ents if not ne_guard.heard(e, wh)]   # 04: "Sochi" synthesised as "Soki"
            ladder = [(0, 0.8, -1), (1, 0.8, -1), (2, 0.6, -1)] + ([(3 + k, 0.8, k) for k in range(3)] if any(e.get("respellings") for e in ents) else [])   # then the NE respellings
            take = None
            for soff, temp, rk in ladder:
                say = ne_guard.apply(c["en"], ents, rk); torch.manual_seed(a.seed + 100 * u["id"] + 7919 * soff)
                with torch.autocast("cuda", dtype=torch.float16):
                    wav = m.generate(say, language_id="en", temperature=temp, cfg_weight=c["cfg"])
                wv = wav.detach().float().cpu()[0].numpy().astype(np.float32); q = qa.assess(say, wv, m.sr, "en"); un = unheard_of(q["whisper"])
                key = (q["bad"] or bool(un), q["score"])
                if take is None or key < take[2]: take = (wv, {**q, "unheard": un, "said": say}, key)
                if not key[0]: break
            wv, q, _ = take; c["qa"] = {"bad": q["bad"], "score": q["score"], "whisper": q["whisper"][:200], "said": q["said"]}
            if q["bad"]: c["rejected"] = "tts_qa"; continue
            if q["unheard"]: c["rejected"] = f"name not heard {q['unheard']}"; continue
            spans = aligner.align(wv, m.sr, q["said"])   # the respelled form when a respelling was spoken
            if not spans: c["rejected"] = "align"; continue
            wv, c["hushed"] = hush_gaps(wv, m.sr, spans)   # non-word murmur between words (kept by the dub gate: voiced) -> -30 dB
            raw = (" ".join(c["pieces"]) if c["pieces"] else c["en"]).split()   # piece candidates: the joiner's inserted commas are not real punctuation
            layouts = [("coarse", r["coarse"], r["coarse_w"], split_by_time(spans, [b[1] - b[0] for b in r["coarse"]], c["en"]))]
            pc_counts = [len(norm(pc)) for pc in c["pieces"]] if c["pieces"] else None
            if pc_counts and sum(pc_counts) == len(spans) and min(pc_counts) >= 1 and len(pc_counts) == len(r["coarse"]): layouts.insert(0, ("coarse", r["coarse"], r["coarse_w"], pc_counts))
            if len(r["bursts"]) != len(r["coarse"]): layouts.append(("fine", r["bursts"], r["srcw"], split_by_time(spans, [b[1] - b[0] for b in r["bursts"]], c["en"])))
            words = c["en"].split()
            for lay, bb, sw, counts in layouts:
                if sum(counts) != len(spans) or min(counts) < 1: continue
                pw = [words[sum(counts[:k]):sum(counts[:k + 1])] for k in range(len(counts))]
                if len(counts) > 1 and any(len(x) == 1 and re.sub(r"[^\w']", "", x[0].lower()) in FUNC for x in pw): continue   # a lone "The" / "and" at a pause
                cuts = np.cumsum(counts)[:-1]
                if a.natural_cuts == "on" and any(raw[e - 1].lower().strip(",.;:!?") in NOCUT or not (pl.classify((raw[e - 1], None, False), (raw[e],))[1] > 0 or spans[e]["start"] - spans[e - 1]["end"] >= 0.15)
                                                  for e in cuts if e < len(raw) and e < len(spans)): continue   # r8 v3: natural cuts only
                parts = cut_parts(wv, m.sr, spans, counts, pre_s=0.12 if a.natural_cuts == "on" else 0.06, post_s=0.12 if a.natural_cuts == "on" else 0.08, tight_inner=True); placed, t = [], None   # placed apart: no gap murmur cut off into silence
                for (seg, s0, s1, _), b in zip(parts, bb):
                    t0 = b[0] if t is None else max(b[0], t + 0.12)
                    if a.natural_cuts == "on" and t is not None: t0 = max(t + 0.12, min(t0, t + float((~S)[int(t / FR):int(b[0] / FR)].sum() * FR) + 0.35))   # gap <= source pause + 0.35
                    placed.append((round(t0, 3), round(t0 + len(seg) / m.sr, 3))); t = placed[-1][1]
                inner = sum(1 for e in cuts if e < len(raw) and pl.classify((raw[e - 1], None, False), (raw[e],))[1] <= 0)   # r6 pb1 boundary? (punctuation / clause start)
                if placed[-1][1] > r["win"] + 0.05: continue   # never into the next unit's take (two voices at once)
                cst, det = cost_of(placed, bb, r["win"]); one = sum(1 for k_ in counts if k_ == 1) if len(counts) > 1 else 0
                cst += 0.8 * inner + 0.5 * one; det = {**det, "layout": lay, "cuts_not_at_punctuation": inner, "one_word_pieces": one}   # a pause off a pb1 boundary (only to follow a speaker who pauses mid-sentence on screen) / a lone word
                pcs = [" ".join(words[sum(counts[:k]):sum(counts[:k + 1])]) for k in range(len(counts))]
                if c.get("cost") is None or cst < c["cost"]: c["cost"] = round(cst, 2); c["detail"] = det
                if best is None or cst < best[0]: best = (cst, c, parts, placed, bb, sw, pcs)
        ok = False
        if best:
            cst, c, parts, placed, bb, sw_, pcs = best
            if "short" in r["why"]: ok = cst <= base_cost - a.min_gain
            if "faith" in r["why"] and not ok: ok = (c["judge"]["score"] or 0) > (old_j.get("score") or 0) and cst <= base_cost + 0.7
            if "added" in r["why"] and not ok: ok = not c["judge"]["added"] and (c["judge"]["score"] or 0) >= (old_j.get("score") or 0) and cst <= base_cost + 0.7
        rec["accepted"] = bool(ok); rec["seconds"] = round(time.perf_counter() - t_start, 1)
        if ok:
            cst, c, parts, placed, bb, sw_, pcs = best
            a0, a1 = span_of(u); i0, i1 = int((a0 - 0.03) * sr), int((a1 + 0.03) * sr); old = z[i0:i1].copy()
            L = lambda x: float(np.percentile(20 * np.log10(np.sqrt((x[:len(x) // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-9), 95))
            gain = 10 ** ((L(old) - L(np.concatenate([p[0] for p in parts]))) / 20)
            k = int(0.02 * sr); g = np.zeros(i1 - i0); g[:k] = 1 - ramp(k); g[-k:] = ramp(k)   # old take out: 1 -> 0 over 20 ms at its head, 0 -> 1 after its tail
            z[i0:i1] = old * g
            for (seg, *_), (t0, t1) in zip(parts, placed):
                j0 = int(t0 * sr); s_ = seg * gain; f = int(0.01 * sr); s_[:f] *= ramp(f); s_[-f:] *= ramp(f)[::-1]
                z[j0:j0 + len(s_)] += s_[:max(0, len(z) - j0)]
            newp = []
            for pc, sw, b, (t0, t1), (seg, *_) in zip(pcs, sw_, bb, placed, parts):
                newp.append({"burst": b, "window": b, "phrases": [len(newp)], "group_s": round(len(seg) / m.sr, 3), "ratio": round((t1 - t0) / max(0.05, b[1] - b[0]), 3), "atempo": 1.0,
                             "placed": [t0, t1], "track_start": t0, "text": pc, "source_words": sw, "r7_retake": True})
            u["alignment"]["placements"] = newp; u["alignment"]["track_start"] = placed[0][0]
            u["translation_r6"] = u.get("translation"); u["translation"] = c["en"]; u["translation_fitted"] = c["en"]
            u["tts_parts"] = [{"text": pc} for pc in pcs]
            rec.update({"new_en": c["en"], "new_cfg": c["cfg"], "new_judge": c["judge"], "new_cost": round(cst, 2), "new_detail": c["detail"], "new_pieces": pcs, "level_gain_db": round(20 * np.log10(gain), 1)})
        u["r7_retake"] = {k: v for k, v in rec.items() if k != "candidates"}
        rep["units"].append(rec); print(json.dumps({k: v for k, v in rec.items() if k != "candidates"}, ensure_ascii=False), flush=True)
        for c in rec["candidates"]: print("    cand", c.get("rejected", "ok"), c.get("cfg"), c.get("judge"), c.get("cost"), c.get("detail"), c.get("qa", {}).get("whisper", "")[:80], "|", c["en"], flush=True)
    for f in W.iterdir():
        if f.name in ("dubbed_24k.wav", "dubbed_16k.wav") or f.name.endswith(".manifest.json"): continue
        (O / f.name).unlink(missing_ok=True); (O / f.name).symlink_to(f)
    sf.write(str(O / "dubbed_24k.wav"), z, sr, subtype="PCM_16")
    import librosa
    sf.write(str(O / "dubbed_16k.wav"), librosa.resample(z, orig_sr=sr, target_sr=16000), 16000, subtype="PCM_16")
    man["r7_retake"] = {"rate_sps": rep["rate_sps"], "changed": [r["id"] for r in rep["units"] if r["accepted"]], "selected": [r["id"] for r in rep["units"]], "qwen": qname}
    (O / mp.name).write_text(json.dumps(man, indent=1, ensure_ascii=False)); (O / "retake.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
