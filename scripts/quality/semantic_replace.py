#!/usr/bin/env python3
"""Offline re-placement of a thought-policy run with pauses at SEMANTIC chunk boundaries (2026-10-14, CPU only).

Reviewer (after R10): the speech of R10 is right (one complete thought, natural tempo), but with one part per sentence the dub is silent for
more than half of the source speech on 03 / 04 (the Russian speaker talks in pieces with long pauses, the English runs through them and then
stops), and every face fix for "source talks, dub silent" failed. Rule from the reviewer: a pause may be inserted only where a meaning chunk
ends ("смысловой кусок закончился — паузу можно хоть 15 минут"), never inside one ("сегодня ……… хороший день" is not how people talk).

Per sentence of an existing run (its whole-sentence TTS part, unchanged audio):
  1. Qwen (CPU) marks the LARGEST self-contained meaning chunks of the English sentence (generic few-shots; the words must come back
     unchanged, else no marks); 2. CTC word spans (wav2vec2, CPU); 3. thought_policy.cover_split restricted to those chunk boundaries puts each
     chunk on the start of a source burst so that the dub covers as much source speech as possible, every chunk at natural speed in its
     window; 4. align_units_thought (slow floor as the run) -> dub track, background bed, listening mp4 (source video + the new mix: the
     lips are NOT re-synced, it is for judging the rhythm).
Writes <out>/<id>_semantic.{json,wav,mp4} and prints source-speech-without-dub before / after.

    CUDA_VISIBLE_DEVICES= python scripts/quality/semantic_replace.py --manifest RUN/final/04_final.manifest.json --out DIR [--slow-floor 0.9]
"""
from __future__ import annotations
import argparse, json, os, pathlib, re, shutil, subprocess, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE))
SR = 24000

CHUNK_SYSTEM = "You split spoken English sentences into meaning chunks for dubbing. You answer with the sentence only, chunks separated by ' | '."
CHUNK_INSTR = ("Split the sentence into the LARGEST self-contained meaning chunks: after a chunk the speaker may pause for a long time and the listener must "
               "not feel that anything is left hanging. A split is allowed after a complete clause, after an introductory phrase (time, place, 'Well,', "
               "'Therefore,', an address like 'Friends,'), between the items of a list, and before 'and / but / which / so that' starting a new clause. Never "
               "split inside a noun phrase, between a verb and its object or complement, between a preposition and its object, inside a number with its "
               "unit, or after 'the / a / to / of / for / with / our / my / how / what'. Keep every word exactly unchanged. If the sentence is one chunk, "
               "return it unchanged.")
CHUNK_SHOTS = [   # generic examples, NOT from the five test videos
    ("After the meeting we can finally discuss the new budget for next year.", "After the meeting | we can finally discuss the new budget for next year."),
    ("Yesterday, in the old town, my brother bought a red bicycle and rode it home.", "Yesterday, | in the old town, | my brother bought a red bicycle | and rode it home."),
    ("This little shop sells fresh bread every morning.", "This little shop sells fresh bread every morning."),
    ("Next week I will send you the report, which explains all the results in detail.", "Next week | I will send you the report, | which explains all the results in detail."),
    ("Friends, hello everyone, welcome to my channel.", "Friends, | hello everyone, | welcome to my channel."),
    ("We will test how our new camera works in the rain.", "We will test how our new camera works in the rain.")]


def chunk_marks(llm, text: str) -> tuple[set[int] | None, str]:
    """Indices of the raw words AFTER which a meaning chunk ends (validated: same words), the raw answer."""
    words = text.split()
    if len(words) < 4:
        return set(), text
    msgs = [{"role": "system", "content": CHUNK_SYSTEM}]
    for a, b in CHUNK_SHOTS:
        msgs += [{"role": "user", "content": f"{CHUNK_INSTR}\n{a}"}, {"role": "assistant", "content": b}]
    msgs.append({"role": "user", "content": f"{CHUNK_INSTR}\n{text}"})
    out = llm.create_chat_completion(msgs, max_tokens=len(words) * 4 + 40, temperature=0.0)["choices"][0]["message"]["content"].strip()
    toks = out.replace("|", " | ").split(); got, marks = [], set()
    for t in toks:
        if t == "|":
            if got: marks.add(len(got) - 1)
        else:
            got.append(t)
    key = lambda w: re.sub(r"[^\w']", "", w.lower())
    if [key(w) for w in got] != [key(w) for w in words]:
        return None, out
    marks.discard(len(words) - 1)
    # the reviewer's rule, enforced even if Qwen breaks it: no pause after a function / wh-word ("what I want to say is that |")
    weak = {"the", "a", "an", "to", "of", "for", "with", "our", "my", "your", "his", "her", "their", "its", "how", "what", "that", "which", "who", "whom",
            "is", "are", "was", "were", "will", "would", "can", "be", "and", "or", "but", "in", "on", "at", "by", "from", "into", "about", "as", "than", "very"}
    marks = {i for i in marks if key(words[i]) not in weak or re.search(r"[,;:.!?…]['\"»)]*$", words[i])}   # "wished for," ends a clause: allowed
    return marks, out


def main() -> int:
    import soundfile as sf
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True); ap.add_argument("--slow-floor", type=float, default=0.9)
    ap.add_argument("--min-pause", type=float, default=0.3, help="a chunk boundary is used only on a source pause at least this long (s)")
    ap.add_argument("--min-gain", type=float, default=0.5, help="split a sentence only when it covers this much more source speech (s)")
    ap.add_argument("--max-parts", type=int, default=6); ap.add_argument("--no-mp4", action="store_true")
    ap.add_argument("--outside-w", type=float, default=0.15, help="penalty per second of dub over source silence in the placement score")
    a = ap.parse_args()
    import torch
    torch.set_num_threads(12)
    import run_clean_pipeline_05 as rp
    import thought_policy as tp
    from tts_align import CtcAligner
    from tts_qa import norm
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    man = json.loads(pathlib.Path(a.manifest).read_text()); out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    vid = pathlib.Path(man["input"]).stem; work = out / f"work_{vid}"; work.mkdir(exist_ok=True); dur = man["source"]["duration_s"]
    units = [dict(u) for u in man["units"]]
    plans = rp.burst_plan(units, man["asr"]["segments"], man["speech_intervals"], dur, 0.0, True)
    w2v = str(rp.MODELS / "wav2vec2-base-960h")
    aligner = CtcAligner(Wav2Vec2Processor.from_pretrained(w2v), Wav2Vec2ForCTC.from_pretrained(w2v).eval(), device="cpu")
    times = {}; llm, _ = rp.load_llm(times)
    o = tp.ThoughtOpts({"tempo"}, slow_floor=a.slow_floor)
    log = []
    for u, p in zip(units, plans):
        part = u["tts_parts"][0]; y, sr = sf.read(part["file"], dtype="float32"); assert sr == SR
        text = " ".join(x["text"] for x in u["tts_parts"]); raw_words = text.split()
        spans = aligner.align(y, sr, text)
        marks, ans = chunk_marks(llm, text)
        rec = {"id": u["id"], "text": text, "qwen": ans, "marks": sorted(marks) if marks is not None else None, "bursts": [list(map(float, b)) for b in p["bursts"]]}
        res = None
        if spans is not None and marks is not None and len(p["bursts"]) >= 2:
            # cover_split with punctuation replaced by the meaning-chunk marks: mark the allowed words with a trailing ',' for the candidate test only
            flagged = [(w if tp.PUNCT_END.search(w) else w + ",") if i in marks else re.sub(r"[,;:—–.!?…]+(['\"»)]*)$", r"\1", w) for i, w in enumerate(raw_words)]
            res = tp.cover_split(spans, flagged, p["bursts"], p["next_start"], norm, o.gap, o.inner_gap, o.tol, max_parts=a.max_parts,
                                 min_gain=a.min_gain, min_pause=a.min_pause, min_words=1, free_start=True, outside_w=a.outside_w, split_cost=0.1, late_cost=0.1)
            # cover_split also accepts real punctuation that Qwen did not mark: keep only the marked boundaries
            if res is not None:
                acc = 0; ok = True; idx = 0; tok_end = []
                for w in raw_words:
                    acc += len(norm(w)); tok_end.append(acc)
                pos = np.cumsum(res[0])[:-1]
                ends = {tok_end[i]: i for i in range(len(raw_words))}
                if any(ends.get(int(k)) not in marks for k in pos): res = None
        if res is None:
            counts, groups = [len(spans) if spans else len(norm(text))], [list(range(len(p["bursts"])))]
        else:
            counts, groups = res
        if spans is not None and sum(counts) == len(spans):
            pieces, _e = tp.cut_thought(y, sr, spans, counts, onset_s=None, pre_s=0.0)
            pieces[0] = (0.0, pieces[0][1]); pieces[-1] = (pieces[-1][0], len(y) / sr)   # the part file is already edge-trimmed
        else:
            pieces = [(0.0, len(y) / sr)]; counts = [counts[0]]; groups = [list(range(len(p["bursts"])))]
        words_per = []; k0 = 0; idx = 0; acc = 0
        for c in counts:
            ws = []
            while idx < len(raw_words) and acc < k0 + c:
                acc += len(norm(raw_words[idx])); ws.append(raw_words[idx]); idx += 1
            words_per.append(" ".join(ws)); k0 += c
        u["tts_parts"] = []
        for k, ((s0, s1), txt) in enumerate(zip(pieces, words_per)):
            seg = tp.cosine_fade(y[int(s0 * sr):int(s1 * sr)], sr, 5.0 if k == 0 else 8.0, 20.0 if k + 1 == len(pieces) else 8.0)
            f = work / f"sem_u{u['id']:02d}_p{k}.wav"; sf.write(str(f), seg, sr)
            u["tts_parts"].append({"file": str(f), "seconds": round(len(seg) / sr, 3), "text": txt})
        p["bursts"] = [(p["bursts"][g[0]][0], p["bursts"][g[-1]][1]) for g in groups]
        p["groups"] = [" ".join(p["groups"][j] for j in g) for g in groups]
        rec["chunks"] = words_per; log.append(rec)
        print(f"u{u['id']:02d} {' | '.join(words_per)}" + ("" if marks is not None else "   [qwen marks rejected]"), flush=True)
    al = tp.align_units_thought(units, plans, work, o, dur)
    track = np.zeros(int(round(dur * SR)), np.float32)
    for u in units:
        yy, _ = sf.read(u["alignment"]["file"], dtype="float32"); s0 = int(round(u["alignment"]["track_start"] * SR)); track[s0:s0 + len(yy)] += yy[: max(0, len(track) - s0)]
    w24 = out / f"{vid}_semantic_dub.wav"; sf.write(str(w24), track, SR, subtype="PCM_16")
    # metrics vs the original run
    from thought_metrics import metrics
    before = metrics(man); man2 = dict(man); man2["units"] = units; after = metrics(man2)
    keys = ("source_without_dub_s", "dub_outside_source_s", "inside_pauses_gt_0.3", "weak_pauses_gt_0.3", "between_pause_max_s", "parts_slowed", "atempo_max")
    res = {"id": vid, "before": {k: before[k] for k in keys}, "after": {k: after[k] for k in keys}, "used": al["used"], "sentences": log,
           "placements": [{"unit": u["id"], "placed": [p_["placed"] for p_ in u["alignment"]["placements"]], "texts": [p_["text"] for p_ in u["alignment"]["placements"]]} for u in units]}
    (out / f"{vid}_semantic.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({"before": res["before"], "after": res["after"]}, indent=1))
    if not a.no_mp4:   # listening clip: source video + dub voice + the run's background bed (duck 0)
        from background_bed import mix_background
        wd = pathlib.Path(man["work_dir"]); sep = {"bg": str(wd / "bg_44k.wav"), "vocals": str(wd / "vocals_44k.wav")}
        dub_iv = [tuple(p_["placed"]) for u in units for p_ in u["alignment"]["placements"]]
        mix = mix_background(sep, w24, [(v["start"], v["end"]) for v in man["speech_intervals"]], dub_iv, dur, work, duck_db=0.0, room_floor_db=-30.0)
        mp4 = out / f"{vid}_semantic_listen.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", man["input"], "-i", mix["aac"], "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy", str(mp4)], check=True)
        print("listening clip:", mp4)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
