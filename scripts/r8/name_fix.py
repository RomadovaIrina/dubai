#!/usr/bin/env python3
"""r8 v4s: proper nouns pronounced right (user on audio-r8-v4 04: "City of SOKI" for Sochi). On a finished PRE-MIX dub (the remix input
dir of a build: dubbed_24k.wav + manifest, placements on the final timeline), per placed piece that contains a named entity of its unit
(u["ne"]: english, respellings from the pipeline's ne_guard):
  * the piece as it plays is checked by the TTS QA's Whisper: is every entity HEARD (ne_guard.heard)?
  * if not, only that piece is re-synthesised: the written form, then each respelling (Chatterbox only: "City of Sochee"; the text / subtitles
    keep "Sochi"), 2 seeds each, until the QA passes AND Whisper hears the name; CTC-trimmed (0.12 s of the take's silence at both ends);
  * it replaces the old piece IN PLACE: the same start, tempo-fitted (0.9-1.1, pitch kept) to the old span, else up to the next piece
    - 0.1 s; level-matched to the old piece (speech p95); 10-ms fades. Every other sample of the dub is bit-identical.
Writes <out>/ (symlinks + the new dub + the manifest with placement["r8_name_fix"]) and <out>/name_fix.json (every checked piece).
    python scripts/r8/name_fix.py --run <remix input dir> --out <dir> [--seed 31]"""
from __future__ import annotations
import argparse, json, pathlib, re, sys, zlib
import numpy as np, soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent
for p in (HERE.parent / "pilot", HERE.parent / "quality", HERE.parent / "r7"): sys.path.insert(0, str(p))


def trim(wv, sr, spans, pre=0.12, post=0.12, floor_db=-40.0):
    """the take from pre s before its first CTC word to post s after the last, the end moved on until the voice has decayed under floor_db
    (CTC ends a final open vowel early: 04 "Sochi" cut at -16 dB), then a 10-ms fade in / 40-ms fade out."""
    h = int(0.02 * sr); db = lambda i: 20 * np.log10(np.sqrt((wv[i:i + h] ** 2).mean()) + 1e-9)
    i0 = max(0, int((spans[0]["start"] - pre) * sr)); i1 = min(len(wv), int((spans[-1]["end"] + post) * sr)); lim = min(len(wv), int((spans[-1]["end"] + 0.8) * sr))
    while i1 + h <= lim and db(i1 - h) > floor_db: i1 += h
    seg = wv[i0:min(len(wv), i1 + int(0.04 * sr))].copy(); f, g = int(0.01 * sr), min(int(0.04 * sr), len(wv[i0:]) // 2)
    seg[:f] *= np.linspace(0, 1, f); seg[-g:] *= np.linspace(1, 0, g); return seg


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--out", required=True); ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--check-only", action="store_true"); a = ap.parse_args()
    import torch, ne_guard
    from tts_qa import TtsQA
    from tts_align import CtcAligner, hush_gaps
    import burst_align
    R, O = pathlib.Path(a.run).resolve(), pathlib.Path(a.out); O.mkdir(parents=True, exist_ok=True)
    mp = next(R.glob("*.manifest.json")); man = json.loads(mp.read_text())
    y, sr = sf.read(str(R / "dubbed_24k.wav"), dtype="float32"); y = y if y.ndim == 1 else y.mean(1); z = y.copy(); h = int(0.02 * sr)
    qa = TtsQA(); aligner = CtcAligner.from_qa(qa); m = None; cur = None; rep = []
    P = sorted([(u, p) for u in man["units"] for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed")], key=lambda x: x[1]["placed"][0])
    for q, (u, p) in enumerate(P):
        ents = [e for e in ne_guard.unit_entities(u) if all(re.search(r"\b" + re.escape(g) + r"\b", p["text"], re.I) for g in e["guard"])]
        if not ents: continue
        i0, i1 = int(p["placed"][0] * sr), min(len(y), int(p["placed"][1] * sr)); old = y[i0:i1]
        wh = qa.assess(p["text"], old, sr, "en")["whisper"]; miss = [e["english"] for e in ents if not ne_guard.heard(e, wh)]
        row = {"unit": u["id"], "piece": p["text"], "t": [round(p["placed"][0], 2), round(p["placed"][1], 2)], "entities": [e["english"] for e in ents], "heard_as": wh[:120], "misheard": miss}
        if not miss or a.check_only: rep.append(row); print(json.dumps(row, ensure_ascii=False), flush=True); continue
        if m is None:
            from chatterbox.mtl_tts import ChatterboxMultilingualTTS
            from chatterbox_fp16 import install_speech_token_guard
            m = ChatterboxMultilingualTTS.from_local(HERE.parents[1] / "models" / "chatterbox", "cuda", t3_model="v3"); install_speech_token_guard(m)
        if u["speaker"] != cur:
            ref = next(iter(sorted(R.glob(f"ref_{u['speaker']}.wav"))), None)
            if ref is None: raise SystemExit(f"no speaker reference ref_{u['speaker']}.wav in {R}")
            m.prepare_conditionals(str(ref)); cur = u["speaker"]
        nresp = max(len(e.get("respellings") or []) for e in ents); take = None
        for rk in [-1] + list(range(nresp)):   # the written form, then each respelling
            say = ne_guard.apply(p["text"], ents, rk)
            for soff in (0, 1):
                torch.manual_seed(a.seed + 100 * u["id"] + 7919 * soff + zlib.crc32(say.encode()) % 1000)
                with torch.autocast("cuda", dtype=torch.float16):
                    wav = m.generate(say, language_id="en", temperature=0.8, cfg_weight=0.5)
                wv = wav.detach().float().cpu()[0].numpy().astype(np.float32); qq = qa.assess(say, wv, m.sr, "en")
                heard = [e["english"] for e in ents if ne_guard.heard(e, qq["whisper"])]
                row.setdefault("tries", []).append({"said": say, "seed_off": soff, "bad": qq["bad"], "whisper": qq["whisper"][:100], "heard": heard})
                if qq["bad"] or len(heard) < len(ents): continue
                spans = aligner.align(wv, m.sr, say)
                if not spans: continue
                wv, _ = hush_gaps(wv, m.sr, spans); take = (trim(wv, m.sr, spans), say); break
            if take: break
        if take is None: row["result"] = "no take with the name heard: kept"; rep.append(row); print(json.dumps(row, ensure_ascii=False), flush=True); continue
        seg, say = take; Lo, Ln = (i1 - i0) / sr, len(seg) / sr
        nxt = P[q + 1][1]["placed"][0] if q + 1 < len(P) else len(y) / sr; room = max(Lo, nxt - 0.1 - p["placed"][0])
        t = Ln / Lo; tp = min(1.1, max(0.9, t)); new = burst_align.atempo(seg, sr, tp); Lnew = len(new) / sr
        if Lnew > room: new = burst_align.atempo(seg, sr, min(1.1, Ln / room)); Lnew = len(new) / sr   # still too long: as fast as allowed
        Lv = lambda x: float(np.percentile(20 * np.log10(np.sqrt((x[:len(x) // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-9), 95)) if len(x) >= h else -60.0
        new = new * 10 ** ((Lv(old) - Lv(new)) / 20)   # faded by trim()
        k = int(0.01 * sr); g = np.ones(i1 - i0); g[:k] = np.linspace(1, 0, k); g[k:] = 0.0; z[i0:i1] *= g   # the old piece out
        j1 = min(len(z), i0 + len(new)); z[i0:j1] += new[:j1 - i0]
        p["r8_name_fix"] = {"said": say, "tempo": round(tp, 3), "old_span": row["t"], "new_end": round(p["placed"][0] + Lnew, 3)}
        p["placed"] = [p["placed"][0], round(p["placed"][0] + Lnew, 3)]
        row.update(result="re-taken", said=say, tempo=round(tp, 3), new_end=round(p["placed"][1], 2))
        rep.append(row); print(json.dumps(row, ensure_ascii=False), flush=True)
    for f in R.iterdir():
        if f.name in ("dubbed_24k.wav", "dubbed_16k.wav") or f.name.endswith(".manifest.json"): continue
        (O / f.name).unlink(missing_ok=True); (O / f.name).symlink_to(f.resolve())
    if any(r.get("result") == "re-taken" for r in rep):
        sf.write(str(O / "dubbed_24k.wav"), z, sr, subtype="FLOAT")
        import librosa
        sf.write(str(O / "dubbed_16k.wav"), librosa.resample(z, orig_sr=sr, target_sr=16000), 16000, subtype="PCM_16")
    else:   # nothing to fix: the very same dub
        for f in ("dubbed_24k.wav", "dubbed_16k.wav"): (O / f).unlink(missing_ok=True); (O / f).symlink_to((R / f).resolve())
    man["r8_name_fix"] = {"checked": len(rep), "re_taken": [r for r in rep if r.get("result") == "re-taken"]}
    (O / mp.name).write_text(json.dumps(man, indent=1, ensure_ascii=False)); (O / "name_fix.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
