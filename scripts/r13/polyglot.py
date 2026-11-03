#!/usr/bin/env python3
"""r13 spec 8d "полиглот-микс" + the generic non-English path (behind DUB_POLYGLOT, default off; the accepted EN audio is untouched).

The accepted RU->EN chain runs first and fixes WHERE every sentence is spoken (its placed intervals, after retake / burst align / lip
placement). Then every unit the map assigns another language is re-voiced IN PLACE:
  1. Qwen 14B (the production prompt) translates the unit's RUSSIAN source into the language, asked to be about as long as the English take;
     one shorter retry when the speech would need atempo > 1.25;
  2. Chatterbox multilingual v3 speaks it with the speaker's own reference (the same ref_<SPEAKER>.wav the EN takes used);
  3. it is fitted to the unit's slot (first placed start .. last placed end): longer -> atempo up to x1.25, then it may run into the gap up to
     0.1 s before the next unit (faded); shorter -> natural pace, the rest is a pause (spec 7.3: x0.75-x1.25, remainder = pause);
  4. the voice track (dubbed_24k / 16k) gets the new speech in that slot, the English one is removed there;
  5. each bed's final mix is rebuilt: the bed the EN mix used (bg_clean_44k: gap-cleaned) ducked under the new voice + the voice, then the
     production static loudness stage (r3_remix.static_loudness: -16 LUFS, true-peak guard). SIMPLER than the EN remix chain (no dub gate /
     gap caps): flagged as experimental in the manifest.
The face generator runs AFTER this (it reads dubbed_16k), so the lips follow the new language; the subtitles read the new placement text.
Map (JSON, env DUB_POLYGLOT or --map):
  {"mode": "all", "lang": "de"}                         every sentence in German (= the generic non-English target)
  {"mode": "cycle", "langs": ["en", "de", "es"]}        sentence k gets langs[k % n] ("en" = keep the accepted English take)
  {"mode": "units", "units": {"0": "de", "3": "fr"}}    by unit index; the rest stays English
  {"mode": "ranges", "ranges": [{"start": 0, "end": 20, "lang": "de"}]}   by the unit's source start time
    python scripts/r13/polyglot.py --run $A/r9_<id>/<id> [--htd $A/r9_<id>_htd/<id>] --map '{"mode": "cycle", "langs": ["en", "de", "es"]}'"""
from __future__ import annotations
import argparse, glob, json, os, pathlib, shutil, subprocess, sys, time
import numpy as np, soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "pilot")); sys.path.insert(0, str(ROOT / "scripts" / "quality")); sys.path.insert(0, str(HERE))
SYSTEM_PROMPT = "You are a professional translator. Return only the translation, with no explanation."


def lang_of(m: dict, k: int, u: dict) -> str:
    mode = m.get("mode", "all")
    if mode == "all": return m["lang"]
    if mode == "cycle": return m["langs"][k % len(m["langs"])]
    if mode == "units": return m.get("units", {}).get(str(k), m.get("default", "en"))
    if mode == "ranges":
        for r in m.get("ranges", []):
            if r["start"] <= u["start"] < r["end"]: return r["lang"]
        return m.get("default", "en")
    raise ValueError(f"unknown polyglot mode {mode}")


def slot_of(u: dict):
    pls = [p for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed")]
    return (min(p["placed"][0] for p in pls), max(p["placed"][1] for p in pls), pls) if pls else None


def atempo(y: np.ndarray, sr: int, t: float) -> np.ndarray:
    if abs(t - 1) < 0.01: return y
    p = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-", "-af", f"atempo={t:.4f}", "-f", "f32le", "-"],
                       input=y.astype(np.float32).tobytes(), capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.float32).copy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--htd"); ap.add_argument("--map", default=os.environ.get("DUB_POLYGLOT", ""))
    ap.add_argument("--seed", type=int, default=1234); a = ap.parse_args()
    import run_clean_pipeline_05 as P
    pm = json.loads(a.map); D = pathlib.Path(a.run); mp = next(iter(sorted(D.glob("*_r4b.manifest.json"))), None) or next(iter(sorted(D.glob("*.manifest.json"))))
    man = json.loads(mp.read_text()); units = man["units"]; t0 = time.time()
    plan = [(k, u, lang_of(pm, k, u)) for k, u in enumerate(units)]; todo = [(k, u, l) for k, u, l in plan if l != "en" and slot_of(u)]
    rep = {"map": pm, "units": [], "experimental_remix": True}
    if not todo:
        print("== polyglot: nothing to re-voice"); return
    for f in ("dubbed_24k.wav", "dubbed_16k.wav", "final_mix_44k.wav", "final_mix_aac.m4a", mp.name):   # the accepted English files are kept
        for d in [D] + ([pathlib.Path(a.htd)] if a.htd else []):
            src = d / f
            if src.exists() and not (d / f"en_{f}").exists(): shutil.copy2(src.resolve(), d / f"en_{f}")
    sys.path.insert(0, str(ROOT / "third_party" / "llama_cuda"))
    from llama_cpp import Llama
    llm = Llama(model_path=sorted(glob.glob(str(ROOT / "models" / "qwen2.5-14b-instruct-gguf" / "*q5_k_m*.gguf")))[0], n_ctx=4096, n_gpu_layers=-1, n_threads=8, verbose=False)
    def chat(msg):
        r = llm.create_chat_completion([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": msg}], max_tokens=400, temperature=0.0)
        t = " ".join(r["choices"][0]["message"]["content"].split()).strip()
        return t[1:-1].strip() if len(t) >= 2 and t[0] in "\"«“" and t[-1] in "\"»”" else t
    texts = {}
    for k, u, L in todo:
        en = u.get("translation_fitted") or u.get("translation") or ""; Ln = P.lang_name(L)
        texts[k] = chat(f"Translate from Russian to {Ln} for dubbing. Make it about as long, when spoken, as this English version: \"{en}\"\nRussian:\n{u['text']}")
    del llm
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    tts = ChatterboxMultilingualTTS.from_local(ROOT / "models" / "chatterbox", "cuda", t3_model="v3"); sr = tts.sr
    v24, s24 = sf.read(str(D / "en_dubbed_24k.wav"), dtype="float32"); assert s24 == sr, (s24, sr)
    voice = v24.copy(); starts = sorted(slot_of(u)[0] for u in units if slot_of(u))
    for k, u, L in todo:
        s0, s1, pls = slot_of(u); slot = s1 - s0; nxt = min([t for t in starts if t > s0 + 1e-3], default=len(voice) / sr); ref = D / f"ref_{u['speaker']}.wav"
        if not ref.exists(): ref = next(iter(sorted(D.glob("ref_*.wav"))))
        tts.prepare_conditionals(str(ref)); torch.manual_seed(a.seed + k)
        txt = texts[k]; y = tts.generate(txt, language_id=L).squeeze(0).cpu().numpy(); retried = False
        if len(y) / sr > slot * 1.25 + 0.3:   # too long even at x1.25: one shorter translation
            retried = True
            llm2 = Llama(model_path=sorted(glob.glob(str(ROOT / "models" / "qwen2.5-14b-instruct-gguf" / "*q5_k_m*.gguf")))[0], n_ctx=4096, n_gpu_layers=-1, n_threads=8, verbose=False)
            r = llm2.create_chat_completion([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content":
                 f"Shorten this {P.lang_name(L)} dubbing line so it can be spoken in {slot:.1f} seconds, keeping the meaning:\n{txt}"}], max_tokens=300, temperature=0.0)
            short = " ".join(r["choices"][0]["message"]["content"].split()).strip().strip("\"«»“”"); del llm2
            if short:
                y2 = tts.generate(short, language_id=L).squeeze(0).cpu().numpy()
                if len(y2) < len(y): y, txt = y2, short
        nat = len(y) / sr; tempo = min(1.25, max(1.0, nat / slot)); y = atempo(y, sr, tempo)
        room = max(slot, (nxt - 0.1) - s0); y = y[:int(room * sr)]
        if len(y) > int(slot * sr):   # spill into the gap: fade the last 60 ms
            f = int(0.06 * sr); y[-f:] *= np.linspace(1, 0, f)
        i0, i1 = int(s0 * sr), int(s1 * sr); fade = int(0.02 * sr)
        voice[i0:i1] = 0.0; e = min(len(voice), i0 + len(y)); voice[i0:e] = y[:e - i0]
        if i0 - fade > 0: voice[i0 - fade:i0] *= np.linspace(1, 0, fade)
        pk = np.abs(y).max() + 1e-9; ek = np.abs(v24[i0:i1]).max() + 1e-9; voice[i0:e] *= min(1.0, ek / pk)   # the English take's peak level
        for p in pls: p["text_en"] = p.get("text_en", p.get("text")); p["text"] = ""
        pls[0]["text"] = txt; pls[0]["placed"] = [s0, round(s0 + len(y) / sr, 3)]
        for p in pls[1:]: p["placed"] = None   # one placement per re-voiced unit (the subtitles read placements[].text)
        u["polyglot"] = {"lang": L, "text": txt, "slot_s": round(slot, 3), "natural_s": round(nat, 3), "atempo": round(tempo, 3), "spill_s": round(max(0, len(y) / sr - slot), 3), "shorter_retry": retried}
        rep["units"].append({"unit": k, **u["polyglot"]}); print(f"  [polyglot] u{k:02d} {L} slot {slot:.2f}s natural {nat:.2f}s atempo {tempo:.2f} | {txt[:70]}", flush=True)
    del tts; torch.cuda.empty_cache()
    def put(path: pathlib.Path):   # never write through a symlink into the shared run dir: replace the link by a file
        path.unlink(missing_ok=True); return path
    for d in [D] + ([pathlib.Path(a.htd)] if a.htd else []):
        sf.write(put(d / "dubbed_24k.wav"), voice, sr, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(d / "dubbed_24k.wav"), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(put(d / "dubbed_16k.wav"))], check=True)
    from r3_remix import static_loudness
    for d in [D] + ([pathlib.Path(a.htd)] if a.htd else []):
        bed, bsr = sf.read(str(d / "bg_clean_44k.wav"), dtype="float32")
        if bed.ndim > 1: bed = bed.mean(axis=1)
        v = np.interp(np.arange(int(len(voice) * bsr / sr)) / bsr, np.arange(len(voice)) / sr, voice).astype(np.float32)
        n = max(len(v), len(bed)); v = np.pad(v, (0, n - len(v))); bed = np.pad(bed, (0, n - len(bed)))
        env = np.convolve((np.abs(v) > 0.01).astype(np.float32), np.ones(int(0.3 * bsr)) / int(0.3 * bsr), "same")
        duck = 10 ** (-6 * np.clip(env * 3, 0, 1) / 20)   # -6 dB under the voice (the EN remix: --src-duck-db -6)
        mix = bed * duck + v
        sf.write(put(d / "final_mix_44k.wav"), mix, bsr, subtype="FLOAT"); put(d / "final_mix_aac.m4a")
        rep[f"loudness_{'htd' if a.htd and d == pathlib.Path(a.htd) else 'spleeter'}"] = static_loudness(d / "final_mix_44k.wav", d / "final_mix_aac.m4a", -16.0, -1.5)
    man["polyglot"] = {**rep, "seconds": round(time.time() - t0, 1)}; man["target_lang"] = pm.get("lang", "multi") if pm.get("mode") == "all" else "multi"
    put(mp).write_text(json.dumps(man, indent=1, ensure_ascii=False))
    if a.htd:
        hm = next(iter(sorted(pathlib.Path(a.htd).glob("*_r4b.manifest.json"))), None)
        if hm: put(hm).write_text(json.dumps(man, indent=1, ensure_ascii=False))
    print(f"== polyglot: {len(rep['units'])} units re-voiced in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
