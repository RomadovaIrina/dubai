#!/usr/bin/env python3
"""r13 spec 7.3 fix proposal (behind DUB_PACE_FIX=1, default OFF; the accepted audio is unchanged without it): a dubbed phrase that sounds
rushed is re-said SHORTER instead of faster.

Trigger, per placement, measured on the final dub (scripts/r13/tempo_audit.py): its speech rate (syllables / s of speech) > --k x the video's
median rate (default 1.25), OR its effective tempo (natural take -> final) outside x0.75-1.25 (the spec corridor). "The voice speeds up in one
place" on old03 = 46.4 s: a fast Chatterbox take x atempo 1.14 = x1.33 the median rate, inside the atempo corridor but heard as a rush.
Fix: Qwen 14B rephrases the line to fit the slot at the video's median pace (a syllable budget), Chatterbox says it with the same speaker
reference, at natural tempo (atempo <= x1.05), in the same placed interval. Only that interval changes:
  - the voice track: the placement's samples replaced;
  - each final mix (Spleeter and htdemucs beds): OUTSIDE the interval bit-identical (wav); inside, the gap-cleaned bed ducked under the new
    voice + the voice, scaled to the original mix's level there, 50-ms crossfades at both edges; the AAC re-encoded;
  - the manifest placement text (the subtitles follow).
    python scripts/r13/pace_fix.py --run $A/r9_<id>/<id> [--htd $A/r9_<id>_htd/<id>] [--audit audit.json] [--k 1.25] [--units 5:2,...]"""
from __future__ import annotations
import argparse, glob, json, pathlib, shutil, subprocess, sys, time
import numpy as np, soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "pilot")); sys.path.insert(0, str(ROOT / "scripts" / "quality")); sys.path.insert(0, str(HERE))
SYSTEM_PROMPT = "You are a professional translator. Return only the translation, with no explanation."


def put(p: pathlib.Path) -> pathlib.Path:
    p.unlink(missing_ok=True); return p


def atempo(y, sr, t):
    if abs(t - 1) < 0.005: return y
    r = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "-", "-af", f"atempo={t:.4f}", "-f", "f32le", "-"],
                       input=y.astype(np.float32).tobytes(), capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.float32).copy()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--htd"); ap.add_argument("--audit"); ap.add_argument("--k", type=float, default=1.25)
    ap.add_argument("--units", default="", help="explicit unit:placement-index list (else from the audit)"); ap.add_argument("--seed", type=int, default=4321); a = ap.parse_args()
    D = pathlib.Path(a.run); dirs = [D] + ([pathlib.Path(a.htd)] if a.htd else []); t0 = time.time()
    mp = next(iter(sorted(D.glob("*_r4b.manifest.json")))); man = json.loads(mp.read_text()); units = {u["id"]: u for u in man["units"]}
    if a.audit is None:
        a.audit = str(D / "tempo_audit.json"); subprocess.run([sys.executable, str(HERE / "tempo_audit.py"), "--run", str(D), "--out", a.audit], check=True, capture_output=True)
    au = json.loads(pathlib.Path(a.audit).read_text()); med = au.get("rate_median_sps")
    todo = []
    if a.units:
        for x in a.units.split(","):
            ui, pi = (int(v) for v in x.split(":")); todo.append((ui, pi, "explicit"))
    else:
        for r in au["rows"]:
            why = []
            if r.get("rate_sps") and med and r["rate_sps"] > a.k * med: why.append(f"rate x{r['rate_sps'] / med:.2f} median")
            if r.get("tempo") and not 0.75 <= r["tempo"] <= 1.25: why.append(f"tempo {r['tempo']}")
            if why:
                pls = [p for p in units[r["unit"]]["alignment"]["placements"] if p.get("placed")]
                pi = next(i for i, p in enumerate(pls) if abs(p["placed"][0] - r["placed"][0]) < 0.02); todo.append((r["unit"], pi, "; ".join(why)))
    rep = {"k": a.k, "median_sps": med, "fixed": []}
    if not todo:
        print("== pace_fix: no rushed phrase"); man["pace_fix"] = rep; put(mp).write_text(json.dumps(man, indent=1, ensure_ascii=False)); return
    for d in dirs:   # the accepted files are kept next to the new ones
        for f in ("dubbed_24k.wav", "dubbed_16k.wav", "final_mix_44k.wav", "final_mix_aac.m4a", mp.name):
            if (d / f).exists() and not (d / f"pre_pace_{f}").exists(): shutil.copy2((d / f).resolve(), d / f"pre_pace_{f}")
    from dur_translate import syllables
    import lang_guard
    sys.path.insert(0, str(ROOT / "third_party" / "llama_cuda"))
    from llama_cpp import Llama
    llm = Llama(model_path=sorted(glob.glob(str(ROOT / "models" / "qwen2.5-14b-instruct-gguf" / "*q5_k_m*.gguf")))[0], n_ctx=4096, n_gpu_layers=-1, n_threads=8, verbose=False)
    chat = lambda msgs, temp: llm.create_chat_completion(msgs, max_tokens=200, temperature=temp)["choices"][0]["message"]["content"]
    plan = []
    for ui, pi, why in todo:
        u = units[ui]; p = [q for q in u["alignment"]["placements"] if q.get("placed")][pi]; slot = p["placed"][1] - p["placed"][0]
        budget = max(2, int(med * slot * 0.9)) if med else None; cur = p["text"]; best = None
        for temp in (0.0, 0.3, 0.6):
            msg = (f"This English dubbing line is spoken too fast for its {slot:.1f}-second slot: \"{cur}\". Rephrase it in natural spoken English with at most "
                   f"{budget} syllables, keeping the meaning (the Russian original: \"{u['text']}\"). Answer with the line only.")
            t = " ".join(chat([{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": msg}], temp).split()).strip().strip("\"«»“”")
            if lang_guard.defects(t, "en"): continue
            if best is None or abs(syllables(t) - budget) < abs(syllables(best) - budget): best = t
            if syllables(t) <= budget: best = t; break
        plan.append((ui, pi, why, p, slot, budget, best or cur))
    del llm
    import torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    tts = ChatterboxMultilingualTTS.from_local(ROOT / "models" / "chatterbox", "cuda", t3_model="v3"); sr = tts.sr
    v24, s24 = sf.read(str(D / "pre_pace_dubbed_24k.wav"), dtype="float32"); voice = v24.copy()
    allp = sorted(q["placed"][0] for uu in man["units"] for q in uu["alignment"]["placements"] if q.get("placed"))
    regions = []
    for ui, pi, why, p, slot, budget, txt in plan:
        u = units[ui]; ref = D / f"ref_{u['speaker']}.wav"
        if not ref.exists(): ref = next(iter(sorted(D.glob("ref_*.wav"))))
        tts.prepare_conditionals(str(ref)); best = None
        for k in range(3):   # up to 3 seeds: the take closest to the slot without exceeding it by more than 5 %
            torch.manual_seed(a.seed + 100 * ui + 10 * pi + k); y = tts.generate(txt, language_id="en").squeeze(0).cpu().numpy()
            if best is None or abs(len(y) / sr - slot) < abs(len(best) / sr - slot): best = y
            if len(y) / sr <= slot * 1.05: best = y; break
        y = best; nat = len(y) / sr; tempo = min(1.05, max(1.0, nat / slot)); y = atempo(y, sr, tempo)
        s0 = p["placed"][0]; nxt = min([t for t in allp if t > s0 + 1e-3], default=len(voice) / sr); room = max(slot, nxt - 0.08 - s0)
        y = y[:int(room * sr)]; f = int(0.03 * sr); y[-f:] *= np.linspace(1, 0, f)
        i0, i1 = int(s0 * sr), int((s0 + slot) * sr); e = min(len(voice), i0 + len(y))
        ek = np.abs(v24[i0:i1]).max() + 1e-9; y = y * min(1.0, ek / (np.abs(y).max() + 1e-9))   # the old take's peak level
        voice[i0:i1] = 0.0; voice[i0:e] = y[:e - i0]; regions.append((s0, max(s0 + slot, s0 + len(y) / sr)))
        p["text_before_pace_fix"] = p["text"]; p["text"] = txt; p["placed"] = [s0, round(s0 + len(y) / sr, 3)]
        r = {"unit": ui, "placement": pi, "why": why, "slot_s": round(slot, 2), "budget_syll": budget, "before": p["text_before_pace_fix"], "after": txt,
             "natural_s": round(nat, 2), "atempo": round(tempo, 3), "rate_before": next((x["rate_sps"] for x in au["rows"] if x["unit"] == ui and abs(x["placed"][0] - s0) < 0.02), None),
             "rate_after": round(syllables(txt) / max(0.3, len(y) / sr * 0.85), 2)}
        rep["fixed"].append(r); print(f"  [pace] u{ui}.{pi} {why}: {r['before'][:50]!r} -> {txt[:50]!r} slot {slot:.2f}s natural {nat:.2f}s atempo {tempo:.3f}", flush=True)
    del tts; torch.cuda.empty_cache()
    for d in dirs:
        sf.write(put(d / "dubbed_24k.wav"), voice, sr, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(d / "dubbed_24k.wav"), "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(put(d / "dubbed_16k.wav"))], check=True)
    for d in dirs:   # the splice: the original mix outside the regions, bed + new voice inside at the original level
        mix, msr = sf.read(str(d / "pre_pace_final_mix_44k.wav"), dtype="float32", always_2d=True); bed, bsr = sf.read(str(d / "bg_clean_44k.wav"), dtype="float32", always_2d=True)
        bed = bed.mean(1); vn = np.interp(np.arange(int(len(voice) * msr / sr)) / msr, np.arange(len(voice)) / sr, voice).astype(np.float32)
        vo = np.interp(np.arange(int(len(v24) * msr / sr)) / msr, np.arange(len(v24)) / sr, v24).astype(np.float32); out = mix.copy(); xf = int(0.05 * msr)
        for r0, r1 in regions:
            a0, a1 = max(0, int(r0 * msr) - xf), min(len(mix), int(r1 * msr) + xf)
            def local(v):
                env = np.convolve((np.abs(v[a0:a1]) > 0.01).astype(np.float32), np.ones(int(0.3 * msr)) / int(0.3 * msr), "same")
                b = bed[a0:a1] if len(bed) >= a1 else np.pad(bed[a0:], (0, a1 - a0 - len(bed[a0:])))
                return b * 10 ** (-6 * np.clip(env * 3, 0, 1) / 20) + v[a0:a1]
            old_local = local(vo); g = np.sqrt(np.mean(mix[a0:a1].mean(1) ** 2) / (np.mean(old_local ** 2) + 1e-12))
            new = (local(vn) * g)[:, None].repeat(mix.shape[1], 1); w = np.ones(a1 - a0, np.float32); w[:xf] = np.linspace(0, 1, xf); w[-xf:] = np.linspace(1, 0, xf)
            out[a0:a1] = mix[a0:a1] * (1 - w[:, None]) + new * w[:, None]
        sf.write(put(d / "final_mix_44k.wav"), out, msr, subtype="FLOAT")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(d / "final_mix_44k.wav"), "-c:a", "aac", "-b:a", "192k", str(put(d / "final_mix_aac.m4a"))], check=True)
        outside = np.ones(len(mix), bool)
        for r0, r1 in regions: outside[max(0, int(r0 * msr) - xf):int(r1 * msr) + xf] = False
        rep.setdefault("outside_identical", {})[d.name + ("_htd" if d != D else "")] = bool(np.array_equal(out[outside], mix[outside]))
    man["pace_fix"] = {**rep, "seconds": round(time.time() - t0, 1)}
    for d in dirs:
        m_ = next(iter(sorted(d.glob("*_r4b.manifest.json"))), None)
        if m_: put(m_).write_text(json.dumps(man, indent=1, ensure_ascii=False))
    print(f"== pace_fix: {len(rep['fixed'])} phrases re-said shorter in {time.time() - t0:.0f} s; outside the regions bit-identical: {rep.get('outside_identical')}")


if __name__ == "__main__":
    main()
