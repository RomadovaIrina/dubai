#!/usr/bin/env python3
"""r13 spec 0.9 / 6.4 / 7.2 / 12.2: RU -> every Chatterbox-multilingual language, measured locally (no external key).
  sents   RU sentences from the real test videos (production ASR, whole file), deduplicated, 4..30 words
  mt      Qwen2.5-14B-Instruct Q5_K_M (llama.cpp CUDA), the PRODUCTION translation prompt (SYSTEM_PROMPT + "Translate from Russian to X:")
          per language: script check (share of letters in the expected script, Cyrillic leak), empty / preamble rate,
          back-translation X -> RU by the same model + chrF(source, back), LLM judge adequacy / fluency 1..5 (the same 14B: self-judge, optimistic)
  tts     Chatterbox Multilingual v3 (the production TTS, voice = a 10 s clip of a test speaker), then faster-whisper large-v3 forced to the language:
          CER(text, ASR) after normalisation (12.2), plus the duration per character (babble / truncation)
    python scripts/r13/lang_eval.py sents --out DIR wav16 ...
    python scripts/r13/lang_eval.py mt  --out DIR [--langs de,es,...] [--n 24]
    python scripts/r13/lang_eval.py tts --out DIR --ref voice.wav [--langs ...] [--n 12]
    python scripts/r13/lang_eval.py report --out DIR       -> DIR/LANGS.md"""
import argparse, glob, json, os, pathlib, re, sys, time, unicodedata
from collections import Counter
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "pilot"))

LANGS = ["en", "de", "es", "fr", "it", "pt", "tr", "pl", "nl", "sv", "da", "no", "fi", "el", "he", "ar", "hi", "zh", "ja", "ko", "ms", "sw"]
SCRIPT = {"el": "GREEK", "he": "HEBREW", "ar": "ARABIC", "hi": "DEVANAGARI", "zh": "CJK", "ja": "JA", "ko": "HANGUL"}   # else LATIN
SYSTEM_PROMPT = "You are a professional translator. Return only the translation, with no explanation."   # = run_clean_pipeline_05.SYSTEM_PROMPT


def script_of(ch):
    n = unicodedata.name(ch, "")
    if "CJK" in n: return "CJK"
    if "HIRAGANA" in n or "KATAKANA" in n: return "KANA"
    for s in ("LATIN", "CYRILLIC", "GREEK", "HEBREW", "ARABIC", "DEVANAGARI", "HANGUL"):
        if n.startswith(s) or f" {s} " in f" {n} ": return s
    return "OTHER"


def script_share(text, lang):
    c = Counter(script_of(ch) for ch in text if ch.isalpha()); tot = sum(c.values()) or 1
    want = SCRIPT.get(lang, "LATIN"); ok = c["CJK"] + c["KANA"] if want == "JA" else c[want]
    return round(ok / tot, 3), round(c["CYRILLIC"] / tot, 3)


def chrf(hyp, ref, n=6, beta=2.0):
    """chrF (character n-grams 1..6, beta 2), whitespace removed, as sacrebleu's default"""
    h, r = re.sub(r"\s+", "", hyp.lower()), re.sub(r"\s+", "", ref.lower()); P = R = 0.0; k = 0
    for i in range(1, n + 1):
        hc = Counter(h[j:j + i] for j in range(len(h) - i + 1)); rc = Counter(r[j:j + i] for j in range(len(r) - i + 1))
        if not hc or not rc: continue
        m = sum((hc & rc).values()); P += m / sum(hc.values()); R += m / sum(rc.values()); k += 1
    if not k: return 0.0
    P, R = P / k, R / k
    return round(100 * (1 + beta ** 2) * P * R / (beta ** 2 * P + R), 1) if P + R else 0.0


def norm_txt(s, lang):
    s = unicodedata.normalize("NFKC", s.lower())
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("P") and ch not in "«»„“”‘’\"'`¿¡")
    s = re.sub(r"\s+", " ", s).strip()
    return s.replace(" ", "") if lang in ("zh", "ja") else s


def cer(ref, hyp):
    a, b = list(ref), list(hyp); d = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        p, d[0] = d[0], i
        for j, y in enumerate(b, 1): p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (x != y))
    return d[len(b)] / max(1, len(a))


def llm():
    sys.path.insert(0, str(ROOT / "third_party" / "llama_cuda"))
    from llama_cpp import Llama
    f = sorted(glob.glob(str(ROOT / "models" / "qwen2.5-14b-instruct-gguf" / "*q5_k_m*.gguf")))[0]
    return Llama(model_path=f, n_ctx=4096, n_gpu_layers=-1, n_threads=8, verbose=False)


def chat(m, msgs, mt=400):
    r = m.create_chat_completion(msgs, max_tokens=mt, temperature=0.0)
    t = " ".join(r["choices"][0]["message"]["content"].split()).strip()
    return t[1:-1].strip() if len(t) >= 2 and t[0] in "\"«“" and t[-1] in "\"»”" else t


def cmd_sents(a):
    import run_clean_pipeline_05 as P
    out = []; seen = set()
    for w in a.wavs:
        speech = P.run_vad(pathlib.Path(w)); P.ASR_OPTS.update(hallu_filter="on", segments="off"); segs, _ = P.run_asr(pathlib.Path(w), "ru", speech)
        text = " ".join(s["text"] for s in segs)
        for s in re.split(r"(?<=[.!?…])\s+", text):
            s = s.strip(); k = norm_txt(s, "ru")
            if 4 <= len(s.split()) <= 30 and k not in seen: seen.add(k); out.append({"src": pathlib.Path(w).stem, "ru": s})
    pathlib.Path(a.out, "sents.json").write_text(json.dumps(out, ensure_ascii=False, indent=1)); print(f"{len(out)} sentences")


JUDGE = ("You grade a translation from Russian to {L}. Source (Russian): {S}\nTranslation ({L}): {T}\n"
         "Answer with two integers 1-5 separated by a space: ADEQUACY (meaning preserved, nothing added or lost) and FLUENCY (natural, correct {L}). Only the two numbers.")


def cmd_mt(a):
    import run_clean_pipeline_05 as P
    sents = json.loads(pathlib.Path(a.out, "sents.json").read_text())[:a.n]; m = llm(); langs = a.langs.split(",") if a.langs else LANGS
    for lg in langs:
        f = pathlib.Path(a.out, f"mt_{lg}.json")
        if f.exists(): continue
        L = P.lang_name(lg); rows = []; t0 = time.time()
        for s in sents:
            tr = chat(m, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Translate from Russian to {L}:\n{s['ru']}"}])
            back = chat(m, [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"Translate from {L} to Russian:\n{tr}"}])
            j = chat(m, [{"role": "user", "content": JUDGE.format(L=L, S=s["ru"], T=tr)}], 10); nums = [int(x) for x in re.findall(r"[1-5]", j)[:2]]
            sh, cy = script_share(tr, lg)
            rows.append({"ru": s["ru"], "tr": tr, "back": back, "chrf_back": chrf(back, s["ru"]), "script_ok": sh, "cyrillic": cy,
                         "judge": nums if len(nums) == 2 else None, "preamble": bool(re.match(r"(?i)(here is|translation:|sure|voici|hier ist|aquí)", tr))})
        pathlib.Path(f).write_text(json.dumps({"lang": lg, "name": L, "s": round(time.time() - t0, 1), "rows": rows}, ensure_ascii=False, indent=1))
        js = [r["judge"] for r in rows if r["judge"]]
        print(f"== mt {lg}: chrF(back) {sum(r['chrf_back'] for r in rows) / len(rows):.1f}  adequacy {sum(x[0] for x in js) / max(1, len(js)):.2f} "
              f"fluency {sum(x[1] for x in js) / max(1, len(js)):.2f}  script {min(r['script_ok'] for r in rows):.2f}  {time.time() - t0:.0f} s", flush=True)


def cmd_tts(a):
    import numpy as np, soundfile as sf, torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from faster_whisper import WhisperModel
    langs = a.langs.split(",") if a.langs else LANGS; jobs = {}
    for lg in langs:
        f = pathlib.Path(a.out, f"mt_{lg}.json")
        if f.exists() and not pathlib.Path(a.out, f"tts_{lg}.json").exists(): jobs[lg] = json.loads(f.read_text())["rows"][:a.n]
    if not jobs: return
    tts = ChatterboxMultilingualTTS.from_local(ROOT / "models" / "chatterbox", "cuda", t3_model="v3"); tts.prepare_conditionals(a.ref)
    wd = pathlib.Path(a.out, "wav"); wd.mkdir(exist_ok=True); made = {}
    for lg, rows in jobs.items():
        made[lg] = []
        for i, r in enumerate(rows):
            torch.manual_seed(1234 + i); t0 = time.time()
            w = tts.generate(r["tr"], language_id=lg).squeeze(0).cpu().numpy(); p = wd / f"{lg}_{i:02d}.wav"; sf.write(p, w, tts.sr)
            made[lg].append({"i": i, "text": r["tr"], "wav": str(p), "dur_s": round(len(w) / tts.sr, 2), "gen_s": round(time.time() - t0, 2)})
        print(f"   tts {lg} {len(rows)} done", flush=True)
    del tts; torch.cuda.empty_cache()
    asr = WhisperModel(str(ROOT / "models" / "whisper-large-v3"), device="cuda", compute_type="int8_float16")
    for lg, rows in made.items():
        for r in rows:
            segs, _ = asr.transcribe(r["wav"], language=lg, beam_size=5, vad_filter=False); hyp = " ".join(s.text for s in segs)
            r["asr"] = hyp.strip(); r["cer"] = round(cer(norm_txt(r["text"], lg), norm_txt(hyp, lg)), 4); r["s_per_char"] = round(r["dur_s"] / max(1, len(norm_txt(r["text"], lg))), 4)
        c = sorted(r["cer"] for r in rows)
        pathlib.Path(a.out, f"tts_{lg}.json").write_text(json.dumps({"lang": lg, "rows": rows}, ensure_ascii=False, indent=1))
        print(f"== tts {lg}: CER mean {sum(c) / len(c):.3f} median {c[len(c) // 2]:.3f} max {c[-1]:.3f}  bad(>0.2) {sum(x > 0.2 for x in c)}/{len(c)}", flush=True)


def cmd_report(a):
    import run_clean_pipeline_05 as P
    rows = []
    for lg in LANGS:
        mt = json.loads(pathlib.Path(a.out, f"mt_{lg}.json").read_text()) if pathlib.Path(a.out, f"mt_{lg}.json").exists() else None
        tt = json.loads(pathlib.Path(a.out, f"tts_{lg}.json").read_text()) if pathlib.Path(a.out, f"tts_{lg}.json").exists() else None
        r = {"lang": lg, "name": P.lang_name(lg)}
        if mt:
            R = mt["rows"]; js = [x["judge"] for x in R if x["judge"]]
            r.update(n=len(R), chrf_back=round(sum(x["chrf_back"] for x in R) / len(R), 1), adequacy=round(sum(x[0] for x in js) / max(1, len(js)), 2),
                     fluency=round(sum(x[1] for x in js) / max(1, len(js)), 2), script_bad=sum(x["script_ok"] < 0.95 for x in R), preamble=sum(x["preamble"] for x in R))
        if tt:
            c = [x["cer"] for x in tt["rows"]]; r.update(tts_n=len(c), cer_mean=round(sum(c) / len(c), 3), cer_median=sorted(c)[len(c) // 2], cer_bad=sum(x > 0.2 for x in c))
        rows.append(r)
    json.dump(rows, open(pathlib.Path(a.out, "langs_table.json"), "w"), indent=1)
    L = ["| lang | chrF back | adequacy | fluency | wrong script | TTS CER mean | median | bad >0.2 |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['lang']} {r['name']} | {r.get('chrf_back', '–')} | {r.get('adequacy', '–')} | {r.get('fluency', '–')} | {r.get('script_bad', '–')}/{r.get('n', '–')} "
                 f"| {r.get('cer_mean', '–')} | {r.get('cer_median', '–')} | {r.get('cer_bad', '–')}/{r.get('tts_n', '–')} |")
    pathlib.Path(a.out, "langs_table.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


# 6.5 routing proposal from the measurements (thresholds = the EN numbers of the same test, which the user accepted by ear, with a margin)
ROUTE_RULES = {"mt_external_if": "adequacy < 4.0 or chrF(back) < 0.85 x EN or wrong script in > 1 sentence",
               "tts_donor_if": "TTS CER mean > 0.10 or > 25 % of the sentences with CER > 0.20"}


def cmd_routes(a):
    rows = {r["lang"]: r for r in json.loads(pathlib.Path(a.out, "langs_table.json").read_text())}; en = rows.get("en", {})
    table = {"source": "r13 local measurements (scripts/r13/lang_eval.py; LANGS.md)", "rules": ROUTE_RULES, "langs": {}}
    for lg, r in rows.items():
        if "adequacy" not in r: continue
        mt_bad = (r["adequacy"] < 4.0 or (en.get("chrf_back") and r["chrf_back"] < 0.85 * en["chrf_back"]) or r.get("script_bad", 0) > 1)
        tts_bad = "cer_mean" in r and (r["cer_mean"] > 0.10 or r["cer_bad"] > 0.25 * r["tts_n"])
        table["langs"][lg] = {"mt": "openai" if mt_bad else "qwen", "tts": "donor_vc" if tts_bad else "chatterbox",
                              "adequacy": r["adequacy"], "chrf_back": r["chrf_back"], "cer_mean": r.get("cer_mean")}
    pathlib.Path(a.out, "lang_routes.json").write_text(json.dumps(table, indent=1)); print(json.dumps(table["langs"], indent=0)[:3000])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["sents", "mt", "tts", "report", "routes"]); ap.add_argument("--out", required=True)
    ap.add_argument("--langs", default=""); ap.add_argument("--n", type=int, default=24); ap.add_argument("--ref"); ap.add_argument("wavs", nargs="*")
    a = ap.parse_args(); pathlib.Path(a.out).mkdir(parents=True, exist_ok=True)
    {"sents": cmd_sents, "mt": cmd_mt, "tts": cmd_tts, "report": cmd_report, "routes": cmd_routes}[a.cmd](a)


if __name__ == "__main__":
    main()
