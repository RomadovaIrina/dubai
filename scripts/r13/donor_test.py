#!/usr/bin/env python3
"""r13 spec 7.5 / 7.4 / 6.5 test against MockOpenAI (no key needed):
  cpu  T1 transient 429 / 500 -> recovered, Retry-After honoured; T2 persistent 429 -> DonorUnavailable -> local fallback, job goes on;
       T3 rate limiter: 16 requests from 4 threads at rpm 60, server cap 60 rpm -> no 429; T4 HTTP 400 -> no retry; T5 no key -> fallback;
       T6 router table (with / without a key)
  gpu  T7 donor (stand-in voice) -> ChatterboxVC -> the user's timbre: speaker similarity (Chatterbox VoiceEncoder cosine) of user vs donor
       and user vs converted, and CER of the converted speech (faster-whisper) per language
    python scripts/r13/donor_test.py cpu --out DIR
    python scripts/r13/donor_test.py gpu --out DIR --ref user_ref.wav [--langs de,fr,...]"""
import argparse, json, os, pathlib, sys, threading, time
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import donor as D  # noqa: E402

TEXT = {"de": "Wir fahren morgen früh ans Meer und bleiben dort eine ganze Woche.", "fr": "Nous partons demain matin à la mer et nous y restons une semaine entière.",
        "es": "Mañana por la mañana nos vamos al mar y nos quedamos allí una semana entera.", "en": "Tomorrow morning we are going to the sea and staying there for a whole week.",
        "tr": "Yarın sabah denize gidiyoruz ve orada bütün bir hafta kalıyoruz.", "ar": "سنذهب غدًا صباحًا إلى البحر وسنبقى هناك أسبوعًا كاملًا.",
        "hi": "कल सुबह हम समुद्र पर जा रहे हैं और वहाँ पूरा एक हफ़्ता रहेंगे।", "ja": "明日の朝、私たちは海に行って、そこで丸一週間過ごします。"}


def cpu(a):
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); R = {}
    os.environ["OPENAI_API_KEY"] = "mock-key"
    m = D.MockOpenAI(schedule=[429, 429, 500], retry_after=0.5); d = D.OpenAITTSDonor(base_url=m.base, rpm=600, base_delay=0.2, seed=1)
    t0 = time.time(); d.synth("hello", out / "t1.wav"); R["T1_transient"] = {"ok": (out / "t1.wav").stat().st_size > 44, "codes": [x["code"] for x in m.log], "stats": d.stats, "s": round(time.time() - t0, 2),
                                                                          "retry_after_honoured": d.stats["backoff_s"] >= 2 * 0.5}; m.close()
    m = D.MockOpenAI(p429=1.0, retry_after=0.2); d = D.OpenAITTSDonor(base_url=m.base, rpm=600, max_retries=3, base_delay=0.1, seed=2)
    stub = lambda text, lang, p: pathlib.Path(p).write_bytes(D._wav_bytes(__import__("numpy").zeros(2400, "float32"), 24000))
    class NoVC:
        def convert(self, *a): raise AssertionError("VC must not run without a donor track")
    rec = D.synth_with_fallback("hallo", "de", out / "t2.wav", out / "none.wav", d, NoVC(), stub)
    R["T2_persistent_429"] = {"requests": len(m.log), "engine": rec["engine"], "flag": rec["flag"], "file": (out / "t2.wav").exists(), "error": rec.get("error"), "stats": d.stats}; m.close()
    m = D.MockOpenAI(rpm_cap=60); d = D.OpenAITTSDonor(base_url=m.base, rpm=60, seed=3); t0 = time.time(); errs = []
    def worker(k):
        for i in range(4):
            try: d.synth(f"t{k}{i}", out / f"t3_{k}_{i}.wav")
            except Exception as e: errs.append(str(e))
    th = [threading.Thread(target=worker, args=(k,)) for k in range(4)]; [t.start() for t in th]; [t.join() for t in th]
    ts = [x["t"] for x in m.log]; gaps = sorted(b - c for b, c in zip(ts[1:], ts[:-1]))
    R["T3_rate_limiter"] = {"requests": len(m.log), "429": sum(x["code"] == 429 for x in m.log), "errors": errs, "wall_s": round(time.time() - t0, 2),
                            "expected_min_wall_s": round((16 - d.limiter.cap) / 1.0, 1), "median_gap_s": round(gaps[len(gaps) // 2], 3)}; m.close()
    m = D.MockOpenAI(schedule=[400]); d = D.OpenAITTSDonor(base_url=m.base, rpm=600, seed=4)
    try: d.synth("x", out / "t4.wav"); R["T4_400"] = {"raised": False}
    except D.DonorUnavailable as e: R["T4_400"] = {"raised": True, "requests": len(m.log), "msg": str(e)[:80]}
    m.close(); del os.environ["OPENAI_API_KEY"]
    d = D.OpenAITTSDonor(base_url="http://127.0.0.1:9/v1"); rec = D.synth_with_fallback("x", "de", out / "t5.wav", out / "none.wav", d, NoVC(), stub)
    R["T5_no_key"] = {"engine": rec["engine"], "flag": rec["flag"], "error": rec.get("error")}
    rt = D.Router(); R["T6_router"] = {"table_langs": len(rt.table.get("langs", {})), "no_key": {l: rt.route(l, False) for l in ("en", "de", "hi", "xx")},
                                       "with_key": {l: rt.route(l, True) for l in ("en", "de", "hi", "xx")}}
    ok = (R["T1_transient"]["ok"] and R["T1_transient"]["retry_after_honoured"] and R["T2_persistent_429"]["engine"] == "chatterbox_fallback" and R["T2_persistent_429"]["file"]
          and R["T3_rate_limiter"]["429"] == 0 and not errs and R["T4_400"]["raised"] and R["T4_400"]["requests"] == 1 and R["T5_no_key"]["engine"] == "chatterbox_fallback")
    R["PASS"] = bool(ok); (out / "donor_cpu.json").write_text(json.dumps(R, indent=1, ensure_ascii=False)); print(json.dumps(R, indent=1, ensure_ascii=False)[:3000])


def gpu(a):
    import numpy as np, soundfile as sf, torch
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    from faster_whisper import WhisperModel
    import lang_eval as LE
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); langs = a.langs.split(",")
    tts = ChatterboxMultilingualTTS.from_local(D.ROOT / "models" / "chatterbox", "cuda", t3_model="v3")   # builtin voice (conds.pt) = the stand-in donor, not the user
    def stand_in(text):
        lg = next((l for l, t in TEXT.items() if t == text), "en"); torch.manual_seed(7)
        return tts.generate(text, language_id=lg).squeeze(0).cpu().numpy().astype(np.float32), tts.sr
    m = D.MockOpenAI(schedule=[429], retry_after=0.3, stand_in=stand_in); os.environ["OPENAI_API_KEY"] = "mock-key"
    donor = D.OpenAITTSDonor(base_url=m.base, rpm=120, base_delay=0.2); vc = D.VoiceConverter("cuda")
    ve = tts.ve
    def emb(p):
        import librosa
        y, _ = librosa.load(str(p), sr=16000); return ve.embeds_from_wavs([y], sample_rate=16000)[0]
    cos = lambda x, y: float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    u = emb(a.ref); rows = {}
    for lg in langs:
        rec = D.synth_with_fallback(TEXT[lg], lg, out / f"vc_{lg}.wav", pathlib.Path(a.ref), donor, vc, lambda *x: None)
        rows[lg] = {"engine": rec["engine"], "sim_user_vs_donor": round(cos(u, emb(out / f"vc_{lg}.wav.donor.wav")), 3), "sim_user_vs_vc": round(cos(u, emb(out / f"vc_{lg}.wav")), 3)}
    del tts; torch.cuda.empty_cache(); asr = WhisperModel(str(D.ROOT / "models" / "whisper-large-v3"), device="cuda", compute_type="int8_float16")
    for lg in langs:
        for k, f in (("cer_donor", out / f"vc_{lg}.wav.donor.wav"), ("cer_vc", out / f"vc_{lg}.wav")):
            segs, _ = asr.transcribe(str(f), language=lg, beam_size=5); rows[lg][k] = round(LE.cer(LE.norm_txt(TEXT[lg], lg), LE.norm_txt(" ".join(s.text for s in segs), lg)), 3)
        print(f"== vc {lg}: {rows[lg]}", flush=True)
    m.close(); (out / "donor_gpu.json").write_text(json.dumps({"ref": a.ref, "rows": rows, "mock_log": m.log}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=["cpu", "gpu"]); ap.add_argument("--out", required=True); ap.add_argument("--ref"); ap.add_argument("--langs", default="de,fr,hi,ar")
    a = ap.parse_args(); {"cpu": cpu, "gpu": gpu}[a.cmd](a)
