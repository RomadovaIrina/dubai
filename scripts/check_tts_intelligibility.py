"""Chatterbox fp16 INTELLIGIBILITY smoke: known phrases -> Chatterbox (production fp16 cast + speech-token guard) -> independent ASR (Whisper + wav2vec2 CTC) -> CER thresholds.
'Waveform is finite and non-silent' (scripts/check_chatterbox.py) does NOT catch the fp16 speech-token corruption (first_res review: unit CTC CER 0.395 vs 0.083 fixed).
Two voices: the built-in voice and a clone of a reference wav (self-clone of the first built-in phrase), so both prepare_conditionals paths are covered.
    python scripts/check_tts_intelligibility.py            # PASS/FAIL (exit code)
    python scripts/check_tts_intelligibility.py --no-guard # demonstrates the failure this test exists for (expected: FAIL)
"""
import argparse, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "pilot"))
from smoke_common import *   # noqa: F401,F403  (MODELS, REPORTS, fail, Timer)
import numpy as np, soundfile as sf, torch
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox_fp16 import install_speech_token_guard
from tts_qa import TtsQA

PHRASES = ["The quick brown fox jumps over the lazy dog near the river bank.",
           "Please send the signed documents to our office before Friday afternoon.",
           "Modern apartments include a bathroom, a balcony and a stretched white ceiling.",
           "We will meet again next week to discuss the results of the project."]
MEAN_MAX, WORST_MAX = 0.15, 0.30      # clean fp16-fixed / fp32 reference: mean ~0.04-0.08, worst < 0.2; unfixed fp16: mean ~0.3-0.4

ap = argparse.ArgumentParser(); ap.add_argument("--no-guard", action="store_true"); a = ap.parse_args()
m = ChatterboxMultilingualTTS.from_local(MODELS / "chatterbox", "cuda", t3_model="v3")
guard = None if a.no_guard else install_speech_token_guard(m)
for mod in (m.t3, m.s3gen, m.ve): mod.to(torch.float16)
m.conds = m.conds.to(device="cuda"); builtin = m.conds
ref = REPORTS / "_tts_intel_ref.wav"; rows = []


def gen(text, seed):
    torch.manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.float16):
        return m.generate(text, language_id="en").detach().float().cpu().numpy()[0]


for voice in ("builtin", "clone"):
    if voice == "clone":
        with torch.autocast("cuda", dtype=torch.float16): m.prepare_conditionals(str(ref))
    else:
        m.conds = builtin.to(device="cuda")
        if guard: guard.use_builtin()
    for k, text in enumerate(PHRASES):
        y = gen(text, 1247 + k)
        if voice == "builtin" and k == 0: sf.write(str(ref), y[: int(10 * m.sr)], m.sr, subtype="PCM_16")
        rows.append((voice, text, y))
del m; torch.cuda.empty_cache()
qa = TtsQA(); res = []
for voice, text, y in rows:
    r = qa.assess(text, y, m_sr := 24000); res.append(r)
    print(f"[{voice:7s}] score {r['score']:.3f} ctc_cer {r['ctc_cer']:.3f} w_cer {r['w_cer']:.3f} bad={r['bad']} {r['reasons']} | {text[:40]!r} -> ctc {r['ctc'][:50]!r}")
ref.unlink(missing_ok=True)
scores = [r["score"] for r in res]; mean, worst = float(np.mean(scores)), max(scores)
print(f"mean score {mean:.3f} (max {MEAN_MAX}), worst {worst:.3f} (max {WORST_MAX}), bad parts {sum(r['bad'] for r in res)}, guard calls {guard.calls if guard else 'OFF'}")
if guard is not None and guard.calls == 0: fail("speech-token guard was never invoked")
if mean > MEAN_MAX or worst > WORST_MAX or any(r["interjections"] for r in res): fail("TTS is not intelligible enough (independent ASR round trip)")
print("PASS: intelligible")
