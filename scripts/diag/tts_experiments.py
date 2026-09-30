#!/usr/bin/env python3
"""Diagnostic: controlled Chatterbox regeneration experiments on the REAL first_res units (text = each unit's full translation, one TTS call).
Axes (everything else fixed to the production call: language_id=en, exaggeration 0.5, cfg 0.5, min_p 0.05, top_p 1.0, rep.penalty 1.2):
  base        fp16 cast + autocast (production), speaker ref = production 10 s concatenated diarized speech, T=0.8
  fp32        full precision (upstream default), same ref / seed
  builtin     fp32 + Chatterbox built-in voice (no cloning)            - ceiling of the model on this text
  cleanref    fp16, reference = cleanest single 3-6 s word-aligned segment of the speaker (see select_clean_ref)
  T0.6/0.5/0.4 fp16, production ref, lower sampling temperature
  fp16fix     fp16 like production, but the speech-token ids are kept int64 (see install_token_fix): S3Token2Mel.forward casts EVERY ref_dict tensor, incl. the
              int64 prompt_token, to fp16 and torch.concat([prompt_token, token]) in flow.inference then promotes the generated speech-token ids to fp16
              as well -> ids >2048 are rounded to even / multiples of 4 (44 % of the 6561 ids change, error up to +-2). fp32 is exact.
  cfg0        fp16, cfg_weight 0.0 (exploratory: upstream advises it when the reference language differs from the target language)
Seeds per condition: production seed of the unit (1247+100*id), +1, +2.   Output: <out>/<id>/<cond>/u<uid>_s<k>.wav (resumable) + refs.

    python scripts/diag/tts_experiments.py --id 01 --repro /workspace/dub/tts_diag/repro --out /workspace/dub/tts_diag/exp [--conds base,fp32,...]
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, soundfile as sf, torch

ROOT = pathlib.Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"
CONDS = {"base": dict(dtype="fp16", ref="cur", T=0.8), "fp32": dict(dtype="fp32", ref="cur", T=0.8), "builtin": dict(dtype="fp32", ref="builtin", T=0.8),
         "cleanref": dict(dtype="fp16", ref="clean", T=0.8), "cleanref32": dict(dtype="fp32", ref="clean", T=0.8),
         "T0.6": dict(dtype="fp16", ref="cur", T=0.6), "T0.5": dict(dtype="fp16", ref="cur", T=0.5), "T0.4": dict(dtype="fp16", ref="cur", T=0.4),
         "cfg0": dict(dtype="fp16", ref="cur", T=0.8, cfg=0.0),
         "fp16fix": dict(dtype="fp16", ref="cur", T=0.8, fix=True), "fp16fix_clean": dict(dtype="fp16", ref="clean", T=0.8, fix=True),
         "fp16fix_T0.6": dict(dtype="fp16", ref="cur", T=0.6, fix=True), "fp16fix_T0.5": dict(dtype="fp16", ref="cur", T=0.5, fix=True), "fp16fix_T0.4": dict(dtype="fp16", ref="cur", T=0.4, fix=True),
         "fp16fix_cfg0": dict(dtype="fp16", ref="cur", T=0.8, cfg=0.0, fix=True)}


def select_clean_ref(man: dict, wav16: np.ndarray, out: pathlib.Path, spk: str) -> dict:
    """Cleanest single 3-6 s segment of `spk`: word-aligned edges (no cut words), inside one diarized turn, high mean whisper word probability,
    a clear but not rushed speaking rate (2.0-3.0 words/s) and at least one short pause; ranked by a transparent score."""
    words = [w for s in man["asr"]["segments"] for w in s["words"]]; turns = [t for t in man["diarization"]["turns"] if t["speaker"] == spk]
    best = None
    for i in range(len(words)):
        for j in range(i + 4, min(len(words), i + 22)):
            s, e = words[i]["start"], words[j]["end"]; d = e - s
            if d < 3.0 or d > 6.0: continue
            if not any(t["start"] <= s and e <= t["end"] for t in turns): continue
            ws = words[i:j + 1]; rate = len(ws) / d; p = float(np.mean([w["p"] for w in ws])); gaps = [ws[k + 1]["start"] - ws[k]["end"] for k in range(len(ws) - 1)]
            if max(gaps) > 0.8: continue                                   # no long silence inside the prompt
            score = p - 0.35 * max(0.0, rate - 3.0) - 0.15 * max(0.0, 2.0 - rate) + (0.05 if max(gaps) >= 0.12 else 0.0) + 0.02 * min(d, 5.0) / 5
            if best is None or score > best[0]: best = (score, s, e, p, rate, max(gaps), [w["word"].strip() for w in ws])
    score, s, e, p, rate, mg, txt = best
    y = wav16[int(s * 16000):int(e * 16000)]; f = out / f"ref_clean_{spk}.wav"; sf.write(str(f), y, 16000, subtype="PCM_16")
    return {"file": str(f), "start": s, "end": e, "seconds": round(e - s, 2), "mean_word_p": round(p, 3), "words_per_s": round(rate, 2), "max_gap_s": round(mg, 2), "text": " ".join(txt), "score": round(score, 3)}


def install_token_fix(m, on: bool) -> None:
    """Experiment-only monkeypatch (no library file is edited): feed flow.inference the ORIGINAL int64 prompt tokens."""
    f = m.s3gen.flow
    if not hasattr(f, "_orig_inference"): f._orig_inference = f.inference
    if not on:
        f.inference = f._orig_inference; return
    store = {"pt": m.conds.gen["prompt_token"].detach().clone().long(), "ptl": m.conds.gen["prompt_token_len"].detach().clone().long()}

    def fixed(*a, **k):
        k["prompt_token"] = store["pt"]; k["prompt_token_len"] = store["ptl"]; return f._orig_inference(*a, **k)
    f.inference = fixed


def load(dtype: str):
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    m = ChatterboxMultilingualTTS.from_local(MODELS / "chatterbox", "cuda", t3_model="v3")
    if dtype == "fp16":
        for mod in (m.t3, m.s3gen, m.ve): mod.to(torch.float16)
        m.conds = m.conds.to(device="cuda")
    return m


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--id", required=True); ap.add_argument("--repro", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--conds", default=",".join(CONDS)); ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--units", default=""); a = ap.parse_args()
    rd = pathlib.Path(a.repro) / a.id; man = json.loads((rd / f"{a.id}.manifest.json").read_text()); out = pathlib.Path(a.out) / a.id; out.mkdir(parents=True, exist_ok=True)
    spk = man["units"][0]["speaker"]; ref_cur = man["tts"]["references"][spk]["file"]
    wav16, sr = sf.read(str(rd / f"work_{a.id}" / "audio16k.wav"), dtype="float32"); assert sr == 16000
    cr = select_clean_ref(man, wav16, out, spk); (out / "refs.json").write_text(json.dumps({"current": ref_cur, "clean": cr}, indent=1, ensure_ascii=False)); print("clean ref:", cr, flush=True)
    models = {}; refs = {"cur": ref_cur, "clean": cr["file"], "builtin": None}; conds = a.conds.split(","); want = {CONDS[c]["dtype"] for c in conds}
    for dt in want: models[dt] = load(dt)
    builtin = {dt: m.conds for dt, m in models.items()}
    units = [u for u in man["units"] if not a.units or str(u["id"]) in a.units.split(",")]
    prepared = {}
    for c in conds:
        cfg = CONDS[c]; m = models[cfg["dtype"]]; dtype = torch.float16 if cfg["dtype"] == "fp16" else torch.float32
        (out / c).mkdir(exist_ok=True)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            if cfg["ref"] == "builtin": m.conds = builtin[cfg["dtype"]].to(device="cuda")
            else: m.prepare_conditionals(refs[cfg["ref"]])
        install_token_fix(m, bool(cfg.get("fix")))
        for u in units:
            for k in range(a.seeds):
                f = out / c / f"u{u['id']:02d}_s{k}.wav"
                if f.exists(): continue
                torch.manual_seed(1247 + 100 * u["id"] + k)
                with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
                    wav = m.generate(u["translation"], language_id="en", temperature=cfg["T"], cfg_weight=cfg.get("cfg", 0.5))
                wav = wav.detach().float().cpu().numpy()[0]; sf.write(str(f), wav, m.sr, subtype="PCM_16")
        print("cond done", c, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
