#!/usr/bin/env python3
"""Diagnostic: score the regenerated wavs of tts_experiments.py with the independent ASRs, sound-level flags and speaker similarity (VoiceEncoder cosine
to the production reference).  Writes <exp>/<id>/scores.json (resumable: already-scored files are reused).

    python scripts/diag/analyze_experiments.py --id 01 --repro /workspace/dub/tts_diag/repro --exp /workspace/dub/tts_diag/exp
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, soundfile as sf, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import intel_metrics as im

ROOT = pathlib.Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--id", required=True); ap.add_argument("--repro", required=True); ap.add_argument("--exp", required=True); ap.add_argument("--sound", action="store_true", help="also pyin stable-voiced run (slow)")
    a = ap.parse_args()
    rd = pathlib.Path(a.repro) / a.id; man = json.loads((rd / f"{a.id}.manifest.json").read_text()); d = pathlib.Path(a.exp) / a.id
    units = {u["id"]: u for u in man["units"]}; asr = im.Asr()
    from chatterbox.models.voice_encoder import VoiceEncoder
    ve = VoiceEncoder(); ve.load_state_dict(torch.load(ROOT / "models" / "chatterbox" / "ve.pt", map_location="cpu", weights_only=True)); ve.eval()
    refs = json.loads((d / "refs.json").read_text())

    def emb(y16):
        return ve.embeds_from_wavs([y16], sample_rate=16000).mean(axis=0)
    cos = lambda x, y: float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y)))
    ref_emb = emb(sf.read(refs["current"], dtype="float32")[0]); clean_emb = emb(sf.read(refs["clean"]["file"], dtype="float32")[0])
    sp = d / "scores.json"; old = {(r["cond"], r["uid"], r["seed"]): r for r in json.loads(sp.read_text())} if sp.exists() else {}; out = []
    for cd in sorted(p for p in d.iterdir() if p.is_dir()):
        for f in sorted(cd.glob("u*_s*.wav")):
            uid = int(f.stem[1:3]); k = int(f.stem.split("_s")[1]); key = (cd.name, uid, k)
            if key in old: out.append(old[key]); continue
            y, sr = sf.read(str(f), dtype="float32"); u = units[uid]; y16 = asr.to16(y, sr); w = asr.whisper(y16, pad_s=0.5); c = asr.ctc(y16); exp = u["translation"]
            r = {"cond": cd.name, "uid": uid, "seed": k, "dur_s": round(len(y) / sr, 3), "ref_words": len(im.norm(exp)), "s_per_word": round(len(y) / sr / max(len(im.norm(exp)), 1), 3), "whisper": w["text"], "ctc": c["text"],
                 **{f"w_{x}": v for x, v in im.score(exp, w["text"]).items()}, **{f"c_{x}": v for x, v in im.score(exp, c["text"]).items()}, "max_same_symbol_s": c["max_same_symbol_s"], **im.unexplained_sound(y16, c),
                 "sim_ref": round(cos(emb(y16), ref_emb), 3), "sim_clean_ref": round(cos(emb(y16), clean_emb), 3)}
            if a.sound: r.update(im.voiced_stable_run(y16))
            out.append(r)
        print(cd.name, "scored", flush=True)
    sp.write_text(json.dumps(out, indent=1, ensure_ascii=False)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
