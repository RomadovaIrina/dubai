#!/usr/bin/env python3
"""Round 3 D: where the TTS clips were cut. For every INNER cut between two parts of one unit (tts_parts[k].clip_span end / [k+1] start),
the level of the raw whole-sentence generation (tts_u<id>_raw.wav) in the 10 ms before / after the cut, relative to the clip's speech level
(p90 of 10 ms frames). A cut within 20 dB of the speech level lands in voiced sound (mid-word / mid-syllable = the "abrupt cut" the user hears).
    python scripts/quality/r3_cutlevel.py MANIFEST [MANIFEST ...]   (the raw clips are looked up next to the manifest or in its work_dir)"""
import json, pathlib, sys
import numpy as np, soundfile as sf


def lvl(y, sr, t0, t1):
    a, b = max(0, int(t0 * sr)), min(len(y), int(t1 * sr))
    return 20 * np.log10(np.sqrt(np.mean(y[a:b] ** 2)) + 1e-9) if b > a else -120.0


for mp in sys.argv[1:]:
    m = json.loads(pathlib.Path(mp).read_text()); d = pathlib.Path(mp).parent; rel = []
    for u in m["units"]:
        raw = u.get("tts_raw_file"); parts = u.get("tts_parts") or []
        if not raw or len(parts) < 2: continue
        rp = d / pathlib.Path(raw).name
        if not rp.exists(): rp = pathlib.Path(raw)
        if not rp.exists(): continue
        y, sr = sf.read(str(rp), dtype="float32"); y = y if y.ndim == 1 else y.mean(1)
        fr = 20 * np.log10(np.sqrt((y[: len(y) // 240 * 240].reshape(-1, 240) ** 2).mean(1)) + 1e-9); ref = np.percentile(fr[fr > -60], 90) if (fr > -60).any() else -20
        for k in range(len(parts) - 1):
            e = parts[k]["clip_span"][1]; s = parts[k + 1]["clip_span"][0]
            rel.append((round(lvl(y, sr, e - 0.01, e) - ref, 1), round(lvl(y, sr, s, s + 0.01) - ref, 1)))
    r = np.array(rel) if rel else np.zeros((0, 2))
    voiced = int(((r[:, 0] > -20) | (r[:, 1] > -20)).sum()) if len(r) else 0
    print(f"{mp}: inner cuts {len(r)}, in voiced sound (within 20 dB of speech) {voiced}, median level before / after cut "
          f"{np.median(r[:, 0]) if len(r) else 0:.1f} / {np.median(r[:, 1]) if len(r) else 0:.1f} dB rel")
