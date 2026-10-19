#!/usr/bin/env python3
"""Extraneous-sound table (Agent C r3, 2026-10-19) from audio_stages.py JSONs: clicks (dub voice / mix-only), hot cut edges, extra voice in TTS
(breath / garble outside the CTC words), overlapping dub voices, Russian voice in the bed (s), bed steps (all / CREATED by the mix = the
separated stem itself moves < 4 dB in the same 100 ms, i.e. ducking pumping rather than the music).

    python scripts/quality/sound_table.py --stages DIR/01_r2.audio_stages.json ... --stem-dir-pattern '/workspace/dub/c/runs/r2/work_{id}' --tag audio-r2 --out OUT
"""
from __future__ import annotations
import argparse, json, pathlib
import numpy as np, soundfile as sf


def stem_step_db(y: np.ndarray, sr: int, t: float, w: float = 0.1) -> float:
    a = y[max(0, int((t - w) * sr)):int(t * sr)]; b = y[int(t * sr):int((t + w) * sr)]
    db = lambda x: 20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-9) if len(x) else -180.0
    return abs(db(a) - db(b))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", action="append", required=True); ap.add_argument("--stem-dir-pattern", default=None)
    ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args(); rows = []
    for sp in a.stages:
        r = json.loads(pathlib.Path(sp).read_text()); vid = pathlib.Path(sp).name[:2]; ev = r["events"]
        steps = [e for e in ev if e["type"] == "bed_step"]; created = None
        if a.stem_dir_pattern:
            f = pathlib.Path(a.stem_dir_pattern.format(id=vid)) / "bg_44k.wav"
            if f.exists():
                y, sr = sf.read(str(f), dtype="float32"); y = y.mean(1) if y.ndim == 2 else y
                created = [e for e in steps if stem_step_db(y, sr, e["t"]) < 4.0]
        c = r["clicks"]; n = lambda t: sum(e["type"] == t for e in ev)
        rows.append({"video": vid, "clicks_dub_voice": c.get("dub_voice", 0), "clicks_mix_only": max(0, c.get("final_mix", 0) - c.get("dub_voice", 0)),
                     "hot_edge": n("hot_edge"), "extra_voice": n("extra_voice"), "overlap": n("overlap"), "bed_voice_s": r["bed"].get("bed_speech_seconds", 0),
                     "bed_steps": len(steps), "bed_steps_mix_created": None if created is None else len(created),
                     "mix_created_at_s": None if created is None else [round(e["t"], 2) for e in created],
                     "extra_voice_at_s": [round(e["t"], 2) for e in ev if e["type"] == "extra_voice"], "hot_edge_at_s": [round(e["t"], 2) for e in ev if e["type"] == "hot_edge"]})
    md = [f"## Extraneous sounds: {a.tag}", "", "| video | clicks dub voice | clicks mix-only | hot cut edges | extra voice in TTS | overlapping voices | Russian voice in bed (s) | bed steps all / created by the mix |",
          "|---|---:|---:|---:|---:|---:|---:|---|"]
    for x in rows:
        md.append(f"| {x['video']} | {x['clicks_dub_voice']} | {x['clicks_mix_only']} | {x['hot_edge']} | {x['extra_voice']} | {x['overlap']} | {x['bed_voice_s']} | "
                  f"{x['bed_steps']} / {x['bed_steps_mix_created'] if x['bed_steps_mix_created'] is not None else 'n/a'} |")
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / f"sounds_{a.tag}.md").write_text("\n".join(md) + "\n"); (out / f"sounds_{a.tag}.json").write_text(json.dumps(rows, indent=1))
    print("\n".join(md)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
