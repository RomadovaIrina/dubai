#!/usr/bin/env python3
"""r7 F (audio v3 variants): closed-mouth-while-the-source-talks seconds per rendered variant.
A frame counts when the OUTPUT mouth is closed (inner opening 13-14 < --closed IOD) while the SOURCE mouth is talking (opening > --talk IOD),
face visible in both. Also the subset where the variant's dub is silent (energy VAD on dubbed_16k.wav, the r7_mouth_probe rule) = the
pressed-lips case, and the converse (output mouth open > --talk while the dub is silent).
    python scripts/r7/r7_closed_mouth.py --ids 05 04 --variants m8b:/workspace/dub/handoff/in/audio-r7-v2 m8v3a:/workspace/dub/handoff/in/audio-r7-v3a
"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); from r7_mouth_probe import silent_spans, geo, B, SRC_LM   # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", nargs="+", default=["05", "04"]); ap.add_argument("--variants", nargs="+", required=True, help="tag:audio_handoff_dir")
    ap.add_argument("--closed", type=float, default=0.02); ap.add_argument("--talk", type=float, default=0.05); ap.add_argument("--out", default="")
    a = ap.parse_args(); R = {}
    print("| video | variant | closed mouth while the source talks (s) | ... of it dub-silent (s) | dub-silent face frames (s) | open mouth while the dub is silent (s) |")
    print("|---|---|---:|---:|---:|---:|")
    for id_ in a.ids:
        so = geo(np.load(SRC_LM / f"{id_}_src_mp.npz")["lm"])[0]
        for var in a.variants:
            tag, aud = var.split(":"); d = B / tag; lmp = d / f"{id_}_{tag}_out_mp.npz"; man = d / f"{id_}_{tag}.manifest.json"
            if not lmp.exists(): print(f"| {id_} | {tag} | no output landmarks | | | |"); continue
            o = geo(np.load(lmp)["lm"])[0]; m = json.loads(man.read_text()); n = min(len(o), len(so)); fps = n / m["source"]["duration_s"]
            _, sil = silent_spans(pathlib.Path(aud) / id_ / "dubbed_16k.wav", fps, n)
            face = np.isfinite(o[:n]) & np.isfinite(so[:n]); closed_talk = face & (o[:n] < a.closed) & (so[:n] > a.talk)
            r = {"closed_while_source_talks_s": round(float(closed_talk.sum() / fps), 2), "closed_while_source_talks_dub_silent_s": round(float((closed_talk & sil).sum() / fps), 2),
                 "dub_silent_face_s": round(float((face & sil).sum() / fps), 2), "open_while_dub_silent_s": round(float((face & sil & (o[:n] > a.talk)).sum() / fps), 2), "fps": round(fps, 3)}
            R.setdefault(id_, {})[tag] = r
            print(f"| {id_} | {tag} | {r['closed_while_source_talks_s']} | {r['closed_while_source_talks_dub_silent_s']} | {r['dub_silent_face_s']} | {r['open_while_dub_silent_s']} |")
    if a.out: pathlib.Path(a.out).write_text(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
