#!/usr/bin/env python3
"""r6 audio: the key gate axes of several tags side by side (one row per video x tag).
    python scripts/r6/r6_table.py pb1 r6v1_14b [...]"""
import json, pathlib, sys
A = pathlib.Path("/workspace/dub/artifacts/r4a")
COLS = [("src-talk/dub-sil", lambda m: m.get("src_speaks_dub_silent_s")), ("dub/src-sil", lambda m: m.get("dub_speaks_src_silent_s")),
        ("face/dub-sil", lambda m: m.get("dub_silent_face_visible_s")), ("gaps>1s", lambda m: m.get("gaps_excess_gt_1s")),
        ("gapmax", lambda m: m.get("gap_excess_max_s")), ("tempo out", lambda m: m["tempo"]["out_s"]), ("fast", lambda m: m["tempo"]["fast_s"]),
        ("inside n", lambda m: m["pause_logic"]["pauses_inside_phrase"]), ("inside s", lambda m: m["pause_logic"]["inside_phrase_s"]),
        ("WER", lambda m: m["asr"]["wer_final_mix"]), ("faith", lambda m: (m.get("faith") or {}).get("mean")), ("<4", lambda m: (m.get("faith") or {}).get("lt4")),
        ("cuts", lambda m: m.get("hard_cuts")), ("LUFS", lambda m: m["loudness"]["integrated_lufs"])]
tags = sys.argv[1:]
print("| id | tag | " + " | ".join(c for c, _ in COLS) + " |"); print("|---|---|" + "---:|" * len(COLS))
for id_ in ["01", "02", "03", "04", "05"]:
    for t in tags:
        f = A / t / f"{id_}.metrics.json"
        if not f.exists(): continue
        m = json.loads(f.read_text()); v = []
        for _, g in COLS:
            try: x = g(m); v.append("" if x is None else (f"{x:.3f}" if isinstance(x, float) and x < 1 else str(x)))
            except Exception: v.append("")
        print(f"| {id_} | {t} | " + " | ".join(v) + " |")
