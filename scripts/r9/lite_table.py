#!/usr/bin/env python3
"""r9: r9 | r9fx | fx-lite | fx-rep per video: SyncNet (dub speech), flashes (total / strong / texture), mismatch, pressed lips, twitch.
    python lite_table.py m03 m18 m15"""
import json, pathlib, sys
B = pathlib.Path("/workspace/dub/artifacts/r4m"); D = pathlib.Path("/workspace/dub/r9/diag"); R = pathlib.Path("/workspace/dub/r9/runs"); TW = pathlib.Path("/workspace/dub/artifacts/r6f/twitch")


def jl(p):
    try: return json.loads(pathlib.Path(p).read_text())
    except Exception: return {}


kind = lambda e: max((abs(e.get("z_hf", 0)), "texture"), (abs(e.get("z_lum", 0)), "brightness"), (abs(e.get("z_col", 0)), "colour"))[1]
rows = ["| video | arm | SyncNet | flashes (strong, texture) | mismatch s | pressed lips s | twitch sum (strong) |", "|---|---|---|---|---|---|---|"]
for i in sys.argv[1:]:
    fx = "fx3" if i == "m03" else "fx"
    for lab, T in [("r9", f"r9f_{i}"), ("r9fx", f"r9f_{i}{fx}"), ("fx-lite (no spike smoothing)", f"r9f_{i}fxlite"), ("fx-rep (full CF detail)", f"r9f_{i}fxrep"), ("fx + stab (detail stabiliser)", f"r9f_{i}fxstab")]:
        s = jl(B / T / f"{i}_{T}.score.json"); fl = jl(D / f"flash_{T}.json").get("flash_list") or []
        st = [e for e in fl if (e.get("mc_ratio") or 0) >= 2 and (e.get("mc_vs_src") or 0) >= 2]
        cl = jl(R / i / "work" / "closed_mouth.json") if lab == "r9" else jl(D / f"closed_{T}.json")
        pl = ((cl.get(i) or {}).get(T) or {}).get("closed_while_source_talks_dub_silent_s", "")
        tw = jl(TW / T / f"{i}.twitch.json"); ev = tw.get("events") or []
        rows.append(f"| {i} | {lab} | {(s.get('sync_speech') or {}).get('conf_speech', '')} | {len(fl)} ({len(st)}, {sum(1 for e in fl if kind(e) == 'texture')}) | "
                    f"{(s.get('mouth_audio') or {}).get('mouth_audio_mismatch_s', '')} | {pl} | {(tw.get('summary') or {}).get('twitch_sum', '')} ({sum(1 for e in ev if (e.get('score') or 0) >= 2)}) |")
print("\n".join(rows))
