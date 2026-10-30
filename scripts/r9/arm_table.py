#!/usr/bin/env python3
"""r9: gate rows of the WATCH panels per video: r9 shipped | r9 + fix | ref shift16 + fix | ref fixed + fix.
    python arm_table.py <out.md> m03:fx3 m19:fx m15:fx"""
import json, pathlib, sys
B = pathlib.Path("/workspace/dub/artifacts/r4m"); D = pathlib.Path("/workspace/dub/r9/diag"); R = pathlib.Path("/workspace/dub/r9/runs")


def jl(p):
    try: return json.loads(pathlib.Path(p).read_text())
    except Exception: return {}


rows = ["| video | panel | SyncNet (dub speech) | flashes | mouth / dub mismatch s | pressed lips s | a stick / unstick | b amplitude | c teeth | d slide | source fallback frames |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
for spec in sys.argv[2:]:
    i, fx = spec.split(":")
    for lab, T, art, closed in [("r9 shipped", f"r9f_{i}", D / f"art_{i}.json", R / i / "work" / "closed_mouth.json"),
                                ("r9 + fix", f"r9f_{i}{fx}", D / f"art_{i}{fx}.json", D / f"closed_r9f_{i}{fx}.json"),
                                ("ref shift16 + fix", f"r9f_{i}shift16L05{fx}", D / f"art_{i}shift16L05{fx}.json", D / f"closed_r9f_{i}shift16L05{fx}.json"),
                                ("ref fixed + fix", f"r9f_{i}fixedL05{fx}", D / f"art_{i}fixedL05{fx}.json", D / f"closed_r9f_{i}fixedL05{fx}.json")]:
        s = jl(B / T / f"{i}_{T}.score.json"); a = (jl(art).get("layers") or {}).get("L4 final") or {}
        c = (jl(closed).get(i) or {}).get(T) or {}
        g = lambda k: (a.get(k) or {}).get("events")
        if "fix" in lab:
            G = "r9g_" + T[4:-len(fx)]; rl = "rep3" if fx == "fx3" else "rep2"
            fbl = jl(B / G / f"work_{i}_{G}" / f"{i}_{G}_25fps_ls_cfd_ia04_srt1_{rl}.repair.json").get("fallback_frames_left")
        else:
            fbl = jl(B / f"r9g_{i}" / f"work_{i}_r9g_{i}" / f"{i}_r9g_{i}_25fps_fb_fb7.json").get("fallback_frames_gt05")
        rows.append(f"| {i} | {lab} | {(s.get('sync_speech') or {}).get('conf_speech', '')} | {s.get('flash_strong', '')} | {(s.get('mouth_audio') or {}).get('mouth_audio_mismatch_s', '')} | "
                    f"{c.get('closed_while_source_talks_dub_silent_s', '')} | {g('a_stick_unstick')} | {g('b_amplitude')} | {g('c_teeth')} | {g('d_slide')} | {fbl if fbl is not None else ''} |")
pathlib.Path(sys.argv[1]).write_text("\n".join(rows) + "\n"); print("\n".join(rows))
