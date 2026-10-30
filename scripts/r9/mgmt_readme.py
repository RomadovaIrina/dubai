#!/usr/bin/env python3
"""r9 management set README: what the three face variants are + the per-video gate rows (SyncNet, flashes, twitch, mismatch)."""
import json, pathlib
B = pathlib.Path("/workspace/dub/artifacts/r4m"); D = pathlib.Path("/workspace/dub/r9/diag"); TW = pathlib.Path("/workspace/dub/artifacts/r6f/twitch")
O = pathlib.Path("/workspace/dub/outbox/mgmt_r9")


def jl(p):
    try: return json.loads(pathlib.Path(p).read_text())
    except Exception: return {}


def row(i, lab, T):
    s = jl(B / T / f"{i}_{T}.score.json"); fl = jl(D / f"flash_{T}.json").get("flash_list") or []
    st = sum(1 for e in fl if (e.get("mc_ratio") or 0) >= 2 and (e.get("mc_vs_src") or 0) >= 2)
    tw = jl(TW / T / f"{i}.twitch.json"); ev = tw.get("events") or []
    sync = (s.get("sync_speech") or {}).get("conf_speech"); mm = (s.get("mouth_audio") or {}).get("mouth_audio_mismatch_s")
    tws = (tw.get("summary") or {}).get("twitch_sum")
    f = lambda v, d=2: "–" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))
    return f"| {i} | {lab} | {f(sync)} | {len(fl) if fl else '–'} ({st if fl else '–'}) | {f(tws, 1)} ({sum(1 for e in ev if (e.get('score') or 0) >= 2) if ev else '–'}) | {f(mm)} |"


names = {"m03": "m03: mother + child (two faces), 40 s", "m18": "m18: older woman, glasses, dim light, 20 s", "old03": "old 03: the man (test set), 90 s", "old05": "old 05: the man (test set), 52 s"}
L = ["# DabAI r9: management set, three face variants (31.10.2026)", "",
     "Each video comes in 3 face variants x 2 background beds. The dub (voice, timing, translation) is identical across all six files of a video;",
     "only the mouth rendering differs. Files: `<id>_<variant>_<bed>.mp4`, full quality, source resolution / fps.", "",
     "| variant | what it is |", "|---|---|",
     "| **r9fix** | the r9 generator + the composite fix: (1) short mouth glitches (the old fallback's 1-frame swaps to the original mouth, 1-2-frame opening jumps, a mouth that slides / doubles) are re-drawn from the neighbouring good frames; longer broken runs ease to the original mouth instead of snapping; (2) CodeFormer's detail inside the mouth at 0.4, so the teeth are no sharper than the real ones |",
     "| **fxlite** | r9fix without the smoothing of 1-2-frame opening jumps (only real breakages are repaired), in case that smoothing hides real consonant articulation |",
     "| **fxstab** | r9fix + a mouth-detail stabiliser: inside the mouth each frame keeps its own shape and motion but takes the median fine detail of its +-2 neighbours, so CodeFormer's frame-to-frame texture pops ('explosions') are outvoted |", "",
     "Beds: `demucs` = htdemucs background (main), `spleeter` = Spleeter background.", "",
     "## Numbers", "SyncNet = lip-sync confidence on the dub speech (higher = better). Flashes = 1-frame mouth texture / brightness / colour jumps vs the source (strong in brackets; lower = better).",
     "Twitch = whole-face twitch detector vs the source (events >= 2 in brackets; lower = better). Mismatch = seconds where the mouth and the dub disagree (open mouth on silence / closed on speech).", "",
     "| video | variant | SyncNet | flashes (strong) | twitch (strong) | mismatch s |", "|---|---|---|---|---|---|"]
for i in ["m03", "m18", "old03", "old05"]:
    r9 = f"r9f_{i}"
    if jl(B / r9 / f"{i}_{r9}.score.json"): L.append(row(i, "r9 (before, for reference)", r9))
    L += [row(i, "r9fix", "r9f_m03fx3" if i == "m03" else f"r9f_{i}fx"), row(i, "fxlite", f"r9f_{i}fxlite"), row(i, "fxstab", f"r9f_{i}fxstab")]
L += ["", "Notes: " + "; ".join(names.values()) + ". The old 03 / 05 are re-dubbed from scratch with the current pipeline (final_r8b audio config, r9 generator)."]
(O / "README.md").write_text("\n".join(L) + "\n"); print("\n".join(L))
