#!/usr/bin/env python3
"""r6 G: full-video table (04 / 05) of r6_full / r6_recomp arms + the m4 reference: SyncNet on dub speech, strong flashes, mouth / dub mismatch,
chin wave (rigid), twitch (sum / events >= 2), LatentSync s/video-min, peak VRAM.   python scripts/r6/r6_full_table.py [arms...]"""
import json, pathlib, sys
F = pathlib.Path("/workspace/dub/artifacts/r6g/full"); M4 = pathlib.Path("/workspace/dub/artifacts/r4m/m4")
arms = ["m4"] + (sys.argv[1:] or sorted(p.name for p in F.iterdir() if p.is_dir()))
print("| arm | id | SyncNet dub speech | strong flashes | mouth/dub s | chin rigid | twitch sum | twitch ev>=2 | LS s/video-min | peak MiB |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for arm in arms:
    for id_ in ("04", "05"):
        d = M4 if arm == "m4" else F / arm; p = d / f"{id_}_{arm}"
        if not p.with_suffix(".score.json").exists(): continue
        r = json.loads(p.with_suffix(".score.json").read_text())
        try: t = json.loads(p.with_suffix(".twitch.json").read_text())["summary"]
        except Exception: t = {}
        v = r.get("manifest_r4v") or {}
        print(f"| {arm} | {id_} | {(r.get('sync_speech') or {}).get('conf_speech')} | {r.get('flash_strong')} | {(r.get('mouth_audio') or {}).get('mouth_audio_mismatch_s')} | "
              f"{(r.get('chinwave') or {}).get('rigid_mean')} | {t.get('twitch_sum')} | {t.get('n_events_ge2')} | {(v.get('s_per_video_min') or {}).get('latentsync_25fps')} | {v.get('peak_vram_mib')} |")
