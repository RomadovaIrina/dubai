#!/usr/bin/env python3
"""Agent C r2 listening clips: ORIGINAL video frames + one audio variant per file (lips NOT re-synced: the mouth is the original Russian one).
Variants per window (file suffix): A_original (source audio) | B_old (cand5 timing + HDEMUCS bed, duck -6 dB / 0.08 s: what was heard before;
HDEMUCS = non-commercial reference, never shipped) | C_new (r2 delivery mix) and special before/after pairs (refit / declick / bed / duck).

    python scripts/quality/r2_sbs.py --out /workspace/dub/artifacts/audio-r2/sbs
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
HERE = pathlib.Path(__file__).resolve().parent
for d in ("pilot", "quality"): sys.path.insert(0, str(HERE.parent / d))
RUNS = pathlib.Path("/workspace/dub/runs"); DEL = pathlib.Path("/workspace/dub/artifacts/audio-r2"); SRC = RUNS / "sbs_src"


def ff(args): subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-nostdin", *args], check=True)


def mix(v: str, dub_dir: pathlib.Path, tag: str, sep: str, **kw) -> str:
    """dub voice of dub_dir + a bed (bed_s4 stems: spleeter | hdemucs) with the given mix settings -> SRC/<v>_<tag>.wav (cached)."""
    import background_bed as bb, run_clean_pipeline_05 as rp
    o = SRC / f"{v}_{tag}"; f = o / "final_mix_44k.wav"
    if f.exists(): return str(f)
    o.mkdir(parents=True, exist_ok=True); man = json.loads(next(dub_dir.glob("*manifest.json")).read_text()) if list(dub_dir.glob("*manifest.json")) else None
    m = json.loads((DEL / v / f"{v}_r2.manifest.json").read_text()); dur = m["source"]["duration_s"]; src = [(x["start"], x["end"]) for x in m["speech_intervals"]]
    dub = [(x["start"], x["end"]) for x in rp.run_vad(dub_dir / "dubbed_16k.wav")]
    st = RUNS / "bed_s4" / v / sep
    bb.mix_background({"bg": str(st / "bg_44k.wav"), "vocals": str(st / "vocals_44k.wav")}, dub_dir / "dubbed_24k.wav", src, dub, dur, o, **kw); return str(f)


def clip(video: str, audio: str, t0: float, t1: float, out: pathlib.Path):
    ff(["-ss", f"{t0:.2f}", "-to", f"{t1:.2f}", "-i", video, "-ss", f"{t0:.2f}", "-to", f"{t1:.2f}", "-i", audio, "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-shortest", str(out)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--out", required=True); a = ap.parse_args()
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    OLD = dict(duck_db=-6.0, attack_s=0.08, release_s=0.35)                                   # the cand5 production mix settings
    NEW = dict(duck_db=-4.0, attack_s=0.25, release_s=0.6, residue_duck_db=-15.0)             # r2 runner defaults
    vids = ["01", "02", "03", "04", "05"]; inp = {v: json.loads((DEL / v / f"{v}_r2.manifest.json").read_text())["input"] for v in vids}
    old = {v: mix(v, RUNS / "e2e_cand5" / "work" / f"work_{v}", "old_cand5_hdemucs", "hdemucs", **OLD) for v in vids}
    new = {v: str(DEL / v / "final_mix_44k.wav") for v in vids}
    rows = []
    def trio(v, t0, t1, why):
        base = f"{v}_{t0:06.2f}-{t1:06.2f}s"
        for tag, aud in (("A_original", inp[v]), ("B_old_cand5", old[v]), ("C_new_r2", new[v])):
            f = out / f"{base}_{tag}_LIPS_NOT_RESYNCED.mp4"; clip(inp[v], aud, t0, t1, f)
        rows.append(f"| `{base}_*` | A original / B old (cand5 + HDEMUCS bed) / C new (r2) | {why} |")
    def pair(v, t0, t1, why, before_tag, before_audio, after_tag, after_audio):
        base = f"{v}_{t0:06.2f}-{t1:06.2f}s"
        for tag, aud in ((before_tag, before_audio), (after_tag, after_audio)):
            clip(inp[v], aud, t0, t1, out / f"{base}_{tag}_LIPS_NOT_RESYNCED.mp4")
        rows.append(f"| `{base}_{before_tag}` / `_{after_tag}` | before / after | {why} |")
    # 1. the two reviewed 03 windows
    trio("03", 24.0, 26.0, "reviewed window: 24.5–25.0 s gap (dub silent while the speaker talks) 0.5 → 0.1 s")
    trio("03", 34.0, 37.0, "reviewed window: 34.3–35.5 s gap 1.2 → 0.9 s (u04 translation length; layer 3 found no faithful rewrite)")
    # 2. worst residual gaps of the delivery (2 per video by LS frames, then seconds)
    for v in vids:
        R = json.loads((DEL / v / f"{v}_r2.residual.json").read_text())["gaps"]
        worst = sorted([g for g in R if g["class"] == "residual"], key=lambda g: (-g["ls_frames"], -g["seconds"]))[:2]
        for g in worst:
            t0, t1 = max(0.0, g["t0"] - 1.5), g["t1"] + 1.5
            trio(v, round(t0, 2), round(t1, 2), f"worst remaining gap {g['t0']}–{g['t1']} s ({g['seconds']} s, {g['ls_frames']} LS frames, {g['position_in_burst']}): "
                 f"source says \"{g['source_words']}\"")
    # 3. refit units (accepted by the timing gate): r2 without vs with the rewrite, same new bed
    for v in ("01", "04"):
        b = json.loads((RUNS / "refit" / v / "build.json").read_text()); man = json.loads((DEL / v / f"{v}_r2.manifest.json").read_text())
        norefit = mix(v, RUNS / "e2e_r2" / "work" / f"work_{v}", "r2_without_refit", "spleeter", **NEW)
        for uid in b["accepted"]:
            u = next(x for x in man["units"] if x["id"] == uid); t0, t1 = u["slot"]["start"] - 1.0, u["slot"]["end"] + 1.0
            pair(v, round(t0, 2), round(t1, 2), f"layer-3 refit u{uid:02d}: \"{u.get('translation_fitted') or u['translation']}\"", "D_r2_without_refit", norefit, "E_r2_with_refit", new[v])
    # 4. declick: r2 track without vs with the click repair (same new bed)
    for v, t in (("01", 42.28), ("01", 14.32), ("05", 41.99)):
        nd = mix(v, RUNS / ("refit/01/recut_final" if v == "01" else "e2e_r2/work/work_05"), "r2_without_declick", "spleeter", **NEW)
        pair(v, t - 1.5, t + 1.5, f"click at {t} s (01 42.28: atempo splice; others: inside the raw Chatterbox take) before / after declick", "F_without_declick", nd, "G_with_declick", new[v])
    # 5. bed: Spleeter leak (02 39 s) without / with the residue duck; old fast duck vs new gentle duck (03 8.3 / 14.3 s, 04 8.5 s)
    nr = mix("02", DEL / "02", "r2_spleeter_no_residue_duck", "spleeter", duck_db=-4.0, attack_s=0.25, release_s=0.6)
    pair("02", 37.5, 41.5, "Spleeter left a vocal sound in the bed at 39.0–39.4 s (dub silent): without / with the residue duck", "H_without_residue_duck", nr, "I_with_residue_duck", new["02"])
    for v, t in (("03", 8.3), ("03", 14.3), ("04", 8.5)):
        od = mix(v, DEL / v, "r2_old_duck", "spleeter", duck_db=-6.0, attack_s=0.08, release_s=0.35, residue_duck_db=-15.0)
        pair(v, t - 2.0, t + 2.0, f"bed step before a dub phrase at {t} s: old duck (-6 dB, 80 ms attack) vs new (-4 dB, 250 ms)", "J_old_duck", od, "K_new_duck", new[v])
    (out / "README.md").write_text("# Agent C r2 listening clips (2026-10-17)\n\nEvery file: ORIGINAL video frames + one audio variant. **Lips are NOT re-synced** "
        "(LatentSync not re-run): the mouth is the original Russian one, judge the AUDIO (timing against the speaker, pauses, foreign sounds, bed).\n\n"
        "B_old = cand5 timing + HDEMUCS bed with the old duck = what was heard before (HDEMUCS is a non-commercial reference, never shipped).\n"
        "C_new = the r2 delivery (`/workspace/dub/artifacts/audio-r2/<id>/final_mix_aac.m4a`): r2 timing + accepted refit units + Spleeter bed, residue duck, "
        "-4 dB / 250 ms duck, declick.\n\n| files | variants | what to listen for |\n|---|---|---|\n" + "\n".join(rows) + "\n")
    print(len(list(out.glob("*.mp4"))), "clips ->", out); return 0


if __name__ == "__main__":
    raise SystemExit(main())
