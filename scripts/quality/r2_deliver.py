#!/usr/bin/env python3
"""Agent C r2 delivery (2026-10-17): one self-contained audio work dir per video for the video agents (LatentSync re-render on the new audio).

Input: a finished audio manifest (run_clean_pipeline_05 --stop-after audio, or a refit_offline.py `recut_final` manifest on top of it).
Output <out>/<id>/:
  dubbed_16k.wav (LatentSync / speech gate input, clean dub voice), dubbed_24k.wav, dubbed_aac.m4a,
  final_mix_44k.wav + final_mix_aac.m4a (what the viewer hears: dub voice + Spleeter 2stems bed, residue duck), aligned_u*.wav, tts_u*_{raw,p*}.wav,
  audio16k.wav (source), <id>_r2.manifest.json (runner schema, all paths inside this dir: units / placements / speech_gate / background / audio_policy),
  <id>_r2.residual.json (remaining "speaker talks, dub silent" gaps with master-frame indices, for the silent-face handling).
candidate_video_stage.py --dubbed-dir <out>/<id> uses dubbed_16k.wav + final_mix_aac.m4a.

    python scripts/quality/r2_deliver.py --manifest RUN/final/03_final.manifest.json --face-ref FACE/03.ref.manifest.json --out /workspace/dub/artifacts/audio-r2
"""
from __future__ import annotations
import argparse, json, pathlib, shutil, subprocess, sys, time
HERE = pathlib.Path(__file__).resolve().parent
for d in ("pilot", "quality"): sys.path.insert(0, str(HERE.parent / d))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--face-ref", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--residue-duck-db", type=float, default=-15.0); ap.add_argument("--duck-db", type=float, default=-4.0)
    ap.add_argument("--attack-s", type=float, default=0.25); ap.add_argument("--release-s", type=float, default=0.6); ap.add_argument("--refit-build", default=None, help="refit_offline build.json (recorded in the manifest)")
    a = ap.parse_args()
    import run_clean_pipeline_05 as rp, background_bed as bb
    from face_aware_latentsync_accel import build_speech_gate
    t0 = time.perf_counter(); man = json.loads(pathlib.Path(a.manifest).read_text()); vid = pathlib.Path(man["input"]).stem
    out = pathlib.Path(a.out) / vid
    if out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True)
    # 1. copy every wav the manifest references (units, parts, aligned, dub track, source) into the delivery dir, rewrite the paths
    W = pathlib.Path(man["work_dir"]); txt = json.dumps(man, ensure_ascii=False); mapping = {}
    def take(p: pathlib.Path):
        if not p.exists() or p.suffix not in (".wav", ".m4a") or str(p) in mapping: return
        q = out / p.name
        if q.exists(): q = out / f"{p.parent.name}_{p.name}"
        shutil.copy(p.resolve(), q); mapping[str(p)] = str(q)
    for u in man["units"]:
        take(pathlib.Path(u["tts_raw_file"]))
        for tp in u.get("tts_parts", []): take(pathlib.Path(tp["file"]))
        al = u.get("alignment") or {}
        for k in ("file", "aligned_file", "track"):
            if isinstance(al.get(k), str): take(pathlib.Path(al[k]))
    for f in ("dubbed_16k.wav", "dubbed_24k.wav", "dubbed_aac.m4a", "audio16k.wav"): take(W / f)
    for f in sorted(W.glob("aligned_u*.wav")): take(f)
    for k in sorted(mapping, key=len, reverse=True): txt = txt.replace(json.dumps(k)[1:-1], json.dumps(mapping[k])[1:-1])
    man = json.loads(txt); man["work_dir"] = str(out)
    # 1b. declick of the dub voice (runner default --declick on; the source runs were made before it existed)
    if not (man.get("dubbed_track") or {}).get("declick"):
        from declick import declick_file
        dc = declick_file(str(out / "dubbed_24k.wav"))
        rp.ff(["-i", str(out / "dubbed_24k.wav"), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(out / "dubbed_16k.wav")])
        rp.ff(["-i", str(out / "dubbed_24k.wav"), "-c:a", "aac", "-b:a", "192k", str(out / "dubbed_aac.m4a")])
        man["dubbed_track"]["declick"] = dc
    # 2. clean bed: Spleeter 2stems + residue duck (runner defaults)
    dur = man["source"]["duration_s"]; src = [(v["start"], v["end"]) for v in man["speech_intervals"]]
    dub = [(v["start"], v["end"]) for v in rp.run_vad(out / "dubbed_16k.wav")]
    sep = bb.separate_background(pathlib.Path(man["input"]), out, separator="spleeter")
    mix = bb.mix_background(sep, out / "dubbed_24k.wav", src, dub, dur, out, residue_duck_db=a.residue_duck_db, duck_db=a.duck_db, attack_s=a.attack_s, release_s=a.release_s)
    for f in ("bg_44k.wav", "vocals_44k.wav"): (out / f).unlink()   # stems: regenerable (Spleeter, deterministic), not needed downstream
    man["background"] = {"enabled": True, "separation": {k: v for k, v in sep.items() if k not in ("bg", "vocals")}, "mix": mix}
    # 3. speech gate exactly as the optimized video backend builds it (source VAD u dub VAD, margin 0.24 s, merge 0.6 s)
    gate = build_speech_gate([man["speech_intervals"], [{"start": s, "end": e} for s, e in dub]], dur, 0.24, 0.6)
    man["speech_gate"] = {"enabled": True, "margin_s": 0.24, "merge_gap_s": 0.6, "intervals": gate, "gate_seconds": round(sum(e - s for s, e in gate), 3),
                          "dubbed_speech_seconds": round(sum(e - s for s, e in dub), 3), "note": "recomputed by the video stage from dubbed_16k.wav; stored for reference"}
    man["delivery"] = {"by": "Agent C r2", "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "source_manifest": a.manifest,
                       "audio_policy": man.get("audio_policy") or {"name": "r2 (recut)"}, "refit": json.loads(pathlib.Path(a.refit_build).read_text()) if a.refit_build else None,
                       "lips": "NOT re-synced: the video stage must re-run LatentSync on dubbed_16k.wav", "mux_audio": str(out / "final_mix_aac.m4a")}
    mp = out / f"{vid}_r2.manifest.json"; mp.write_text(json.dumps(man, indent=1, ensure_ascii=False))
    # 4. residual gaps for the video side
    py = sys.executable; tmp = out / "_gaps"
    subprocess.run([py, str(HERE / "dub_gaps.py"), "--manifest", str(mp), "--ref-manifest", a.face_ref, "--tag", "r2", "--out", str(tmp)], check=True, capture_output=True)
    subprocess.run([py, str(HERE / "residual_for_video.py"), "--gaps", str(tmp / f"{vid}_r2.gaps.json"), "--manifest", str(mp), "--policy", "r2",
                    "--flags", "--audio-policy r2 (+ accepted layer-3 refit units)", "--out", str(tmp)], check=True, capture_output=True)
    shutil.move(str(tmp / f"{vid}_r2.residual.json"), out / f"{vid}_r2.residual.json"); shutil.move(str(tmp / f"{vid}_r2.gaps.json"), out / f"{vid}_r2.gaps.json"); shutil.rmtree(tmp)
    r = json.loads((out / f"{vid}_r2.residual.json").read_text())["summary"]
    dc = man["dubbed_track"].get("declick") or {}
    print(f"{vid}: declick {len(dc.get('repaired', []))} repaired / {len(dc.get('left_after', []))} left;", end=" ")
    print(f"{vid}: {len(mapping)} files, bed {sep['model'][:22]} residue-duck {mix['residue_duck']['seconds']} s, residual {r['residual']} ({r['residual_s']} s, {r['residual_ls_frames']} LS frames), "
          f"{time.perf_counter() - t0:.1f} s -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
