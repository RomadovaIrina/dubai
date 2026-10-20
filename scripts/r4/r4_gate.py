#!/usr/bin/env python3
"""Round 4 regression gate (video agent): NEW vs PREVIOUS PUBLISHED on all five videos, same rules for every video, no per-video flags.
A set = a label + a dir holding <id>_<label>.mp4 (+ <id>_<label>.manifest.json; a set without manifests, e.g. the v4 reference finals, borrows
the manifest of --manifest-from, the reproduction of the same version, for the manifest-dependent metrics: dub-voice SyncNet, LS segments).
  score : r3_scorecard (9.4, SyncNet dub / mix, v4 face metric, slide, WER, bg ring, LUFS / TP, speed) + r3_flash (strong single-frame flashes)
          + r3_joints (window joints) + r4_face_metrics (half-face colour shift, jaw jitter, below-jaw band) + audio stream md5
  compare: a metric clearly worse than PREV on any video (beyond the noise tolerance below, calibrated on v4-repro vs v4) -> FAIL, no publish
    python scripts/r4/r4_gate.py score --label L --dir D [--manifest-from DIR:LABEL] [--ids 01,02,03,04,05]
    python scripts/r4/r4_gate.py compare --new D:L --prev D:L --out r4_video_vN_regression.md [--ids ...]"""
from __future__ import annotations
import argparse, hashlib, json, pathlib, subprocess, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; Q = HERE.parent / "quality"; PY = sys.executable
TIMING = pathlib.Path("/workspace/dub/artifacts/r3d/timing")   # r3_scorecard's fixed source-landmark location
IDS = ["01", "02", "03", "04", "05"]
# name, getter, direction (+1 = higher is worse), absolute tolerance, relative tolerance, hard limit (FAIL beyond it regardless of PREV)
RULES = [
    ("frames == source", lambda r: r["spec_9_4"]["frames_equal"], "eq", 0, 0, None),
    ("duration Δ s", lambda r: abs(r["spec_9_4"]["duration_delta_s"]), +1, 0.02, 0, 0.15),
    ("SyncNet on dub-speech seconds (conf at 0)", lambda r: r["sync_speech"]["conf_speech"], -1, 0.3, 0, None),
    ("SyncNet dub-only (whole video, info)", lambda r: r["syncnet"]["dub_only"][0], 0, 0, 0, None),
    ("SyncNet mix (whole video, info)", lambda r: r["syncnet"]["mix"][0], 0, 0, 0, None),
    ("dub silent while face tracked, s (audio info)", lambda r: r["sync_speech"]["dub_silent_face_s"], 0, 0, 0, None),
    ("|AV offset| dub", lambda r: abs(r["syncnet"]["dub_only"][1]), +1, 1, 0, None),
    ("mouth / dub-audio mismatch s (talking mouth + dub silent, closed mouth + dub speaks)", lambda r: r["mouth_audio"]["mouth_audio_mismatch_s"], +1, 0.5, 0, None),
    ("v4 resid mouth", lambda r: r["face_v4"]["resid_mouth_ls_core"], +1, 0, 0.05, None),
    ("v4 twitch mouth", lambda r: r["face_v4"]["twitch_mouth_ls_core"], +1, 0, 0.05, None),
    ("v4 shear95 lower face", lambda r: r["face_v4"]["shear95_lowerface_ls_core"], +1, 0, 0.05, None),
    ("v4 resid mouth, own frames (info)", lambda r: (r.get("face_v4_own") or r["face_v4"])["resid_mouth_ls_core"], 0, 0, 0, None),
    ("v4 common lip-synced seconds (info)", lambda r: r["face_v4"].get("frames_common_s"), 0, 0, 0, None),
    ("v4 sharpness mouth (src=1)", lambda r: r["face_v4"]["sharp_mouth_ls_core"], -1, 0.03, 0, None),
    ("slide nose-base px", lambda r: float(np.mean([w["nosebase"]["slide_px"]["mean"] for w in r["slide"]])), +1, 0.02, 0.10, None),
    ("window joints ratio", lambda r: r["joints"]["raw"]["resid"]["joint_over_inner"], +1, 0, 0.05, None),
    ("window joints LF ratio", lambda r: r["joints"]["raw"]["lowfreq"]["joint_over_inner"], +1, 0, 0.05, None),
    ("bg ring lower-upper", lambda r: r["bgring"]["lower_minus_upper"], +1, 0.2, 0, 0.5),
    ("lip flash strong", lambda r: r["flash_strong"], +1, 2, 0.2, None),
    ("colour worst 1 s |dL| (half face)", lambda r: r["r4face"]["colour_worst_1s_abs_dL"], +1, 0.5, 0, 4.0),
    ("colour frames |dL|>6", lambda r: r["r4face"]["colour_left"]["frames_abs_dL_gt6"] + r["r4face"]["colour_right"]["frames_abs_dL_gt6"], +1, 3, 0.2, None),
    ("jaw jitter out/src", lambda r: r["r4face"]["jaw_jitter"]["ratio_out_over_src"], +1, 0, 0.05, None),
    ("chin wave (lower-chin residual std, 2 s)", lambda r: r["chinwave"]["mean"], +1, 0.1, 0.1, None),
    ("chin wave worst 2 s", lambda r: r["chinwave"]["worst_2s"], +1, 0.15, 0.1, None),
    ("below-jaw band |out-src|", lambda r: r["r4face"]["below_jaw"]["mean"], +1, 0.3, 0, None),
    ("WER final mix", lambda r: r["asr"]["wer_final_mix"], +1, 0.02, 0, None),
    ("audio hard cuts", lambda r: r["audio_gate"]["audio_hard_cuts"]["cuts_stop_restart"], +1, 0, 0, None),
    ("|LUFS + 16|", lambda r: abs(r["loudness"]["integrated_lufs"] + 16), +1, 0.5, 0, 1.0),
    ("true peak dBTP", lambda r: r["loudness"]["true_peak_dbtp"], +1, 0.3, 0, -1.0),
    ("video s / video-min", lambda r: r["manifest_r4v"]["s_per_video_min"]["total"], +1, 0, 0.15, 300),
    ("peak VRAM MiB", lambda r: r["manifest_r4v"]["peak_vram_mib"], 0, 0, 0, None),
    ("audio stream md5", lambda r: r["audio_md5"][:8], 0, 0, 0, None),
]


def run_json(cmd, env=None):
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    for line in reversed(p.stdout.strip().splitlines()):
        if line.startswith("{"):
            try: return json.loads(line)
            except Exception: pass
    return {"error": (p.stderr or p.stdout)[-400:]}


def audio_md5(p):
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(p), "-map", "0:a:0", "-c", "copy", "-f", "data", "-"], capture_output=True)
    return hashlib.md5(r.stdout).hexdigest()


def add_chinwave(r, id_, label, d, video):   # task 5: lower-chin low-frequency residual std vs the source (r4_chin_wave.py)
    src = next(pathlib.Path("/workspace/dub/inputs/test_videos_raw").glob(f"{id_}.*"))
    cw = run_json([PY, str(HERE / "r4_chin_wave.py"), "--video", str(video), "--src", str(src), "--lm", str(TIMING / f"{id_}_source_lm_F.npz"),
                   "--out", str(d / f"{id_}_{label}.chinwave.json")])
    r["chinwave"] = cw if "mean" in cw else {"mean": None, "worst_2s": None, **cw}
    return r


def add_sync(r, id_, label, d, video, man):
    am = pathlib.Path(man["dubbed_track"]["aac"]).parent
    if "sync_speech" not in r and ((man.get("lipsync") or {}).get("by_action") or {}).get("LATENT_SYNC"):
        # per-second SyncNet (sync_perframe.py) on a 25 fps rendition with the dub voice: the mean conf at offset 0 over seconds where the dub
        # speaks. A whole-video score also counts dub-silent seconds (a correctly closed mouth scores ~0), so it moves with the audio placement.
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            v25 = pathlib.Path(td) / "v25.mp4"; sp = d / f"{id_}_{label}.sync_perframe.json"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video), "-vf", "fps=25", "-c:v", "libx264", "-crf", "12", "-an", str(v25)], check=True)
            subprocess.run([PY, str(Q / "sync_perframe.py"), "--video", str(v25), "--audio", str(am / "dubbed_16k.wav"), "--out", str(sp)], capture_output=True, text=True)
        ps = json.loads(sp.read_text())["per_second"] if sp.exists() else []
        sp_ = [x["conf_at0"] for x in ps if x.get("conf_at0") is not None and (x.get("dub_act") or 0) >= 0.5]
        r["sync_speech"] = {"conf_speech": round(float(np.mean(sp_)), 3) if sp_ else None, "speech_seconds": len(sp_),
                            "dub_silent_face_s": sum(1 for x in ps if x.get("conf_at0") is not None and (x.get("dub_act") or 0) < 0.1)}
    return r


def add_mouth_audio(r, id_, label, d, video, man):
    """r4 (user on 05 m2): the visible mouth follows the DUB audio (r4_mouth_audio.py: talking mouth while the dub is silent + closed mouth while it speaks)."""
    dub = pathlib.Path(man["dubbed_track"].get("wav16k") or pathlib.Path(man["dubbed_track"]["aac"]).with_name("dubbed_16k.wav"))
    r["mouth_audio"] = run_json([PY, str(HERE / "r4_mouth_audio.py"), "--video", str(video), "--dub", str(dub), "--out", str(pathlib.Path(d) / f"{id_}_{label}.mouthaudio.json")])


def score_one(id_, label, d, man_p, force=False, asr_from=None):
    d = pathlib.Path(d); video = d / f"{id_}_{label}.mp4"; sj = d / f"{id_}_{label}.score.json"
    if sj.exists() and not force:
        r = json.loads(sj.read_text())
        if "r4face" in r:
            if "sync_speech" not in r:
                add_sync(r, id_, label, d, video, json.loads(pathlib.Path(man_p).read_text())); sj.write_text(json.dumps(r, indent=1))
            if "chinwave" not in r:
                add_chinwave(r, id_, label, d, video); sj.write_text(json.dumps(r, indent=1))
            if "mouth_audio" not in r: add_mouth_audio(r, id_, label, d, video, json.loads(pathlib.Path(man_p).read_text())); sj.write_text(json.dumps(r, indent=1))
            return r
    src = next(pathlib.Path("/workspace/dub/inputs/test_videos_raw").glob(f"{id_}.*")); lm = TIMING / f"{id_}_source_lm_F.npz"
    subprocess.run([PY, str(Q / "r3_scorecard.py"), "score", "--id", id_, "--label", label, "--video", str(video), "--manifest", str(man_p), "--out", str(d)] + (["--skip-wer"] if asr_from else []),
                   capture_output=True, text=True)
    r = json.loads(sj.read_text()); man = json.loads(pathlib.Path(man_p).read_text())
    if asr_from:   # composite-only arm: the muxed audio is byte-identical to that set's (checked by md5 below), so its WER is reused
        ref = json.loads((pathlib.Path(asr_from.split(":")[0]) / f"{id_}_{asr_from.split(':')[1]}.score.json").read_text()); r["asr"] = {**ref["asr"], "reused_from": asr_from}
    fl = d / f"{id_}_{label}.flash.json"
    subprocess.run([PY, str(Q / "r3_flash.py"), "--video", str(video), "--src", str(src), "--lm", str(lm), "--out", str(fl)], capture_output=True, text=True)
    full = json.loads(fl.read_text()) if fl.exists() else {"flash_list": []}
    r["flash_strong"] = sum(1 for e in full["flash_list"] if (e.get("mc_ratio") or 0) >= 2.0 and (e.get("mc_vs_src") or 0) >= 2.0)
    jj = d / f"{id_}_{label}.joints.json"; mfps = float((man.get("lipsync") or {}).get("master_fps") or 25)
    jlm = TIMING / f"{id_}_source_lm_F{int(mfps)}.npz" if mfps != 25 else lm   # a 30 fps LatentSync run counts its segments in 30 fps master frames
    j = run_json([PY, str(Q / "r3_joints.py"), "--video", str(video), "--manifest", str(man_p), "--lm", str(jlm), "--master-fps", str(mfps), "--out", str(jj)])
    jd = json.loads(jj.read_text()) if jj.exists() else j
    r["joints"] = {"ratio": jd.get("joint_over_inner") or jd.get("ratio") or (jd.get("all") or {}).get("joint_over_inner"), "raw": {k: v for k, v in jd.items() if not isinstance(v, list)}}
    r["r4face"] = run_json([PY, str(HERE / "r4_face_metrics.py"), "--video", str(video), "--src", str(src), "--src-lm", str(lm), "--out", str(d / f"{id_}_{label}.r4face.json")])
    am = pathlib.Path(man["dubbed_track"]["aac"]).parent
    import importlib; sys.path.insert(0, str(Q)); hc = importlib.import_module("r3_hardcut")
    dub = am / "dubbed_24k.wav"
    if dub.exists():
        ev = hc.events(dub); cuts = [(e1, e2) for e1, e2 in zip(ev, ev[1:]) if e1[0] == "OFF" and e2[0] == "ON" and e2[1] - e1[1] <= 0.6]
        r["audio_gate"] = {"audio_hard_cuts": {"cuts_stop_restart": len(cuts), "cut_times": [round(c[0][1], 3) for c in cuts]}}
    add_sync(r, id_, label, d, video, man); add_chinwave(r, id_, label, d, video); add_mouth_audio(r, id_, label, d, video, man)
    r["manifest_r4v"] = man.get("r4v") or {"s_per_video_min": {"total": None}, "peak_vram_mib": None}
    r["audio_md5"] = audio_md5(video)
    sj.write_text(json.dumps(r, indent=1)); return r


def cmd_score(a):
    ids = a.ids.split(","); mf = a.manifest_from.split(":") if a.manifest_from else None
    for id_ in ids:
        own = pathlib.Path(a.dir) / f"{id_}_{a.label}.manifest.json"
        man = own if own.exists() else pathlib.Path(mf[0]) / f"{id_}_{mf[1]}.manifest.json"
        r = score_one(id_, a.label, a.dir, man, a.force, a.asr_from)
        print(id_, a.label, "sync", (r.get("syncnet") or {}).get("dub_only"), "flash", r.get("flash_strong"), "colour", r["r4face"].get("colour_worst_1s_abs_dL"),
              "jaw", (r["r4face"].get("jaw_jitter") or {}).get("ratio_out_over_src"), flush=True)


def verdict(rule, n, p):
    name, f, dirn, at, rt, hard = rule
    try: vn = f(n)
    except Exception: vn = None
    try: vp = f(p)
    except Exception: vp = None
    if vn is None and vp is None: return vn, vp, "n/a"   # e.g. 02: no lip-sync frames in either
    if dirn == "eq": return vn, vp, "PASS" if vn else "FAIL"
    if dirn == 0 or vn is None or vp is None: return vn, vp, "INFO" if vn is not None else "MISSING"
    over = lambda v: v * dirn > hard * dirn
    # a hard limit fails a video only when PREV was inside it (crossing the limit = regression); a baseline already beyond it is reported, and
    # NEW is then held to the relative rule below (v4 itself runs 390-475 s/video-min on 03 / 04 / 05 against the 300 s norm)
    if hard is not None and over(vn) and not over(vp): return vn, vp, "FAIL (limit)"
    tol = max(at, rt * abs(vp)); worse = (vn - vp) * dirn
    v = "FAIL" if worse > tol else ("better" if worse < -tol else "=")
    return vn, vp, v + (" (over limit)" if hard is not None and over(vn) and not v.startswith("FAIL") else "")


def fmt(x):
    if isinstance(x, float): return f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}"
    return str(x)


def face_same_frames(id_, sn, sp, cache):
    """r4 (coordinator 20.10): per-frame video metrics on the SAME frame set for NEW and PREV = the intersection of the frames lip-synced in both.
    A NEW run that lip-syncs fewer frames (e.g. dub-gated segments) would otherwise be averaged over a different (speech-only) frame set: 04 m2g
    'v4 resid mouth' 1.106 vs 0.999 on own frames, 1.106 vs m2 1.110 on identical frames. Returns {label: face_v4 dict} (cached)."""
    if cache.exists(): return json.loads(cache.read_text())
    mn, mp = (json.loads(pathlib.Path(x["manifest"]).read_text()) for x in (sn, sp))
    seg = lambda m: [(s["start_frame"], s["end_frame"]) for s in (m.get("lipsync") or {}).get("segments", []) if s.get("action") == "LATENT_SYNC"]
    inter = [(max(a0, b0), min(a1, b1)) for a0, a1 in seg(mn) for b0, b1 in seg(mp) if min(a1, b1) - max(a0, b0) >= 16]
    if not inter: cache.write_text("{}"); return {}
    tmpl = json.loads(json.dumps(mn)); segs = [s for s in tmpl["lipsync"]["segments"] if s.get("action") == "LATENT_SYNC"][:1]
    tmpl["lipsync"]["segments"] = [{**segs[0], "start_frame": a0, "end_frame": a1, "action": "LATENT_SYNC"} for a0, a1 in inter]
    mpath = cache.with_suffix(".manifest.json"); mpath.write_text(json.dumps(tmpl))
    src = next(pathlib.Path("/workspace/dub/inputs/test_videos_raw").glob(f"{id_}.*")); out = {}
    for tag, sc in (("new", sn), ("prev", sp)):
        j = cache.with_name(cache.stem + f"_{tag}.mjhr.json")
        subprocess.run([PY, str(Q / "mouth_jitter_hr.py"), "--video", sc["video"], "--reference", str(src), "--manifest", str(mpath), "--out", str(j), "--tag", tag], capture_output=True, text=True)
        if not j.exists(): continue
        d = json.loads(j.read_text()); core = (d.get("classes") or {}).get("ls_core") or d["all"]; g = lambda x: None if x is None else x.get("ratio")
        out[tag] = {"resid_mouth_ls_core": g(core["resid"]["mouth"]), "twitch_mouth_ls_core": g((core.get("twitch_index") or {}).get("mouth")),
                    "shear95_lowerface_ls_core": g(core.get("shear95", {}).get("lowerface")), "sharp_mouth_ls_core": g(core.get("sharp", {}).get("mouth")),
                    "frames_common_s": round(sum(b - a for a, b in inter) / 25, 1)}
    cache.write_text(json.dumps(out)); return out


def cmd_compare(a):
    nd, nl = a.new.split(":"); pd, pl = a.prev.split(":"); ids = a.ids.split(",")
    L = [f"# Regression gate: NEW `{nl}` vs PREV `{pl}`\n", f"commit {subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, cwd=HERE).stdout.strip()}\n",
         "| metric | " + " | ".join(f"{i} new / prev" for i in ids) + " |", "|---|" + "---|" * len(ids)]
    fails = []
    S = {i: (json.loads((pathlib.Path(nd) / f"{i}_{nl}.score.json").read_text()), json.loads((pathlib.Path(pd) / f"{i}_{pl}.score.json").read_text())) for i in ids}
    for i in ids:   # same-frame face metrics: the gate rows read the intersection values, the own-frame values stay as info
        sn, sp = S[i]
        if not ((sn.get("face_v4") or {}).get("resid_mouth_ls_core") is not None and (sp.get("face_v4") or {}).get("resid_mouth_ls_core") is not None): continue
        fs = face_same_frames(i, sn, sp, pathlib.Path(nd) / f"{i}_{nl}__vs_{pl}.same.json")
        if "new" in fs and "prev" in fs:
            for sc, k in ((sn, "new"), (sp, "prev")):
                sc["face_v4_own"] = dict(sc["face_v4"]); sc["face_v4"] = {**sc["face_v4"], **fs[k]}
    L.insert(2, "Face rows `v4 *` are computed on the SAME frames for NEW and PREV (intersection of the frames lip-synced in both); own-frame values: info rows.\n")
    AUDIO = ("WER final mix", "audio hard cuts", "|LUFS + 16|", "true peak dBTP")   # --audio-info: merged gate on the video axes, audio rows reported only
    for rule in RULES:
        if a.audio_info and rule[0] in AUDIO: rule = (rule[0] + " (audio, info)", rule[1], 0, 0, 0, None)
        cells = []
        for i in ids:
            vn, vp, v = verdict(rule, *S[i]); cells.append(f"{fmt(vn)} / {fmt(vp)} **{v}**" if v.startswith("FAIL") else f"{fmt(vn)} / {fmt(vp)} {v}")
            if v.startswith("FAIL") or v == "MISSING" and rule[2] != 0: fails.append(f"{i} {rule[0]}: {fmt(vn)} vs {fmt(vp)} ({v})")
        L.append(f"| {rule[0]} | " + " | ".join(cells) + " |")
    L += ["", f"## Verdict: {'**FAIL — no publish**' if fails else 'PASS'}", *[f"- {x}" for x in fails]]
    pathlib.Path(a.out).write_text("\n".join(L) + "\n"); print("\n".join(L)); return 1 if fails else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("score"); s.add_argument("--label", required=True); s.add_argument("--dir", required=True); s.add_argument("--manifest-from", default=None)
    s.add_argument("--ids", default=",".join(IDS)); s.add_argument("--force", action="store_true"); s.add_argument("--asr-from", default=None, help="DIR:LABEL whose WER to reuse (same audio)")
    c = sp.add_parser("compare"); c.add_argument("--new", required=True); c.add_argument("--prev", required=True); c.add_argument("--out", required=True); c.add_argument("--ids", default=",".join(IDS)); c.add_argument("--audio-info", action="store_true", help="audio rows = info (the merged gate judges video)")
    a = ap.parse_args(); return cmd_score(a) if a.cmd == "score" else cmd_compare(a)


if __name__ == "__main__":
    raise SystemExit(main())
