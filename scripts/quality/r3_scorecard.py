#!/usr/bin/env python3
"""Round 3 D: one QA scorecard per candidate run over 01-05 (JSON + Markdown), every spec item that binds the candidate.

Per video (`score`), from the output mp4 + its run manifest:
  spec 9.4   output w x h / fps / frame count / duration / codec vs the SOURCE
  coverage   LatentSync frames (master grid) / face frames / master frames
  sync       SyncNet (LatentSync evaluator, syncnet_fps.py: a non-25 fps output is scored on its fps=25 rendition): final mix, dub-only, AV offset
  face       common twitch metric mouth_jitter_hr v4 (resid / twitch / frozen mouth / lower-face shear) against the source frames
  slide      common slide metric on the shared shaky windows (/workspace/dub/common/shaky_windows.json), lower face vs the same-instant reference
  joints 8.5 LatentSync window starts: share with a mouth d1 jump > 1.5x (v4 latentsync_window_boundaries) + LS <-> original switches
  feather 8.4  configured 10-90 % edge width of the mouth mask in output px (clean0 Gaussian: 2.563 sigma of the kernel at the median face width)
  audio      Whisper large-v3 back-transcription WER of the FINAL MIX vs the English text; EBU R128 integrated loudness / true peak
  speed      stage seconds -> s per video-minute (run manifest stage_seconds + any extra stage the caller adds)

    python scripts/quality/r3_scorecard.py score --id 03 --label clean0 --video V.mp4 --manifest V.manifest.json --out DIR [--extra-stage name=s]
    python scripts/quality/r3_scorecard.py table --label clean0 --dir DIR [--label r2m --dir DIR2 ...] --out card.md
"""
from __future__ import annotations
import argparse, json, math, pathlib, re, subprocess, sys, tempfile, time
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent
for d in ("pilot", "quality", "diag"): sys.path.insert(0, str(HERE.parent / d))
PY = sys.executable
SRC_DIR = pathlib.Path("/workspace/dub/inputs/test_videos_raw"); SHAKY = pathlib.Path("/workspace/dub/common/shaky_windows.json")


def src_of(id_):
    return next(SRC_DIR.glob(f"{id_}.*"))


def probe(p):
    o = json.loads(subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                                   "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_read_frames", "-show_entries", "format=duration", "-of", "json", str(p)],
                                  capture_output=True, text=True).stdout)
    s = o["streams"][0]; n, d = s["avg_frame_rate"].split("/")
    return {"codec": s["codec_name"], "w": s["width"], "h": s["height"], "fps": round(float(n) / float(d), 3), "r_frame_rate": s["r_frame_rate"],
            "frames": int(s["nb_read_frames"]), "duration": round(float(o["format"]["duration"]), 3)}


def loudness(p):
    e = subprocess.run(["ffmpeg", "-nostats", "-hide_banner", "-i", str(p), "-map", "0:a:0", "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True).stderr
    tail = e[e.rfind("Summary:"):]
    I = re.search(r"I:\s+(-?[\d.]+) LUFS", tail); LRA = re.search(r"LRA:\s+(-?[\d.]+) LU", tail); TP = re.search(r"Peak:\s+(-?[\d.]+|-inf) dBFS", tail)
    return {"integrated_lufs": float(I.group(1)) if I else None, "lra_lu": float(LRA.group(1)) if LRA else None,
            "true_peak_dbtp": (float(TP.group(1)) if TP and TP.group(1) != "-inf" else None)}


def norm_words(t):
    t = t.lower().replace("’", "'"); t = re.sub(r"[^a-z0-9' ]+", " ", t)
    return [w for w in t.split() if w]


def wer(ref, hyp):
    r, h = norm_words(ref), norm_words(hyp)
    if not r: return None
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1])); prev, d[j] = d[j], cur
    return round(d[len(h)] / len(r), 4)


def unit_text(u):
    """the English text actually synthesised (tts_parts after fit / refit), else the unit translation."""
    if u.get("tts_parts"): return " ".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in u["tts_parts"])
    t = u.get("translation")
    if isinstance(t, dict): t = t.get("text") or t.get("en") or ""
    return t or ""


def back_transcribe(p, man):
    import torch  # noqa: F401  maps libcublas before ctranslate2 loads (as run_clean_pipeline_05.py)
    import faster_whisper
    root = pathlib.Path(__file__).resolve().parents[2]
    cand = [root / "models" / "faster-whisper-large-v3", root / "models" / "whisper-large-v3"]
    mdl = next((str(c) for c in cand if c.exists()), "large-v3")
    with tempfile.TemporaryDirectory() as td:
        w = pathlib.Path(td) / "a.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(p), "-map", "0:a:0", "-ac", "1", "-ar", "16000", str(w)], check=True)
        m = faster_whisper.WhisperModel(mdl, device="cuda", compute_type="float16")
        segs, _ = m.transcribe(str(w), language="en", beam_size=5, vad_filter=False, condition_on_previous_text=False)
        hyp = " ".join(s.text.strip() for s in segs); del m
    ref = " ".join(unit_text(u) for u in man["units"])
    return {"model": pathlib.Path(mdl).name, "wer_final_mix": wer(ref, hyp), "ref_words": len(norm_words(ref)), "hyp_words": len(norm_words(hyp)), "hyp": hyp}


def feather_px(man, video_params):
    """configured 10-90 % edge width of the mask in output px."""
    vp = video_params or {}
    if man.get("fix30"): return {"kind": "fps_composite explicit edge", "edge_10_90_px": man["fix30"].get("edge_px")}
    f = vp.get("mask_feather", 0.08); fs = man.get("face_stack") or {}
    wmed = None
    mb = pathlib.Path(man["output"]).with_name(pathlib.Path(man["output"]).stem + "_maskboxes.npz")
    if mb.exists():
        z = np.load(mb); b = z["boxes"]; wmed = float(np.median(b[:, 2] - b[:, 0]))
    if wmed is None: return {"kind": "gaussian (face width unknown)", "mask_feather_frac": f}
    k = max(3, int(f * wmed) | 1); sig = 0.3 * ((k - 1) * 0.5 - 1) + 0.8
    return {"kind": "clean0 gaussian ellipse", "median_face_w_px": round(wmed, 1), "kernel": k, "sigma": round(sig, 2), "edge_10_90_px": round(2.563 * sig, 1)}


def slide_windows(id_, video, ref_video, ref_fps_is_source, master_timing, out_dir):
    """common slide metric on the shaky windows; output frame k vs reference frame k (source when the output is at source fps, else the master)."""
    sw = json.loads(SHAKY.read_text())["videos"].get(id_)
    if not sw: return None
    import av, cv2
    import clean_face_stack as cfs
    def times(p):
        c = av.open(str(p)); s = c.streams.video[0]; t = [float(f.pts * s.time_base) for f in c.decode(s)]; c.close(); return np.array(t)
    tv = times(video); res = []
    for t0, t1 in sw["windows_s"]:
        k = np.flatnonzero((tv >= t0) & (tv < t1)); a, b = int(k.min()), int(k.max()) + 1
        cuts = {}
        for name, p in (("v", video), ("r", ref_video)):
            c = av.open(str(p)); s = c.streams.video[0]; fr = []
            for i, f in enumerate(c.decode(s)):
                if i >= b: break
                if i >= a: fr.append(f.to_ndarray(format="bgr24"))
            c.close()
            q = out_dir / f".slide_{id_}_{name}_{a}_{b}.mp4"
            import imageio.v2 as iio
            w = iio.get_writer(str(q), fps=25, codec="libx264rgb", macro_block_size=None, pixelformat="rgb24", ffmpeg_params=["-crf", "0", "-preset", "veryfast"], ffmpeg_log_level="error")
            for f in fr: w.append_data(f[:, :, ::-1])
            w.close(); cuts[name] = q
        npz = cfs.run_mediapipe(cuts["r"], cuts["r"].with_suffix(".mediapipe.npz"))
        j = out_dir / f"{id_}_slide_{t0}-{t1}.json"
        subprocess.run([PY, str(HERE / "slide_metric.py"), "--video", str(cuts["v"]), "--master", str(cuts["r"]), "--npz", str(npz), "--offset", "0", "--frames", f"0:{b - a}", "--out", str(j)],
                       capture_output=True, text=True, env={**__import__("os").environ, "OPENCV_FOR_THREADS_NUM": "8"})
        for q in list(cuts.values()) + [pathlib.Path(npz)]: pathlib.Path(q).unlink(missing_ok=True)
        if j.exists():
            d = json.loads(j.read_text()); res.append({"window_s": [t0, t1], "frames": [a, b], **{r: d.get("regions", d).get(r) for r in ("nosebase", "cheeks", "upper") if isinstance(d.get("regions", d), dict)}})
    return res


def cmd_score(a):
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); t_start = time.perf_counter()
    man = json.loads(pathlib.Path(a.manifest).read_text()); video = pathlib.Path(a.video); src = src_of(a.id)
    prev = out / f"{a.id}_{a.label}.score.json"
    r = json.loads(prev.read_text()) if a.update and prev.exists() else {}
    r |= {"id": a.id, "label": a.label, "video": str(video), "manifest": str(a.manifest), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    ps, po = probe(src), probe(video); r["source"], r["output"] = ps, po
    r["spec_9_4"] = {"resolution_equal": (ps["w"], ps["h"]) == (po["w"], po["h"]), "fps_equal": abs(ps["fps"] - po["fps"]) < 0.05,
                     "frames_equal": ps["frames"] == po["frames"], "duration_delta_s": round(po["duration"] - ps["duration"], 3), "h264": po["codec"] == "h264"}
    r["spec_9_4"]["pass"] = all(r["spec_9_4"][k] for k in ("resolution_equal", "fps_equal", "frames_equal", "h264")) and abs(r["spec_9_4"]["duration_delta_s"]) <= 0.15
    ls = man["lipsync"]; ba = ls.get("by_action", {})
    r["coverage"] = {"ls_frames": (ba.get("LATENT_SYNC") or {}).get("frames", 0), "master_frames": ls.get("master_frames"),
                     "face_frames": (ls.get("frame_classes") or {}).get("VALID_FACE"), "segments": (ba.get("LATENT_SYNC") or {}).get("segments"),
                     "switches": sum(1 for s in ls.get("segments", []) if s.get("action") == "LATENT_SYNC") * 2}
    vp = (man.get("candidate") or {}).get("video_params")
    r["feather"] = feather_px(man, vp)
    if not a.skip_sync and r["coverage"]["ls_frames"]:
        dub = man["dubbed_track"]["aac"]
        dub_voice = str(pathlib.Path(dub).with_name("dubbed_aac.m4a")) if pathlib.Path(dub).name.startswith("final_mix") else dub
        p = subprocess.run([PY, str(HERE / "syncnet_fps.py"), str(video), "--dub", dub_voice], capture_output=True, text=True)
        try: r["syncnet"] = json.loads([l for l in p.stdout.splitlines() if l.startswith("{")][-1])
        except Exception: r["syncnet"] = {"error": p.stderr[-400:]}
    if not a.skip_face:
        j = out / f"{a.id}_{a.label}.mjhr.json"
        p = None if (a.update and j.exists()) else subprocess.run([PY, str(HERE / "mouth_jitter_hr.py"), "--video", str(video), "--reference", str(src), "--manifest", str(a.manifest), "--out", str(j), "--tag", f"{a.id}_{a.label}"],
                           capture_output=True, text=True)
        if j.exists():
            d = json.loads(j.read_text()); A = d["all"]; cl = d.get("classes", {}); wb = d.get("latentsync_window_boundaries") or {}
            g = lambda x, k="ratio": None if x is None else x.get(k)
            core = cl.get("ls_core") or A
            fm = [s["frozen_mouth"] for s in d.get("per_second", []) if s.get("frozen_mouth") is not None]
            r["face_v4"] = {"resid_mouth_ls_core": g(core["resid"]["mouth"]), "twitch_mouth_ls_core": g((core.get("twitch_index") or {}).get("mouth")),
                            "shear95_lowerface_ls_core": g(core.get("shear95", {}).get("lowerface")), "sharp_mouth_ls_core": g(core.get("sharp", {}).get("mouth")),
                            "jerk_upper_all": g(A["jerk"]["upper"]), "frozen_mouth_p95_per_s": round(float(np.percentile(fm, 95)), 3) if fm else None,
                            "frozen_mouth_secs_gt2": int(sum(1 for x in fm if x > 2.0)), "window_boundaries": wb.get("boundary_d1_jump"),
                            "window_phase0_over_inner_d1": (wb.get("d1") or {}).get("video", {}).get("phase0_over_inner")}
        else: r["face_v4"] = {"error": p.stderr[-400:] if p else "no mjhr json"}
    if not a.skip_slide:
        same = abs(po["fps"] - ps["fps"]) < 0.05
        ref = src if same else pathlib.Path(man["work_dir"]) / "master_25fps.mp4"
        try: r["slide"] = slide_windows(a.id, video, ref, same, None, out)
        except Exception as e: r["slide"] = {"error": repr(e)[:300]}
    lmF = pathlib.Path("/workspace/dub/artifacts/r3d/timing") / f"{a.id}_source_lm_F.npz"
    if abs(po["fps"] - ps["fps"]) < 0.05 and lmF.exists() and r["coverage"]["ls_frames"]:   # background ring beside the face (user defect 3)
        p = subprocess.run([PY, str(HERE / "r3_bgring.py"), "--video", str(video), "--ref", str(src), "--lm", str(lmF)], capture_output=True, text=True)
        try: r["bgring"] = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception: r["bgring"] = {"error": p.stderr[-300:]}
    r["loudness"] = loudness(video)
    if not a.skip_wer:
        try: r["asr"] = back_transcribe(video, man)
        except Exception as e: r["asr"] = {"error": repr(e)[:300]}
    st = dict(man.get("stage_seconds") or {})
    for x in a.extra_stage or []:
        k, v = x.split("="); st[k] = float(v)
    vm = ps["duration"] / 60.0
    r["speed"] = {"stage_seconds": st, "s_per_video_min": {k: round(v / vm, 1) for k, v in st.items() if isinstance(v, (int, float))}, "video_min": round(vm, 3)}
    r["scorecard_seconds"] = round(time.perf_counter() - t_start, 1)
    (out / f"{a.id}_{a.label}.score.json").write_text(json.dumps(r, indent=1)); print(json.dumps({k: r[k] for k in ("id", "label", "spec_9_4", "coverage")} | {"syncnet": r.get("syncnet")}))
    return 0


def audio_source(man):
    """which audio a run muxes: the canonical cand5 track (HDEMUCS bed: video-only CONTROL, not a candidate) or a delivered clean track."""
    dd = (man.get("candidate") or {}).get("dubbed_dir") or man.get("audio_dir") or man.get("work_dir") or ""
    if "/handoff/in/C/" in dd or "audio-r" in dd: return "C " + pathlib.Path(dd).parent.name + " (Spleeter bed, clean)"
    if "/artifacts/baseline/" in dd or "/canonical/" in dd: return "cand5 canonical (HDEMUCS bed: control only)"
    return "own run: " + dd


def fmt(x, nd=2):
    if x is None: return "–"
    if isinstance(x, bool): return "✓" if x else "✗"
    if isinstance(x, float): return f"{x:.{nd}f}"
    return str(x)


def cmd_table(a):
    rows = {}
    for lab, d in zip(a.label, a.dir):
        for p in sorted(pathlib.Path(d).glob(f"*_{lab}.score.json")):
            r = json.loads(p.read_text()); rows.setdefault(r["id"], {})[lab] = r
    ids = sorted(rows); L = []
    L.append(f"# Scorecard: {' vs '.join(a.label)}\n")
    def line(name, f, nd=2):
        cells = []
        for i in ids:
            for lab in a.label:
                r = rows[i].get(lab)
                try: cells.append(fmt(f(r), nd) if r else "n/a")
                except Exception: cells.append("–")
        L.append(f"| {name} | " + " | ".join(cells) + " |")
    L.append("| metric | " + " | ".join(f"{i} {lab}" for i in ids for lab in a.label) + " |"); L.append("|---|" + "---:|" * (len(ids) * len(a.label)))
    line("audio source", lambda r: audio_source(json.loads(pathlib.Path(r["manifest"]).read_text())))
    line("9.4 pass (res / fps / frames / H.264 / dur)", lambda r: r["spec_9_4"]["pass"])
    line("output fps (source)", lambda r: f"{r['output']['fps']:g} ({r['source']['fps']:g})")
    line("frames out / source", lambda r: f"{r['output']['frames']}/{r['source']['frames']}")
    line("duration Δ s", lambda r: r["spec_9_4"]["duration_delta_s"], 3)
    line("LS frames / master", lambda r: f"{r['coverage']['ls_frames']}/{r['coverage']['master_frames']}")
    line("SyncNet mix", lambda r: r["syncnet"]["mix"][0]); line("SyncNet dub-only", lambda r: r["syncnet"]["dub_only"][0]); line("AV offset (dub)", lambda r: r["syncnet"]["dub_only"][1])
    line("v4 resid mouth (LS core, src=1)", lambda r: r["face_v4"]["resid_mouth_ls_core"]); line("v4 twitch mouth", lambda r: r["face_v4"]["twitch_mouth_ls_core"])
    line("v4 shear95 lower face", lambda r: r["face_v4"]["shear95_lowerface_ls_core"]); line("v4 sharpness mouth", lambda r: r["face_v4"]["sharp_mouth_ls_core"])
    line("v4 frozen mouth p95 / s", lambda r: r["face_v4"]["frozen_mouth_p95_per_s"])
    line("8.5 window starts > 1.5x mouth jump (source)", lambda r: f"{r['face_v4']['window_boundaries']['video']['share_gt_1_5']:.2f} ({r['face_v4']['window_boundaries']['reference']['share_gt_1_5']:.2f})")
    line("slide nose-base mean px (shaky win)", lambda r: np.mean([w["nosebase"]["slide_px"]["mean"] for w in r["slide"]]), 3)
    line("8.4 feather 10-90 px", lambda r: r["feather"].get("edge_10_90_px"), 1)
    line("WER final mix (Whisper)", lambda r: r["asr"]["wer_final_mix"], 3)
    line("bg ring beside face: lower - upper (grey, clean0 04 = 0.70)", lambda r: r["bgring"]["lower_minus_upper"], 2)
    line("loudness LUFS", lambda r: r["loudness"]["integrated_lufs"], 1); line("true peak dBTP", lambda r: r["loudness"]["true_peak_dbtp"], 1)
    line("total s / video-min", lambda r: r["speed"]["s_per_video_min"].get("total"), 0)
    line("LatentSync s / video-min", lambda r: r["speed"]["s_per_video_min"].get("latentsync"), 0)
    pathlib.Path(a.out).write_text("\n".join(L) + "\n"); print("\n".join(L))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps(rows, indent=1))
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter); sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("score"); s.add_argument("--id", required=True); s.add_argument("--label", required=True); s.add_argument("--video", required=True)
    s.add_argument("--manifest", required=True); s.add_argument("--out", required=True); s.add_argument("--extra-stage", action="append")
    for k in ("sync", "face", "slide", "wer"): s.add_argument(f"--skip-{k}", action="store_true")
    s.add_argument("--update", action="store_true", help="keep the parts of an existing score JSON that are skipped this time")
    t = sp.add_parser("table"); t.add_argument("--label", action="append", required=True); t.add_argument("--dir", action="append", required=True)
    t.add_argument("--out", required=True); t.add_argument("--json", default=None)
    a = ap.parse_args()
    return cmd_score(a) if a.cmd == "score" else cmd_table(a)


if __name__ == "__main__":
    raise SystemExit(main())
