#!/usr/bin/env python3
"""Stage decomposition of the cand5 video stage on one window of a finished run: where does the mouth / face jitter come from?

Re-runs ONLY the video stage of a baseline run (same master frames, same dub track, same LatentSync / CodeFormer / paste-back settings
as cand5) on a frame window and keeps every intermediate, all on the same frames:

  S0_ls_raw        LatentSync output frames (its own 512 face-square paste-back into the frame), cand5 landmark smoothing (+-2)
  S1_ls_cf         S0 + optimized CodeFormer (w 1.0, fp16, insightface 5 points) — what cand5 feeds into the composite
  S2_composite     S1 composited into the master inside the feathered lower-face ellipse (mask smooth +-3)          (= cand5, no crossfade)
  S3_final         S2 + the 4-frame LS <-> original crossfade of the assembly                                         (= cand5 output)
  S2nc / S3nc      the same without CodeFormer (S0 composited)
  master           the master frames through the same encoder (noise floor of the comparison)

LatentSync segments = the run's LATENT_SYNC segments clipped to the window; a clipped segment starts on the run's own 16-frame window
grid (segment start + 16 k), so window boundaries fall on the same frames as in the run. A segment that lies completely inside the
window is bit-for-bit the run's input (frames, dub slice, detections) -> also a reproducibility check against the run's output.
Every stage is written losslessly (libx264rgb crf 0) as <out>/<stage>.mp4 plus <out>/window_manifest.json (segments in window frame
indices, for mouth_jitter_hr.py --manifest). Optional --sbs: labelled side-by-side review clip (with the run's muxed audio).

    python scripts/quality/decompose_video_stage.py --manifest RUN/final/03_final.manifest.json --frames 1050:1506 --out DIR [--sbs]
"""
from __future__ import annotations
import argparse, json, os, pathlib, subprocess, sys, time
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "optim"))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
FPS = 25; SR = 16000; P = 16


def write_video(path: pathlib.Path, frames_bgr: list) -> None:
    import imageio, cv2
    w = imageio.get_writer(str(path), fps=FPS, codec="libx264rgb", macro_block_size=None, pixelformat="rgb24",
                           ffmpeg_params=["-crf", "0", "-preset", "veryfast"], ffmpeg_log_level="error")
    for f in frames_bgr: w.append_data(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    w.close()


def clip_segments(man: dict, a: int, b: int) -> list:
    """LATENT_SYNC segments of the run clipped to [a, b), start moved forward onto the run's own 16-frame window grid. Crossfade in / out =
    min(4, frames // 3) only where the run's neighbouring segment is not LATENT_SYNC (process_video's rule)."""
    out = []; allseg = sorted([s for s in man["lipsync"]["segments"] if s["action"] != "SPLIT"], key=lambda s: s["start_frame"])
    for j, s in enumerate(allseg):
        if s["action"] != "LATENT_SYNC" or s["end_frame"] <= a or s["start_frame"] >= b: continue
        k = min(4, (s["end_frame"] - s["start_frame"]) // 3)
        prev_ls = j > 0 and allseg[j - 1]["action"] == "LATENT_SYNC"; next_ls = j + 1 < len(allseg) and allseg[j + 1]["action"] == "LATENT_SYNC"
        st = s["start_frame"]
        if st < a: st = s["start_frame"] + P * int(np.ceil((a - s["start_frame"]) / P))
        en = min(s["end_frame"], b)
        if en - st < P: continue
        out.append({"run_index": s["index"], "start": st, "end": en, "full": st == s["start_frame"] and en == s["end_frame"],
                    "xf_in": k if st == s["start_frame"] and j > 0 and not prev_ls else 0, "xf_out": k if en == s["end_frame"] and j + 1 < len(allseg) and not next_ls else 0})
    return out


def crossfade(rep: list, master: list, k_in: int, k_out: int) -> list:
    """assemble_frames() of face_aware_latentsync_accel: alpha (k+1)/(xf+1) ramps at LS <-> original switches."""
    import cv2
    n = len(rep); res = []
    for k in range(n):
        a = 1.0
        if k_in and k < k_in: a = (k + 1) / (k_in + 1)
        if k_out and k >= n - k_out: a = min(a, (n - k) / (k_out + 1))
        res.append(cv2.addWeighted(rep[k], a, master[k], 1.0 - a, 0.0) if a < 1.0 else rep[k])
    return res


def main() -> int:
    import cv2, soundfile as sf
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--frames", required=True, help="a:b master frame window (end exclusive) or 'all'")
    ap.add_argument("--stages-out", nargs="*", default=None, help="stages to keep / write (default: all); master is always written")
    ap.add_argument("--out", required=True); ap.add_argument("--ls-landmark-smooth", type=int, default=2); ap.add_argument("--mask-smooth", type=int, default=3)
    ap.add_argument("--mask-top", type=float, default=0.52); ap.add_argument("--mask-feather", type=float, default=0.08)
    ap.add_argument("--codeformer-w", type=float, default=1.0); ap.add_argument("--sbs", action="store_true")
    a = ap.parse_args()
    man = json.loads(pathlib.Path(a.manifest).read_text()); work = pathlib.Path(man["work_dir"])
    A, B = (0, 10 ** 9) if a.frames == "all" else [int(x) for x in a.frames.split(":")]; out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import face_aware_latentsync as fa, face_aware_latentsync_accel as faa
    from candidate_video_stage import MouthOnlyRunner
    from codeformer_accel import CodeFormerAccel, lm5_from_insightface106
    times = {}; t_all = time.perf_counter()
    master_all = fa.read_frames(work / "master_25fps.mp4"); B = min(B, len(master_all)); master = master_all[A:B]; del master_all
    audio, sr = sf.read(str(work / "dubbed_16k.wav"), dtype="float32"); audio = audio.mean(1) if audio.ndim > 1 else audio; assert sr == SR
    segs = clip_segments(man, A, B)
    print(f"window {A}:{B} ({(B - A) / FPS:.2f} s), LS pieces: {[(s['start'], s['end'], s['full']) for s in segs]}", flush=True)
    t0 = time.perf_counter(); inner = faa.AccelRunner(2, True, "none", "auto", steps=20, guidance=1.5, deepcache_interval=5); times["ls_load"] = round(time.perf_counter() - t0, 2)
    runner = MouthOnlyRunner(inner, a.mask_top, a.mask_feather, a.mask_smooth, defer=True)
    cf = CodeFormerAccel(fidelity_weight=a.codeformer_w, batch_size=8, landmark_source="insightface", autocast=True, parse_autocast=True)
    keep = a.stages_out or ["S0_ls_raw", "S1_ls_cf", "S2_composite", "S3_final", "S2nc_composite_nocf", "S3nc_final_nocf"]
    stages = {k: [f.copy() for f in master] for k in keep}
    t_ls = t_cf = 0.0
    for s in segs:
        st, en = s["start"], s["end"]; rgb = np.stack([cv2.cvtColor(master[i - A], cv2.COLOR_BGR2RGB) for i in range(st, en)])
        a0 = int(round(st / FPS * SR)); seg_audio = audio[a0:a0 + int(round((en - st) / FPS * SR))]
        wav = out / f"seg_{st}_in.wav"; sf.write(str(wav), seg_audio, SR, subtype="PCM_16")
        dets = fa.detect_frames(rgb)
        if not all(d[0] is not None for d in dets): print(f"  piece {st}:{en}: LatentSync detector rejects {sum(d[0] is None for d in dets)} frames -> skipped"); continue
        if a.ls_landmark_smooth > 0: dets = faa.smooth_detections(dets, a.ls_landmark_smooth)
        t0 = time.perf_counter(); frs_rgb, _ = runner.run_frames(rgb, wav, dets); t_ls += time.perf_counter() - t0; wav.unlink(missing_ok=True)
        n = min(len(frs_rgb), en - st); raw = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frs_rgb[:n]]; boxes = runner._boxes
        t0 = time.perf_counter(); lms = [lm5_from_insightface106(dets[k][1]) for k in range(n)]; cfo, st_cf = cf.restore_frames([f.copy() for f in raw], lms); t_cf += time.perf_counter() - t0
        mst = [master[i - A] for i in range(st, st + n)]
        comp = runner.composite(cfo, mst, boxes); comp_nc = runner.composite(raw, mst, boxes)
        fin = crossfade(comp, mst, s["xf_in"], s["xf_out"]); fin_nc = crossfade(comp_nc, mst, s["xf_in"], s["xf_out"])
        for k, lst in (("S0_ls_raw", raw), ("S1_ls_cf", cfo), ("S2_composite", comp), ("S3_final", fin), ("S2nc_composite_nocf", comp_nc), ("S3nc_final_nocf", fin_nc)):
            if k in stages:
                for j in range(n): stages[k][st - A + j] = lst[j]
        print(f"  piece {st}:{en} done: LS {n} frames, CodeFormer {st_cf['faces_restored']} faces", flush=True)
    times["latentsync"] = round(t_ls, 2); times["codeformer"] = round(t_cf, 2)
    t0 = time.perf_counter()
    write_video(out / "master.mp4", master)
    for k, lst in stages.items(): write_video(out / f"{k}.mp4", lst)
    times["write"] = round(time.perf_counter() - t0, 2)
    # run-output reproducibility: the run's final frames on the window vs S3_final (pieces that are complete run segments)
    cap = cv2.VideoCapture(man["output"]); cap.set(cv2.CAP_PROP_POS_FRAMES, A); diffs = []
    for i in range(A, B):
        ok, f = cap.read()
        if not ok: break
        if "S3_final" in stages and any(s["full"] and s["start"] <= i < s["end"] for s in segs): diffs.append(float(np.abs(f.astype(np.float32) - stages["S3_final"][i - A].astype(np.float32)).mean()))
    cap.release()
    wman = {"source_manifest": a.manifest, "window": [A, B], "fps": FPS, "pieces": segs, "times": times,
            "settings": {"ls_landmark_smooth": a.ls_landmark_smooth, "mask_smooth": a.mask_smooth, "mask_top": a.mask_top, "mask_feather": a.mask_feather,
                         "codeformer_w": a.codeformer_w, "latentsync": inner.config},
            "repro_vs_run_output_full_segments": {"frames": len(diffs), "mean_abs_diff": round(float(np.mean(diffs)), 3) if diffs else None,
                                                   "note": "run output is libx264 crf 13 yuv420p, stages are lossless: ~1-2 levels = encoder noise"},
            "lipsync": {"segments": []}}
    prev = 0
    for s in segs:   # segments in WINDOW frame indices (mouth_jitter_hr.py --manifest)
        if s["start"] - A > prev: wman["lipsync"]["segments"].append({"index": len(wman["lipsync"]["segments"]), "action": "PASS_THROUGH", "start_frame": prev, "end_frame": s["start"] - A})
        wman["lipsync"]["segments"].append({"index": len(wman["lipsync"]["segments"]), "action": "LATENT_SYNC", "start_frame": s["start"] - A, "end_frame": s["end"] - A})
        prev = s["end"] - A
    if prev < B - A: wman["lipsync"]["segments"].append({"index": len(wman["lipsync"]["segments"]), "action": "PASS_THROUGH", "start_frame": prev, "end_frame": B - A})
    times["total"] = round(time.perf_counter() - t_all, 2)
    (out / "window_manifest.json").write_text(json.dumps(wman, indent=1))
    if a.sbs and all(k in stages for k in ("S0_ls_raw", "S1_ls_cf", "S3_final")):
        aud = man["dubbed_track"].get("aac") or str(work / "final_mix_aac.m4a")
        flt = ("[0:v]drawtext=text='ORIGINAL':x=8:y=8:fontsize=26:fontcolor=yellow:box=1:boxcolor=black@0.5[a];"
               "[1:v]drawtext=text='LS raw':x=8:y=8:fontsize=26:fontcolor=yellow:box=1:boxcolor=black@0.5[b];"
               "[2:v]drawtext=text='LS + CodeFormer':x=8:y=8:fontsize=26:fontcolor=yellow:box=1:boxcolor=black@0.5[c];"
               "[3:v]drawtext=text='cand5 final':x=8:y=8:fontsize=26:fontcolor=yellow:box=1:boxcolor=black@0.5[d];[a][b][c][d]hstack=inputs=4[v]")
        sbs = out / "sbs_stages.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(out / "master.mp4"), "-i", str(out / "S0_ls_raw.mp4"), "-i", str(out / "S1_ls_cf.mp4"),
                        "-i", str(out / "S3_final.mp4"), "-ss", f"{A / FPS:.3f}", "-t", f"{(B - A) / FPS:.3f}", "-i", aud, "-filter_complex", flt,
                        "-map", "[v]", "-map", "4:a:0", "-c:v", "libx264", "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(sbs)], check=False)
    print(json.dumps({"times": times, "repro": wman["repro_vs_run_output_full_segments"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
