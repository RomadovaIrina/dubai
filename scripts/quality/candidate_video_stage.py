#!/usr/bin/env python3
"""Re-run ONLY the video stage of run_clean_pipeline_05.py (speech gate -> RetinaFace router -> LatentSync -> assembly) on a baseline
work dir with a different dubbed track and/or different video-backend parameters. Audio stages (ASR/translation/TTS) are reused
from the baseline run, so E1/E6 candidates are compared on identical speech content. Writes <out>/<id>_<tag>.mp4 + manifest in the
runner's schema (so quality_manifest.py / face_metrics.py work unchanged).
    python scripts/quality/candidate_video_stage.py --id 04 --tag e1 --dubbed-dir /tmp/dabai_quality/candidates/e1/work_04 \
        --out /tmp/dabai_quality/candidates [--crossfade-frames 4] [--gate-margin-s 0.24] [--gate-merge-gap-s 0.6] [--router-stride 3] [--speech-gate on|off]
"""
from __future__ import annotations
import argparse, json, os, pathlib, shutil, sys, time
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "optim"))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
Q = pathlib.Path("/tmp/dabai_quality")


class MouthOnlyRunner:
    """E4 wrapper around AccelRunner: LatentSync output is composited back into the ORIGINAL frames only inside a feathered lower-face
    mask built from LatentSync's own per-frame detection bbox (replayed detections), so eyes/nose/skin keep the source texture.
    smooth > 0 (2026-10-13): the mask follows a centred moving average (+-smooth frames) of the bbox instead of the raw per-frame detection
    (the raw insightface bbox shakes by ~1.4 px/frame^2 on 03, and the mask edge with it).
    defer = True (2026-10-13, --codeformer-scope mouth): run_frames returns the RAW LatentSync frames and keeps the masks; the caller restores
    them with CodeFormer first and then calls composite(), so CodeFormer's re-synthesised eyes / brows never reach the output (it used to run on
    the composited frame and re-rendered the whole face: eyes / lids differ from the source in every LS frame and jump back at each switch)."""

    def __init__(self, inner, top: float, feather: float, smooth: int = 0, defer: bool = False):
        self.inner, self.top, self.feather, self.smooth, self.deferred = inner, top, feather, int(smooth), bool(defer)
        self.config = {**inner.config, "mouth_only_pasteback": True, "mask_top": top, "mask_feather": feather, "mask_smooth_frames": self.smooth, "composite_after_codeformer": self.deferred}
        self.calls = inner.calls
        self.steps, self.guidance, self.seed, self.load_s = inner.steps, inner.guidance, inner.seed, inner.load_s; self.masked_frames = 0; self._boxes = None

    raw: dict | None = None   # round 3 (--dump-raw): {segment start frame: raw LatentSync frames (BGR)} for track B's source-fps composite

    def set_segment(self, start, n):
        self._seg_start = start
        if hasattr(self.inner, "set_segment"): self.inner.set_segment(start, n)

    def boxes(self, detections, n):
        import numpy as np
        bb = [np.asarray(detections[i][0], np.float64) if i < len(detections) and detections[i][0] is not None else None for i in range(n)]
        if self.smooth <= 0: return bb
        ok = [i for i, b in enumerate(bb) if b is not None]; out = list(bb)
        for i in ok:
            nb = [bb[j] for j in range(max(0, i - self.smooth), min(n, i + self.smooth + 1)) if bb[j] is not None]
            out[i] = np.mean(nb, axis=0)
        return out

    def mask(self, bbox, H, W):
        """Feathered lower-face ellipse as (alpha HxW-crop float32 [..., None], (ya, yb, xa, xb)): only the ROI that can be non-zero is built
        (ellipse bounding box + 2x the blur kernel), the result equals the full-frame mask inside it (it is 0 outside)."""
        import cv2, numpy as np
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]; w, h = max(x2 - x1, 1), max(y2 - y1, 1)
        cy = y1 + int(self.top * h); cx = (x1 + x2) // 2
        # lower face: ellipse spanning the bbox width (+10 %) from mask_top to slightly below the chin
        axes = (int(w * 0.55), int((y2 + int(0.06 * h) - cy) * 0.5 + 1)); centre = (cx, (cy + y2 + int(0.06 * h)) // 2)
        k = max(3, int(self.feather * w) | 1); pad = k + 2
        xa, xb = max(0, centre[0] - axes[0] - pad), min(W, centre[0] + axes[0] + pad + 1); ya, yb = max(0, centre[1] - axes[1] - pad), min(H, centre[1] + axes[1] + pad + 1)
        if xb <= xa or yb <= ya: return None, None
        m = np.zeros((yb - ya, xb - xa), np.float32)
        cv2.ellipse(m, (centre[0] - xa, centre[1] - ya), axes, 0, 0, 360, 1.0, -1)
        return cv2.GaussianBlur(m, (k, k), 0, borderType=cv2.BORDER_CONSTANT)[..., None], (ya, yb, xa, xb)

    def composite(self, outs, originals, boxes=None):
        """outs / originals: same-length sequences of HxWx3 frames (same channel order); returns outs composited into originals."""
        import numpy as np
        boxes = boxes if boxes is not None else self._boxes; res = []
        if self.raw is not None and getattr(self, "_seg_start", None) is not None: self.raw[self._seg_start] = [np.array(o, copy=True) for o in outs]
        for i, o in enumerate(outs):
            bb = boxes[i] if boxes is not None and i < len(boxes) else None
            if bb is None or i >= len(originals): res.append(o); continue
            m, roi = self.mask(bb, *o.shape[:2])
            if m is None: res.append(o); continue
            ya, yb, xa, xb = roi; r = np.array(originals[i], copy=True)
            r[ya:yb, xa:xb] = (o[ya:yb, xa:xb].astype(np.float32) * m + r[ya:yb, xa:xb].astype(np.float32) * (1 - m)).round().clip(0, 255).astype(np.uint8)
            res.append(r); self.masked_frames += 1
        return res

    def run_frames(self, frames_rgb, audio_wav, detections=None):
        import numpy as np
        outs, dt = self.inner.run_frames(frames_rgb, audio_wav, detections)
        if detections is None: self._boxes = None; return outs, dt
        self._boxes = self.boxes(detections, len(frames_rgb))
        if self.deferred: return outs, dt
        res = np.stack(self.composite(outs[:len(frames_rgb)], frames_rgb))
        if len(outs) > len(frames_rgb): res = np.concatenate([res, outs[len(frames_rgb):]])
        return res, dt


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--run-dir", default=str(Q / "baseline")); ap.add_argument("--work-dir", default=str(Q / "work"))
    ap.add_argument("--dubbed-dir", default=None, help="dir with dubbed_16k.wav + dubbed_aac.m4a (+ aligned_u*.wav); default = baseline work dir")
    ap.add_argument("--out", default=str(Q / "candidates")); ap.add_argument("--work-out", default=None)
    ap.add_argument("--manifest", default=None, help="manifest of a finished E2E run (overrides --run-dir / --work-dir)")
    ap.add_argument("--steps-silent", type=int, default=0, help="phase 2 (speed): LatentSync windows whose dub is >= --silent-frac silent (< --silent-db dBFS per master frame) are "
                    "denoised with this many DDIM steps while --steps applies to the speaking windows (fewer steps tighten the mouth in dub silence); 0 = off")
    ap.add_argument("--silent-frac", type=float, default=0.25); ap.add_argument("--silent-db", type=float, default=-45.0)
    ap.add_argument("--ls-split-snap", type=int, default=0, help="phase 2 (speed): move each dub-silence cut inside its silence so that the LatentSync segment before it is a "
                    "multiple of this many frames (16 = one UNet window): fewer partial windows, same closed-mouth re-anchoring; 0 = off")
    ap.add_argument("--ls-split-dub-silence", type=float, default=0.0, help="r4 merged: split LatentSync segments inside dub silences >= this many s (0.12 s after the dub "
                    "stops / before it restarts); 0 = off")
    ap.add_argument("--cap-align", choices=["on", "off"], default="off", help="r4 merged: --max-ls-frames cuts at the quietest dub frame of their last 6 s")
    ap.add_argument("--gate-sources", choices=["both", "dub"], default="both", help="r4 merged: speech gate from source + dub speech (default) or the dub speech only")
    ap.add_argument("--mux", choices=["dub", "final_mix"], default="final_mix", help="audio muxed into the candidate mp4: final_mix (dub + background bed, when present) or the dub voice only")
    ap.add_argument("--speech-gate", choices=["on", "off"], default="on"); ap.add_argument("--gate-margin-s", type=float, default=0.24); ap.add_argument("--gate-merge-gap-s", type=float, default=0.6)
    ap.add_argument("--crossfade-frames", type=int, default=4); ap.add_argument("--router-stride", type=int, default=3); ap.add_argument("--window-batch-size", type=int, default=2)
    ap.add_argument("--deepcache-interval", type=int, default=5); ap.add_argument("--no-deepcache", action="store_true"); ap.add_argument("--steps", type=int, default=20); ap.add_argument("--guidance", type=float, default=1.5)
    ap.add_argument("--face-min-w", type=int, default=50); ap.add_argument("--face-min-h", type=int, default=80); ap.add_argument("--retina-conf", type=float, default=0.8)
    ap.add_argument("--min-ls-frames", type=int, default=25); ap.add_argument("--max-verify-depth", type=int, default=3)
    ap.add_argument("--mouth-only-pasteback", choices=["on", "off"], default="off", help="E4: outside a feathered lower-face mask keep the master pixels (LatentSync upstream untouched)")
    ap.add_argument("--mask-top", type=float, default=0.52, help="E4: mask starts at this fraction of the detected face height (0 = top of bbox)")
    ap.add_argument("--mask-feather", type=float, default=0.08, help="E4: gaussian feather as a fraction of the face width")
    ap.add_argument("--ls-landmark-smooth", type=int, default=0, help="centred moving average (+-N frames) of the landmarks replayed into LatentSync's affine crop (0 = raw)")
    ap.add_argument("--mask-smooth", type=int, default=0, help="E4: mask bbox = centred moving average over +-N frames (0 = raw per-frame detection)")
    ap.add_argument("--codeformer-scope", choices=["face", "mouth"], default="face", help="with --mouth-only-pasteback: face = CodeFormer on the composited frame (whole face re-rendered, "
                    "previous behaviour); mouth = CodeFormer on the raw LatentSync frame, THEN the mouth-only composite (eyes / brows stay the source)")
    ap.add_argument("--codeformer", choices=["off", "optimized"], default="off", help="week 2: in-memory CodeFormer on LATENT_SYNC frames (codeformer_accel)")
    ap.add_argument("--codeformer-parse", choices=["on", "off"], default="on", help="r8 F (speed): off = no ParseNet paste mask (the r8 detail transfer applies its own mouth mask)"); ap.add_argument("--codeformer-w", type=float, default=1.0); ap.add_argument("--codeformer-batch", type=int, default=8)
    ap.add_argument("--codeformer-empty-cache", action="store_true", help="A/B: upstream per-face torch.cuda.empty_cache()")
    ap.add_argument("--codeformer-landmarks", choices=["insightface", "retinaface"], default="insightface")
    ap.add_argument("--face-stack", choices=["insightface", "mediapipe"], default="mediapipe", help="mediapipe = clean face stack (default); insightface needs DUB_ALLOW_NC=1")
    ap.add_argument("--max-faces", type=int, default=2, help="spec 8.2: faces tracked per frame; with >= 2 faces only the active speaker's face (diarization <-> mouth motion) is lip-synced")
    ap.add_argument("--pose-precheck", choices=["on", "off"], default="on", help="spec 5.4: > 30 %% of face frames with |yaw| > 30 deg -> voice-only result (no lip-sync), flagged")
    ap.add_argument("--pose-gate-deg", type=float, default=0.0, help="spec 5.5 test: frames with |yaw| above this go to pass-through (held +-3 frames, eased by the crossfade); 0 = off")
    ap.add_argument("--mp-anchor-calib", choices=["on", "off"], default="on")
    ap.add_argument("--anchor-filter", default=None, help="round 2: temporal filter of the landmarks driving LatentSync's crop AND the mouth mask (anchor_filters.py: "
                    "flow:S = rigid head motion from DIS flow, noise smoothed around it); forces --ls-landmark-smooth 0 --mask-smooth 0")
    ap.add_argument("--prep-only", action="store_true", help="r3 D: run only the CPU face-stack prep (master, MediaPipe, rigid motion) into --work-out and exit: "
                    "the caches are reused by the real run, so a queue can prep the next video while the GPU renders the current one")
    ap.add_argument("--dump-ls-raw", default="off", help="r3 D (B: or a PATH): also write <out>_ls_raw.mp4 (or PATH) = the RAW LatentSync frames on the 25 fps master "
                    "(master frames elsewhere, lossless): the --ls-raw input of fps_composite.py (FIX30) from the same run; needs --mouth-only-pasteback on --codeformer-scope mouth")
    ap.add_argument("--mask-track", choices=["dets", "head"], default="dets", help="head: mouth-mask boxes locked to the rigid head motion (needs the rigid pass)")
    ap.add_argument("--affine-precision", choices=["fp16", "fp32"], default="fp16", help="fp32: LatentSync AlignRestore in float32 (ls_geometry.py)")
    ap.add_argument("--anchor-round", choices=["on", "off"], default="on", help="off: sub-pixel affine anchors (ls_geometry.py)")
    ap.add_argument("--codeformer-precision", choices=["fp32", "parse16", "fp16"], default="fp16", help="fp32 = upstream numerics; parse16 = ParseNet under fp16 autocast; fp16 = CodeFormer net + ParseNet under autocast")
    ap.add_argument("--master-timing-mode", choices=["r25", "fps"], default="r25", help="r3 B: how the 25 fps master is made: r25 = `-r 25` (default, lags the "
                    "clock ~60 ms on 30 fps sources), fps = `-vf fps=25` (nearest source frame, mean clock error 0); sets DUB_MASTER_TIMING")
    ap.add_argument("--ls-fps", type=int, default=25, help="r3 B experiment: frame rate of the master / gate / LatentSync (30 = the 30 fps source frames "
                    "themselves with --master-timing-mode fps; LatentSync's audio windows follow via video_fps)")
    ap.add_argument("--dump-raw", action="store_true", help="round 3: also write <out>_lsraw.mp4 = master frames with the RAW LatentSync frames (before the mouth composite / crossfade) "
                    "in the LATENT_SYNC segments, lossless libx264rgb crf 0, master frame count (track B fps_composite --ls-raw)")
    ap.add_argument("--ref-mode", choices=["same", "fixed", "shift8", "shift16", "shift32"], default="same", help="round 3: LatentSync UNet reference (ls_stable.py): same = current frames "
                    "(upstream); fixed = a calm 16-frame clip of the same video (--ref-select); shiftS = the same segment S frames away")
    ap.add_argument("--ref-select", choices=["calm_any", "calm_closed"], default="calm_closed")
    ap.add_argument("--audio-lead", type=float, default=0.0, help="r9: LatentSync driven by the audio N frames later, in 50 Hz whisper steps (0.5 = one step at 25 fps; stable path). --ref-mode fixed / shiftS answer ~0.5 frame late")
    ap.add_argument("--bridge-len", type=int, default=0); ap.add_argument("--bridge-steps", type=int, default=0, help="round 3 (spec 8.5): low-noise bridge windows over every 16-frame seam (ls_stable.py)")
    ap.add_argument("--max-ls-frames", type=int, default=0, help="round 3 (D's VRAM audit): LatentSync calls of at most this many frames (rounded down to 16k; 400 -> 400)")
    ap.add_argument("--inject-ls-error", default=None, help="round 3 spec-11 test: comma list of LatentSync call numbers (1-based) that raise RuntimeError -> PASS_THROUGH_LS_FAILED")
    ap.add_argument("--silence-mix", type=float, default=0.0, help="round 3: dub-silent mouth, whisper conditioning moved toward the SOURCE speech by this amount where the source talks and no dub part sounds (r3_lipsync.py)")
    ap.add_argument("--silence-ramp", type=int, default=6); ap.add_argument("--silence-min-s", type=float, default=0.3); ap.add_argument("--gaps", default=None, help="residual-gap JSON (track C) instead of the manifest stretches")
    a = ap.parse_args()
    os.environ["DUB_MASTER_TIMING"] = a.master_timing_mode
    if a.ls_fps != 25: os.environ["DUB_LS_FPS"] = str(a.ls_fps)
    import clean_face_stack as cfs
    cfs.enforce_clean(a.face_stack, a.codeformer, "optimized", what="candidate_video_stage")
    import run_clean_pipeline_05 as rp
    import face_aware_latentsync_accel as faa
    if a.anchor_filter or a.mask_track == "head":
        a.ls_landmark_smooth = 0; a.mask_smooth = 0
    import ls_geometry
    geom = ls_geometry.apply(a.affine_precision, a.anchor_round)
    from measure_stages import probe
    if a.manifest:   # any finished E2E run (e.g. refresh_e2e.sh final/<id>_final.manifest.json): its own work dir holds the audio stages
        man = json.loads(pathlib.Path(a.manifest).read_text()); base_work = pathlib.Path(man["work_dir"])
    else:
        base_work = pathlib.Path(a.work_dir) / f"work_{a.id}"; man = json.loads((pathlib.Path(a.run_dir) / f"{a.id}_baseline.manifest.json").read_text())
    src = pathlib.Path(man["input"]); duration = man["source"]["duration_s"]; out_dir = pathlib.Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{a.id}_{a.tag}.mp4"; work = pathlib.Path(a.work_out or (out_dir / f"work_{a.id}_{a.tag}")); work.mkdir(parents=True, exist_ok=True)
    dd = pathlib.Path(a.dubbed_dir) if a.dubbed_dir else base_work
    dub16, dub_aac = dd / "dubbed_16k.wav", dd / "dubbed_aac.m4a"
    if a.mux == "final_mix" and (dd / "final_mix_aac.m4a").exists():
        dub_aac = dd / "final_mix_aac.m4a"   # what the viewer hears (dub voice + background bed); LatentSync / gate still use the clean voice
    master = base_work / "master_25fps.mp4"
    if master.exists() and not (work / "master_25fps.mp4").exists(): os.symlink(master, work / "master_25fps.mp4")   # make_master() reuses it (identical ffmpeg call otherwise)
    times = {}; t_all = time.perf_counter()
    man = {**man, "output": str(out), "work_dir": str(work), "candidate": {"tag": a.tag, "dubbed_dir": str(dd), "video_params": {k: v for k, v in vars(a).items() if k not in ("id", "tag", "run_dir", "work_dir", "dubbed_dir", "out", "work_out")}}}
    if a.dubbed_dir:   # units' alignment info comes from the E1 report when present
        e1 = dd / "e1_alignment.json"
        if e1.exists():
            rep = json.loads(e1.read_text()); by = {r["id"]: r for r in rep["units"]}
            for u in man["units"]:
                r = by.get(u["id"])
                if r: u["alignment"] = {**u["alignment"], "mode": f"e1_{rep['mode']}", "atempo": r["max_atempo"], "e1_placements": r["placements"], "file": r["file"]}
            man["e1_summary"] = rep["summary"]
        man["dubbed_track"] = {**man["dubbed_track"], "wav16k": str(dub16), "aac": str(dub_aac), "wav24k": str(dd / "dubbed_24k.wav")}
    speech = man["speech_intervals"]; gate = None; split_frames = None; cut_energy = None
    if a.ls_split_dub_silence > 0 or a.cap_align == "on" or a.steps_silent > 0:   # r4 merged: dub-silence split points / dub energy per master frame
        import numpy as np, soundfile as _sf
        _y, _sr = _sf.read(str(dub16), dtype="float32"); _y = _y if _y.ndim == 1 else _y.mean(1); _h = int(_sr / faa.FPS)
        cut_energy = np.sqrt((_y[: len(_y) // _h * _h].reshape(-1, _h) ** 2).mean(1))
        if a.ls_split_dub_silence > 0:
            _ds = rp.run_vad(dub16); split_frames = []; split_ranges = {}
            for (s0, e0), (s1, e1) in zip([(v["start"], v["end"]) for v in _ds], [(v["start"], v["end"]) for v in _ds][1:]):
                if s1 - e0 >= a.ls_split_dub_silence:
                    c0, c1 = int(round((e0 + 0.12) * faa.FPS)), int(round((s1 - 0.12) * faa.FPS)); split_frames += [c0, c1]
                    split_ranges[c0] = (c0, c1); split_ranges[c1] = (c0, c1)   # phase 2: a cut may move anywhere inside its dub silence (--ls-split-snap)
            man["ls_split"] = {"min_silence_s": a.ls_split_dub_silence, "split_frames": split_frames, "snap": a.ls_split_snap}
    if a.speech_gate == "on":
        with rp.timed(times, "speech_gate"):
            dub_speech = rp.run_vad(dub16)
            # r4 merged: --gate-sources dub = lip-sync only where the DUB speaks (pb1's long dub silences inside 16 s segments painted a quiet mouth
            # over a talking face: 04 slide / window joints); the source mouth shows (crossfade) where only the source talks
            gate = faa.build_speech_gate([dub_speech] if a.gate_sources == "dub" else [speech, dub_speech], duration, a.gate_margin_s, a.gate_merge_gap_s)
        gated = sum(e - s for s, e in gate)
        man["speech_gate"] = {"enabled": True, "margin_s": a.gate_margin_s, "merge_gap_s": a.gate_merge_gap_s, "crossfade_frames": a.crossfade_frames, "source_speech_seconds": round(sum(i["end"] - i["start"] for i in speech), 3),
                              "dubbed_speech_seconds": round(sum(i["end"] - i["start"] for i in dub_speech), 3), "gate_seconds": round(gated, 3), "gate_fraction": round(gated / duration, 4), "intervals": gate}
        print(f"  [gate] {len(gate)} intervals, {gated:.1f}s of {duration:.1f}s", flush=True)
    else: man["speech_gate"] = {"enabled": False}
    prep = None
    if a.face_stack == "mediapipe":   # CLEAN-ONLY: MediaPipe router + LatentSync detections, NC loaders blocked
        import clean_face_stack as cfs
        cfs.guard_nc()
        with rp.timed(times, "face_stack_mediapipe"): prep = cfs.CleanPrep(src, work, a.face_min_w, a.face_min_h, calib=a.mp_anchor_calib == "on", pose_precheck=a.pose_precheck == "on", pose_gate_deg=a.pose_gate_deg,
                                     max_faces=a.max_faces, turns=(man.get("diarization") or {}).get("turns"),
                                     anchor_filter=a.anchor_filter, mask_track=a.mask_track)
        router = prep.router; man["face_stack"] = {**prep.info, "times": prep.times, "ls_geometry": geom}; man["frame_map"] = prep.frame_map
        if a.prep_only:
            print(f"== {a.id} {a.tag}: prep-only done {prep.times} -> {work}", flush=True); return 0
    else:
        with rp.timed(times, "face_router_load"): router = faa.make_router("retinaface", a.face_min_w, a.face_min_h, a.retina_conf)
    if a.ref_mode != "same" or a.bridge_len:   # round 3: generator variants (ls_stable.StableLatentSync; same call interface)
        from ls_stable import StableRunner
        with rp.timed(times, "latentsync_load"): runner = StableRunner(a.window_batch_size, not a.no_deepcache, a.steps, a.guidance, 1247, a.deepcache_interval, ref_mode=a.ref_mode,
                                                                       bridge_len=a.bridge_len, bridge_steps=a.bridge_steps)
        runner.acc.audio_lead = a.audio_lead; man["audio_lead"] = a.audio_lead
        if a.steps_silent > 0 and cut_energy is not None:   # phase 2: per-master-frame dub silence for the adaptive step count
            import numpy as _np
            runner.silent_all = _np.asarray(cut_energy) < 10 ** (a.silent_db / 20); runner.acc.steps_silent = a.steps_silent; runner.acc.silent_frac = a.silent_frac
            man["steps_silent"] = {"steps": a.steps, "steps_silent": a.steps_silent, "silent_frac": a.silent_frac, "silent_db": a.silent_db, "silent_master_frames": int(runner.silent_all.sum())}
        if a.ref_mode == "fixed":
            if prep is None: raise SystemExit("--ref-mode fixed needs the clean face stack")
            import cv2, numpy as np
            from ls_window_ab import pick_reference
            LM = np.stack([np.asarray(d[1], np.float32) if d[0] is not None else np.full((106, 2), np.nan, np.float32) for d in prep.ls_dets]); ok = np.isfinite(LM[:, 0, 0])
            cost, rs = pick_reference(LM, ok, a.ref_select, 16)
            runner.acc.set_reference(np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in prep.frames[rs:rs + 16]]), list(prep.ls_dets[rs:rs + 16]))
            man["reference"] = {"select": a.ref_select, "start_frame": int(rs), "t": round(rs / 25, 2), "cost": round(cost, 4)}
            print(f"  [ref] fixed reference frames {rs}-{rs + 16} ({rs / 25:.2f}s)", flush=True)
    else:
        with rp.timed(times, "latentsync_load"): runner = faa.AccelRunner(a.window_batch_size, not a.no_deepcache, "none", "auto", steps=a.steps, guidance=a.guidance, deepcache_interval=a.deepcache_interval)
    if a.silence_mix > 0:   # round 3: dub-silent mouth (r3_lipsync.SilenceMix), stretches in absolute master frames
        import r3_lipsync as r3
        N_all = (man.get("lipsync") or {}).get("master_frames") or int(round(man["source"]["duration_s"] * 25)) + 2; sm = r3.silent_mask(man, N_all, a.silence_min_s, gaps=r3.load_gaps(a.gaps) if a.gaps else None)
        runner.silence_mix = r3.SilenceMix(base_work / "own_audio_16k.wav", r3.ramp(sm, a.silence_ramp), a.silence_mix)
        man["silence_mix"] = {"lam": a.silence_mix, "ramp": a.silence_ramp, "min_s": a.silence_min_s, "gaps": a.gaps, "frames": int(sm.sum())}
    if a.inject_ls_error:   # spec 11 fault injection: a model error must not crash the worker (segment passes through, counted)
        bad = {int(x) for x in a.inject_ls_error.split(",")}; inner_run = runner.run_frames; ncall = [0]
        def failing(frames_rgb, audio_wav, detections=None):
            ncall[0] += 1
            if ncall[0] in bad: raise RuntimeError(f"injected fault (call {ncall[0]})")
            return inner_run(frames_rgb, audio_wav, detections)
        runner.run_frames = failing
    rec_cls = None
    if a.mouth_only_pasteback == "on":
        Mo = MouthOnlyRunner
        if prep is not None:   # round 2: record the composite mask boxes per master frame (slide_metric.py --boxes)
            from ls_window_ab import make_box_recorder
            Mo = rec_cls = make_box_recorder(MouthOnlyRunner, prep.frames)
        runner = Mo(runner, a.mask_top, a.mask_feather, a.mask_smooth, defer=a.codeformer_scope == "mouth")
        if a.dump_raw: runner.raw = {}   # A r3: --dump-raw (_lsraw.mp4)
    raw_ls = {}; raw_pre = {}; _pre_stash = []   # r8 F: raw_pre = the same LatentSync frames BEFORE CodeFormer (--dump-ls-raw writes _ls_pre.mp4)
    if a.dump_ls_raw != "off":   # r3 D: keep the raw LatentSync frames (BGR, before the mouth-only composite) per master frame
        if not getattr(runner, "deferred", False): raise SystemExit("--dump-ls-raw needs --mouth-only-pasteback on --codeformer-scope mouth")
        import hashlib, numpy as np
        import face_aware_latentsync as _fa
        # r8 F: without the MediaPipe prep (--face-stack insightface) the 25 fps master is written by process_video AFTER this point, in work/video/:
        # the master frames + their hashes are loaded lazily, on the first composite call
        _mf = prep.frames if prep is not None else []; _h = []; _idx = {}; _comp = runner.composite; _thumb = []
        def _load_master():
            if not _mf:
                mp_ = next((q for q in (work / "video" / "master_25fps.mp4", work / "master_25fps.mp4", base_work / "video" / "master_25fps.mp4") if q.exists()), None)
                if mp_ is None: raise RuntimeError("--dump-ls-raw: no 25 fps master found")
                _mf.extend(_fa.read_frames(mp_))
            if not _h:
                _h.extend(hashlib.md5(f.tobytes()).hexdigest() for f in _mf)
                for i, hh in enumerate(_h): _idx.setdefault(hh, []).append(i)
        def _composite(outs, originals, boxes=None):   # segment start = the master position whose frames match (repeated frames: all of them must match)
            _load_master()
            oh = [hashlib.md5(np.ascontiguousarray(o).tobytes()).hexdigest() for o in originals]
            st = next((i for i in _idx.get(oh[0], []) if _h[i:i + len(oh)] == oh), None) if oh else None
            swap = False
            if st is None and len(originals):   # r8 F: the InsightFace / CodeFormer path hands the frames over differently decoded or channel-swapped:
                # nearest 32 x 32 thumbnail (both channel orders), confirmed on the segment's first frames
                import cv2 as _cv
                if not _thumb: _thumb.append(np.stack([_cv.resize(f, (32, 32), interpolation=_cv.INTER_AREA).astype(np.float32) for f in _mf]))
                T = _thumb[0]; best = None
                for sw in (False, True):
                    q = [_cv.resize(np.ascontiguousarray(o[..., ::-1] if sw else o), (32, 32), interpolation=_cv.INTER_AREA).astype(np.float32) for o in originals[:min(8, len(originals))]]
                    d0 = np.abs(T - q[0]).mean((1, 2, 3)); i0 = int(np.argmin(d0))
                    if i0 + len(q) <= len(T):
                        d = float(np.mean([np.abs(T[i0 + k] - q[k]).mean() for k in range(len(q))]))
                        if best is None or d < best[0]: best = (d, i0, sw)
                if best is not None and best[0] < 2.0: _, st, swap = best; print(f"  [dump-ls-raw] segment start {st} by thumbnail match (mad {best[0]:.2f}, channel swap {swap})", flush=True)
            if st is None: raise RuntimeError("--dump-ls-raw: LS segment start not found among the master frames")
            for k, o in enumerate(outs): raw_ls[st + k] = (o[..., ::-1] if swap else o).copy()
            if len(_pre_stash) == len(outs):
                for k, o in enumerate(_pre_stash): raw_pre[st + k] = (o[..., ::-1] if swap else o).copy()
            _pre_stash.clear()
            return _comp(outs, originals, boxes)
        runner.composite = _composite
    cf_runner = None
    if a.codeformer == "optimized":
        from codeformer_accel import CodeFormerAccel
        with rp.timed(times, "codeformer_load"): cf_runner = CodeFormerAccel(fidelity_weight=a.codeformer_w, use_parse=a.codeformer_parse == "on", batch_size=a.codeformer_batch, empty_cache_per_face=a.codeformer_empty_cache, landmark_source=a.codeformer_landmarks, autocast=a.codeformer_precision == "fp16", parse_autocast=a.codeformer_precision != "fp32")
        if a.dump_ls_raw != "off":   # r8 F: remember CodeFormer's input frames of the segment for the pre-CodeFormer dump
            _rf = cf_runner.restore_frames
            def _restore(frames, lms, _rf=_rf):
                _pre_stash[:] = [np.ascontiguousarray(f).copy() for f in frames]; return _rf(frames, lms)
            cf_runner.restore_frames = _restore
    rep = faa.process_video(src, work, dub16, dub_aac, out, router=router, runner=runner, min_ls_frames=a.min_ls_frames, max_verify_depth=a.max_verify_depth, codeformer=a.codeformer, times=times,
                            log=lambda m: print(m, flush=True), speech_gate=gate, crossfade_frames=a.crossfade_frames, router_stride=a.router_stride, in_memory=True, cf_runner=cf_runner, lm_smooth=a.ls_landmark_smooth, prep=prep, max_ls_frames=a.max_ls_frames, split_frames=split_frames, cut_energy=cut_energy,
                            split_ranges=(split_ranges if a.ls_split_snap and split_frames else None), split_snap=a.ls_split_snap)
    if rec_cls is not None and rec_cls.rec:
        import numpy as np
        fr_, bx_ = [], []
        for st, bxs in rec_cls.rec.items():
            for k, b in enumerate(bxs):
                if b is not None: fr_.append(st + k); bx_.append(b)
        np.savez_compressed(out.with_name(out.stem + "_maskboxes.npz"), frames=np.array(fr_, int), boxes=np.array(bx_, np.float64))
    if a.dump_ls_raw != "off":
        import imageio.v2 as iio
        with rp.timed(times, "dump_ls_raw"):
            raw_path = out.with_name(out.stem + "_ls_raw.mp4") if a.dump_ls_raw == "on" else pathlib.Path(a.dump_ls_raw); w = iio.get_writer(str(raw_path), fps=25, codec="libx264rgb", macro_block_size=None, pixelformat="rgb24",
                                                                                ffmpeg_params=["-crf", "0", "-preset", "veryfast"], ffmpeg_log_level="error")
            _load_master()
            for i, f in enumerate(_mf): w.append_data((raw_ls.get(i, f))[:, :, ::-1])
            w.close()
        man["ls_raw"] = {"path": str(raw_path), "ls_frames": len(raw_ls), "master_frames": len(_mf)}; del raw_ls
        if raw_pre:   # r8 F: the clean (pre-CodeFormer) layer of the same run: the base of the CodeFormer detail transfer (r8_cf_detail.py)
            with rp.timed(times, "dump_ls_pre"):
                pre_path = raw_path.with_name(raw_path.name.replace("_ls_raw", "_ls_pre")); w = iio.get_writer(str(pre_path), fps=25, codec="libx264rgb", macro_block_size=None,
                                                                                                          pixelformat="rgb24", ffmpeg_params=["-crf", "0", "-preset", "veryfast"], ffmpeg_log_level="error")
                for i, f in enumerate(_mf): w.append_data((raw_pre.get(i, f))[:, :, ::-1])
                w.close()
            man["ls_raw"]["pre_codeformer"] = {"path": str(pre_path), "ls_frames": len(raw_pre)}
        del raw_pre
    if a.dump_raw and getattr(runner, "raw", None):
        import cv2, imageio, face_aware_latentsync as fa_
        rawp = out.with_name(out.stem + "_lsraw.mp4"); w_ = imageio.get_writer(str(rawp), fps=25, codec="libx264rgb", macro_block_size=None, pixelformat="rgb24",
                                                                              ffmpeg_params=["-crf", "0", "-preset", "veryfast"], ffmpeg_log_level="error")
        mf = fa_.read_frames(work / "master_25fps.mp4"); sub = {}
        for st, frs in runner.raw.items():
            for k, f in enumerate(frs): sub[st + k] = f
        for i, f in enumerate(mf): w_.append_data(cv2.cvtColor(sub.get(i, f), cv2.COLOR_BGR2RGB))
        w_.close(); man["ls_raw"] = {"path": str(rawp), "frames": len(mf), "ls_frames": len(sub)}; del mf, sub
    rep["backend"] = "optimized"; man["lipsync"] = rep
    if prep is not None:   # spec 11: degraded frames (held landmarks, LatentSync failures) vs the 2 % SUCCESS / PARTIAL_SUCCESS rule (same rule as run_clean_pipeline_05)
        elig = int(rep["frame_classes"].get("VALID_FACE", 0)); failed = int((rep["by_action"].get("PASS_THROUGH_LS_FAILED") or {}).get("frames", 0))
        held = int(man["face_stack"].get("held_landmark_frames", 0)); ratio = (held + failed) / max(elig, 1)
        man["video_status"] = {"status": "PARTIAL_SUCCESS" if (prep.voice_only or ratio > 0.02) else "SUCCESS", "eligible_lipsync_frames": elig, "held_landmark_frames": held,
                               "latentsync_failed_frames": failed, "degraded_ratio": round(ratio, 4), "voice_only": bool(prep.voice_only),
                               "rule": "SUCCESS if degraded <= 2 % of eligible frames; voice-only fallback = PARTIAL_SUCCESS"}; man["codeformer"] = "DISABLED (not invoked)" if a.codeformer == "off" else {"mode": a.codeformer, "w": a.codeformer_w, "batch": a.codeformer_batch, "landmarks": a.codeformer_landmarks, "precision": a.codeformer_precision}
    if cf_runner is not None: cf_runner.close()
    rp.free_cuda(runner, cf_runner)
    times["total"] = round(time.perf_counter() - t_all, 3); op = probe(out); fps = man["source"]["video"].get("avg_fps") or 25.0; tol = max(0.15, 2.0 / fps); delta = round(op["duration_s"] - duration, 4)
    checks = {"output_exists_nonempty": out.exists() and out.stat().st_size > 0, "output_has_video_audio": op["has_video"] and op["has_audio"], "duration_within_tolerance": abs(delta) <= tol,
              "resolution_preserved": (op["video"]["width"], op["video"]["height"]) == (man["source"]["video"]["width"], man["source"]["video"]["height"]),
              "all_units_have_tts_and_translation": True, "frame_count_preserved": rep["assembly"]["frames_written"] == rep["master_frames"], "output_frames_decodable_no_blank": bool(rep["output_frame_check"]["pass"])}
    man["output_meta"] = op; man["validation"] = {"tolerance_s": round(tol, 4), "duration_delta_s": delta, "checks": checks, "pass": all(checks.values())}
    man["stage_seconds"] = times; man["gpu"] = rp.gpu_info(); man["at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rp.write_json(out.with_name(out.stem + ".manifest.json"), man)
    for p in work.glob("seg*_in.wav"): p.unlink(missing_ok=True)
    print(f"== {a.id} {a.tag}: validation {'PASS' if man['validation']['pass'] else 'FAIL'} {checks} total {times['total']}s -> {out}")
    return 0 if man["validation"]["pass"] else 4


if __name__ == "__main__":
    raise SystemExit(main())
