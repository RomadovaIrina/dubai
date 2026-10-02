#!/usr/bin/env python3
"""Pilot 0.7 refresh (2026-10-02) — CURRENT optimized in-memory CodeFormer vs SyncNet: controlled A/B on ONE LatentSync pass.

A  = the current optimized LatentSync output (speech gate, RetinaFace router, batched windows, DeepCache) assembled with CodeFormer OFF.
B  = the EXACT same LatentSync frames + landmarks restored by scripts/optim/codeformer_accel.CodeFormerAccel at every weight of
     --weights (production settings otherwise: batch 8, fp16 autocast, insightface landmarks), assembled with the same crossfade and
     the same dubbed audio. LatentSync runs ONCE: `MultiWeightCodeFormer` stands in for CodeFormerAccel inside
     face_aware_latentsync_accel.process_video, records every restore_frames() call (raw frames + resolved landmarks), restores the
     frames with one real CodeFormerAccel per weight and hands the production weight back to the pipeline. Every variant is then
     assembled from the same master frames with faa.assemble_frames (A = raw LatentSync frames).
Audio stages are reused from the E2E run (<run-dir>/<id>_final.manifest.json, <work-dir>/work_<id>/dubbed_16k.wav + dubbed_aac.m4a),
exactly like scripts/quality/candidate_video_stage.py; the speech gate is rebuilt with the E2E parameters from the manifest.
SyncNet = upstream evaluator via syncnet_common.eval_syncnet (methodology unchanged).
Strict criterion (pilot 0.7, unchanged): confidence must not decrease, |AV offset| must not worsen.

    source scripts/env.sh
    python scripts/pilot/refresh_07_codeformer_sweep.py --id 04 --run-dir /tmp/pilot_refresh/final --work-dir /tmp/pilot_refresh/work \
        --out /tmp/pilot_refresh/r07 --weights 0.3 0.5 0.7 0.9 1.0
"""
from __future__ import annotations
import argparse, hashlib, json, os, pathlib, shutil, sys, time
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "optim")); sys.path.insert(0, str(HERE.parent / "quality"))
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
import numpy as np
from pilot_common import PILOT, ROOT, gpu_info, git_info, write_json


class MultiWeightCodeFormer:
    """Same call surface as codeformer_accel.CodeFormerAccel (restore_frames / landmark_source / fallback_detector / config / load_s /
    totals / close). Landmarks are resolved ONCE per call (fallback detector included) so every weight sees identical input."""

    def __init__(self, weights, prod_w, batch, precision, landmark_source):
        import torch
        from codeformer_accel import CodeFormerAccel
        self.torch = torch; self.weights = list(weights); self.prod_w = prod_w; self.landmark_source = landmark_source
        self.runners = {w: CodeFormerAccel(fidelity_weight=w, batch_size=batch, landmark_source=landmark_source,
                                           autocast=precision == "fp16", parse_autocast=precision != "fp32") for w in self.weights}
        self.fallback_detector = None
        self.load_s = round(sum(r.load_s for r in self.runners.values()), 3)
        self.config = {**self.runners[prod_w].config, "sweep_weights": self.weights, "production_weight": prod_w}
        self.totals = self.runners[prod_w].totals
        self.calls: list[dict] = []

    def restore_frames(self, frames_bgr, landmarks5, inplace: bool = False):
        n = len(frames_bgr); raw = [np.ascontiguousarray(f).copy() for f in frames_bgr]; lms = []; fallback = 0
        for i in range(n):
            lm = landmarks5[i] if i < len(landmarks5) else None
            if lm is None and self.fallback_detector is not None:
                lm = self.fallback_detector(raw[i]); fallback += 1
            lms.append(None if lm is None else np.asarray(lm, dtype=np.float32).copy())
        call = {"call_id": len(self.calls), "frames_in": n, "fallback_detections": fallback, "landmarks_available": sum(l is not None for l in lms),
                "raw": raw, "lms": lms, "out": {}, "stats": {}, "wall_s": {}}
        for w in self.weights:
            r = self.runners[w]; r.fallback_detector = None
            self.torch.cuda.synchronize(); t0 = time.perf_counter()
            out, st = r.restore_frames(raw, lms)
            self.torch.cuda.synchronize(); call["wall_s"][w] = round(time.perf_counter() - t0, 3)
            call["out"][w] = out; call["stats"][w] = st
        self.calls.append(call)
        st = dict(call["stats"][self.prod_w]); st["call_id"] = call["call_id"]; st["fallback_detections"] = fallback
        return call["out"][self.prod_w], st

    def close(self) -> None:
        for r in self.runners.values():
            r.close()


def md5(p: pathlib.Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", default="04"); ap.add_argument("--run-dir", default="/tmp/pilot_refresh/final"); ap.add_argument("--work-dir", default="/tmp/pilot_refresh/work")
    ap.add_argument("--out", default="/tmp/pilot_refresh/r07")
    ap.add_argument("--weights", nargs="+", type=float, default=[0.3, 0.5, 0.7, 0.9, 1.0])
    ap.add_argument("--production-w", type=float, default=1.0, help="weight handed back to the pipeline (production default)")
    ap.add_argument("--codeformer-batch", type=int, default=8); ap.add_argument("--codeformer-precision", choices=["fp32", "parse16", "fp16"], default="fp16")
    ap.add_argument("--codeformer-landmarks", choices=["insightface", "retinaface"], default="insightface")
    ap.add_argument("--max-confidence-drop", type=float, default=0.0, help="strict pilot 0.7 policy: 0")
    ap.add_argument("--max-offset-worsening-frames", type=int, default=0)
    ap.add_argument("--skip-syncnet", action="store_true")
    ap.add_argument("--json", default=str(PILOT / "refresh_0.7_current_codeformer.json")); ap.add_argument("--md", default=str(PILOT / "refresh_0.7_current_codeformer.md"))
    a = ap.parse_args()
    if a.production_w not in a.weights:
        a.weights.append(a.production_w)
    a.weights = sorted(set(a.weights))
    import torch
    import run_clean_pipeline_05 as rp
    import face_aware_latentsync_accel as faa
    import face_aware_latentsync as fa
    from measure_stages import probe
    from syncnet_common import eval_syncnet

    man_path = pathlib.Path(a.run_dir) / f"{a.id}_final.manifest.json"; man = json.loads(man_path.read_text())
    src = pathlib.Path(man["input"]); duration = float(man["source"]["duration_s"])
    base_work = pathlib.Path(a.work_dir) / f"work_{a.id}"; out_dir = pathlib.Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / f"work_{a.id}"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    master_src = base_work / "master_25fps.mp4"
    if master_src.exists():
        os.symlink(master_src, work / "master_25fps.mp4")        # make_master() reuses it (identical ffmpeg call otherwise)
    dub16, dub_aac = base_work / "dubbed_16k.wav", base_work / "dubbed_aac.m4a"
    for p in (dub16, dub_aac):
        if not p.exists():
            raise SystemExit(f"missing dubbed track of the E2E run: {p}")
    sg = man.get("speech_gate") or {}; ls_man = man["lipsync"]; runner_cfg = ls_man.get("runner") or {}; rpar = ls_man.get("router_params") or {}
    gate_margin, gate_merge, crossfade = sg.get("margin_s", 0.24), sg.get("merge_gap_s", 0.6), sg.get("crossfade_frames", 4)
    stride = int(ls_man.get("router_stride", 3))
    times: dict = {}; t_all = time.perf_counter()
    with rp.timed(times, "speech_gate"):
        dub_speech = rp.run_vad(dub16); gate = faa.build_speech_gate([man["speech_intervals"], dub_speech], duration, gate_margin, gate_merge)
    print(f"  [gate] {len(gate)} intervals, {sum(e - s for s, e in gate):.1f}s of {duration:.1f}s (E2E manifest gate: {sg.get('gate_seconds')}s)", flush=True)
    with rp.timed(times, "face_router_load"):
        router = faa.make_router("retinaface", int(rpar.get("face_min_w", 50)), int(rpar.get("face_min_h", 80)), float(rpar.get("retina_conf", 0.8)))
    with rp.timed(times, "latentsync_load"):
        runner = faa.AccelRunner(int(runner_cfg.get("window_batch_size", 2)), bool(runner_cfg.get("deepcache", True)), runner_cfg.get("compile_backend", "none"),
                                 runner_cfg.get("sdpa_backend", "auto"), steps=int(runner_cfg.get("steps", 20)), guidance=float(runner_cfg.get("guidance", 1.5)),
                                 seed=int(runner_cfg.get("seed", 1247)), deepcache_interval=int(runner_cfg.get("deepcache_interval", 5)))
    with rp.timed(times, "codeformer_load"):
        cf = MultiWeightCodeFormer(a.weights, a.production_w, a.codeformer_batch, a.codeformer_precision, a.codeformer_landmarks)
    torch.cuda.reset_peak_memory_stats()
    prod_pipeline_out = out_dir / f"{a.id}_cf_w{a.production_w:g}_pipeline.mp4"
    rep = faa.process_video(src, work, dub16, dub_aac, prod_pipeline_out, router=router, runner=runner, min_ls_frames=25, max_verify_depth=3,
                            codeformer="optimized", times=times, log=lambda m: print(m, flush=True), speech_gate=gate, crossfade_frames=crossfade,
                            router_stride=stride, in_memory=True, cf_runner=cf)
    torch_peak_mib = round(torch.cuda.max_memory_allocated() / 2 ** 20)
    rp.free_cuda(runner)
    segs = rep["segments"]; n = rep["master_frames"]
    seg_call = {s["index"]: s["codeformer"]["call_id"] for s in segs if isinstance(s.get("codeformer"), dict) and "call_id" in s["codeformer"]}
    ls_segs = [s for s in segs if s["action"] == "LATENT_SYNC"]
    missing = [s["index"] for s in ls_segs if s["index"] not in seg_call]
    if missing:
        raise SystemExit(f"LATENT_SYNC segments without a recorded CodeFormer call: {missing}")
    calls = {c["call_id"]: c for c in cf.calls}
    master_frames = fa.read_frames(work / "master_25fps.mp4")
    if len(master_frames) != n:
        raise SystemExit(f"master re-read {len(master_frames)} frames, pipeline saw {n}")

    def assemble(tag: str, repl: dict) -> tuple[pathlib.Path, dict, float]:
        o = out_dir / f"{a.id}_{tag}.mp4"; t0 = time.perf_counter()
        asm = faa.assemble_frames(master_frames, dub_aac, segs, repl, o, work)
        return o, asm, round(time.perf_counter() - t0, 3)

    with rp.timed(times, "assembly_variants"):
        off_out, off_asm, _ = assemble("cf_off", {i: calls[c]["raw"] for i, c in seg_call.items()})
        w_outs = {w: assemble(f"cf_w{w:g}", {i: calls[c]["out"][w] for i, c in seg_call.items()}) for w in a.weights}
    cf.close(); del master_frames
    prod_md5 = {"pipeline": md5(prod_pipeline_out), "reassembled": md5(w_outs[a.production_w][0])}

    def syncnet(p: pathlib.Path):
        if a.skip_syncnet:
            return None
        r = eval_syncnet(p); return {"confidence": r["confidence"], "av_offset_frames": r["av_offset_frames"], "seconds": r["seconds"]}

    with rp.timed(times, "syncnet_before"):
        before = syncnet(off_out)
    off_meta = probe(off_out); off_chk = faa.check_output_frames(off_out, n)
    print(f"  [syncnet] A (CodeFormer OFF): {before}", flush=True)
    rows = []
    for w in a.weights:
        o, asm, dt_asm = w_outs[w]; meta = probe(o); chk = faa.check_output_frames(o, n)
        t0 = time.perf_counter(); after = syncnet(o); t_sn = round(time.perf_counter() - t0, 3)
        cf_wall = round(sum(c["wall_s"][w] for c in cf.calls), 3); cf_stage = round(sum(c["stats"][w]["seconds"]["total"] for c in cf.calls), 3)
        faces = sum(c["stats"][w]["faces_restored"] for c in cf.calls); frames_in = sum(c["frames_in"] for c in cf.calls)
        peak = max([c["stats"][w].get("peak_vram_mib") or 0 for c in cf.calls] or [0])
        stage_sec = {}
        for c in cf.calls:
            for k, v in c["stats"][w]["seconds"].items():
                stage_sec[k] = round(stage_sec.get(k, 0.0) + v, 3)
        geom_same = (meta["video"]["width"], meta["video"]["height"]) == (off_meta["video"]["width"], off_meta["video"]["height"])
        frames_same = chk["decoded_frames"] == off_chk["decoded_frames"] == n
        dur_same = abs(meta["duration_s"] - off_meta["duration_s"]) <= 0.02
        row = {"w": w, "output": str(o), "codeformer_wall_s": cf_wall, "codeformer_stage_s": cf_stage, "codeformer_stage_breakdown_s": stage_sec,
               "faces_restored": faces, "frames_in": frames_in, "s_per_face": round(cf_wall / faces, 4) if faces else None,
               "codeformer_s_per_video_min": round(cf_wall / duration * 60, 3), "codeformer_torch_peak_mib": peak,
               "output_meta": {"width": meta["video"]["width"], "height": meta["video"]["height"], "frames": chk["decoded_frames"], "duration_s": meta["duration_s"]},
               "frame_check": chk, "validation": {"geometry_same_as_A": geom_same, "frame_count_same_as_A": frames_same, "duration_same_as_A": dur_same,
                                                  "frames_decodable_no_blank": bool(chk["pass"]), "pass": geom_same and frames_same and dur_same and bool(chk["pass"])},
               "assembly_s": dt_asm, "syncnet_seconds": t_sn}
        if after and before:
            delta = round(after["confidence"] - before["confidence"], 4); worse = abs(after["av_offset_frames"]) - abs(before["av_offset_frames"])
            row.update(confidence_before=before["confidence"], confidence_after=after["confidence"], confidence_delta=delta,
                       av_offset_before=before["av_offset_frames"], av_offset_after=after["av_offset_frames"], offset_abs_worsening_frames=worse,
                       strict_pass=bool(delta >= -a.max_confidence_drop and worse <= a.max_offset_worsening_frames))
        rows.append(row); print({k: v for k, v in row.items() if k not in ("frame_check", "codeformer_stage_breakdown_s", "output_meta")}, flush=True)
    times["total"] = round(time.perf_counter() - t_all, 3)
    passing = [r for r in rows if r.get("strict_pass")]
    selected = min(passing, key=lambda r: r["w"])["w"] if passing else None
    best = max((r for r in rows if "confidence_delta" in r), key=lambda r: r["confidence_delta"], default=None)
    old = None
    try:
        old = json.loads((PILOT / "0.7_codeformer_syncnet.json").read_text())
    except Exception:
        pass
    ls_calls = rep.get("runner_calls") or []
    out = {"task": "0.7-refresh", "video": str(src), "id": a.id, "source_duration_s": duration, "e2e_manifest": str(man_path), "e2e_manifest_at": man.get("at"),
           "git": git_info(ROOT), "gpu": gpu_info(), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "method": {"latentsync_passes": 1, "a": "same LatentSync frames, CodeFormer OFF (raw frames assembled)", "b": "same LatentSync frames restored per weight in memory (codeformer_accel.CodeFormerAccel)",
                      "audio": "dubbed track of the E2E run (reused, identical for every variant)", "speech_gate": {"margin_s": gate_margin, "merge_gap_s": gate_merge, "crossfade_frames": crossfade, "intervals": len(gate)},
                      "codeformer": {"batch": a.codeformer_batch, "precision": a.codeformer_precision, "landmarks": a.codeformer_landmarks, "production_w": a.production_w},
                      "latentsync": rep.get("runner"), "syncnet": "third_party/latentsync/eval/eval_sync_conf.py via syncnet_common.eval_syncnet (unchanged)"},
           "policy": {"max_confidence_drop": a.max_confidence_drop, "max_offset_worsening_frames": a.max_offset_worsening_frames, "selection": "lowest w that passes strict"},
           "latentsync": {"stage_seconds": times.get("latentsync"), "verify_seconds": times.get("latentsync_verify"), "inference_total_s": rep.get("latentsync_inference_total_s"),
                          "calls": len(ls_calls), "s_per_video_min": round(times["latentsync"] / duration * 60, 3) if times.get("latentsync") else None,
                          "frames_lipsynced": sum(s["frames"] for s in ls_segs), "frame_classes": rep.get("frame_classes"), "by_action": rep.get("by_action"),
                          "runner_calls": ls_calls, "load_s": times.get("latentsync_load")},
           "codeformer_calls": [{k: v for k, v in c.items() if k not in ("raw", "lms", "out", "stats")} | {"stats": c["stats"]} for c in cf.calls],
           "a_codeformer_off": {"output": str(off_out), "syncnet": before, "meta": {"width": off_meta["video"]["width"], "height": off_meta["video"]["height"], "frames": off_chk["decoded_frames"], "duration_s": off_meta["duration_s"]}, "frame_check": off_chk},
           "weights": rows, "selected_fidelity_weight": selected, "best_effort_w": best["w"] if best else None,
           "verdict": "PASS" if selected is not None else "FAIL — NO_STRICT_SAFE_WEIGHT",
           "production_output_md5": prod_md5, "production_reassembly_bit_identical": prod_md5["pipeline"] == prod_md5["reassembled"],
           "stage_seconds": times, "torch_peak_mib_video_stage": torch_peak_mib, "pipeline_report": {k: v for k, v in rep.items() if k not in ("segments", "runner_calls")},
           "segments": segs, "old_0_7": None if old is None else {"before": old.get("before"), "weights": [{k: r.get(k) for k in ("w", "confidence", "confidence_delta", "av_offset_frames", "codeformer_seconds")} for r in old.get("weights", [])],
                                                                  "verdict": old.get("verdict"), "at": old.get("at"), "implementation": "legacy third_party/CodeFormer/inference_codeformer.py CLI (512x936 output)"}}
    write_json(pathlib.Path(a.json), out)
    b = before or {}
    md = [f"# Pilot 0.7 refresh — CURRENT optimized CodeFormer vs SyncNet (video {a.id}, {time.strftime('%Y-%m-%d')})", "",
          f"Verdict (strict policy unchanged: Δconf >= {-a.max_confidence_drop:g}, |AV offset| worsening <= {a.max_offset_worsening_frames}): **formal 0.7 = {'PASS' if selected is not None else 'FAIL'}** "
          f"({'selected w = ' + str(selected) if selected is not None else 'NO_STRICT_SAFE_WEIGHT'}); best effort w = {best['w'] if best else '-'} (Δconf {best['confidence_delta']:+.2f})" if best else "", "",
          f"Method: ONE LatentSync pass (current optimized backend, {len(ls_segs)} LATENT_SYNC segments, {sum(s['frames'] for s in ls_segs)} frames) on the dubbed track of the E2E run; "
          f"A = those frames assembled with CodeFormer OFF; B = the same frames restored in memory by `codeformer_accel.CodeFormerAccel` (batch {a.codeformer_batch}, {a.codeformer_precision}, "
          f"{a.codeformer_landmarks} landmarks) at each w, same crossfade / audio / encoder. SyncNet: upstream evaluator, unchanged. "
          f"Production output (w {a.production_w:g}) re-assembled bit-identical to the pipeline's own file: {prod_md5['pipeline'] == prod_md5['reassembled']}.", "",
          f"A (CodeFormer OFF): SyncNet confidence **{b.get('confidence')}**, AV offset **{b.get('av_offset_frames')}** frames; {off_meta['video']['width']}x{off_meta['video']['height']}, {off_chk['decoded_frames']} frames, {off_meta['duration_s']:.2f} s", "",
          "| w | SyncNet before | SyncNet after | Δconf | offset before | offset after | Δ\\|offset\\| | strict pass | CodeFormer s | s/face | CF s/video-min | faces | geometry / frames / duration vs A |",
          "|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---|"]
    for r in rows:
        v = r["validation"]; om = r["output_meta"]
        md.append(f"| {r['w']:g} | {r.get('confidence_before', '-')} | {r.get('confidence_after', '-')} | {r.get('confidence_delta', 0):+.2f} | {r.get('av_offset_before', '-')} | {r.get('av_offset_after', '-')} | "
                  f"{r.get('offset_abs_worsening_frames', 0):+d} | {'YES' if r.get('strict_pass') else 'NO'} | {r['codeformer_wall_s']:.2f} | {r['s_per_face']} | {r['codeformer_s_per_video_min']:.1f} | {r['faces_restored']} | "
                  f"{om['width']}x{om['height']} {om['frames']}f {om['duration_s']:.2f}s {'OK' if v['pass'] else 'MISMATCH'} |")
    md += ["", f"LatentSync (this pass): stage {times.get('latentsync')} s = {out['latentsync']['s_per_video_min']} s/video-min, verify {times.get('latentsync_verify')} s, load {times.get('latentsync_load')} s; "
               f"torch peak {torch_peak_mib} MiB during the video stage (LatentSync + {len(a.weights)} CodeFormer instances resident).", ""]
    if old:
        md += ["## Comparison with the legacy 0.7 (2026-09-15, upstream CLI, output 512x936)", "",
               f"old before {old.get('before', {}).get('confidence')} / offset {old.get('before', {}).get('av_offset_frames')}; old verdict `{old.get('verdict')}`", "",
               "| w | old conf after (legacy CLI) | old Δconf | new conf after (optimized) | new Δconf |", "|---:|---:|---:|---:|---:|"]
        om_ = {r["w"]: r for r in old.get("weights", [])}
        for r in rows:
            o_ = om_.get(r["w"])
            md.append(f"| {r['w']:g} | {o_['confidence'] if o_ else '-'} | {('%+.2f' % o_['confidence_delta']) if o_ else '-'} | {r.get('confidence_after', '-')} | {r.get('confidence_delta', 0):+.2f} |")
    md += ["", "Evidence: this file + `" + str(pathlib.Path(a.json).relative_to(ROOT) if str(a.json).startswith(str(ROOT)) else a.json) + "`; media in `" + str(out_dir) + "` (not tracked)."]
    pathlib.Path(a.md).write_text("\n".join(md) + "\n"); print("\n".join(md))
    return 0 if selected is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
