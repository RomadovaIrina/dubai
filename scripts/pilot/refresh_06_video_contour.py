#!/usr/bin/env python3
"""Pilot 0.6 refresh (2026-10-02) — CURRENT video contour timing: optimized LatentSync + optimized in-memory CodeFormer.

Samples (runner manifests, same schema):
  (1) the E2E run of the video (run_clean_pipeline_05.py --alignment burst --video-backend optimized --codeformer optimized);
  (2) video-stage-only repeats on the SAME prepared audio / work dir (scripts/quality/candidate_video_stage.py --codeformer optimized:
      speech gate -> router -> LatentSync -> CodeFormer -> assembly; no ASR / Qwen / TTS rerun);
  (3) optionally the 0.7 sweep JSON (one more LatentSync pass; CodeFormer timing there is per weight -> production weight, informational).
Formula (pilot 0.6, unchanged): stage_seconds / source_duration_seconds * 60; target combined <= 300 s/video-min.
NEW = median over the production samples (1)+(2). The old legacy numbers (whole-video LatentSync CLI + upstream CodeFormer CLI,
2026-09-15) are read from reports/pilot/0.6_latentsync.json / 0.6_codeformer.json for the comparison. VRAM is informational (0.4 stays closed).

    python scripts/pilot/refresh_06_video_contour.py --e2e-manifest /tmp/pilot_refresh/final/04_final.manifest.json \
        --repeat-manifest /tmp/pilot_refresh/r06/04_cf_prod_r2.manifest.json --sweep-json reports/pilot/refresh_0.7_current_codeformer.json \
        --nvsmi e2e=/tmp/pilot_refresh/final/04_final.nvsmi.log r2=/tmp/pilot_refresh/r06/04_cf_prod_r2.nvsmi.log
"""
from __future__ import annotations
import argparse, json, pathlib, statistics, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT, ROOT, gpu_info, git_info, write_json

TARGET = 300.0


def nvsmi_peak(p: str | None) -> int | None:
    if not p or not pathlib.Path(p).exists():
        return None
    vals = []
    for line in pathlib.Path(p).read_text().splitlines():
        parts = [x.strip() for x in line.split(",")]
        try:
            vals.append(int(parts[-1]))
        except Exception:
            pass
    return max(vals) if vals else None


def sample_from_manifest(path: pathlib.Path, label: str, nvsmi: str | None) -> dict:
    man = json.loads(path.read_text()); st = man["stage_seconds"]; ls = man["lipsync"]; cf = ls.get("codeformer"); dur = float(man["source"]["duration_s"])
    cf_dict = cf if isinstance(cf, dict) else {}
    segs = ls.get("segments") or []
    seg_cf = [s["codeformer"] for s in segs if isinstance(s.get("codeformer"), dict)]
    ls_s = float(st.get("latentsync") or 0.0); cf_s = float(st.get("codeformer") or 0.0)
    video_stage_keys = ("face_router_load", "latentsync_load", "codeformer_load", "ls_master_25fps", "ls_master_decode", "face_classify", "latentsync", "codeformer", "final_assembly", "speech_gate")
    video_stage_total = round(sum(float(st.get(k) or 0.0) for k in video_stage_keys), 3)
    faces = int(cf_dict.get("totals", {}).get("faces_restored", 0) or 0)
    ls_video_s = float(ls.get("latentsync_seconds_of_video") or 0.0)
    calls = ls.get("runner_calls") or []
    unet = round(sum(c.get("stages", {}).get("unet_denoise", 0.0) for c in calls), 3)
    return {"label": label, "manifest": str(path), "at": man.get("at"), "kind": "e2e" if man.get("task") == "0.5-runner" and "candidate" not in man else "video_stage_repeat",
            "source_duration_s": dur, "validation_pass": bool((man.get("validation") or {}).get("pass")), "duration_delta_s": (man.get("validation") or {}).get("duration_delta_s"),
            "frames": {"master": ls.get("master_frames"), "written": (ls.get("assembly") or {}).get("frames_written"), "lipsynced": int(round(ls_video_s * 25)), "ls_fraction": ls.get("ls_fraction")},
            "latentsync": {"stage_s": ls_s, "verify_s": st.get("latentsync_verify"), "inference_total_s": ls.get("latentsync_inference_total_s"), "calls": ls.get("latentsync_calls"), "unet_denoise_s": unet,
                           "load_s": st.get("latentsync_load"), "s_per_video_min": round(ls_s / dur * 60, 3), "s_per_lipsynced_min": round(ls_s / ls_video_s * 60, 3) if ls_video_s else None,
                           "config": ls.get("runner")},
            "codeformer": {"stage_s": cf_s, "load_s": st.get("codeformer_load"), "faces_restored": faces, "frames_in": cf_dict.get("totals", {}).get("frames_in"), "eligible_frames": cf_dict.get("eligible_frames"),
                           "fallback_detections": cf_dict.get("totals", {}).get("fallback_detections"), "frames_without_landmarks": cf_dict.get("totals", {}).get("frames_without_landmarks"),
                           "s_per_video_min": round(cf_s / dur * 60, 3), "s_per_face": round(cf_s / faces, 4) if faces else None, "faces_per_s": round(faces / cf_s, 2) if cf_s else None,
                           "torch_peak_mib_max_segment": max([c.get("peak_vram_mib") or 0 for c in seg_cf] or [0]) or None, "config": cf_dict.get("config"), "mode": man.get("codeformer")},
            "combined": {"ls_plus_cf_s": round(ls_s + cf_s, 3), "s_per_video_min": round((ls_s + cf_s) / dur * 60, 3), "within_target": (ls_s + cf_s) / dur * 60 <= TARGET},
            "video_stage_total_s": video_stage_total, "video_stage_s_per_video_min": round(video_stage_total / dur * 60, 3), "pipeline_total_s": st.get("total"),
            "stage_seconds": st, "peak_vram_nvsmi_mib": nvsmi_peak(nvsmi), "gpu": {k: (man.get("gpu") or {}).get(k) for k in ("name", "capability")}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--e2e-manifest", required=True); ap.add_argument("--repeat-manifest", nargs="*", default=[])
    ap.add_argument("--sweep-json", default=None); ap.add_argument("--nvsmi", nargs="*", default=[], metavar="LABEL=PATH")
    ap.add_argument("--json", default=str(PILOT / "refresh_0.6_current_pipeline.json")); ap.add_argument("--md", default=str(PILOT / "refresh_0.6_current_pipeline.md"))
    a = ap.parse_args()
    nv = dict(x.split("=", 1) for x in a.nvsmi)
    samples = [sample_from_manifest(pathlib.Path(a.e2e_manifest), "e2e", nv.get("e2e"))]
    for i, m in enumerate(a.repeat_manifest, 2):
        samples.append(sample_from_manifest(pathlib.Path(m), f"r{i}", nv.get(f"r{i}")))
    sweep = None
    if a.sweep_json and pathlib.Path(a.sweep_json).exists():
        sj = json.loads(pathlib.Path(a.sweep_json).read_text()); prod = sj["method"]["codeformer"]["production_w"]
        row = next((r for r in sj["weights"] if r["w"] == prod), None)
        sweep = {"json": a.sweep_json, "latentsync_stage_s": sj["latentsync"]["stage_seconds"], "latentsync_s_per_video_min": sj["latentsync"]["s_per_video_min"],
                 "codeformer_prod_w": prod, "codeformer_wall_s": row["codeformer_wall_s"] if row else None, "codeformer_s_per_video_min": row["codeformer_s_per_video_min"] if row else None,
                 "faces": row["faces_restored"] if row else None, "torch_peak_mib_video_stage": sj.get("torch_peak_mib_video_stage"), "note": "5 CodeFormer instances resident; informational sample"}
    dur = samples[0]["source_duration_s"]
    ls_vals = [s["latentsync"]["stage_s"] for s in samples]; cf_vals = [s["codeformer"]["stage_s"] for s in samples]
    ls_new, cf_new = statistics.median(ls_vals), statistics.median(cf_vals)
    faces = samples[0]["codeformer"]["faces_restored"]
    new = {"samples": len(samples), "latentsync_s": round(ls_new, 3), "codeformer_s": round(cf_new, 3), "latentsync_s_per_video_min": round(ls_new / dur * 60, 3),
           "codeformer_s_per_video_min": round(cf_new / dur * 60, 3), "combined_s_per_video_min": round((ls_new + cf_new) / dur * 60, 3),
           "latentsync_s_per_video_min_min_max": [round(min(ls_vals) / dur * 60, 3), round(max(ls_vals) / dur * 60, 3)],
           "codeformer_s_per_video_min_min_max": [round(min(cf_vals) / dur * 60, 3), round(max(cf_vals) / dur * 60, 3)],
           "codeformer_faces": faces, "codeformer_s_per_face": round(cf_new / faces, 4) if faces else None,
           "peak_vram_nvsmi_mib": max([s["peak_vram_nvsmi_mib"] or 0 for s in samples] or [0]) or None,
           "video_stage_total_s_per_video_min": round(statistics.median([s["video_stage_total_s"] for s in samples]) / dur * 60, 3)}
    new["within_target"] = new["combined_s_per_video_min"] <= TARGET
    old = {}
    try:
        lj = json.loads((PILOT / "0.6_latentsync.json").read_text()); cj = json.loads((PILOT / "0.6_codeformer.json").read_text())
        cf_old = cj.get("codeformer") or {}; comb = cj.get("combined") or {}
        old = {"latentsync_s": lj["steady_state"]["wall_s"], "latentsync_s_per_video_min": lj["steady_state"]["s_per_video_min"], "latentsync_peak_mib": lj["steady_state"].get("peak_device_mib"),
               "codeformer_s": cf_old.get("wall_s"), "codeformer_s_per_video_min": cf_old.get("s_per_video_min"), "codeformer_faces": cf_old.get("input_frames"), "codeformer_peak_mib": cf_old.get("peak_device_mib"),
               "codeformer_w": cf_old.get("fidelity_weight"), "combined_s_per_video_min": comb.get("total_s_per_video_min"), "within_target": comb.get("verdict") == "PASS_TARGET", "old_verdict": comb.get("verdict"),
               "implementation": "LatentSync CLI on the whole video (no gate/router) + upstream inference_codeformer.py CLI (w 0.5, 512x936 output, PNG round trips)", "at": cj.get("at")}
    except Exception as e:
        old = {"error": str(e)}
    out = {"task": "0.6-refresh", "video": samples[0]["manifest"], "source_duration_s": dur, "target_s_per_video_min": TARGET, "formula": "stage_seconds / source_duration_seconds * 60",
           "new": new, "verdict": "PASS_TARGET" if new["within_target"] else "FAIL_TARGET", "samples": samples, "sweep_sample": sweep, "old_legacy": old,
           "git": git_info(ROOT), "gpu": gpu_info(), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "notes": ["LatentSync time = stage wall incl. LatentSync's own detector verify pass and in-memory segment prep, excl. model load (reported separately);",
                     "CodeFormer time = in-memory stage wall on the LATENT_SYNC frames only (model load reported separately);",
                     "the current pipeline lip-syncs only speech-gated VALID_FACE frames, so s/video-min is also given per lip-synced minute for comparison with the old whole-video number;",
                     "VRAM informational only (0.4 not reopened)."]}
    write_json(pathlib.Path(a.json), out)
    s0 = samples[0]
    md = [f"# Pilot 0.6 refresh — current video contour timing (LatentSync optimized + CodeFormer optimized), {time.strftime('%Y-%m-%d')}", "",
          f"Video: `{pathlib.Path(s0['manifest']).name}` source {dur:.3f} s · GPU {s0['gpu'].get('name')} / {s0['gpu'].get('capability')} · HEAD `{out['git'].get('commit', '')[:8]}`  ",
          f"Samples: {len(samples)} production runs (e2e + {len(samples) - 1} video-stage repeat(s) on the same audio/work dir){' + 1 sweep pass (informational)' if sweep else ''}. Formula: `{out['formula']}`. Target: combined <= {TARGET:.0f} s/video-min.", "",
          "```text", f"LatentSync seconds        = {new['latentsync_s']:.2f}   (median; samples {', '.join(f'{v:.2f}' for v in ls_vals)})",
          f"CodeFormer seconds        = {new['codeformer_s']:.2f}   (median; samples {', '.join(f'{v:.2f}' for v in cf_vals)})",
          f"LatentSync s/video-min    = {new['latentsync_s']:.2f} / {dur:.2f} * 60 = {new['latentsync_s_per_video_min']:.1f}",
          f"CodeFormer s/video-min    = {new['codeformer_s']:.2f} / {dur:.2f} * 60 = {new['codeformer_s_per_video_min']:.1f}",
          f"combined video contour    = {new['combined_s_per_video_min']:.1f} s/video-min", f"target                    = {TARGET:.0f} s/video-min", f"verdict                   = {out['verdict']}", "",
          f"peak VRAM (nvidia-smi, device-wide, whole run)   = {new['peak_vram_nvsmi_mib']} MiB   (informational)",
          f"CodeFormer faces                                 = {faces} (eligible LATENT_SYNC frames {s0['codeformer']['eligible_frames']} of {s0['frames']['master']} master frames)",
          f"CodeFormer seconds / processed face              = {new['codeformer_s_per_face']}", "```", "",
          "| sample | kind | LS stage s | LS verify s | LS s/video-min | LS s/lipsynced-min | CF stage s | CF faces | CF s/video-min | CF s/face | combined s/video-min | video stage total s | pipeline total s | nvsmi peak MiB | CF torch peak MiB | valid |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for s in samples:
        L, C = s["latentsync"], s["codeformer"]
        md.append(f"| {s['label']} | {s['kind']} | {L['stage_s']:.2f} | {L['verify_s']} | {L['s_per_video_min']:.1f} | {L['s_per_lipsynced_min']} | {C['stage_s']:.2f} | {C['faces_restored']} | {C['s_per_video_min']:.1f} | {C['s_per_face']} | "
                  f"{s['combined']['s_per_video_min']:.1f} | {s['video_stage_total_s']:.1f} | {s['pipeline_total_s']} | {s['peak_vram_nvsmi_mib']} | {C['torch_peak_mib_max_segment']} | {'PASS' if s['validation_pass'] else 'FAIL'} |")
    if sweep:
        md.append(f"| sweep (0.7) | video stage, 5 CF instances | {sweep['latentsync_stage_s']} | - | {sweep['latentsync_s_per_video_min']} | - | {sweep['codeformer_wall_s']} | {sweep['faces']} | {sweep['codeformer_s_per_video_min']} | - | - | - | - | - | {sweep['torch_peak_mib_video_stage']} (torch) | - |")
    md += ["", "## old legacy -> new optimized", "", "| metric | old legacy (2026-09-15) | new optimized (this refresh) |", "|---|---:|---:|",
           f"| LatentSync s/video-min | {old.get('latentsync_s_per_video_min')} (whole video, CLI) | {new['latentsync_s_per_video_min']:.1f} (speech-gated {s0['frames']['ls_fraction']:.0%} of frames; {s0['latentsync']['s_per_lipsynced_min']:.1f} per lip-synced min) |",
           f"| CodeFormer s/video-min | {old.get('codeformer_s_per_video_min')} (upstream CLI, {old.get('codeformer_faces')} faces, 512x936) | {new['codeformer_s_per_video_min']:.1f} (in-memory, {faces} faces, geometry preserved) |",
           f"| combined s/video-min | {old.get('combined_s_per_video_min')} | {new['combined_s_per_video_min']:.1f} |",
           f"| target 300 | {'PASS' if old.get('within_target') else 'FAIL_TARGET'} | {out['verdict']} |",
           f"| CodeFormer s/face | {round(old['codeformer_s'] / old['codeformer_faces'], 4) if old.get('codeformer_faces') else '-'} | {new['codeformer_s_per_face']} |",
           f"| peak VRAM MiB | LS {old.get('latentsync_peak_mib')} / CF {old.get('codeformer_peak_mib')} (separate processes) | {new['peak_vram_nvsmi_mib']} (one process, LatentSync + CodeFormer resident) |", "",
           "Notes: " + " ".join(out["notes"]), "", f"Evidence: `{pathlib.Path(a.json).relative_to(ROOT) if str(a.json).startswith(str(ROOT)) else a.json}`, manifests listed in the samples; media under /tmp/pilot_refresh (not tracked)."]
    pathlib.Path(a.md).write_text("\n".join(md) + "\n"); print("\n".join(md)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
