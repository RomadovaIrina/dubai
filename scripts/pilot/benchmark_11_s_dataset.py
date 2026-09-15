#!/usr/bin/env python3
"""Pilot 0.11 — run the timeline-preserving S measurement across a dataset and aggregate it.

Delegates each file to measure_stages.py so the stage definition remains single-source:
S = T_vad_split + T_concat + T_subtitle_burn (all chunks kept, full-length output, cues on the original timeline).
Aggregates per-video 0.11_stages_<name>.json into reports/pilot/0.11_s_dataset.{json,md}; the primary aggregate is the
normalized S per source minute (raw seconds are not averaged across videos of different length).
"""
from __future__ import annotations
import argparse, json, pathlib, statistics, subprocess, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT, gpu_info, write_json

EXT = {'.mp4', '.mov', '.mkv', '.avi', '.webm'}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument('videos')
    ap.add_argument('--repeat', type=int, default=1); ap.add_argument('--asr', action='store_true')
    ap.add_argument('--split-mode', choices=['reencode', 'copy'], default='reencode')
    ap.add_argument('--tag', default='', help="suffix for per-video json (informational runs, e.g. 'copy')")
    ap.add_argument('--no-aggregate', action='store_true', help="run only; do not (re)write 0.11_s_dataset.*")
    a = ap.parse_args(); root = pathlib.Path(__file__).resolve().parents[2]
    p = pathlib.Path(a.videos).resolve(); vids = [p] if p.is_file() else sorted(x for x in p.iterdir() if x.suffix.lower() in EXT)
    if not vids:
        raise SystemExit('no videos found')
    fail = []
    for i, v in enumerate(vids, 1):
        print(f"\n===== 0.11 {i}/{len(vids)} {v.name} =====", flush=True)
        cmd = ['/venv/dabai/bin/python', str(root / 'scripts/pilot/measure_stages.py'), str(v), '--repeat', str(a.repeat), '--split-mode', a.split_mode]
        if a.asr: cmd.append('--asr')
        if a.tag: cmd += ['--tag', a.tag]
        rc = subprocess.run(cmd, cwd=str(root)).returncode
        if rc != 0: fail.append((v.name, rc))
    if a.no_aggregate or a.tag:
        return 1 if fail else 0

    rows = []
    for v in vids:
        f = PILOT / f"0.11_stages_{v.stem}.json"
        rows.append(json.load(open(f)) if f.exists() else {"input": v.name, "missing": True})
    ok = [r for r in rows if not r.get("missing") and r.get("validation_all_runs_pass")]
    spm = [r["S_per_video_min"] for r in ok]
    def agg(vals): return {"mean": round(statistics.mean(vals), 3), "median": round(statistics.median(vals), 3), "min": round(min(vals), 3), "max": round(max(vals), 3)} if vals else None
    stages = {k: agg([r["median"][k] for r in ok]) for k in ("vad_split", "concat", "subtitle_burn")}
    worst = max(ok, key=lambda r: r["S_per_video_min"]) if ok else None
    reasons = []
    for r in rows:
        if r.get("missing"): reasons.append(f"{r['input']}: pipeline failure, no result json")
        elif not r.get("validation_all_runs_pass"): reasons.append(f"{r['input']}: validation FAIL {[k for k, v in r['validation']['checks'].items() if not v]} (assembled Δ {r['assembled_delta_s']:+.3f}s, final Δ {r['final_delta_s']:+.3f}s)")
    for n, rc in fail: reasons.append(f"{n}: measure_stages.py exit {rc}")
    closed = len(ok) == len(vids) == 5 and not reasons
    rep = {"task": "0.11", "closed": closed,
           "verdict": "CLOSED — 5/5 TIMELINE-PRESERVING S MEASURED" if closed else "OPEN/BLOCKED — " + "; ".join(reasons),
           "methodology": "S = T_vad_split (Silero + full timeline + cutting every SPEECH/PASS_THROUGH chunk) + T_concat (all chunks, original order) + T_subtitle_burn (cues on original timeline, full-length clip); demux and ASR excluded",
           "split_mode": a.split_mode, "asr": a.asr, "repeat": a.repeat, "median_of_runs": True, "gpu": gpu_info(),
           "videos": len(vids), "validated": len(ok),
           "per_video": [{k: r.get(k) for k in ("input", "S_seconds", "S_per_video_min", "median", "speech_intervals", "speech_seconds", "speech_pct",
                                                 "pass_through_intervals", "pass_through_seconds", "chunks_total", "assembled_seconds", "final_seconds",
                                                 "assembled_delta_s", "final_delta_s", "final_resolution", "final_fps", "validation_all_runs_pass", "excluded_from_S", "env_snapshot")}
                         | {"source_duration_s": r.get("source", {}).get("duration_s"), "source_resolution": f"{r.get('source', {}).get('video', {}).get('width')}x{r.get('source', {}).get('video', {}).get('height')}",
                            "source_fps": r.get("source", {}).get("video", {}).get("avg_fps"), "vfr": r.get("source", {}).get("video", {}).get("vfr"),
                            "tolerance_s": r.get("validation", {}).get("tolerance_s")} for r in rows],
           "aggregate": {"S_per_video_min": agg(spm), "stages_seconds": stages,
                         "worst_case_video": worst["input"] if worst else None, "worst_case_S_per_video_min": worst["S_per_video_min"] if worst else None},
           "timeline_validation": f"{len(ok)}/{len(vids)} PASS", "reasons_open": reasons,
           "legacy_note": "reports/pilot/stages_04.json, stages.jsonl, stages.md = LEGACY_INVALID_TIMELINE (speech-only concat, 53.9 s -> 29.5 s); not used here",
           "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    write_json(PILOT / "0.11_s_dataset.json", rep)
    md = ["# Pilot 0.11 — S on the real-content dataset (timeline-preserving)", "",
          f"**Verdict:** `{rep['verdict']}`  ", f"**Split mode:** {a.split_mode} (frame-accurate), ASR={a.asr}, repeat={a.repeat}, per-stage median  ",
          f"**Timeline validation:** {rep['timeline_validation']}  ", "",
          "| video | src dur | res | fps | speech int. | speech s (%) | pass-through int. / s | T_vad_split | T_concat | T_subtitle_burn | **S** | **S/min** | assembled Δ | final Δ | final res @ fps | valid |",
          "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for r in rep["per_video"]:
        if r.get("S_seconds") is None: md.append(f"| {r['input']} | - | - | - | - | - | - | - | - | - | - | - | - | - | - | MISSING |"); continue
        m = r["median"]
        md.append(f"| {r['input']} | {r['source_duration_s']:.2f} | {r['source_resolution']} | {r['source_fps']}{' VFR' if r['vfr'] else ''} | {r['speech_intervals']} | {r['speech_seconds']:.1f} ({r['speech_pct']:.0f}%) | {r['pass_through_intervals']} / {r['pass_through_seconds']:.1f} | {m['vad_split']:.2f} | {m['concat']:.2f} | {m['subtitle_burn']:.2f} | **{r['S_seconds']:.2f}** | **{r['S_per_video_min']:.2f}** | {r['assembled_delta_s']:+.3f} | {r['final_delta_s']:+.3f} | {r['final_resolution']} @ {r['final_fps']} | {'PASS' if r['validation_all_runs_pass'] else 'FAIL'} |")
    ag = rep["aggregate"]
    md += ["", "Excluded from S (per video, median): " + ", ".join(f"{r['input']} demux {r['excluded_from_S']['demux_s']}s" + (f" / asr {r['excluded_from_S']['asr_s']}s" if r['excluded_from_S'].get('asr_s') else "") for r in rep["per_video"] if r.get("excluded_from_S")), "",
           "## Aggregate (normalized S per source minute is the primary figure)", "",
           f"S/video-min: mean **{ag['S_per_video_min']['mean']}**, median **{ag['S_per_video_min']['median']}**, min {ag['S_per_video_min']['min']}, max {ag['S_per_video_min']['max']}  " if ag['S_per_video_min'] else "S/video-min: n/a  ",
           "".join(f"T_{k}: mean {v['mean']} s / median {v['median']} s  \n" for k, v in stages.items() if v),
           f"Worst case: **{ag['worst_case_video']}** ({ag['worst_case_S_per_video_min']} s/video-min)", "",
           f"Legacy: {rep['legacy_note']}"]
    (PILOT / "0.11_s_dataset.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md)); print("-> reports/pilot/0.11_s_dataset.json / .md")
    return 0 if closed else 2


if __name__ == '__main__':
    raise SystemExit(main())
