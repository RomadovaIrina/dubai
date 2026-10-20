#!/usr/bin/env python3
"""Round 4 audio regression gate: NEW vs PREVIOUS per video, the same thresholds for every video. A metric clearly worse on any video -> NO PUBLISH.
PREVIOUS = a published reference (mix-only metrics of its mp4, e.g. ref_v4/) + optionally the run that reproduces it (stem metrics: WER, hard cuts,
joins), because a published mp4 carries no dub stem / text.
    python scripts/quality/r4_audio_gate.py --prev ref_v4 [--prev-run v4repro] --new <tag> [--out r4_audio_vN_regression.md] [--ids 01 ...]"""
import argparse, json, pathlib, sys
A = pathlib.Path("/workspace/dub/artifacts/r4a")
# (label, getter, worse-if, tolerance): worse-if "up" = higher is worse
AX = [("duration - source (s)", lambda m: round(m["duration_s"] - m["source_duration_s"], 3), "abs", 0.1),
      ("LUFS", lambda m: m["loudness"]["integrated_lufs"], "lufs", 1.0),
      ("true peak dBTP", lambda m: m["loudness"]["true_peak_dbtp"], "tp", -1.0),
      ("residue under dub (dB rel. voice)", lambda m: m.get("residue_under_dub_db"), "up", 1.5),
      ("residue exposed, dub silent (dB rel. voice)", lambda m: m.get("residue_exposed_db"), "up", 1.5),
      ("bed speech heard (VAD, s)", lambda m: m.get("bed_vad_s"), "up", 1.0),
      ("bed in pauses vs source (dB)", lambda m: m.get("pause_bed_rel_db"), "up", 1.0),
      ("bed p95 in pauses vs source (dB)", lambda m: m.get("pause_bed_p95_rel_db"), "up", 1.0),
      ("level jump at dub-pause onsets p95 (dB)", lambda m: m.get("pause_jump_p95_db"), "up", 1.5),
      ("pause distribution W1 vs source (s)", lambda m: m.get("pauses_w1_s"), "up", 0.1),
      ("dub pauses > 1 s (source in brackets)", lambda m: m["pauses_dub"]["gt_1s"], "up", 1),
      ("source speaks, dub silent (s)", lambda m: m.get("src_speaks_dub_silent_s"), "up", 2.0),
      ("dub speaks, source silent (s)", lambda m: m.get("dub_speaks_src_silent_s"), "up", 2.0),
      ("WER final mix", lambda m: (m.get("asr") or {}).get("wer_final_mix"), "up", 0.03),
      ("hard cuts (stop/restart)", lambda m: m.get("hard_cuts"), "up", 0),
      ("joins F0 jump > 3 st", lambda m: (m.get("joins") or {}).get("f0_jump_gt_3st"), "up", 1),
      ("joins energy jump median (dB)", lambda m: (m.get("joins") or {}).get("e_jump_median_db"), "info", 0),
      ("joins F0 excess over the uncut take (st)", lambda m: (m.get("joins") or {}).get("f0_excess_over_raw_median_st"), "up", 0.5),
      ("joins energy excess over the uncut take (dB)", lambda m: (m.get("joins") or {}).get("e_excess_over_raw_median_db"), "up", 1.5),
      ("sentence-to-sentence level jump p90 (dB)", lambda m: (m.get("units") or {}).get("level_jump_p90_db"), "up", 1.0),
      ("sentence-to-sentence F0 jump p90 (st)", lambda m: (m.get("units") or {}).get("f0_jump_p90_st"), "up", 1.0),
      ("named entities misheard", lambda m: (m.get("ne") or {}).get("misheard"), "up", 0)]


def get(f, m):
    try: return f(m)
    except Exception: return None


def verdict(kind, tol, p, n):
    if n is None: return ""
    if kind == "abs": return "FAIL" if abs(n) > tol else ""
    if kind == "lufs": return "FAIL" if abs(n + 16) > tol else ""
    if kind == "tp": return "FAIL" if n > tol else ""
    if kind == "info" or p is None: return ""
    return "WORSE" if n - p > tol else ("better" if p - n > tol else "")


def load(d, id_):
    f = A / d / f"{id_}.metrics.json"
    return json.loads(f.read_text()) if f.exists() else None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prev", required=True); ap.add_argument("--prev-run", default=None); ap.add_argument("--new", required=True)
    ap.add_argument("--ids", nargs="+", default=["01", "02", "03", "04", "05"]); ap.add_argument("--out", default=None); ap.add_argument("--title", default="")
    a = ap.parse_args(); L = [f"# Audio regression gate {a.title}: NEW `{a.new}` vs PREVIOUS `{a.prev}`" + (f" (+ stem axes from `{a.prev_run}`)" if a.prev_run else ""), ""]
    L += ["Tolerance = the fixed per-axis tolerance + |published previous - its reproduction by the same code on this server| (mix axes), i.e. a change is",
          "\"clearly worse\" only beyond what re-running the previous version itself produces. Stem axes (WER, cuts, joins, continuity, names) compare with the reproduction.", ""]
    rd = A / a.new / "README.txt"
    if rd.exists(): L += ["```", rd.read_text().strip(), "```", ""]
    bad = []
    for id_ in a.ids:
        n, p = load(a.new, id_), load(a.prev, id_); pr = load(a.prev_run, id_) if a.prev_run else None
        if n is None: bad.append(f"{id_}: no NEW metrics"); L += [f"## {id_}: NEW missing", ""]; continue
        L += [f"## {id_}", "", "| axis | previous | new | previous reproduced | tolerance | verdict |", "|---|---:|---:|---:|---:|---|"]
        for lab, f, kind, tol in AX:
            stem = lab.startswith(("WER", "hard cuts", "joins", "named", "sentence-to"))   # need the run's text / dub stem: from the reproducing run only
            pv = (get(f, pr) if pr else None) if stem else (get(f, p) if p else None)
            nv = get(f, n)
            if lab.startswith("dub pauses"):
                lab = f"dub pauses > 1 s (source {get(lambda m: m['pauses_source']['gt_1s'], n)})"
            t_ = tol   # noise-aware: "clearly worse" = beyond tol + the difference between the published PREVIOUS and its own reproduction (same code)
            if not stem and pr and p and kind == "up" and get(f, pr) is not None and pv is not None: t_ = tol + abs(get(f, pr) - pv)
            v = verdict(kind, t_, pv, nv)
            if v in ("WORSE", "FAIL"): bad.append(f"{id_}: {lab} {pv} -> {nv}")
            L.append(f"| {lab} | {'' if pv is None else pv} | {'' if nv is None else nv} | {'' if (pr is None or stem) else get(f, pr)} | {round(t_, 3) if kind == 'up' else tol} | {v} |")
        L.append("")
    L += ["## Verdict", "", "**PASS: no axis clearly worse on any video.**" if not bad else "**NO PUBLISH:**", ""] + [f"- {b}" for b in bad]
    s = "\n".join(L) + "\n"
    if a.out: pathlib.Path(a.out).write_text(s)
    print(s); return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
