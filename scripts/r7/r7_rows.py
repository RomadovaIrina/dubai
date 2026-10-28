#!/usr/bin/env python3
"""r7 review rows for one video: lip placement (what moved, where the dub now follows the source) and the gap axes PREV vs NEW.
    python scripts/r7/r7_rows.py --prev PREV --new NEW --id 05      (markdown to stdout, appended to <id>_review.md by r7_publish.sh)"""
import argparse, json, pathlib
A = pathlib.Path("/workspace/dub/artifacts/r4a")


def tc(t): return f"{int(t // 60)}:{t % 60:05.2f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prev", required=True); ap.add_argument("--new", required=True); ap.add_argument("--id", required=True); a = ap.parse_args()
    P = json.loads((A / a.prev / f"{a.id}.metrics.json").read_text()); N = json.loads((A / a.new / f"{a.id}.metrics.json").read_text())
    man = json.loads((A / a.new / a.id / f"{a.id}_r4b.manifest.json").read_text()); lp = man.get("lip_place") or {}; gc = man.get("gap_clean") or {}
    L = ["", f"## r7 lip placement ({a.id})", ""]
    if lp:
        L += [f"- face visible on the source: {round(100 * (lp.get('face_visible_frac') or 0))} % of the video; blocks {lp['blocks']}, moved {lp['moved_blocks']}, mean |shift| {lp['shift_abs_mean_s']} s, max {lp['shift_abs_max_s']} s",
              f"- new pauses (splits at an English boundary, aligned to a source pause): " + (", ".join(f"{tc(t)} ({c})" for t, c in lp["splits"]) or "none"),
              f"- face-visible dub/source speech mismatch (frame level, the DP objective): {lp['mismatch_face_visible_s_before']} s -> {lp['mismatch_face_visible_s_after']} s"]
        mv = [s for s in lp["shifts"] if abs(s[2]) >= 0.2]
        if mv: L += ["- moves >= 0.2 s (r6 time -> shift): " + ", ".join(f"{tc(s[0])}-{s[1]:.2f} {s[2]:+.2f} s" for s in mv[:20])]
    for k, lab in [("dub_silent_face_visible_s", "speaker on screen talks, dub silent, runs >= 1 s (s)"), ("src_speaks_dub_silent_s", "source speaks, dub silent (s)"),
                   ("dub_speaks_src_silent_s", "dub speaks, source silent (s)"), ("pauses_w1_s", "pause distribution W1 vs source (s)"),
                   ("gap_excess_max_s", "longest dub gap beyond the source's own pause (s)"), ("residue_exposed_db", "residue exposed, dub silent (dB rel. voice)")]:
        L += [f"- {lab}: {P.get(k)} -> {N.get(k)}"]
    L += ["", f"## r7 gap clean ({a.id})", ""]
    if gc:
        L += [f"- room tone: {'yes' if gc.get('room_tone') else 'no'} (floor {gc.get('floor_db')} dBFS, ambience spread {gc.get('ambience_spread_db')} dB); detected leaked voice {gc['voice_s']} s, "
              f"separator bursts {gc['burst_s']} s -> {gc['replaced_s']} s replaced; denoised {gc.get('denoised_s', 0)} s of gaps where the source speaks"]
        if gc.get("runs"): L += ["- replaced: " + ", ".join(f"{tc(r[0])}-{r[1]:.2f} {r[2]}" for r in gc["runs"][:25])]
    for t in (a.prev, a.new):
        f = A / t / f"{a.id}.gaps.json"
        if f.exists():
            g = json.loads(f.read_text())
            L += [f"- {t}: in dub gaps where the source speaks: leaked voice {g.get('gap_voice_s_src_speaks')} s, bursts {g.get('gap_burst_s_src_speaks')} s, voice-band excess "
                  f"{g.get('gap_vband_db_src_speaks')} dB; all gaps: words heard {g.get('gap_words')}"]
    print("\n".join(L))


if __name__ == "__main__":
    main()
