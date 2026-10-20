#!/usr/bin/env python3
"""r4: listening package for a NOT gate-passing candidate next to the published version (user decides by ear):
  outbox/<id>_r4_audio_<cand>_LISTEN_over-original.mp4, outbox/<id>_r4_audio_<prev>-vs-<cand>_AB.mp4 (2-3 excerpts of --len s where the pauses differ
  most: the excerpt with PREVIOUS audio, then the same excerpt with the candidate, labelled), outbox/<cand>_listen.md (5 most-changed pauses per video
  + the added "source talks, dub silent" seconds).
    python scripts/quality/r4_listen_pack.py --prev cand2 --prev-label v1 --cand pb1 [--len 18]"""
import argparse, json, pathlib, subprocess
A = pathlib.Path("/workspace/dub/artifacts/r4a"); OB = pathlib.Path("/workspace/dub/outbox"); F = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def changes(pr, nr):
    used, ch = set(), []
    for p in pr:
        j = min((k for k in range(len(nr)) if k not in used and abs(nr[k]["t"] - p["t"]) <= 1.0), key=lambda k: abs(nr[k]["t"] - p["t"]), default=None)
        if j is None: ch.append((p["gap_s"], p["t"], f"removed: {p['gap_s']:.2f} s ({p['class']}) ...{p['before']} | {p['after']}..."))
        else:
            used.add(j); q = nr[j]; ch.append((abs(q["gap_s"] - p["gap_s"]), q["t"], f"{p['gap_s']:.2f} -> {q['gap_s']:.2f} s ({p['class']} -> {q['class']}) ...{q['before']} | {q['after']}..."))
    for k, q in enumerate(nr):
        if k not in used: ch.append((q["gap_s"], q["t"], f"new: {q['gap_s']:.2f} s ({q['class']}) ...{q['before']} | {q['after']}..."))
    return sorted(ch, key=lambda x: -x[0])


def tc(t): return f"{int(t) // 60}:{t % 60:04.1f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prev", required=True); ap.add_argument("--prev-label", required=True); ap.add_argument("--cand", required=True)
    ap.add_argument("--len", type=float, default=18.0); a = ap.parse_args()
    L = [f"# {a.cand}: LISTENING candidate (NOT gate-passing) next to {a.prev_label}", "",
         f"`{a.cand}` = {a.prev_label} + `--pause-boundary-only on`: inside a sentence the dub pauses only after punctuation or before a clause word",
         "(and / but / so / which / because / that ...); any other source pause is not reproduced and the phrase runs through it.",
         f"Gate vs {a.prev_label}: pauses inside an English phrase much fewer, but the dub drifts from the source timing (see the seconds below) -> NOT published as v2.", "",
         f"Files: `<id>_r4_audio_{a.cand}_LISTEN_over-original.mp4` (whole video), `<id>_r4_audio_{a.prev_label}-vs-{a.cand}_AB.mp4` (excerpts: {a.prev_label} first, then {a.cand}).", ""]
    for id_ in ["01", "02", "03", "04", "05"]:
        pm, nm = (json.loads((A / t / f"{id_}.metrics.json").read_text()) for t in (a.prev, a.cand))
        ch = changes(pm.get("pause_rows") or [], nm.get("pause_rows") or [])
        src = next(pathlib.Path("/workspace/dub/inputs/test_videos_raw").glob(f"{id_}.*"))
        subprocess.run(["cp", str(A / a.cand / f"{id_}_over-original.mp4"), str(OB / f"{id_}_r4_audio_{a.cand}_LISTEN_over-original.mp4")], check=True)
        dur = pm["duration_s"]; ex = []
        for _, t, _ in ch:   # excerpts centred on the biggest changes, non-overlapping
            s0 = max(0.0, min(dur - a.len, t - a.len / 2))
            if all(abs(s0 - e) >= a.len for e in ex): ex.append(s0)
            if len(ex) == 3: break
        ex.sort(); parts = []; work = A / a.cand / "_ab"; work.mkdir(exist_ok=True)
        for k, s0 in enumerate(ex):
            for lab, t in ((a.prev_label, a.prev), (a.cand, a.cand)):
                o = work / f"{id_}_{k}_{lab}.mp4"
                txt = f"{lab.upper()}  {tc(s0)}-{tc(s0 + a.len)}"
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{s0:.2f}", "-t", f"{a.len}", "-i", str(A / t / f"{id_}_over-original.mp4"),
                                "-vf", f"scale=-2:720,drawtext=fontfile={F}:text='{txt}':x=20:y=20:fontsize=40:fontcolor=yellow:box=1:boxcolor=black@0.6",
                                "-c:v", "libx264", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2", str(o)], check=True)
                parts.append(o)
        lst = work / f"{id_}.txt"; lst.write_text("".join(f"file '{p}'\n" for p in parts))
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c", "copy", str(OB / f"{id_}_r4_audio_{a.prev_label}-vs-{a.cand}_AB.mp4")], check=True)
        d_src = nm["src_speaks_dub_silent_s"] - pm["src_speaks_dub_silent_s"]; d_dub = nm["dub_speaks_src_silent_s"] - pm["dub_speaks_src_silent_s"]
        pi, ni = (x.get("pause_logic") or {} for x in (pm, nm))
        L += [f"## {id_}", "", f"- pauses inside an English phrase: {pi.get('pauses_inside_phrase')} ({pi.get('inside_phrase_s')} s) -> {ni.get('pauses_inside_phrase')} ({ni.get('inside_phrase_s')} s)",
              f"- source talks, dub silent: {pm['src_speaks_dub_silent_s']} -> {nm['src_speaks_dub_silent_s']} s ({d_src:+.1f} s); dub talks, source silent: {d_dub:+.1f} s",
              f"- A/B excerpts: " + ", ".join(f"{tc(s0)}-{tc(s0 + a.len)}" for s0 in ex), "", "| time | pause change (v1 -> pb1; INSIDE = inside an English phrase) |", "|---|---|"]
        L += [f"| {tc(t)} | {d} |" for _, t, d in ch[:5]] + [""]
    (OB / f"{a.cand}_listen.md").write_text("\n".join(L) + "\n"); print(OB / f"{a.cand}_listen.md")


if __name__ == "__main__":
    main()
