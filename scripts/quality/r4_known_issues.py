#!/usr/bin/env python3
"""Round 4 A: the N worst remaining audio moments per video of a published audio version (timecode + what is wrong), ranked by a severity
score from the version's own measurements, so the user only confirms or refutes:
  source talks, dub silent (run >= 0.4 s)        severity = seconds (x1.5 if the bed's speech band there is within 20 dB of the dub voice)
  dub talks over source silence (run >= 0.4 s)   severity = seconds x 0.7
  TTS QA flagged the unit (qa_bad_final)         severity 3 (+ the QA reasons)
  tempo: atempo >= 1.15 (rushed)                 severity 1.5
  hard cut (stop / restart within 0.6 s)         severity 2
  named entity not heard in the final mix        severity 3
    python scripts/quality/r4_known_issues.py --tag cand2 --ver v1 --out /workspace/dub/outbox/r4_audio_v1_known_issues.md [--n 10]"""
import argparse, json, pathlib, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "pilot"))
A = pathlib.Path("/workspace/dub/artifacts/r4a")


def tc(t): return f"{int(t) // 60}:{t % 60:04.1f}"


def runs(m, hop, min_s):
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]])); on, off = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(a * hop, b * hop) for a, b in zip(on, off) if (b - a) * hop >= min_s]


def unit_at(man, t):
    return next((u for u in man["units"] if u["start"] - 0.3 <= t <= u["end"] + 0.3), None)


def issues(tag, id_):
    S_ = np.load(A / tag / f"{id_}.series.npz"); hop = float(S_["hop"]); S, D = S_["S"], S_["D"]; voice = float(S_["voice"])
    man = json.loads((A / tag / id_ / f"{id_}_r4b.manifest.json").read_text()); met = json.loads((A / tag / f"{id_}.metrics.json").read_text()); out = []
    bed_db = 10 * np.log10(S_["p_bed_sb"] + 1e-12) - voice
    for a, b in runs(S & ~D, hop, 0.4):
        lvl = float(np.median(bed_db[int(a / hop):int(b / hop)])); u = unit_at(man, a); loud = lvl > -20
        what = f"source speaks {b - a:.1f} s with no dub" + (f"; the original voice / bed is audible ({lvl:.0f} dB rel. dub voice)" if loud else "")
        if u: what += f" (sentence u{u['id']}: \"{(u.get('translation_fitted') or u.get('translation') or '')[:50]}\")"
        out.append(((b - a) * (1.5 if loud else 1.0), a, b, "pause / bed", what))
    for a, b in runs(D & ~S, hop, 0.4):
        out.append(((b - a) * 0.7, a, b, "pause", f"dub speaks {b - a:.1f} s while the speaker is silent (lips closed)"))
    for u in man["units"]:
        pl = (u.get("alignment") or {}).get("placements") or []
        t0, t1 = (pl[0]["placed"][0], pl[-1]["placed"][1]) if pl else (u["start"], u["end"])
        rs = [r for p in (u.get("tts_parts") or [])[:1] for a_ in (p.get("attempts") or [])[-1:] for r in (a_.get("reasons") or [])]
        if any(p.get("qa_bad_final") for p in u.get("tts_parts") or []):
            out.append((3.0, t0, t1, "voice", f"TTS QA flags u{u['id']} as hard to understand ({'; '.join(rs)[:80]}): \"{(u.get('translation_fitted') or u.get('translation') or '')[:60]}\""))
        for p in pl:
            at = p.get("atempo", 1.0)
            if at >= 1.15:   # 0.875 = the r2 slow-down floor, used on most short parts: not an outlier
                out.append((1.5, p["placed"][0], p["placed"][1], "voice", f"speech rushed (tempo x{at:.2f}) in u{u['id']}"))
    import r3_hardcut as hc
    ev = hc.events(A / tag / id_ / "dubbed_24k.wav")
    for e1, e2 in zip(ev, ev[1:]):
        if e1[0] == "OFF" and e2[0] == "ON" and e2[1] - e1[1] <= 0.6: out.append((2.0, e1[1], e2[1], "voice", "voice stops dead and restarts (hard cut)"))
    for x in (met.get("ne") or {}).get("list") or []:
        uid = int(x.split(":")[0][1:]); u = next((v for v in man["units"] if v["id"] == uid), None)
        if u: out.append((3.0, u["start"], u["end"], "name", f"name '{x.split(':', 1)[1]}' not heard in the final mix"))
    return sorted(out, key=lambda r: -r[0])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--ver", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=10); a = ap.parse_args()
    L = [f"# r4_audio_{a.ver}: known remaining audio issues (top {a.n} per video, automatic ranking)", "",
         f"Files: `outbox/<id>_r4_audio_{a.ver}_over-original.mp4`. Please mark each row: confirmed / not heard. Severity = seconds affected, weighted by type.", ""]
    for id_ in ["01", "02", "03", "04", "05"]:
        L += [f"## {id_}", "", "| # | time | type | what | severity | confirmed? |", "|---|---|---|---|---:|---|"]
        for k, (sev, t0, t1, typ, what) in enumerate(issues(a.tag, id_)[: a.n], 1):
            L.append(f"| {k} | {tc(t0)}-{tc(t1)} | {typ} | {what} | {sev:.1f} | |")
        L.append("")
    pathlib.Path(a.out).write_text("\n".join(L) + "\n"); print(a.out)


if __name__ == "__main__":
    main()
