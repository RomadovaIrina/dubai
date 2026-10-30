#!/usr/bin/env python3
"""r5 review: every known audio moment the user was given (briefs r4_audio_v1_known_issues.md + pb1_listen.md) re-measured on PREVIOUS and NEW
with the same 50 ms series of r4_audio_metrics.py (S = source speech, D = dub speech) and the placements of the manifests:
  "source speaks, no dub" / "dub speaks while the speaker is silent" -> seconds of S&~D / D&~S inside the moment's interval (+-0.2 s)
  "speech rushed (tempo x1.15)"                                       -> max atempo of the placements overlapping the interval
  "new / removed pause" (pb1_listen)                                   -> the longest dub gap (~D) within +-1 s of the timecode
Verdict per row: fixed (<= 25 % of PREVIOUS and < 0.3 s / tempo <= 1.05), better (<= 75 %), same, worse (> 125 %).
    python scripts/r5/r5_known_check.py --prev pb1 --new r5v2 --id 04 [--out file.md]   (prints a markdown section)"""
import argparse, json, pathlib, re
import numpy as np
A = pathlib.Path("/workspace/dub/artifacts/r4a"); B = pathlib.Path("/workspace/dub/briefs")


def sec(t: str) -> float:
    m, s = t.split(":"); return int(m) * 60 + float(s)


def moments(id_: str) -> list[dict]:
    out = []
    for name in ("r4_audio_v1_known_issues.md", "pb1_listen.md"):
        txt = (B / name).read_text(); blk = re.split(r"\n## ", txt)
        sect = next((b for b in blk if b.startswith(id_ + "\n") or b.startswith(id_ + " ")), "")
        for line in sect.splitlines():
            c = [x.strip() for x in line.strip().strip("|").split("|")]
            if name.startswith("r4") and len(c) >= 4 and re.match(r"\d+:\d", c[1]):
                a, b = (sec(x) for x in c[1].split("-")); out.append({"src": "v1 known issue", "a": a, "b": b, "what": c[3][:90]})
            elif name.startswith("pb1") and len(c) >= 2 and re.match(r"^\d+:\d", c[0]):
                t = sec(c[0]); out.append({"src": "pb1 listen", "a": t - 1.0, "b": t + 1.0, "what": c[1][:90]})
    return out


def load(tag, id_):
    z = np.load(A / tag / f"{id_}.series.npz"); man = json.loads((A / tag / id_ / f"{id_}_r4b.manifest.json").read_text())
    return z["S"].astype(bool), z["D"].astype(bool), float(z["hop"]), man


def measure(m, S, D, hop, man):
    i0, i1 = max(0, int((m["a"] - 0.2) / hop)), min(len(S), int((m["b"] + 0.2) / hop) + 1)
    w = m["what"].lower()
    if "rushed" in w:
        t = [pl.get("atempo", 1.0) for u in man["units"] for pl in (u.get("alignment") or {}).get("placements", [])
             if pl.get("placed") and pl["placed"][0] < m["b"] and pl["placed"][1] > m["a"]]
        return "tempo", max(t, default=1.0)
    if m["src"] == "pb1 listen" or "pause" in w:
        g = ~D[i0:i1]; best = cur = 0
        for x in g:
            cur = cur + 1 if x else 0; best = max(best, cur)
        return "dub gap s", round(best * hop, 2)
    if "dub speaks" in w:
        return "dub over silence s", round(float((D[i0:i1] & ~S[i0:i1]).sum() * hop), 2)
    return "source without dub s", round(float((S[i0:i1] & ~D[i0:i1]).sum() * hop), 2)


def verdict(kind, p, n):
    if kind == "tempo":
        return "fixed" if p > 1.05 and n <= 1.05 else ("same" if abs(n - p) < 0.02 else ("better" if n < p else "worse"))
    if kind == "dub gap s":
        return "same" if abs(n - p) < 0.3 else ("shorter" if n < p else "longer")
    if p < 0.1: return "ok (was ok)" if n < 0.3 else "worse"
    return "fixed" if n <= 0.25 * p and n < 0.3 else "better" if n <= 0.75 * p else "worse" if n > 1.25 * p + 0.1 else "same"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--prev", required=True); ap.add_argument("--new", required=True); ap.add_argument("--id", required=True)
    ap.add_argument("--out", default=None); a = ap.parse_args()
    P, N = load(a.prev, a.id), load(a.new, a.id)
    L = [f"## Known moments re-measured ({a.prev} -> {a.new})", "", "| time | from | what | measure | previous | new | verdict |", "|---|---|---|---|---:|---:|---|"]
    cnt = {}
    for m in moments(a.id):
        k, p = measure(m, *P); _, n = measure(m, *N); v = verdict(k, p, n); cnt[v] = cnt.get(v, 0) + 1
        tc = lambda t: f"{int(t) // 60}:{t % 60:04.1f}"
        L.append(f"| {tc(max(0, m['a']))}-{tc(m['b'])} | {m['src']} | {m['what']} | {k} | {p} | {n} | {v} |")
    L += ["", "Summary: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())), ""]
    s = "\n".join(L) + "\n"
    if a.out: pathlib.Path(a.out).write_text(s)
    print(s)


if __name__ == "__main__":
    main()
