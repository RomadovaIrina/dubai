#!/usr/bin/env python3
"""r9 audio speed gate: a build vs the ENVELOPE of baseline draws (scripts/r8/envelope_gate.py rule: an axis fails only when worse than the
WORST draw by more than the tolerance), on the Spleeter mix AND the htdemucs mix.
    python scripts/r9/r9_gate.py --new r9fast --ids "01 02 03 04 05 m01 m04" [--draws "m01:r9_m01,r9b1_m01,r9b2_m01 ..."] [--out gate.md]
Draws:
  - 01-05: the r8 envelope (reports/experiments/r8-audio/envelope_gate_r8b5.md: 3 sequential draws, seeds as r8v4 / +1 / +2) plus the accepted
    batched build r8b5 as a 4th draw (Spleeter-mix axes; that file has no htdemucs rows);
  - m01 / m04 (and any id given in --draws): artifact tags on this server (the r9 E2E baseline r9_<id> + the extra-seed baselines r9b1_ / r9b2_).
Axes: the r8 nine (faith, coverage, gap excess, pause W1, dub over silence, WER, cut-word ends, inserted gaps, dead air) on the Spleeter mix;
on the htdemucs mix (same dub, other bed): dead air (mixdrop) and gap words (Whisper words in the mix gaps, gap_metrics), plus gap words of the
Spleeter mix. A new build's artifacts: tag <new>_<id> (+ _htd)."""
import argparse, json, pathlib, re, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "r8"))
import envelope_gate as eg

A = eg.A
MD = pathlib.Path(__file__).resolve().parents[2] / "reports/experiments/r8-audio/envelope_gate_r8b5.md"
EXTRA = [("gw_s", "gap words, Spleeter mix", 1, 0), ("dead_d", "dead air after words, htdemucs mix", 1, 0), ("gw_d", "gap words, htdemucs mix", 1, 0)]


def jl(p):
    try: return json.loads(pathlib.Path(p).read_text())
    except Exception: return None


def rows(tag, id_):
    r = eg.rows(tag, id_)
    g = jl(A / tag / f"{id_}.gaps.json"); gd = jl(A / f"{tag}_htd" / f"{id_}.gaps.json"); md = jl(A / f"{tag}_htd" / "cov" / f"mix_{id_}.json")
    r["gw_s"] = g.get("gap_words") if g else None; r["gw_d"] = gd.get("gap_words") if gd else None; r["dead_d"] = (md or {}).get("summary", {}).get("dead")
    return r


def md_draws():
    """{id: {axis key: [draw values + the r8b5 value]}} from the r8 envelope report"""
    names = {name: k for k, name, _, _ in eg.AXES}; out = {}; cur = None
    for ln in MD.read_text().splitlines():
        m = re.match(r"## (\S+) \(draws", ln)
        if m: cur = m.group(1); out[cur] = {}; continue
        if cur and ln.startswith("| ") and not ln.startswith("| axis") and not ln.startswith("|---"):
            c = [x.strip() for x in ln.strip("|").split("|")]
            if c[0] in names and c[1] != "n/a":
                vals = [float(x) for x in c[1].split(" / ")]
                try: vals.append(float(c[3]))
                except ValueError: pass
                out[cur][names[c[0]]] = vals
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--new", required=True); ap.add_argument("--ids", default="01 02 03 04 05 m01 m04")
    ap.add_argument("--draws", default=""); ap.add_argument("--out"); a = ap.parse_args()
    dr = {s.split(":")[0]: s.split(":")[1].split(",") for s in a.draws.split()}
    for id_ in ("m01", "m04"): dr.setdefault(id_, [t for t in (f"r9_{id_}", f"r9b1_{id_}", f"r9b2_{id_}") if (A / t / f"{id_}.metrics.json").exists()])
    env = md_draws(); fails = 0; md = [f"# r9 audio gate: `{a.new}` vs the envelope of the baseline draws (both mixes)", ""]; summ = []
    for id_ in a.ids.split():
        N = rows(f"{a.new}_{id_}", id_)
        if id_ in dr:
            D = [rows(t, id_) for t in dr[id_]]; src = ", ".join(dr[id_]); vals = {k: [d[k] for d in D if d.get(k) is not None] for k, *_ in eg.AXES + EXTRA}
        else:
            src = "r8 envelope (r8e2e0b / +1 / +2 seeds) + r8b5"; vals = {k: v for k, v in env.get(id_, {}).items()}
            vals.update({"gw_s": [0], "gw_d": [0], "dead_d": [N["dead"]] if N.get("dead") is not None else []})   # no htdemucs draws for 01-05: 0 gap words; dead air as on its own Spleeter mix
        md += [f"## {id_} (draws: {src})", "", "| axis | draws | envelope | new | verdict |", "|---|---|---|---:|---|"]; f_id = 0
        for k, name, sign, tol in eg.AXES + EXTRA:
            v = N.get(k); vs = vals.get(k) or []
            if v is None or not vs: md.append(f"| {name} | {' / '.join(map(str, vs)) or 'n/a'} | | {v} | n/a |"); continue
            lo, hi = min(vs), max(vs); worst = hi if sign > 0 else lo
            bad = (v > worst + tol) if sign > 0 else (v < worst - tol); better = (v < lo) if sign > 0 else (v > hi); f_id += bad
            md.append(f"| {name} | {' / '.join(str(x) for x in vs)} | {lo} .. {hi} | {v} | {'FAIL' if bad else 'better' if better else 'inside'} |")
        fails += f_id; summ.append(f"{id_}: {'PASS' if not f_id else f'FAIL {f_id}'}"); md.append("")
    md.insert(1, f"Result: {'PASS' if not fails else f'FAIL ({fails} axis rows)'} ({'; '.join(summ)})"); text = "\n".join(md) + "\n"
    if a.out: pathlib.Path(a.out).write_text(text)
    print(text); sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
