#!/usr/bin/env python3
"""r8-sep: one gate table per video for r8v4c and the sep_mix tags: gap words / leaked voice (gap_metrics), dead air (mixdrop), new clicks vs r8v4c
(clicks.py), bed naturalness in the dub gaps (mix_nat.json).
    python scripts/r8/sep_table.py --nat /workspace/dub/r8a/sep_mix_nat.json --tags r8v4c sepH0 ... [--md out.md]"""
import argparse, json, pathlib
A = pathlib.Path("/workspace/dub/artifacts/r4a")


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tags", nargs="+", required=True); ap.add_argument("--nat"); ap.add_argument("--md"); a = ap.parse_args()
    nat = json.loads(pathlib.Path(a.nat).read_text()) if a.nat and pathlib.Path(a.nat).exists() else {}
    H = ["tag", "gap words", "gap voice s (src speaks)", "gap burst s", "gap vband dB (src speaks)", "dead / abrupt", "clicks in gaps (new / new not in source)", "cont dB", "shape dB", "air dB (src)"]
    out = []
    for id_ in ["01", "02", "03", "04", "05"]:
        out += [f"\n### {id_}\n", "| " + " | ".join(H) + " |", "|" + "---|" * len(H)]
        for t in a.tags:
            g = json.loads((A / t / f"{id_}.gaps.json").read_text()) if (A / t / f"{id_}.gaps.json").exists() else {}
            g = g.get("new", g) if isinstance(g, dict) and "new" in g else g
            m = json.loads((A / t / "cov" / f"mix_{id_}.json").read_text())["summary"] if (A / t / "cov" / f"mix_{id_}.json").exists() else {}
            c = json.loads((A / t / "cov" / "clicks.json").read_text()).get(id_, {}) if (A / t / "cov" / "clicks.json").exists() else {}
            n = nat.get(id_, {}).get(t, {})
            out.append(f"| {t} | {g.get('gap_words', '?')} | {g.get('gap_voice_s', '?')} ({g.get('gap_voice_s_src_speaks', '?')}) | {g.get('gap_burst_s', '?')} | "
                       f"{g.get('gap_vband_db', '?')} ({g.get('gap_vband_db_src_speaks', '?')}) | {m.get('dead', '?')} / {m.get('abrupt', '?')} | "
                       f"{len(n.get('gap_clicks', []))} ({len(n['new_gap_clicks']) if 'new_gap_clicks' in n else '-'}/{len(n['new_gap_clicks_not_in_src']) if 'new_gap_clicks_not_in_src' in n else '-'}) | {n.get('cont_db', '?')} | {n.get('shape_db', '?')} | {n.get('air_db', '?')} ({n.get('src_air_db', '?')}) |")
    s = "\n".join(out); print(s)
    if a.md: pathlib.Path(a.md).write_text(s + "\n")


if __name__ == "__main__":
    main()
