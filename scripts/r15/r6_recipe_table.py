#!/usr/bin/env python3
"""r6: the lip-recipe table (scripts/r15/r6_recipe.sh outputs): sync_speech, mouth opening / speed vs q2u + recipe, F's lip_flicker
(W excess dL, W jumps over the source p99 [the source's own], blink excess per 10 s) per video and arm.
    python scripts/r15/r6_recipe_table.py --root /workspace/dub/r15x --arms q2u s4p1500c8u r6c8u --ids old03 m03"""
import argparse, json, pathlib, re


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--ids", nargs="+", required=True); a = ap.parse_args(); R = pathlib.Path(a.root); E = R / "eval"
    lf = {}
    for l in (R / "lipflick_r6.jsonl").read_text().splitlines() if (R / "lipflick_r6.jsonl").exists() else []:
        try: d = json.loads(l); lf[d["label"]] = d   # the last line per label wins
        except Exception: pass
    print("### With agent F's lip recipe (CF off + adaptive lipcon 0.45 + lip colour lock r5 + flow median r2), same server\n")
    print("| video | arm | sync_speech | opening vs q2u+recipe (mean / speed) | lip flicker W excess dL | W jumps [src] | blink excess / 10 s |\n|---|---|---|---|---|---|---|")
    for v in a.ids:
        try: mo = json.load(open(R / f"recipe_{v}.mouth.json"))
        except Exception: mo = {}
        for arm in a.arms:
            t = f"{arm}_{v}rec"; st = E / f"{t}.sync.txt"; sy = re.search(r": ([-0-9.]+) over", st.read_text()) if st.exists() else None
            m = mo.get(f"{arm}rec", {}); vs = m.get(f"vs_{a.arms[0]}rec", {}); op = "1" if arm == a.arms[0] else (f"{vs.get('mean', 0):.3f} / {vs.get('speed', 0):.3f}" if vs else "–")
            d = lf.get(t, {})
            print(f"| {v} | {arm} + recipe | {sy.group(1) if sy else '–'} | {op} | {d.get('W_excess_dL', '–')} | {d.get('W_jumps_p99', '–')} [{d.get('W_src_jumps_p99', '–')}] | {d.get('excess_per_10s', '–')} |")


if __name__ == "__main__":
    main()
