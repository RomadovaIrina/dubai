#!/usr/bin/env python3
"""Round 4 review sheet outbox/<id>_review.md: one row per 10 s, PREVIOUS vs NEW, columns voice / background / pauses / mouth / face / other.
Filled automatically from the two r4_audio_metrics.py frame series (--series) + the run manifests; rows where something changed are marked
"LOOK", so the listener only confirms or refutes. Audio round: mouth / face are untouched (new audio over the original video).
    python scripts/quality/r4_review.py --id 04 --prev-series P.npz --new-series N.npz --new-manifest N.json [--prev-manifest P.json] --out R.md --title ..."""
import argparse, json, pathlib
import numpy as np

W = 10.0


def db(p): return float(10 * np.log10(np.mean(p) + 1e-12)) if np.size(p) else None


def units_in(man, a, b):
    if not man: return []
    return [u for u in man["units"] if a <= u.get("start", 0) < b]


def text_of(u):
    return u.get("translation_fitted") or u.get("translation") or ""


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--id", required=True); ap.add_argument("--prev-series", required=True); ap.add_argument("--new-series", required=True)
    ap.add_argument("--new-manifest", required=True); ap.add_argument("--prev-manifest", default=None); ap.add_argument("--out", required=True); ap.add_argument("--title", default="")
    a = ap.parse_args()
    P, N = np.load(a.prev_series), np.load(a.new_series); hop = float(N["hop"]); nf = min(len(P["S"]), len(N["S"]))
    nm = json.loads(pathlib.Path(a.new_manifest).read_text()); pm = json.loads(pathlib.Path(a.prev_manifest).read_text()) if a.prev_manifest else None
    S = N["S"][:nf]; L = [f"# {a.id} review sheet {a.title}", "",
                          "PREVIOUS vs NEW, one row per 10 s. LOOK = a measured change worth listening to. Mouth / face: original video, untouched (audio round).",
                          "exposed = the source speaks while the dub is silent (the bed, incl. the original voice residue, is heard alone).", "",
                          "| time | voice | background | pauses | mouth | face | other |", "|---|---|---|---|---|---|---|"]
    look = 0
    for k in range(int(np.ceil(nf * hop / W))):
        i0, i1 = int(k * W / hop), min(nf, int((k + 1) * W / hop)); sl = slice(i0, i1); flags = []
        # voice: dub speech seconds and the sentences that start here
        dp, dn = P["D"][sl].sum() * hop, N["D"][sl].sum() * hop
        nu = units_in(nm, k * W, (k + 1) * W); pu = units_in(pm, k * W, (k + 1) * W)
        changed = [u for u in nu if pm and text_of(u).strip().lower() != next((text_of(v) for v in pu if v["id"] == u["id"]), "").strip().lower()]
        voice = f"dub {dp:.1f} -> {dn:.1f} s"
        if changed: voice += "; new text: " + " / ".join(f'"{text_of(u)[:60]}"' for u in changed[:2]); flags.append("text")
        if abs(dn - dp) > 1.0: flags.append("dub time")
        # background: bed speech-band level where exposed (rel. the dub voice), bed level in joint pauses
        def expo(X):
            sel = X["S"][sl] & ~X["D"][sl]; v = db(X["p_bed_sb"][:nf][sl][sel]); return (v - float(X["voice"])) if v is not None else None
        def paus(X):
            sel = ~X["S"][sl] & ~X["D"][sl]; v = db(X["p_mix"][:nf][sl][sel]); return (v - float(X["voice"])) if v is not None else None
        ep, en, pp, pn = expo(P), expo(N), paus(P), paus(N); bg = []
        if ep is not None or en is not None:
            bg.append(f"exposed residue {'' if ep is None else f'{ep:.0f}'} -> {'' if en is None else f'{en:.0f}'} dB")
            if ep is not None and en is not None and abs(en - ep) > 2: flags.append("residue")
        if pp is not None and pn is not None:
            bg.append(f"bed in pauses {pp:.0f} -> {pn:.0f} dB"); flags += ["pause bed"] if abs(pn - pp) > 2 else []
        # pauses: source talks / dub silent, dub talks / source silent
        xp, xn = (P["S"][sl] & ~P["D"][sl]).sum() * hop, (S[sl] & ~N["D"][sl]).sum() * hop
        yp, yn = (~P["S"][sl] & P["D"][sl]).sum() * hop, (~S[sl] & N["D"][sl]).sum() * hop
        pauses = f"src-only {xp:.1f} -> {xn:.1f} s, dub-only {yp:.1f} -> {yn:.1f} s"
        if abs(xn - xp) > 1.0 or abs(yn - yp) > 1.0: flags.append("placement")
        other = []
        for u in nu:
            for e in u.get("ne") or []: other.append(f"name '{e['english']}' (u{u['id']})")
        if other: flags.append("name")
        look += bool(flags)
        t = f"{int(k * W) // 60}:{int(k * W) % 60:02d}-{int((k + 1) * W) // 60}:{int((k + 1) * W) % 60:02d}"
        L.append(f"| {t}{' **LOOK** (' + ', '.join(flags) + ')' if flags else ''} | {voice} | {'; '.join(bg)} | {pauses} | unchanged | unchanged | {'; '.join(other)} |")
    L += ["", f"{look} of {int(np.ceil(nf * hop / W))} rows marked LOOK."]
    pathlib.Path(a.out).write_text("\n".join(L) + "\n"); print(a.out, look)


if __name__ == "__main__":
    main()
