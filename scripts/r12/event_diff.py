#!/usr/bin/env python3
"""r14 F: what a candidate ADDS over a reference in the gate's event rows (old03 c8 vs F3: flashes 35 -> 51, pressed lips 5.07 -> 6.27 s).
Strong flashes (r3_flash, mc_ratio >= 2 and mc_vs_src >= 2) of the candidate with no reference flash within 2 frames; pressed-lip spans
(output closed < 0.02 while the source talks > 0.05 and the dub is silent = the gate's pressed_lips_s) and closed-while-the-DUB-is-voiced spans
present in the candidate but not the reference. Prints the lists and the top-N windows for clip_at.
    python scripts/r12/event_diff.py old03 Yf3 Yc8 [N]"""
import sys, json, pathlib
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "r7")); import r7_closed_mouth as cm
idn, ref, cand = sys.argv[1:4]; N = int(sys.argv[4]) if len(sys.argv) > 4 else 5; B = pathlib.Path("/workspace/dub/artifacts/r4m")
AUD = pathlib.Path("/workspace/dub/artifacts/r4a/r9_" + idn)
def strong(tag):
    d = json.loads((B / f"r9f_{idn}{tag}" / f"{idn}_r9f_{idn}{tag}.flash.json").read_text())
    return [e for e in d["flash_list"] if (e.get("mc_ratio") or 0) >= 2.0 and (e.get("mc_vs_src") or 0) >= 2.0]
fr, fc = strong(ref), strong(cand); rf = np.array([e["frame"] for e in fr])
new_fl = [e for e in fc if not len(rf) or np.min(np.abs(rf - e["frame"])) > 2]
so = cm.geo(np.load(cm.SRC_LM / f"{idn}_src_mp.npz")["lm"])[0]
def masks(tag):
    t = f"r9f_{idn}{tag}"; d = B / t; o = cm.geo(np.load(d / f"{idn}_{t}_out_mp.npz")["lm"])[0]
    m = json.loads((d / f"{idn}_{t}.manifest.json").read_text()); n = min(len(o), len(so)); fps = n / m["source"]["duration_s"]
    _, sil = cm.silent_spans(AUD / idn / "dubbed_16k.wav", fps, n)
    face = np.isfinite(o[:n]) & np.isfinite(so[:n]); closed = face & (o[:n] < 0.02)
    return closed & (so[:n] > 0.05) & sil, closed & ~sil, fps, n
pr, vr, fps, n = masks(ref); pc, vc, _, n2 = masks(cand); n = min(n, n2)
def spans(mask, minf=3):
    out, k = [], 0
    while k < len(mask):
        if mask[k]:
            e = k
            while e + 1 < len(mask) and mask[e + 1]: e += 1
            if e - k + 1 >= minf: out.append((round(k / fps, 2), round((e + 1) / fps, 2), e - k + 1))
            k = e + 1
        else: k += 1
    return out
new_pr = spans(pc[:n] & ~pr[:n]); new_vc = spans(vc[:n] & ~vr[:n])
print(f"== {idn} {cand} vs {ref}: strong flashes {len(fr)} -> {len(fc)}, new {len(new_fl)}")
for e in sorted(new_fl, key=lambda e: -e["mc_ratio"]): print(f"   flash {e['t']:7.2f} s  mc_ratio {e['mc_ratio']:.2f} vs_src {e['mc_vs_src']:.2f} z_lum {e['z_lum']}")
print(f"== pressed lips (closed, source talks, dub silent): {pr[:n].sum() / fps:.2f} -> {pc[:n].sum() / fps:.2f} s; new spans >= 3 frames:")
for s in sorted(new_pr, key=lambda s: -s[2]): print(f"   pressed {s[0]:7.2f}-{s[1]:.2f} s ({s[2]} frames)")
print(f"== closed while the DUB is voiced: {vr[:n].sum() / fps:.2f} -> {vc[:n].sum() / fps:.2f} s; new spans >= 3 frames:")
for s in sorted(new_vc, key=lambda s: -s[2]): print(f"   closed-voiced {s[0]:7.2f}-{s[1]:.2f} s ({s[2]} frames)")
ev = [(e["mc_ratio"] * 3, e["t"]) for e in new_fl] + [(s[2], (s[0] + s[1]) / 2) for s in new_vc] + [(s[2] * 0.5, (s[0] + s[1]) / 2) for s in new_pr]
ts = []
for _, t in sorted(ev, reverse=True):
    if all(abs(t - u) > 1.2 for u in ts): ts.append(t)
print("== windows", ",".join(f"{max(0, t - 0.6):.2f}-{t + 0.6:.2f}" for t in sorted(ts[:N])))
