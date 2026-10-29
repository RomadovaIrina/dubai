import sys, json, pathlib, numpy as np
sys.path.insert(0, "/workspace/dub/dubai/scripts/r7")
from r7_mouth_probe import silent_spans, geo, B, SRC_LM
id_ = sys.argv[1]
so = geo(np.load(SRC_LM / f"{id_}_src_mp.npz")["lm"])[0]
for tag, aud in [a.split(":") for a in sys.argv[2:]]:
    d = B / tag; lm = np.load(d / f"{id_}_{tag}_out_mp.npz")["lm"]; man = json.loads((d / f"{id_}_{tag}.manifest.json").read_text())
    fps = len(lm) / man["source"]["duration_s"]; o = geo(lm)[0]; n = min(len(o), len(so))
    spans, sil = silent_spans(pathlib.Path(aud) / id_ / "dubbed_16k.wav", fps, n)
    sp = [(float(s.get("start", s[0] if isinstance(s, list) else 0)), float(s.get("end", s[1] if isinstance(s, list) else 0))) if isinstance(s, dict) else (float(s[0]), float(s[1])) for s in man.get("speech_intervals", [])]
    m = sil[:n] & np.isfinite(o[:n]) & np.isfinite(so[:n]) & (o[:n] < 0.02) & (so[:n] > 0.05)
    print(f"== {id_} {tag}: pressed {m.sum() / fps:.2f} s; dub-silent spans >= 0.3 s: {len(spans)} total {sum(b - a for a, b in spans):.2f} s")
    k = 0; rows = []
    while k < n:
        if not m[k]: k += 1; continue
        j = k
        while j < n and (m[j] or (j + 1 < n and m[j + 1])): j += 1
        t0, t1 = k / fps, j / fps; ov = sum(max(0, min(t1, b) - max(t0, a)) for a, b in sp)
        sil_span = next(((a, b) for a, b in spans if a <= t0 + 0.05 and b >= t1 - 0.05), None)
        rows.append((t0, t1, ov, sil_span)); k = j
    for t0, t1, ov, ss in rows:
        print(f"   {t0:6.2f}-{t1:6.2f} ({t1 - t0:.2f} s)  source speech overlap {ov:.2f} s  dub-silent span {('%.2f-%.2f (%.2f s)' % (ss[0], ss[1], ss[1] - ss[0])) if ss else '?'}")
