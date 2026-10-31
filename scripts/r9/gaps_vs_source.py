"""02 'pauses are long': every dub silence >= 0.3 s vs the source silence at the same place.
dub speech = the dub track (dubbed_16k) above its own speech level - 35 dB (10 ms frames, 0.15 s holes closed);
source speech = the source vocal stem (vocals_44k) the same way. Per dub gap: its length, how much of it the source is ALSO silent,
excess = dub gap - source silence inside it (the source talks while the dub pauses), and the source pause nearest in time."""
import json, sys
import numpy as np, soundfile as sf

def speech(path, hop=0.01, rel=35.0, close=0.15):
    y, sr = sf.read(path); y = y.mean(1) if y.ndim > 1 else y; h = int(hop * sr); n = len(y) // h
    db = 20 * np.log10(np.sqrt((y[: n * h].reshape(n, h) ** 2).mean(1)) + 1e-9); ref = np.percentile(db[db > -70], 90) if (db > -70).any() else -20
    on = db > ref - rel; k = int(close / hop); i = 0
    while i < n:   # close short holes
        if not on[i]:
            j = i
            while j < n and not on[j]: j += 1
            if 0 < i and j < n and j - i <= k: on[i:j] = True
            i = j
        else: i += 1
    return on, hop

def gaps(on, hop, min_s=0.3):
    out, i, n = [], 0, len(on)
    while i < n:
        if not on[i]:
            j = i
            while j < n and not on[j]: j += 1
            if (j - i) * hop >= min_s and i > 0 and j < n: out.append((i * hop, j * hop))
            i = j
        else: i += 1
    return out

dub, h = speech(sys.argv[1]); src, _ = speech(sys.argv[2]); n = min(len(dub), len(src)); dub, src = dub[:n], src[:n]
rows = []
for a, b in gaps(dub, h):
    i0, i1 = int(a / h), int(b / h); ssil = float((~src[i0:i1]).sum() * h); rows.append((a, b, b - a, ssil, (b - a) - ssil))
tot = sum(r[2] for r in rows); ex = sum(r[4] for r in rows)
sg = gaps(src, h)
print(f"{sys.argv[3]}: dub gaps >= 0.3 s: {len(rows)}, total {tot:.1f} s; source silent inside them {sum(r[3] for r in rows):.1f} s; EXCESS (source talks, dub pauses) {ex:.1f} s;"
      f" source pauses >= 0.3 s: {len(sg)}, total {sum(b - a for a, b in sg):.1f} s")
for a, b, L, s, e in rows:
    if L >= 0.8 or e >= 0.3: print(f"   {int(a // 60)}:{a % 60:05.2f}-{int(b // 60)}:{b % 60:05.2f}  dub gap {L:4.1f}  source silent {s:4.1f}  excess {e:4.1f}")
