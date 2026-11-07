#!/usr/bin/env python3
"""r14 F: a tone_flicker '== tone_flicker {json}' line on stdin -> the top-N lower-face events as clip_at windows "t0-t1,..." (±0.5 s, >1.2 s apart)."""
import sys, json
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
d = json.loads(sys.stdin.read().split("tone_flicker ", 1)[-1]); ts = []
for t, _ in sorted(d["low"]["top"], key=lambda e: -e[1]):
    if all(abs(t - u) > 1.2 for u in ts): ts.append(t)
print(",".join(f"{max(0, t - 0.5):.2f}-{t + 0.5:.2f}" for t in sorted(ts[:n])))
