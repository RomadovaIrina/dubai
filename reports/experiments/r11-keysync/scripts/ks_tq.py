#!/usr/bin/env python3
"""tqdm totals per bar name in a KeySync run.log (seconds): ks_tq.py run.log"""
import re, sys
order = []; cur = None
for l in open(sys.argv[1], errors="ignore").read().replace("\r", "\n").splitlines():
    m = re.match(r"(.*?):\s+\d+%\|.*?\| *(\d+)/(\d+) \[(\d+):(\d+)(?::(\d+))?<", l)
    if not m: continue
    name, n = m.group(1).strip(), int(m.group(2))
    s = int(m.group(4)) * 60 + int(m.group(5)) if m.group(6) is None else int(m.group(4)) * 3600 + int(m.group(5)) * 60 + int(m.group(6))
    if n == 0 or cur is None or cur[0] != name or n < cur[1]: order.append([name, 0])
    order[-1][1] = s; cur = (name, n)
tot = {}
for k, s in order: tot.setdefault(k, [0, 0]); tot[k][0] += s; tot[k][1] += 1
for k, v in tot.items(): print(f"{k[:50]:50s} bars {v[1]:3d}  s {v[0]}")
