"""a lever vs the APPROVED stack (r9fast-audio): per video and axis, ref -> new, marked when worse by more than the envelope tolerance"""
import sys
sys.path.insert(0, "/workspace/dub/dubai/scripts/r9"); import r9_gate as g
ref, new, ids = sys.argv[1], sys.argv[2], sys.argv[3].split()
worse = 0
for i in ids:
    R, N = g.rows(f"{ref}_{i}", i), g.rows(f"{new}_{i}", i); cells = []
    for k, name, sign, tol in g.eg.AXES + g.EXTRA:
        r, n = R.get(k), N.get(k)
        if r is None or n is None: continue
        w = (n > r + tol) if sign > 0 else (n < r - tol); worse += w
        cells.append(f"{k} {r}->{n}{' WORSE' if w else ''}")
    print(i, "; ".join(cells))
print(f"{new} vs {ref}: {worse} rows worse than the approved stack by more than the tolerance")
