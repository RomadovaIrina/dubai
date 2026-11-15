import sys, re
# segcmp2.py <release ks_fast.log> <floorv ks_fast.log>: 8-step segments per generator job (probe, then the render or its chunks), as (job, segment)
def segs(p):
    S = set(); job = -1
    for ln in open(p, errors="ignore"):
        m = re.match(r"KS_GATE_SEG off (\d+) hi \[(.*)\]", ln)
        if not m: continue
        off = int(m.group(1)); job += int(off == 0)
        if job >= 1: S |= {(job, off + int(x)) for x in m.group(2).split(",") if x.strip()}   # job 0 = the delay probe
    return S
a, b = segs(sys.argv[1]), segs(sys.argv[2])
print("release %d segments, floorv %d | both %d | only release %s | only floorv %s" % (len(a), len(b), len(a & b), sorted(a - b), sorted(b - a)))
