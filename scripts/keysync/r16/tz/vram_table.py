#!/usr/bin/env python3
"""r16 TZ 0.4 / 0.5 on KeySync: per-stage VRAM and GPU time of finished jobs, from the job's own samplers (no GPU needed):
  work/stages.tsv (e2e stage starts), logs/face_keysync.log (KS_T sub-stage marks of the face stage: master ... engines, probe, full, check,
  post, paste), work/nvml.jsonl (10 Hz: total GPU memory, util and the memory of THIS job's processes, so another tenant on the GPU does
  not pollute the job's numbers).
Per stage: wall s, the job's peak / mean VRAM (MiB), the GPU's total peak, mean util, GPU-busy s (util > 0 samples x 0.1 s).
    python vram_table.py E2E_DIR [E2E_DIR ...] --out table.json [--md table.md]"""
import argparse, json, pathlib, re, sys


def stages(e2e):
    w = pathlib.Path(e2e) / "work"; st = []
    for ln in (w / "stages.tsv").read_text().split("\n"):
        if "\t" in ln: n, t = ln.split("\t")[:2]; st.append((n, float(t)))
    ks = pathlib.Path(e2e) / "logs" / "face_keysync.log"
    sub = []
    if ks.exists():
        for m in re.finditer(r"^KS_T (\S+) (\d+\.\d+)", ks.read_text(errors="replace"), re.M): sub.append(("ks:" + m.group(1), float(m.group(2))))
    return st, sub


def rows(e2e, job):
    w = pathlib.Path(e2e) / "work"; S = [json.loads(l) for l in open(w / "nvml.jsonl") if l.strip()]
    st, sub = stages(e2e); t_end = S[-1]["t"] if S else 0
    def agg(name, a, b):
        xs = [s for s in S if a <= s["t"] < b]
        if not xs: return None
        jm = [sum(v for k, v in (s.get("proc_mem_mib") or {}).items() if job is None or k == job) for s in xs]
        return {"stage": name, "wall_s": round(b - a, 1), "job_vram_peak_mib": max(jm), "job_vram_mean_mib": round(sum(jm) / len(jm)),
                "gpu_total_peak_mib": max(s["mem_mib"] for s in xs), "util_mean": round(sum(s["util"] for s in xs) / len(xs), 1),
                "gpu_busy_s": round(sum(0.1 for s in xs if s["util"] > 0), 1)}
    out = []
    for i, (n, t) in enumerate(st):
        b = st[i + 1][1] if i + 1 < len(st) else t_end; r = agg(n, t, b)
        if r: out.append(r)
        if n == "face_keysync":
            for j, (sn, stt) in enumerate(sub):
                sb = sub[j + 1][1] if j + 1 < len(sub) else b; r = agg(sn, stt, min(sb, b))
                if r: out.append(r)
    tot = agg("TOTAL", st[0][1], t_end) if st else None
    return out, tot


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+"); ap.add_argument("--out", required=True); ap.add_argument("--md"); a = ap.parse_args()
    res, md = {}, []
    for d in a.dirs:
        p = pathlib.Path(d); job = None
        try: job = json.loads((p.parent / "status.json").read_text())["job_id"]
        except Exception: pass
        r, tot = rows(p, job); res[str(p)] = {"job": job, "stages": r, "total": tot}
        md += [f"\n### {job or p}\n", "| stage | wall s | job VRAM peak MiB | job VRAM mean MiB | GPU total peak MiB | util % | GPU-busy s |", "|---|---|---|---|---|---|---|"]
        for x in r + ([tot] if tot else []):
            md.append(f"| {x['stage']} | {x['wall_s']} | {x['job_vram_peak_mib']} | {x['job_vram_mean_mib']} | {x['gpu_total_peak_mib']} | {x['util_mean']} | {x['gpu_busy_s']} |")
    json.dump(res, open(a.out, "w"), indent=1)
    if a.md: open(a.md, "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
