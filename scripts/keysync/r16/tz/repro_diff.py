#!/usr/bin/env python3
"""r16 TZ reproducibility: two audio-stage dirs of the SAME input (artifacts/r4a/r9_<id>/<id>), compared stage by stage in pipeline order,
so the FIRST stage that differs is the source of the drift: source decode -> separation -> ASR -> diarization -> translation -> speaker refs
-> TTS takes -> alignment -> dub track -> final mix. A wav that differs also gets its max |diff| and SNR (a tiny numeric difference vs another take).
    python repro_diff.py DIR_A DIR_B [--out diff.json]"""
import argparse, glob, hashlib, json, os, sys
import numpy as np


def md5(p): return hashlib.md5(open(p, "rb").read()).hexdigest() if os.path.exists(p) else None


def wav(p):
    import soundfile as sf
    x, sr = sf.read(p, dtype="float32", always_2d=True); return x, sr


def wcmp(a, b):
    if md5(a) == md5(b): return {"same": True}
    try:
        x, sa = wav(a); y, sb = wav(b); n = min(len(x), len(y)); d = x[:n] - y[:n]
        snr = 10 * np.log10((x[:n] ** 2).sum() / max((d ** 2).sum(), 1e-20))
        return {"same": False, "len_a": len(x), "len_b": len(y), "max_abs": round(float(np.abs(d).max()), 6), "snr_db": round(float(snr), 1)}
    except Exception as e: return {"same": False, "err": repr(e)[:120]}


def man(d): f = sorted(glob.glob(os.path.join(d, "*_r4b.manifest.json")) or glob.glob(os.path.join(d, "*.manifest.json"))); return json.load(open(f[0])) if f else {}


def units(m, keys): return [{k: u.get(k) for k in keys} for u in m.get("units", [])]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("--out"); a = ap.parse_args()
    A, B = man(a.a), man(a.b); res = []
    def step(name, same, detail=None): res.append({"stage": name, "same": bool(same), **({"detail": detail} if detail is not None else {})})
    for f in ("audio16k.wav", "vocals_44k.wav", "bg_44k.wav"):
        r = wcmp(os.path.join(a.a, f), os.path.join(a.b, f)); step(f"1 source / separation: {f}", r["same"], None if r["same"] else r)
    ua, ub = units(A, ("start", "end", "text")), units(B, ("start", "end", "text")); step("2 ASR units (start, end, text)", ua == ub, None if ua == ub else {"a": ua[:3], "b": ub[:3]})
    wa, wb = units(A, ("words",)), units(B, ("words",)); step("2b ASR word timings", wa == wb)
    sa, sb = units(A, ("speaker",)), units(B, ("speaker",)); step("3 diarization (speaker per unit)", sa == sb, None if sa == sb else {"a": sa, "b": sb})
    ta, tb = units(A, ("translation",)), units(B, ("translation",)); step("4 translation (Qwen)", ta == tb, None if ta == tb else [(x, y) for x, y in zip(ta, tb) if x != y][:3])
    pa, pb = units(A, ("parts", "dur_translate", "gap_qa", "split")), units(B, ("parts", "dur_translate", "gap_qa", "split")); step("4b duration-fit rewrites / parts", pa == pb)
    for f in sorted(set(os.path.basename(x) for x in glob.glob(os.path.join(a.a, "ref_*.wav")))):
        r = wcmp(os.path.join(a.a, f), os.path.join(a.b, f)); step(f"5 speaker ref {f}", r["same"], None if r["same"] else r)
    xa, xb = units(A, ("tts",)), units(B, ("tts",)); step("6 TTS take metadata (seed / temperature / attempts / QA)", xa == xb, None if xa == xb else [(x, y) for x, y in zip(xa, xb) if x != y][:1])
    for pat in ("tts_u*_raw.wav", "tts_u*_p*.wav", "aligned_u*.wav"):
        fs = sorted(set(os.path.basename(x) for x in glob.glob(os.path.join(a.a, pat))) | set(os.path.basename(x) for x in glob.glob(os.path.join(a.b, pat))))
        rs = {f: wcmp(os.path.join(a.a, f), os.path.join(a.b, f)) for f in fs}; diff = {f: r for f, r in rs.items() if not r["same"]}
        step(f"7 {pat} ({len(fs)} files)", not diff, None if not diff else {"differ": len(diff), "first": dict(list(diff.items())[:3])})
    for f in ("dubbed_16k.wav", "dubbed_24k.wav", "final_mix_44k.wav", "final_mix_aac.m4a"):
        r = wcmp(os.path.join(a.a, f), os.path.join(a.b, f)) if f.endswith(".wav") else {"same": md5(os.path.join(a.a, f)) == md5(os.path.join(a.b, f))}
        step(f"8 {f}", r["same"], None if r["same"] else r)
    first = next((r["stage"] for r in res if not r["same"]), None)
    # the whole r9_<id> tree (placed / hushed / soft / mix_src / the id dir): every differing file, in the order run A wrote them (mtime =
    # pipeline order), names normalised by the job id -> the earliest differing file is where the drift enters
    pa, ia = os.path.dirname(os.path.abspath(a.a)), os.path.basename(os.path.abspath(a.a)); pb, ib = os.path.dirname(os.path.abspath(a.b)), os.path.basename(os.path.abspath(a.b))
    tree = []
    if os.path.basename(pa).startswith("r9_") and os.path.basename(pb).startswith("r9_"):
        for root, _, fs in os.walk(pa):
            for f in fs:
                fa = os.path.join(root, f); rel = os.path.relpath(fa, pa); fb = os.path.join(pb, rel.replace(ia, ib))
                if not os.path.exists(fb) or f.endswith((".log", ".mp4")): continue
                if md5(fa) == md5(fb): continue
                r = wcmp(fa, fb) if f.endswith(".wav") else {"same": False}
                if f.endswith(".json"):
                    try: r["json_equal"] = json.load(open(fa)) == json.load(open(fb))
                    except Exception: pass
                tree.append({"file": rel, "mtime": os.path.getmtime(fa), **r})
        tree.sort(key=lambda x: x["mtime"])
        print(f"TREE {len(tree)} differing files (earliest first):"); [print("  ", x["file"], {k: v for k, v in x.items() if k not in ("file", "mtime", "same")}) for x in tree[:12]]
    out = {"a": a.a, "b": a.b, "bit_identical_final": res[-1]["same"], "first_difference": first, "stages": res, "tree_differing_earliest_first": tree}
    if a.out: json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    for r in res: print(("SAME " if r["same"] else "DIFF ") + r["stage"] + ("" if r["same"] or "detail" not in r else "  " + json.dumps(r["detail"], ensure_ascii=False)[:300]))
    print("REPRO", json.dumps({"bit_identical_final": out["bit_identical_final"], "first_difference": first}, ensure_ascii=False))


if __name__ == "__main__":
    main()
