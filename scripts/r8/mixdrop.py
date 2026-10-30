#!/usr/bin/env python3
"""r8 v4 mix-level detector (user: "a word, and then IMMEDIATE silence"): the WHOLE MIX after every dub word end, not only the dub voice.
For each dub speech end e (Silero on dubbed_16k.wav) followed by >= 0.3 s of dub silence, 20-ms frames of the final mix (final_mix_44k.wav):
  pre    mean level over [e - 0.15, e - 0.02]       post80  level at [e + 0.06, e + 0.10]       post  median over [e + 0.10, e + 0.30]
  amb    the local ambience: median mix level over the frames within +-3 s where neither the dub nor the source speaks (joint pauses; else the
         raw bed stem bg_44k.wav, scaled like the mix, over the same window)
  dead   post < amb - --dead-db, or post < -60 dBFS: the mix goes (near) silent after the word
  abrupt pre - post80 > 12 dB AND post < amb - 4 dB: the background falls away with the voice
Per flagged end the bed levels at [e + 0.1, e + 0.3] are given for every bed stage present in the run dir (bg_44k raw, bg_capped_44k gap cap,
bg_clean_44k gap clean) so the cause shows (the stage where the bed drops), plus the dub gate run if one starts there.
    python scripts/r8/mixdrop.py --run <remix dir> --id 04 --out X"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot"))
FR = 0.02


def lev(path, n):
    y, sr = sf.read(str(path), dtype="float32"); y = y if y.ndim == 1 else y.mean(1); h = int(FR * sr); m = min(n, len(y) // h)
    L = np.full(n, -120.0); L[:m] = 20 * np.log10(np.sqrt((y[:m * h].reshape(m, h) ** 2).mean(1)) + 1e-9); return L


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--id", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--label", default=""); ap.add_argument("--dead-db", type=float, default=8.0); a = ap.parse_args()
    import run_clean_pipeline_05 as rp
    R = pathlib.Path(a.run); man = json.loads(next(R.glob(f"{a.id}_*.manifest.json")).read_text()); D = man["source"]["duration_s"]; n = int(D / FR) + 1
    mix = R / "final_mix_44k.wav"
    if not mix.exists():   # a handoff dir: decode the AAC
        import subprocess, tempfile
        mix = pathlib.Path(tempfile.mkdtemp()) / "m.wav"; subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(R / "final_mix_aac.m4a"), "-ac", "1", str(mix)], check=True)
    M = lev(mix, n)
    dub = [(s["start"], s["end"]) for s in rp.run_vad(R / "dubbed_16k.wav")]
    Dm = np.zeros(n, bool); Sm = np.zeros(n, bool)
    for s, e in dub: Dm[int(s / FR):min(n, int(np.ceil(e / FR)))] = True
    for s in man["speech_intervals"]: Sm[int(s["start"] / FR):min(n, int(np.ceil(s["end"] / FR)))] = True
    beds = {k: lev(R / f, n) for k, f in (("raw", "bg_44k.wav"), ("capped", "bg_capped_44k.wav"), ("clean", "bg_clean_44k.wav")) if (R / f).exists()}
    gate = [(g0, g1) for g0, g1 in (man.get("dub_gate") or {}).get("runs", [])]
    fr = lambda t: min(n - 1, max(0, int(t / FR))); ev = []
    for q, (s, e) in enumerate(dub):
        nxt = dub[q + 1][0] if q + 1 < len(dub) else D
        if nxt - e < 0.3: continue
        pre = float(M[fr(e - 0.15):fr(e - 0.02) + 1].mean()); post80 = float(M[fr(e + 0.06):fr(e + 0.10) + 1].mean()); post = float(np.median(M[fr(e + 0.10):fr(e + 0.30) + 1]))
        w0, w1 = fr(e - 3), fr(e + 3); jp = np.flatnonzero(~Dm[w0:w1] & ~Sm[w0:w1]) + w0
        if len(jp) >= 10: amb = float(np.median(M[jp])); src = "joint pauses"
        elif "raw" in beds: amb = float(np.median(beds["raw"][w0:w1])) + float(np.median(M[w0:w1] - beds["raw"][w0:w1])); src = "raw bed"
        else: continue
        dead = post < amb - a.dead_db or post < -60; abrupt = pre - post80 > 12 and post < amb - 4
        if not (dead or abrupt): continue
        b = {k: round(float(np.median(v[fr(e + 0.10):fr(e + 0.30) + 1])), 1) for k, v in beds.items()}
        cause = []
        if "raw" in b and "capped" in b and b["capped"] < b["raw"] - 4: cause.append("gap_cap")
        if "clean" in b and b["clean"] < b.get("capped", b.get("raw", b["clean"])) - 4: cause.append("gap_clean")
        if any(abs(g0 - e) < 0.4 for g0, g1 in gate): cause.append("dub_gate")
        if not cause: cause.append("duck/mix" if Sm[fr(e + 0.1):fr(e + 0.3) + 1].any() else "bed itself quiet")
        ev.append({"t": round(e, 2), "kind": "dead" if dead else "abrupt", "pre_db": round(pre, 1), "post80_db": round(post80, 1), "post_db": round(post, 1),
                   "amb_db": round(amb, 1), "amb_from": src, "src_speaks": bool(Sm[fr(e + 0.1):fr(e + 0.3) + 1].any()), "beds": b, "cause": cause})
    summ = {"dead": sum(1 for x in ev if x["kind"] == "dead"), "abrupt": sum(1 for x in ev if x["kind"] == "abrupt"), "ends_checked": sum(1 for q, (s, e) in enumerate(dub) if (dub[q + 1][0] if q + 1 < len(dub) else D) - e >= 0.3),
            "by_cause": {c: sum(1 for x in ev if c in x["cause"]) for c in ("gap_cap", "gap_clean", "dub_gate", "duck/mix", "bed itself quiet")}}
    out = pathlib.Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"id": a.id, "label": a.label, "summary": summ, "events": ev}, indent=1))
    mm = lambda t: f"{int(t // 60)}:{t % 60:05.2f}"
    md = [f"# {a.id} {a.label}: the mix after dub word ends", "", "summary: " + json.dumps(summ), "", "| word end | kind | mix pre / +80 ms / +100-300 ms (dB) | ambience | source speaks | bed stages | cause |", "|---|---|---|---|---|---|---|"]
    md += [f"| {mm(x['t'])} | {x['kind']} | {x['pre_db']} / {x['post80_db']} / {x['post_db']} | {x['amb_db']} ({x['amb_from']}) | {x['src_speaks']} | {x['beds']} | {', '.join(x['cause'])} |" for x in ev]
    out.with_suffix(".md").write_text("\n".join(md) + "\n"); print(a.id, a.label, json.dumps(summ))


if __name__ == "__main__":
    main()
