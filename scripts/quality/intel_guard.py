#!/usr/bin/env python3
"""Intelligibility regression guard (Agent C r3, 2026-10-19): per-unit back-transcription of the dub voice -> per-video / per-unit WER / CER
and the worst units as listening clips with names a person understands.

Input: eval_cpu.py / eval_run.evaluate JSONs (Whisper large-v3 WER + wav2vec2 CTC CER of every unit against its English text) + the manifests.
Output <out>/:
  intel_<tag>.md / .json   per-video table + per-unit rows (worst first), cause hints (tempo, short unit, number, onset/offset loss)
  listen/<tag>_worst<NN>_video<id>_<m>m<ss>s_<what>.wav   dub voice of the unit (+-0.3 s), 24 kHz; listen/README.md = the same list
                            with "expected" vs "Whisper heard" vs "CTC heard"

    python scripts/quality/intel_guard.py --eval DIR/01.eval.json ... --manifest M/01.manifest.json ... --tag audio-r2 --out /workspace/dub/c/outbox
"""
from __future__ import annotations
import argparse, json, pathlib, re
import numpy as np
import soundfile as sf


def _vid(man: dict) -> str:
    return pathlib.Path(man["input"]).stem


def _unit_tempo(u: dict) -> tuple[float, float]:
    pls = (u.get("alignment") or {}).get("placements") or []
    at = [float(p.get("atempo", 1.0)) for p in pls if p.get("atempo")]
    return (max(at) if at else 1.0, min(at) if at else 1.0)


def hints(row: dict, u: dict | None) -> list[str]:
    h = []
    exp, wh, ctc = row["expected"], row.get("whisper", ""), row.get("ctc", "")
    if u is not None:
        hi, lo = _unit_tempo(u)
        if hi > 1.1: h.append(f"sped up x{hi:.2f}")
        if lo < 0.9: h.append(f"slowed x{lo:.3f}")
    if re.search(r"\d", exp): h.append("number in text")
    if row.get("final_word_missing"): h.append("last word lost")
    ew, ww = exp.lower().split(), wh.lower().split()
    if ew and ww and re.sub(r"\W", "", ew[0]) != re.sub(r"\W", "", ww[0]) and row["wer"] > 0: h.append("first word differs (onset?)")
    if row.get("interjections"): h.append("extra sound: " + ",".join(row["interjections"]))
    if row.get("words", 99) <= 3: h.append("short unit")
    if row["cer_ctc"] >= 0.3 and row["wer"] < 0.1: h.append("CTC only (accent / rare word, Whisper fine)")
    return h


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", action="append", required=True); ap.add_argument("--manifest", action="append", required=True)
    ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True); ap.add_argument("--worst", type=int, default=20)
    a = ap.parse_args()
    mans = {}
    for mp in a.manifest:
        m = json.loads(pathlib.Path(mp).read_text()); mans[_vid(m)] = m
    out = pathlib.Path(a.out); lst = out / "listen"; lst.mkdir(parents=True, exist_ok=True)
    for old in lst.glob(f"{a.tag}_worst*.wav"): old.unlink()
    rows, per_video = [], {}
    for ep in a.eval:
        r = json.loads(pathlib.Path(ep).read_text()); vid = r["summary"]["label"].split("_")[-1]; vid = vid if vid in mans else pathlib.Path(ep).stem.split(".")[0][-2:]
        m = mans[vid]; units = {u["id"]: u for u in m["units"]}
        per_video[vid] = {k: r["summary"].get(k) for k in ("units", "wer_whisper_mean", "cer_ctc_mean", "cer_whisper_mean", "units_bad(cer_ctc>=0.3)", "interjection_units", "final_word_missing_units", "atempo_max")}
        for row in r["units"]:
            u = units.get(row["unit"])
            score = max(row["wer"], row["cer_whisper"]) + 0.5 * row["cer_ctc"]       # Whisper is the "listener", CTC breaks ties / catches slurring
            rows.append({"video": vid, **row, "score": round(score, 3), "hints": hints(row, u), "wav": str(pathlib.Path(m["work_dir"]) / "dubbed_24k.wav")})
    rows.sort(key=lambda x: -x["score"])
    md = [f"# Intelligibility guard: {a.tag}", "", "Whisper large-v3 WER and wav2vec2 CTC CER of every dub unit against its English text.",
          "Unit mean per video; `bad` = CTC CER ≥ 0.3.", "", "| video | units | WER (Whisper) | CER (CTC) | CER (Whisper) | bad | extra-sound units | last word lost | max speed-up |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for v in sorted(per_video):
        p = per_video[v]
        md.append(f"| {v} | {p['units']} | {p['wer_whisper_mean']:.3f} | {p['cer_ctc_mean']:.3f} | {p['cer_whisper_mean']:.3f} | {p['units_bad(cer_ctc>=0.3)']} | "
                  f"{p['interjection_units']} | {p['final_word_missing_units']} | x{p['atempo_max']} |")
    allw = [x["wer"] for x in rows]; allc = [x["cer_ctc"] for x in rows]
    md += ["", f"All {len(rows)} units: WER mean {np.mean(allw):.3f}, units with WER > 0: {sum(w > 0 for w in allw)}; CTC CER mean {np.mean(allc):.3f}.", "",
           f"## Worst {a.worst} units (clips in `listen/`, names start with `{a.tag}_worst`)", "",
           "| # | clip | expected (English text) | Whisper heard | CTC heard | WER | CER ctc | hints |", "|---:|---|---|---|---|---:|---:|---|"]
    cache = {}
    for i, x in enumerate(rows[: a.worst], 1):
        if x["wav"] not in cache: cache[x["wav"]] = sf.read(x["wav"], dtype="float32")
        y, sr = cache[x["wav"]]; s0, s1 = max(0, int((x["t0"] - 0.3) * sr)), min(len(y), int((x["t1"] + 0.3) * sr))
        what = "number" if "number in text" in x["hints"] else ("fast" if any(h.startswith("sped") for h in x["hints"]) else ("extra-sound" if any(h.startswith("extra") for h in x["hints"]) else "unclear"))
        name = f"{a.tag}_worst{i:02d}_video{x['video']}_{int(x['t0'] // 60)}m{int(x['t0'] % 60):02d}s_{what}.wav"
        sf.write(lst / name, y[s0:s1], sr); x["clip"] = f"listen/{name}"
        cell = lambda t: t.replace("|", "/")
        md.append(f"| {i} | `{name}` | {cell(x['expected'])} | {cell(x['whisper'])} | {cell(x['ctc'])} | {x['wer']:.2f} | {x['cer_ctc']:.2f} | {', '.join(x['hints'])} |")
    (out / f"intel_{a.tag}.md").write_text("\n".join(md) + "\n")
    (out / f"intel_{a.tag}.json").write_text(json.dumps({"per_video": per_video, "units": rows}, indent=1, ensure_ascii=False))
    readme = lst / "README.md"; prev = readme.read_text() if readme.exists() else ""
    sect = f"<!-- {a.tag} -->"; body = "\n".join(md[md.index(f"## Worst {a.worst} units (clips in `listen/`, names start with `{a.tag}_worst`)"):])
    prev = re.sub(rf"{re.escape(sect)}.*?{re.escape(sect)}\n", "", prev, flags=re.S)
    readme.write_text(prev + f"{sect}\n{body}\n{sect}\n")
    print("\n".join(md[:6 + len(per_video)])); print(f"-> {out / f'intel_{a.tag}.md'}, {min(a.worst, len(rows))} clips in {lst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
