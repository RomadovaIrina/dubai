#!/usr/bin/env python3
"""Diagnostic: merge every analysis into one per-chunk verdict (issue class, severity, probable stage), write the review tables and the
A/B/C/D(+E,+fixed) diagnostic dataset of the worst chunks.   python scripts/diag/build_review.py
Rules are explicit and printed in review.md; nothing here is a human listening judgement - every row carries the numbers it was derived from."""
from __future__ import annotations
import csv, json, pathlib, re, shutil
import numpy as np

T = pathlib.Path("/workspace/dub/tts_diag"); OUT = T / "review"; WORST = T / "worst_chunks"
INT = re.compile(r"\b(um+|uh+|ah+|aah+|oh+|hmm+|mm+|erm?|eh+|ha+|huh|aha)\b")
BAD_CER = 0.30


def inter(txt: str, exp: str) -> list[str]:
    e = set(re.findall(r"[a-z']+", exp.lower())); return [w for w in INT.findall(txt.lower()) if w not in e]


def nearest(d: dict, t: float) -> str:
    return min(d, key=lambda k: abs(float(k) - t))


def main() -> int:
    OUT.mkdir(exist_ok=True); rows = []
    for v in ("01", "03"):
        ch = {r["tag"]: r for r in json.loads((T / f"chunks/{v}/chunks.json").read_text())}; an = {r["tag"]: r for r in json.loads((T / f"chunks/{v}/analysis.json").read_text())}
        rg = {r["tag"]: r for r in json.loads((T / f"chunks/{v}/regen_scores.json").read_text())}
        for tag, c in ch.items():
            a, g = an[tag], rg[tag]; S = a["stages"]; exp = c["translated_text"]; words = S["A"]["w_ref_words"]
            prod, fix = g["prod_fp16"], g["fp16fix"]
            cerA, cerB, cerC, cerE = (S[k]["c_cer"] for k in "ABCE"); cerFix = fix["c_cer"]
            ints_prod = inter(prod["whisper"], exp) + inter(prod["ctc"], exp); ints_E = inter(S["E"]["whisper"], exp) + inter(S["E"]["ctc"], exp)
            tempo = c["atempo"]; h3 = g["fix_h3"]; tk = nearest(h3, tempo); d_tempo = h3[tk]["c_cer"] - h3["1.0"]["c_cer"] if tempo > 1.05 else 0.0
            d_trim = cerB - cerA; d_tempo_prod = cerC - cerB
            bad_final = cerE >= BAD_CER or bool(ints_E)
            causes = []                      # (cause, evidence)
            if cerA >= BAD_CER - 0.05 and cerFix <= cerA - 0.15: causes.append(("RAW_TTS_fp16_token_corruption", f"raw CER {cerA:.2f} -> {cerFix:.2f} with int64 speech tokens (same text/seed/ref)"))
            if ints_prod and not (inter(fix["whisper"], exp) + inter(fix["ctc"], exp)) and not causes: causes.append(("RAW_TTS_fp16_token_corruption", f"interjection {ints_prod[:2]} in raw, gone with int64 speech tokens"))
            if cerFix >= BAD_CER and words <= 3: causes.append(("BAD_CHUNKING_fragment", f"{words}-word fragment still CER {cerFix:.2f} after numeric fix"))
            if d_trim >= 0.25: causes.append(("VAD_TRIM", f"A->B CER {cerA:.2f} -> {cerB:.2f}; head cut {c['trim_left_ms']} ms"))
            if tempo >= 1.25 and (d_tempo >= 0.12 or d_tempo_prod >= 0.15): causes.append(("ATEMPO", f"x{tempo} on clean audio dCER {d_tempo:+.2f}; prod B->C {d_tempo_prod:+.2f}"))
            primary = causes[0][0] if causes else ("UNKNOWN" if bad_final else "")
            # issue class (human-facing taxonomy)
            cls = []
            if ints_prod or (c["raw_duration"] / max(words, 1) >= 1.0 and words <= 2): cls.append("TTS_HALLUCINATION")
            if prod["c_final_word_missing"] or not prod["c_final_suffix_kept"]: cls.append("SWALLOWED_ENDING")
            if cerA >= 0.45 and "TTS_HALLUCINATION" not in cls: cls.append("UNCLEAR_WORD")
            if "VAD_TRIM" in [x[0] for x in causes] and c["trim_left_ms"] >= 120: cls.append("CUT_WORD")
            if "ATEMPO" in [x[0] for x in causes] and tempo >= 1.4: cls.append("OVER_SPEED")
            if "BAD_CHUNKING_fragment" in [x[0] for x in causes]: cls.append("BAD_CHUNK_BOUNDARY")
            if c["split_method"] == "proportional_fallback" and "BAD_CHUNK_BOUNDARY" not in cls and words <= 4: cls.append("BAD_CHUNK_BOUNDARY")
            if exp[:1].islower() or exp.startswith("."): cls.append("BAD_TRANSLATION_SEGMENT")
            sev = "high" if (cerE >= 0.6 or ints_E) else ("medium" if cerE >= BAD_CER else "low")
            rows.append({"video": v, "tag": tag, "unit_id": c["unit_id"], "part_id": c["part_id"], "t_start": c["placed_start"], "t_end": c["placed_end"], "expected": exp, "source_text": c["source_text"],
                         "heard_whisper_final": S["E"]["whisper"], "heard_ctc_final": S["E"]["ctc"], "issue_classes": cls, "severity": sev, "primary_cause": primary, "causes": causes,
                         "cer_raw_prod": cerA, "cer_raw_fixed": cerFix, "cer_trim": cerB, "cer_atempo": cerC, "cer_final": cerE, "wer_final_whisper": S["E"]["w_wer"], "interjections_raw": ints_prod, "interjections_final": ints_E,
                         "atempo": tempo, "raw_duration": c["raw_duration"], "trimmed_duration": c["trimmed_duration"], "trim_left_ms": c["trim_left_ms"], "trim_right_ms": c["trim_right_ms"], "burst_window_s": c["burst_window_s"],
                         "spill": c["spill"], "fallback_used": c["fallback_used"], "seed": c["seed"], "translated_text": exp, "n_parts": c["n_parts"], "split_method": c["split_method"], "bad_in_final": bad_final})
    for i, r in enumerate(rows):   # second pass: a chunk that is fine itself but scores badly only through the +-0.15 s context of a bad neighbour placed < 0.25 s away
        if r["bad_in_final"] and r["primary_cause"] == "UNKNOWN" and r["cer_atempo"] < 0.2:
            nb = [q for j, q in ((i - 1, rows[i - 1] if i else None), (i + 1, rows[i + 1] if i + 1 < len(rows) else None)) if q and q["video"] == r["video"] and q["cer_final"] >= 0.4
                  and (abs(q["t_start"] - r["t_end"]) < 0.25 or abs(r["t_start"] - q["t_end"]) < 0.25)]
            if nb: r["primary_cause"] = "NEIGHBOR_BLEED(context of adjacent bad chunk)"
    (OUT / "chunks_verdict.json").write_text(json.dumps(rows, indent=1, ensure_ascii=False))
    bad = [r for r in rows if r["bad_in_final"]]
    from collections import Counter
    print("chunks:", len(rows), "| bad in final audio (E CER>=0.30 or Whisper WER>=0.5 or interjection):", len(bad))
    print("primary cause counts (bad chunks):", dict(Counter(r["primary_cause"] for r in bad)))
    cc = Counter(c for r in bad for c, _ in r["causes"]); print("any-contributing cause counts (bad chunks):", dict(cc))
    print("issue-class counts (all chunks):", dict(Counter(c for r in rows for c in r["issue_classes"])))
    print("severity:", dict(Counter(r["severity"] for r in rows)))
    # ---- review table (worst first)
    order = sorted(rows, key=lambda r: ({"high": 0, "medium": 1, "low": 2}[r["severity"]], -(r["cer_final"] + r["wer_final_whisper"] * 0.5)))
    with open(OUT / "problem_timestamps.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["video", "t_start", "t_end", "unit", "part", "expected_text", "heard_whisper", "heard_ctc", "issue_classes", "severity", "probable_stage", "cer_raw_prod", "cer_raw_fixed", "atempo"])
        for r in order:
            if r["severity"] != "low" or r["issue_classes"]: w.writerow([r["video"], r["t_start"], r["t_end"], r["unit_id"], r["part_id"], r["expected"], r["heard_whisper_final"], r["heard_ctc_final"], "+".join(r["issue_classes"]), r["severity"], r["primary_cause"], r["cer_raw_prod"], r["cer_raw_fixed"], r["atempo"]])
    # ---- worst-chunk diagnostic dataset (A/B/C/D + E + fixed)
    top = order[:20]; WORST.mkdir(exist_ok=True); fields = []
    for i, r in enumerate(top):
        src = T / f"chunks/{r['video']}/{r['tag']}"; dst = WORST / f"{i + 1:02d}_{r['video']}_{r['tag']}_t{r['t_start']:.1f}"; dst.mkdir(exist_ok=True)
        for s, d in (("A_raw_tts.wav", "A_raw_tts.wav"), ("B_after_trim.wav", "B_after_trim.wav"), ("C_after_atempo.wav", "C_after_atempo.wav"), ("D_final_track.wav", "D_final_track_fragment.wav"), ("E_first_res.wav", "E_first_res_video_audio.wav"), ("A_fp16fix.wav", "A_FIXED_fp16_int64_tokens.wav")):
            if (src / s).exists(): shutil.copy(src / s, dst / d)
        fields.append({k: r[k] for k in ("video", "t_start", "t_end", "unit_id", "part_id", "source_text", "translated_text", "seed", "raw_duration", "trimmed_duration", "trim_left_ms", "trim_right_ms", "burst_window_s", "atempo", "spill", "fallback_used",
                                         "severity", "primary_cause", "cer_raw_prod", "cer_raw_fixed", "cer_trim", "cer_atempo", "cer_final", "heard_whisper_final", "heard_ctc_final", "issue_classes")} | {"folder": dst.name})
    (WORST / "worst_chunks.json").write_text(json.dumps(fields, indent=1, ensure_ascii=False))
    with open(WORST / "worst_chunks.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields[0].keys())); w.writeheader(); [w.writerow({k: (json.dumps(v) if isinstance(v, list) and v and not isinstance(v[0], str) else ("+".join(v) if isinstance(v, list) else v)) for k, v in x.items()}) for x in fields]
    print("\nTOP worst:")
    for r in top: print(f" {r['video']} {r['t_start']:6.1f}-{r['t_end']:6.1f} [{r['severity']:6s}] {r['primary_cause'][:28]:28s} {'+'.join(r['issue_classes'])[:55]:55s} exp={r['expected'][:32]!r} heard={r['heard_whisper_final'][:40]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
