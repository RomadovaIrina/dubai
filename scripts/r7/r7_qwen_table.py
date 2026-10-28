#!/usr/bin/env python3
"""r7 7B-vs-14B table from the arms of scripts/r7/r7_qwen_table.sh (each = scripts/r6/r6_audio.sh, all five videos, one video at a time):
speed (audio stage s/video-min, LLM time split candidates / judge (= back-translation + score in one call) / pipeline LLM steps, tokens/s), peak VRAM,
quality (14B faithfulness judge for every arm, NE, WER, tempo / gap / placement axes). Sums / means over the five videos.
    python scripts/r7/r7_qwen_table.py q14 q7 q7j14 q14r1 > table.md"""
import json, pathlib, re, subprocess, sys
A = pathlib.Path("/workspace/dub/artifacts/r4a"); IDS = ["01", "02", "03", "04", "05"]


def dur(id_):
    f = next(pathlib.Path("/workspace/dub/inputs/test_videos_raw").glob(f"{id_}.*"))
    return float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(f)], capture_output=True, text=True).stdout)


def arm(tag, mins):
    r = {"tag": tag, "total": 0.0, "llm_pipe": 0.0, "cand": 0.0, "judge": 0.0, "other_dur": 0.0, "tok": 0, "dur_q": 0.0, "tts": 0.0, "faith": [], "lt4": 0, "ne": 0,
         "wer": [], "tempo_out": 0.0, "src_sil": 0.0, "dub_over": 0.0, "face_sil": 0.0, "gapx": 0.0, "w1": [], "cands": 0, "switched": 0}
    for id_ in IDS:
        man = json.loads((A / tag / "runs" / f"{id_}_r4a.manifest.json").read_text()); st = man["stage_seconds"]; ds = (man["tts"].get("stats") or {}).get("dur_translate") or {}
        r["total"] += st["total"]; r["tts"] += st.get("tts", 0)
        r["llm_pipe"] += sum(st.get(k, 0) for k in ("translation_load", "segmentation", "translation", "ne_extract")) + ((man["tts"].get("stats") or {}).get("judge_seconds") or 0)
        r["cand"] += sum(v for k, v in ds.items() if k.startswith("qwen_s_") and "candidates" in k); r["judge"] += ds.get("qwen_s__judge", 0.0)
        r["dur_q"] += ds.get("qwen_s", 0.0); r["tok"] += ds.get("qwen_tok_out", 0); r["cands"] += ds.get("candidates", 0); r["switched"] += ds.get("switched", 0)
        m = json.loads((A / tag / f"{id_}.metrics.json").read_text())
        f = m.get("faith") or {}
        if f.get("mean") is not None: r["faith"].append(f["mean"]); r["lt4"] += f.get("lt4", 0)
        r["ne"] += (m.get("ne") or {}).get("misheard") or 0; w = (m.get("asr") or {}).get("wer_final_mix")
        if w is not None: r["wer"].append(w)
        r["tempo_out"] += (m.get("tempo") or {}).get("out_s") or 0; r["src_sil"] += m.get("src_speaks_dub_silent_s") or 0; r["dub_over"] += m.get("dub_speaks_src_silent_s") or 0
        r["face_sil"] += m.get("dub_silent_face_visible_s") or 0; r["gapx"] = max(r["gapx"], m.get("gap_excess_max_s") or 0)
        if m.get("pauses_w1_s") is not None: r["w1"].append(m["pauses_w1_s"])
    r["other_dur"] = r["dur_q"] - r["cand"] - r["judge"]
    rd = (A / tag / "README.txt").read_text(); v = re.findall(r"peak VRAM MiB\s+(\d+)", rd); r["vram"] = int(v[-1]) if v else None
    return r


def main():
    tags = sys.argv[1:]; mins = sum(dur(i) for i in IDS) / 60; R = [arm(t, mins) for t in tags]
    mean = lambda x: round(sum(x) / len(x), 3) if x else None; pm = lambda s: round(s / mins, 1)
    rows = [("audio stage s/video-min (whole r6 stage)", lambda r: pm(r["total"])),
            ("  LLM s/video-min (all LLM work)", lambda r: pm(r["llm_pipe"] + r["dur_q"])),
            ("    pipeline steps (load, segmentation, translation, NE, fit judge)", lambda r: pm(r["llm_pipe"])),
            ("    duration stage: candidates", lambda r: pm(r["cand"])),
            ("    duration stage: judge (back-translation + score)", lambda r: pm(r["judge"])),
            ("    duration stage: other Qwen calls", lambda r: pm(r["other_dur"])),
            ("  TTS s/video-min (incl. candidate takes)", lambda r: pm(r["tts"])),
            ("duration-stage tokens/s (generated)", lambda r: round(r["tok"] / r["dur_q"], 1) if r["dur_q"] else None),
            ("peak VRAM MiB", lambda r: r["vram"]),
            ("candidates written / units switched", lambda r: f"{r['cands']} / {r['switched']}"),
            ("faithfulness mean (14B judge, 1-5)", lambda r: mean(r["faith"])), ("units scored < 4", lambda r: r["lt4"]),
            ("named entities misheard", lambda r: r["ne"]), ("WER final mix (mean)", lambda r: mean(r["wer"])),
            ("tempo outside 0.95-1.05 (s)", lambda r: round(r["tempo_out"], 2)), ("source speaks, dub silent (s)", lambda r: round(r["src_sil"], 2)),
            ("dub speaks, source silent (s)", lambda r: round(r["dub_over"], 2)), ("speaker on screen talks, dub silent (s)", lambda r: round(r["face_sil"], 2)),
            ("longest dub gap beyond the source pause (s, max)", lambda r: r["gapx"]), ("pause W1 vs source (s, mean)", lambda r: mean(r["w1"]))]
    print(f"Five videos, {mins:.2f} video-min, one video at a time on one RTX 5090.\n")
    print("| axis | " + " | ".join(tags) + " |"); print("|---|" + "---:|" * len(tags))
    for lab, f in rows: print(f"| {lab} | " + " | ".join(str(f(r)) for r in R) + " |")


if __name__ == "__main__":
    main()
