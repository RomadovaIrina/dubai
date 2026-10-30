#!/usr/bin/env python3
"""r7 spec 6.2 / 5.2 check on the five test videos, with the production calls (run_clean_pipeline_05.run_vad / run_asr):
  1. whole-file ASR (filter off) vs the r6_audio_v1 manifest transcript (the input of r7_audio_v1): identical?
  2. the hallucination guard on that transcript: every dropped / trimmed / flagged segment, a listen clip of each drop (clips/<id>_drop<k>.m4a)
  3. --asr-segments vad: the piece table, every cut checked against Silero speech (+-50 ms), and the transcript diff vs the whole-file call
    python scripts/r7/asr_check.py --out /workspace/dub/artifacts/r4a/asrchk [ids]       (writes asr_check.json + asr_check.md)"""
import argparse, difflib, json, pathlib, subprocess, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot"))
import run_clean_pipeline_05 as P  # noqa: E402
import asr_guard  # noqa: E402

RAW = pathlib.Path("/workspace/dub/inputs/test_videos_raw"); R6 = pathlib.Path("/workspace/dub/handoff/in/audio-r6-v1")


def tc(t): return f"{int(t // 60)}:{t % 60:05.2f}"


def cut(c): return f"{c['t']} / {c.get('pause_s')} / {c.get('pause_db')} / {c['kind']}"


def wdiff(a, b):
    """words changed between two transcripts (difflib on lower-cased word tokens): (changed, total of a)"""
    x = [w.lower().strip(".,!?") for s in a for w in s["text"].split()]; y = [w.lower().strip(".,!?") for s in b for w in s["text"].split()]
    sm = difflib.SequenceMatcher(None, x, y, autojunk=False)
    wdiff.ops = [f"{' '.join(x[i1:i2])} -> {' '.join(y[j1:j2])}" for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"]
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal"), len(x)


def strip(segs): return [(s["start"], s["end"], s["text"], tuple((w["word"], w["start"], w["end"]) for w in s["words"])) for s in segs]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("ids", nargs="*", default=["01", "02", "03", "04", "05"]); a = ap.parse_args()
    out = pathlib.Path(a.out); (out / "clips").mkdir(parents=True, exist_ok=True); R = {}
    for id_ in a.ids:
        src = next(RAW.glob(f"{id_}.*")); wav = out / f"{id_}_16k.wav"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)], check=True)
        speech = P.run_vad(wav); r = {"speech_s": round(sum(v["end"] - v["start"] for v in speech), 2)}
        P.ASR_OPTS.update(hallu_filter="off", segments="off"); t = time.perf_counter(); whole, meta = P.run_asr(wav, None, speech); r["asr_s_whole"] = round(time.perf_counter() - t, 2)
        r6 = json.loads(next((R6 / id_).glob("*manifest.json")).read_text())["asr"]["segments"]
        r["same_as_r6v1"] = strip(whole) == strip(r6); r["same_text_as_r6v1"] = [s["text"] for s in whole] == [s["text"] for s in r6]
        r["words_diff_vs_r6v1"] = wdiff(r6, whole); r["diff_vs_r6v1"] = wdiff.ops; again, _ = P.run_asr(wav, None, speech); r["rerun_identical"] = strip(again) == strip(whole)
        r["segments"] = len(whole); r["max_no_speech_prob"] = max(s["no_speech_prob"] for s in whole)
        r["nsp_gt_06"] = [{"id": s["id"], "t": tc(s["start"]), "text": s["text"], "nsp": s["no_speech_prob"], "lp": s["avg_logprob"]} for s in whole if s["no_speech_prob"] > 0.6]
        kept, rep = asr_guard.guard(whole, speech); r["guard"] = rep; r["guard_identity"] = strip(kept) == strip(whole)
        for k, d in enumerate(rep["dropped"] + rep["trimmed"]):
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", str(max(0, d["start"] - 1)), "-to", str(d["end"] + 1), "-i", str(src), "-vn", "-c:a", "aac", "-b:a", "128k",
                            str(out / "clips" / f"{id_}_drop{k}.m4a")], check=True)
        P.ASR_OPTS.update(segments="vad"); t = time.perf_counter(); seg, smeta = P.run_asr(wav, None, speech); r["asr_s_vad"] = round(time.perf_counter() - t, 2)
        r["pieces"] = smeta["chunks"]["pieces"]; r["cuts"] = smeta["chunks"]["cuts"]; r["vad_identity"] = strip(seg) == strip(whole); r["words_diff_pieces"] = wdiff(whole, seg); r["diff_pieces"] = wdiff.ops
        for c in r["cuts"]:   # also inside a word of the whole-file transcript or of the r6v1 one?
            c["in_word"] = c["t"] is not None and any(w["start"] < c["t"] < w["end"] for sg in whole + r6 for w in sg["words"])
        if not r["vad_identity"]:
            r["vad_text_whole"] = " ".join(s["text"] for s in whole); r["vad_text_pieces"] = " ".join(s["text"] for s in seg)
        R[id_] = r; print(id_, json.dumps({k: v for k, v in r.items() if k not in ("guard",)}, ensure_ascii=False)[:600], flush=True)
    (out / "asr_check.json").write_text(json.dumps(R, ensure_ascii=False, indent=1))
    L = ["| video | dur s | segments | = r6v1 transcript | no_speech > 0.6 | dropped | trimmed | flagged, kept (VAD speech) | pieces (s) | cuts: t / pause s / dB / kind | cut in speech / in a word | pieces = whole-file (words changed) | rerun = | words changed vs r6v1 |",
         "|---|---:|---:|---|---:|---:|---:|---:|---|---|---|---|---|---|"]
    for id_, r in R.items():
        g = r["guard"]; D = r["pieces"][-1][1] or 0
        L.append(f"| {id_} | {D} | {r['segments']} | {'yes' if r['same_as_r6v1'] else ('text yes' if r['same_text_as_r6v1'] else 'NO')} | {len(r['nsp_gt_06'])} | {len(g['dropped'])} | {len(g['trimmed'])} | "
                 f"{len(g['flagged_kept'])} | {', '.join(f'{p[0]}-{p[1]}' for p in r['pieces'])} | {'; '.join(cut(c) for c in r['cuts']) or '-'} | "
                 f"{'YES' if any(c.get('in_speech') for c in r['cuts']) else 'no'} / {'YES' if any(c.get('in_word') for c in r['cuts']) else 'no'} | "
                 f"{'yes' if r['vad_identity'] else 'NO'} ({r['words_diff_pieces'][0]}/{r['words_diff_pieces'][1]}) | {'yes' if r['rerun_identical'] else 'NO'} | {r['words_diff_vs_r6v1'][0]}/{r['words_diff_vs_r6v1'][1]} |")
    L += ["", "Word changes (whole-file today vs r6v1 | VAD pieces vs whole-file):", ""]
    for id_, r in R.items(): L.append(f"- {id_}: vs r6v1 {r['diff_vs_r6v1'] or 'none'} | pieces {r['diff_pieces'] or 'none'}")
    L += ["", "Every segment the guard dropped, trimmed or flagged:", ""]
    for id_, r in R.items():
        for kind in ("dropped", "trimmed", "flagged_kept"):
            for d in r["guard"][kind]:
                L.append(f"- {id_} {kind} {tc(d['start'])}-{tc(d['end'])} \"{d['text']}\" why={d['why']} nsp={d['no_speech_prob']} logprob={d['avg_logprob']} VAD={d['vad_coverage']}" + (f" removed \"{d['removed']}\"" if d.get("removed") else ""))
    (out / "asr_check.md").write_text("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
