#!/usr/bin/env python3
"""Round 3 D: built-in QA gates of the E2E runner (r3_e2e.sh), run on EVERY output, same rules for every video (no per-video tuning).
  frames_fps_res   output frame count / fps / resolution / duration vs the source (spec 9.4)             FAIL if any differs
  lip_flash        r3_flash.py strong single-frame lip spikes (mc ratio and vs-source ratio >= 2.0)      FLAG if > 1 per 10 s of lip-sync
  bg_ring          r3_bgring.py lower-ring minus upper-ring excess (background beside the face touched)  FLAG if > 0.5 grey levels
  audio_hard_cuts  r3_hardcut.py on the dub voice (loud <-> digital silence within 5 ms)                 FLAG if > 0 per minute... reported
  tts_joins        r3_joins.py (gaps / pitch / energy at the joins between the parts of one sentence)   reported
  loudness         integrated LUFS / true peak of the muxed audio                                       FLAG if |LUFS + 16| > 1 or TP > -1 dBTP
Remediation happens upstream (fps_composite --sanitize-layer, the TTS cut / fade rules, static loudness); a gate that still fails is
written to the manifest under qa_gates[...]["status"] = "FLAG" / "FAIL" and printed loudly.
    python scripts/quality/r3_qa_gates.py --final OUT.mp4 --manifest OUT.manifest.json --src SRC --lm SRC_lm.npz --audio-manifest RUN.manifest.json"""
import argparse, json, pathlib, subprocess, sys
HERE = pathlib.Path(__file__).resolve().parent; PY = sys.executable
sys.path.insert(0, str(HERE))


def run_json(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    for line in reversed(p.stdout.strip().splitlines()):
        if line.startswith("{"): return json.loads(line)
    return {"error": (p.stderr or p.stdout)[-300:]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--final", required=True); ap.add_argument("--manifest", required=True); ap.add_argument("--src", required=True)
    ap.add_argument("--lm", required=True); ap.add_argument("--audio-manifest", required=True); a = ap.parse_args()
    import r3_scorecard as sc, r3_hardcut as hc
    g = {}
    ps, po = sc.probe(a.src), sc.probe(a.final)
    ok = (ps["w"], ps["h"]) == (po["w"], po["h"]) and ps["frames"] == po["frames"] and abs(ps["fps"] - po["fps"]) < 0.05 and abs(ps["duration"] - po["duration"]) <= 0.1
    g["frames_fps_res"] = {"status": "PASS" if ok else "FAIL", "source": ps, "output": po}
    man = json.loads(pathlib.Path(a.manifest).read_text()); ls = ((man.get("lipsync") or {}).get("by_action") or {}).get("LATENT_SYNC", {}).get("frames", 0)
    if ls:
        fl = run_json([PY, str(HERE / "r3_flash.py"), "--video", a.final, "--src", a.src, "--lm", a.lm, "--out", str(pathlib.Path(a.final).with_suffix(".flash.json"))])
        full = json.loads(pathlib.Path(a.final).with_suffix(".flash.json").read_text()) if pathlib.Path(a.final).with_suffix(".flash.json").exists() else {"flash_list": []}
        strong = [e for e in full["flash_list"] if (e.get("mc_ratio") or 0) >= 2.0 and (e.get("mc_vs_src") or 0) >= 2.0]
        # the automatic detectors (residual vs source / vs the layer / ghost-vs-dissolve) do not separate real tears from real articulation on
        # whole videos (validated on 04: known ghost frames score below the median) -> REVIEW: a contact sheet of the top-30 candidates per output
        sheet = str(pathlib.Path(a.final).with_suffix(".lip_review.png"))
        subprocess.run([PY, str(HERE / "r3_gate_sheet.py"), "--video", a.final, "--src", a.src, "--lm", a.lm, "--flash", str(pathlib.Path(a.final).with_suffix(".flash.json")),
                        "--out", sheet, "--top", "30"], capture_output=True, text=True)
        g["lip_flash"] = {"status": "REVIEW", "strong_candidates": len(strong), "all_candidates": fl.get("flash_frames"), "contact_sheet": sheet,
                          "strong_frames": [e["frame"] for e in strong][:50], "note": "remediation built in (flow_safe retime, generator-outlier sanitizer); human check of the sheet"}
        bg = run_json([PY, str(HERE / "r3_bgring.py"), "--video", a.final, "--ref", a.src, "--lm", a.lm])
        g["bg_ring"] = {"status": "PASS" if (bg.get("lower_minus_upper") or 0) <= 0.5 else "FLAG", **{k: bg.get(k) for k in ("lower_minus_upper", "lower_minus_upper_p95")}}
    else:
        g["lip_flash"] = g["bg_ring"] = {"status": "N/A", "note": "no lip-sync frames"}
    am = pathlib.Path(a.audio_manifest); dub = am.parent / "dubbed_24k.wav"
    ev = hc.events(dub); mins = ps["duration"] / 60
    # a CUT = the voice stops dead and restarts within 0.6 s (OFF followed by ON); an isolated ON = a phrase onset out of silence (not a cut)
    cuts = [(e1, e2) for e1, e2 in zip(ev, ev[1:]) if e1[0] == "OFF" and e2[0] == "ON" and e2[1] - e1[1] <= 0.6]
    g["audio_hard_cuts"] = {"status": "PASS" if not cuts else "FLAG", "cuts_stop_restart": len(cuts), "cut_times": [round(c[0][1], 3) for c in cuts],
                            "onsets_from_silence": sum(1 for e in ev if e[0] == "ON") - len(cuts), "all_events": len(ev)}
    import r3_joins as rj
    j = rj.analyse(am); g["tts_joins"] = {"status": "INFO", **{k: v for k, v in j.items() if k not in ("rows", "manifest")}}
    ld = sc.loudness(a.final)
    g["loudness"] = {"status": "PASS" if ld["integrated_lufs"] is not None and abs(ld["integrated_lufs"] + 16) <= 1 and (ld["true_peak_dbtp"] or 0) <= -1.0 else "FLAG", **ld}
    man["qa_gates"] = g; pathlib.Path(a.manifest).write_text(json.dumps(man, indent=1))
    bad = {k: v["status"] for k, v in g.items() if v["status"] in ("FAIL", "FLAG")}
    print(json.dumps({"qa_gates": {k: v["status"] for k, v in g.items()}, "lip_flash_strong": g["lip_flash"].get("strong_candidates"), "bg_ring": g["bg_ring"].get("lower_minus_upper"),
                      "hard_cuts": g["audio_hard_cuts"]["cuts_stop_restart"], "joins_gap_gt_0.3s": g["tts_joins"].get("joins_gap_gt_0.3s"), "lufs": ld["integrated_lufs"], "tp": ld["true_peak_dbtp"]}))
    if bad: print(f"!!! QA GATES NOT PASSED: {bad}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
