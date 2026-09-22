#!/usr/bin/env python3
"""E1/E3 — burst-aligned TTS placement (optional path; the production aligner in run_clean_pipeline_05.py is untouched).

Baseline: one TTS clip per unit placed at the slot start, sped up only when longer than the slot, silence elsewhere.
This variant, per unit:
  1. original speech bursts = Silero VAD intervals of the source inside the unit slot (fallback: the whole slot);
  2. the TTS clip is split into phrases at its own pauses (Silero VAD on the TTS, 16 kHz);
  3. phrases are grouped onto bursts by cumulative-duration matching (in order, no reordering);
  4. every group is fitted into its burst window; the window may extend into the following silence up to the next burst
     (--extend on) and the speed-up is capped (--atempo-cap); overflow shifts the following groups; the unit is still cut
     to its slot (timeline never moves). Groups shorter than the window keep natural speed (no slow-down).
Writes <out>/work_<id>/aligned_u*.wav + dubbed_24k/16k.wav + dubbed_aac.m4a + e1_alignment.json (per-burst placements).
    python scripts/quality/e1_burst_align.py --id 04 --out /tmp/dabai_quality/candidates/e1 [--atempo-cap 1.3] [--extend on|off] [--mode burst|slot]
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys
import numpy as np
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot"))
Q = pathlib.Path("/tmp/dabai_quality"); TTS_SR = 24000; SR16 = 16000


def ff(args):
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-nostdin", *args], check=True)


def vad_intervals(wav16: pathlib.Path, min_silence_ms=50, min_speech_ms=60) -> list[tuple[float, float]]:
    import torch
    from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
    torch.set_num_threads(4); model = load_silero_vad(); audio = read_audio(str(wav16), sampling_rate=SR16)
    ts = get_speech_timestamps(audio, model, sampling_rate=SR16, return_seconds=True, min_silence_duration_ms=min_silence_ms, min_speech_duration_ms=min_speech_ms, speech_pad_ms=40)
    return [(float(t["start"]), float(t["end"])) for t in ts]


def valley_split(y: np.ndarray, sr: int, span: tuple[float, float], k: int) -> list[tuple[float, float]]:
    """Split TTS span into k pieces at the lowest-energy points near the proportional cut positions (+-20 % of a piece)."""
    s0, e0 = span
    if k <= 1 or e0 - s0 < 0.6: return [span]
    hop = int(0.01 * sr); seg = y[int(s0 * sr):int(e0 * sr)]; n = len(seg) // hop
    rms = np.array([np.sqrt(np.mean(seg[i * hop:(i + 1) * hop] ** 2)) for i in range(n)]); rms = np.convolve(rms, np.ones(5) / 5, mode="same")
    piece = n / k; cuts = []
    for j in range(1, k):
        c = int(j * piece); lo, hi = max(1, int(c - 0.2 * piece)), min(n - 1, int(c + 0.2 * piece))
        if hi <= lo: continue
        cuts.append(lo + int(np.argmin(rms[lo:hi])))
    pts = [0] + sorted(set(cuts)) + [n]
    return [(s0 + a * hop / sr, s0 + b * hop / sr) for a, b in zip(pts, pts[1:]) if b > a]


def group_phrases(phrases: list[tuple[float, float]], n_bursts: int, burst_dur: list[float]) -> list[list[int]]:
    """Assign consecutive phrases to bursts so cumulative TTS duration follows cumulative burst duration."""
    if n_bursts <= 1 or len(phrases) <= 1: return [list(range(len(phrases)))] + [[] for _ in range(n_bursts - 1)]
    tot_t = sum(e - s for s, e in phrases); tot_b = sum(burst_dur); groups = [[] for _ in range(n_bursts)]; j = 0; acc_t = 0.0; acc_b = burst_dur[0]
    for i, (s, e) in enumerate(phrases):
        d = e - s
        # move to the next burst when adding this phrase would overshoot the burst's share more than leaving it out (and bursts remain)
        while j < n_bursts - 1 and groups[j] and (acc_t + d / 2) / tot_t > acc_b / tot_b:
            j += 1; acc_b += burst_dur[j]
        groups[j].append(i); acc_t += d
    return groups


def main() -> int:
    import soundfile as sf
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--id", required=True); ap.add_argument("--run-dir", default=str(Q / "baseline")); ap.add_argument("--work-dir", default=str(Q / "work"))
    ap.add_argument("--out", required=True); ap.add_argument("--atempo-cap", type=float, default=1.3); ap.add_argument("--hard-cap", type=float, default=1.6)
    ap.add_argument("--extend", choices=["on", "off"], default="on"); ap.add_argument("--mode", choices=["burst", "slot"], default="burst", help="slot = baseline placement but with extension/cap (E3 only)")
    ap.add_argument("--gap", type=float, default=0.06, help="minimum silence kept between groups (s)")
    ap.add_argument("--fill-slowdown", type=float, default=1.0, help="E1b: allow atempo down to this value (<1 = slow down) so a short group fills its burst (1.0 = never slow down)")
    a = ap.parse_args()
    man = json.loads((pathlib.Path(a.run_dir) / f"{a.id}_baseline.manifest.json").read_text()); src_work = pathlib.Path(a.work_dir) / f"work_{a.id}"
    out = pathlib.Path(a.out) / f"work_{a.id}"; out.mkdir(parents=True, exist_ok=True)
    duration = man["source"]["duration_s"]; units = man["units"]; vad = [(v["start"], v["end"]) for v in man["speech_intervals"]]
    report = {"id": a.id, "mode": a.mode, "atempo_cap": a.atempo_cap, "extend": a.extend, "fill_slowdown": a.fill_slowdown, "units": []}
    for i, u in enumerate(units):
        slot_s, slot_e = u["slot"]["start"], u["slot"]["end"]; next_start = units[i + 1]["slot"]["start"] if i + 1 < len(units) else duration
        tts = pathlib.Path(u["tts"]["file"]); tts_s = u["tts"]["seconds"]
        bursts = [(max(s, slot_s), min(e, slot_e)) for s, e in vad if min(e, slot_e) - max(s, slot_s) > 0.15] if a.mode == "burst" else []
        if not bursts: bursts = [(slot_s, slot_e)]
        # phrases of the TTS
        t16 = out / f"tts16_u{u['id']:02d}.wav"; ff(["-i", str(tts), "-ar", str(SR16), "-ac", "1", "-c:a", "pcm_s16le", str(t16)])
        y, sr = sf.read(str(tts), dtype="float32"); assert sr == TTS_SR
        if y.ndim > 1: y = y.mean(axis=1)
        raw = vad_intervals(t16) or [(0.0, tts_s)]
        speech_span = (max(0.0, raw[0][0] - 0.04), min(tts_s, raw[-1][1] + 0.06))          # leading / trailing silence of the TTS is dropped
        trimmed_s = speech_span[1] - speech_span[0]
        # phrases = VAD chunks, merged over pauses < 0.15 s; if still fewer than bursts, cut at energy valleys
        phrases = []
        for s_, e_ in raw:
            if phrases and s_ - phrases[-1][1] < 0.15: phrases[-1] = (phrases[-1][0], e_)
            else: phrases.append((s_, e_))
        if len(phrases) < len(bursts):
            need = len(bursts); pieces = []
            for s_, e_ in phrases:   # split the longest phrases first, proportionally
                pieces.append([s_, e_])
            while len(pieces) < need:
                j = max(range(len(pieces)), key=lambda k: pieces[k][1] - pieces[k][0]); s_, e_ = pieces[j]
                sub = valley_split(y, sr, (s_, e_), 2)
                if len(sub) < 2: break
                pieces[j:j + 1] = [list(x) for x in sub]
            phrases = [tuple(p) for p in pieces]
        phrases = [(max(0.0, s_ - 0.03), min(tts_s, e_ + 0.05)) for s_, e_ in phrases]
        groups = group_phrases(phrases, len(bursts), [e - s for s, e in bursts])
        # a group that would need more than the cap while the NEXT burst got nothing: hand it the next burst's window (merge) instead of rushing
        for j in range(len(bursts) - 1):
            if groups[j] and not groups[j + 1]:
                g = sum(phrases[k][1] - phrases[k][0] for k in groups[j]); win_end_j = min(next_start - a.gap, bursts[j + 1][0] - a.gap) if a.extend == "on" else bursts[j][1]
                if g / max(win_end_j - bursts[j][0], 0.2) > a.atempo_cap:
                    bursts[j + 1] = (bursts[j][0], bursts[j + 1][1]); groups[j + 1] = groups[j]; groups[j] = []; bursts[j] = (bursts[j][0], bursts[j][0])
        bursts = [b for b, g in zip(bursts, groups) if g or b[1] > b[0]]; groups = [g for g in groups if g] if all(b[1] > b[0] for b in bursts) else groups
        if len(groups) != len(bursts):   # keep them aligned after merges
            pairs = [(b, g) for b, g in zip(bursts, groups) if g]; bursts = [b for b, _ in pairs]; groups = [g for _, g in pairs]
        n_slot = int(round((slot_e - slot_s) * TTS_SR)); track = np.zeros(n_slot, dtype=np.float32); cursor = slot_s; placements = []
        for j, (bs, be) in enumerate(bursts):
            idx = groups[j]
            if not idx: placements.append({"burst": [bs, be], "phrases": [], "note": "no phrase assigned"}); continue
            seg = np.concatenate([y[int(phrases[k][0] * sr):int(phrases[k][1] * sr)] for k in idx]); g = len(seg) / sr
            start = max(bs, cursor)
            win_end = (min(next_start - a.gap, bursts[j + 1][0] - a.gap) if j + 1 < len(bursts) else min(slot_e, next_start - a.gap)) if a.extend == "on" else min(be, slot_e)
            win_end = max(win_end, start + 0.2); win = win_end - start; ratio = g / win; tempo = 1.0
            if ratio < 0.995 and a.fill_slowdown < 1.0:   # E1b: stretch a short group towards the ORIGINAL burst end (never beyond it), floor at --fill-slowdown
                target = min(be, win_end) - start
                if target > g: tempo = max(a.fill_slowdown, g / target)
                p_in = out / f"grp_u{u['id']:02d}_{j}.wav"; p_out = out / f"grp_u{u['id']:02d}_{j}_t.wav"; sf.write(str(p_in), seg, sr, subtype="PCM_16")
                if tempo < 0.999:
                    ff(["-i", str(p_in), "-af", f"atempo={tempo:.6f}", "-ar", str(TTS_SR), "-ac", "1", "-c:a", "pcm_s16le", str(p_out)]); seg, _ = sf.read(str(p_out), dtype="float32"); p_out.unlink()
                p_in.unlink()
            if ratio > 1.005:
                tempo = min(ratio, a.atempo_cap)
                if g / tempo > win and j + 1 < len(bursts) and (g / tempo - win) > 0.4:   # collides with the next burst by > 0.4 s: hard cap; smaller overflows just shift the next group
                    tempo = min(ratio, a.hard_cap)
                p_in = out / f"grp_u{u['id']:02d}_{j}.wav"; p_out = out / f"grp_u{u['id']:02d}_{j}_t.wav"; sf.write(str(p_in), seg, sr, subtype="PCM_16")
                ff(["-i", str(p_in), "-af", f"atempo={tempo:.6f}", "-ar", str(TTS_SR), "-ac", "1", "-c:a", "pcm_s16le", str(p_out)]); seg, _ = sf.read(str(p_out), dtype="float32")
                p_in.unlink(); p_out.unlink()
            a0 = int(round((start - slot_s) * sr)); b0 = min(n_slot, a0 + len(seg))
            if b0 > a0: track[a0:b0] += seg[:b0 - a0]
            end = start + len(seg) / sr; cut = max(0.0, end - slot_e)
            placements.append({"burst": [round(bs, 3), round(be, 3)], "window": [round(start, 3), round(win_end, 3)], "phrases": idx, "group_s": round(g, 3), "ratio": round(ratio, 3),
                               "atempo": round(tempo, 3), "placed": [round(start, 3), round(min(end, slot_e), 3)], "overflow_into_next_burst_s": round(max(0.0, end - win_end), 3), "cut_at_slot_end_s": round(cut, 3)})
            cursor = end + a.gap
        aligned = out / f"aligned_u{u['id']:02d}.wav"; sf.write(str(aligned), track, sr, subtype="PCM_16"); t16.unlink()
        report["units"].append({"id": u["id"], "slot": [slot_s, slot_e], "tts_s": tts_s, "tts_speech_s": round(trimmed_s, 3), "bursts": len(bursts), "phrases": len(phrases), "baseline_ratio": u["alignment"]["ratio_tts_to_slot"],
                                "baseline_atempo": u["alignment"]["atempo"], "placements": placements, "max_atempo": max(p.get("atempo", 1.0) for p in placements), "file": str(aligned)})
        print(f"u{u['id']:02d} slot {slot_s:.2f}-{slot_e:.2f} tts {tts_s:.2f}s bursts {len(bursts)} phrases {len(phrases)} -> " + " ".join(f"[{p['placed'][0]:.2f}-{p['placed'][1]:.2f} x{p['atempo']}]" for p in placements if p.get("placed")), flush=True)
    # dubbed track (same construction as the runner)
    n = int(round(duration * TTS_SR)); track = np.zeros(n, dtype=np.float32)
    for r in report["units"]:
        yy, sr = sf.read(r["file"], dtype="float32"); a0 = int(round(r["slot"][0] * sr)); b0 = min(n, a0 + len(yy)); track[a0:b0] += yy[:b0 - a0]
    peak = float(np.abs(track).max()); track *= (0.99 / peak) if peak > 0.99 else 1.0
    w24 = out / "dubbed_24k.wav"; sf.write(str(w24), track, TTS_SR, subtype="PCM_16"); w16 = out / "dubbed_16k.wav"; ff(["-i", str(w24), "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", str(w16)])
    m4a = out / "dubbed_aac.m4a"; ff(["-i", str(w24), "-c:a", "aac", "-b:a", "192k", str(m4a)])
    ats = [p["atempo"] for r in report["units"] for p in r["placements"] if p.get("placed")]
    report["summary"] = {"groups": len(ats), "atempo_max": max(ats), "atempo_min": min(ats), "slowed_lt_1": sum(x < 0.995 for x in ats), "atempo_gt_1_15": sum(x > 1.15 for x in ats), "atempo_gt_1_25": sum(x > 1.25 for x in ats),
                         "overflow_groups": sum(1 for r in report["units"] for p in r["placements"] if p.get("overflow_into_next_burst_s", 0) > 0.05),
                         "cut_groups": sum(1 for r in report["units"] for p in r["placements"] if p.get("cut_at_slot_end_s", 0) > 0.05), "dubbed_24k": str(w24), "dubbed_16k": str(w16), "aac": str(m4a)}
    (out / "e1_alignment.json").write_text(json.dumps(report, indent=1, ensure_ascii=False)); print("summary", json.dumps(report["summary"])); return 0


if __name__ == "__main__":
    raise SystemExit(main())
