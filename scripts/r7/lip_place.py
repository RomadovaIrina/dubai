#!/usr/bin/env python3
"""r7 --lip-place: while the face is visible, put the dub speech INSIDE the source speech spans (m5 = m4 + r6 audio lost lip sync because of where
r6 places speech: 05 mouth/dub mismatch 1.74 -> 2.64 s, 04 0.77 -> 2.60 s; e.g. 04 u2: one 6.5 s English phrase laid over a 2.3 s source pause,
then silent while the speaker talks). r6's takes, tempo and fades stay exactly as they are:
  * the finished dub track (dubbed_24k.wav) is cut into BLOCKS = runs between real dub pauses (>= 80 ms, 35 dB below the take's p95);
  * where the face is visible a block may also be SPLIT at a logical English boundary (Whisper word times on the dub + r4_pause_logic.classify:
    sentence end, ; : —, comma, clause start: exactly the places pb1 allows a pause) at a quiet point between the two words;
  * every block moves by up to +-max_shift_s (a piece after a new split by up to +-split_shift_s); a DP over the blocks (20 ms grid) minimises
        sum over frames of  w(t) * ([dub speaks & source silent] + [source speaks & dub silent])   (r4_mouth_audio's two error kinds; dub speech
        dilated by 0.15 s as that metric does)  +  lam * |shift|
    with w = 1 where MediaPipe sees a face on the source, w_off (0: r6 placement kept off screen) elsewhere;
  * order kept; an r6 gap never shrinks below min(its length, 0.15 s) nor grows by more than grow_s inside one sentence (grow_unit_s = 0.8 s
    between sentences: r7t3 05 12.8 s, a block moved 1 s onto its source span left a 6.3 s dub gap over a slow speaker); a new split opens
    0..split_gap_s (1.5 s) and leaves >= min_piece_s (0.6 s) of speech on both sides (r7t4 05 17.3 s: a 0.2 s orphan, then a 1.7 s pause); splits are added greedily while each one removes >= split_gain_s of mismatch (max max_splits).
Writes <out>/ = symlinks of the work dir + shifted dubbed_24k.wav / dubbed_16k.wav + the manifest with every placement moved (placed, track_start,
spread pause times; "lip_shift_s") and man["lip_place"] (blocks, shifts, splits, dub_speech = the new dub speech spans).
    python scripts/r7/lip_place.py --work W --out W2 --manifest W/<id>_r4a.manifest.json --face A/sep/<id>/source_mediapipe.npz"""
from __future__ import annotations
import argparse, json, pathlib, sys
import numpy as np, soundfile as sf
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot")); sys.path.insert(0, str(HERE.parent / "quality"))
FR = 0.02


def frames_db(y, sr):
    h = int(FR * sr); n = len(y) // h
    return 20 * np.log10(np.sqrt((y[:n * h].reshape(n, h) ** 2).mean(1)) + 1e-9)


def blocks_of(L, min_gap_s=0.08, cut_rel_db=-35.0, speech_rel_db=-30.0):
    """blocks [a, b) in frames between real pauses (>= min_gap_s below p95 + cut_rel_db); act = speech frames (p95 + speech_rel_db)"""
    n = len(L); ref = np.percentile(L[L > -100], 95) if (L > -100).any() else 0.0; loud = L > ref + cut_rel_db; act = L > ref + speech_rel_db
    d = np.diff(np.concatenate([[0], loud.astype(np.int8), [0]])); bl = []
    for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        if bl and a - bl[-1][1] < int(round(min_gap_s / FR)): bl[-1][1] = int(b)
        else: bl.append([int(a), int(b)])
    # blocks tile the whole track, cut in the middle of each pause (tails / breaths / the take's own low-level noise move with their block, and a
    # video where nothing moves is rendered bit for bit as r6)
    tiles = [[(bl[i - 1][1] + a) // 2 if i else 0, (b + bl[i + 1][0]) // 2 if i + 1 < len(bl) else n] for i, (a, b) in enumerate(bl)]
    return tiles, bl, act, ref


def face_mask(face_npz, n, dur):
    if not face_npz or not pathlib.Path(face_npz).exists(): return np.ones(n, bool), None
    lm = np.load(face_npz)["lm"]; vis = np.isfinite(lm.reshape(len(lm), -1)).all(1); fps = len(vis) / dur
    return vis[np.minimum((np.arange(n) * FR * fps).astype(int), len(vis) - 1)], round(float(vis.mean()), 3)


def plan(B, act, c, n, rule):
    """B = [[a, b, bound_frames, kind_of_gap_before, core_a, core_b]] (tile [a, b) is rendered; gaps / limits use the loud core;
    kind: 'r6' / 'unit' / 'split'); returns (shifts in frames, cost)"""
    K = max(b[2] for b in B); sh = np.arange(-K, K + 1); dil = 7   # dub speech dilated by 0.15 s (r4_mouth_audio)
    cost = np.full((len(B), len(sh)), np.inf); arg = np.zeros((len(B), len(sh)), int)
    for i, (a, b, bd, kind, ca, cb) in enumerate(B):
        m = np.flatnonzero(np.convolve(act[a:b].astype(float), np.ones(2 * dil + 1), "same") > 0) + a
        ok = (np.abs(sh) <= bd) & (ca + sh >= 0) & (cb + sh <= n) & (m.min() + sh >= 0 if len(m) else True) & (m.max() + sh < n if len(m) else True)
        bc = np.array([float(c[m + k].sum()) if o else np.inf for k, o in zip(sh, ok)]) + rule["lam"] * np.abs(sh)
        if i == 0: cost[0] = bc; continue
        g0 = ca - B[i - 1][5]
        gmin = 0 if kind == "split" else min(g0, rule["gmin"]); gmax = g0 + (rule["split_gap"] if kind == "split" else rule["grow"] if kind == "r6" else rule["grow_unit"])
        for j, k in enumerate(sh):
            if not np.isfinite(bc[j]): continue
            lo, hi = max(-K, k + g0 - gmax), min(K, k + g0 - gmin)
            if lo > hi: continue
            seg = cost[i - 1, lo + K:hi + K + 1]; mm = int(np.argmin(seg))
            if np.isfinite(seg[mm]): cost[i, j] = seg[mm] + bc[j]; arg[i, j] = lo + K + mm
    j = int(np.argmin(cost[-1])); best = float(cost[-1, j]); out = [0] * len(B)
    for i in range(len(B) - 1, -1, -1): out[i] = int(sh[j]); j = arg[i, j]
    return out, best


def split_candidates(dub16, man, L, ref, blocks, dbg=None, quiet_rel_db=-25.0):
    """logical English boundaries inside blocks: [(frame of the quietest point between the two words, class)]"""
    import r4_pause_logic as pl, difflib
    W = pl.whisper_words(dub16); S = pl.script_tokens(man)
    sm = difflib.SequenceMatcher(None, [pl.norm(s[0]) for s in S], [pl.norm(w[0]) for w in W], autojunk=False); w2s = {}
    for a, b, k in sm.get_matching_blocks():
        for q in range(k): w2s[b + q] = a + q
    out = []
    for i in range(len(W) - 1):
        if i not in w2s or i + 1 not in w2s or w2s[i + 1] != w2s[i] + 1: continue
        cls, st = pl.classify(S[w2s[i]], S[w2s[i] + 1])
        if dbg is not None: dbg.append([round(W[i][2], 2), W[i][0], W[i + 1][0], cls, st])
        if st <= 0: continue
        lo, hi = int((W[i][2] - 0.06) / FR), int((W[i + 1][1] + 0.06) / FR) + 1
        if hi - lo < 1 or lo < 0 or hi > len(L): continue
        f = lo + int(np.argmin(L[lo:hi]))
        if dbg is not None: dbg[-1].append(round(float(L[f] - ref), 1))
        if L[f] > ref + quiet_rel_db: continue   # no quiet point between the words: a cut there would click / clip a phoneme
        if any(a + 10 < f < b - 10 for a, b in blocks): out.append((f, cls))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--work", required=True); ap.add_argument("--out", required=True); ap.add_argument("--manifest", required=True)
    ap.add_argument("--face"); ap.add_argument("--max-shift", type=float, default=1.0); ap.add_argument("--split-shift", type=float, default=2.5)
    ap.add_argument("--lam", type=float, default=0.15); ap.add_argument("--w-off", type=float, default=0.0); ap.add_argument("--grow", type=float, default=0.35); ap.add_argument("--grow-unit", type=float, default=0.8)
    ap.add_argument("--split-gap", type=float, default=1.5); ap.add_argument("--min-piece", type=float, default=0.6); ap.add_argument("--split-gain", type=float, default=0.3); ap.add_argument("--max-splits", type=int, default=12)
    ap.add_argument("--splits", choices=["on", "off"], default="on"); a = ap.parse_args()
    W, O = pathlib.Path(a.work).resolve(), pathlib.Path(a.out); O.mkdir(parents=True, exist_ok=True); man = json.loads(pathlib.Path(a.manifest).read_text())
    y, sr = sf.read(str(W / "dubbed_24k.wav"), dtype="float32"); y = y if y.ndim == 1 else y.mean(1)
    L = frames_db(y, sr); n = len(L); blocks, cores, act, ref = blocks_of(L)
    S = np.zeros(n, bool)
    for s in man["speech_intervals"]: S[int(s["start"] / FR):int(np.ceil(s["end"] / FR))] = True
    fv, fvis = face_mask(a.face, n, man["source"]["duration_s"]); w = np.where(fv, 1.0, a.w_off); c = w * (1 - 2 * S.astype(float))
    pls = [(u["id"], p) for u in man["units"] for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed")]
    def unit_at(f0, f1):
        t0, t1 = f0 * FR, f1 * FR; ov = [(min(t1, p["placed"][1]) - max(t0, p["placed"][0]), uid) for uid, p in pls]; o = max(ov, default=(0, None))
        return o[1] if o[0] > 0 else None
    mx, ms = int(round(a.max_shift / FR)), int(round(a.split_shift / FR))
    B = [[b0, b1, mx, "r6" if i and unit_at(*cores[i]) == unit_at(*cores[i - 1]) and unit_at(*cores[i]) is not None else "unit", *cores[i]] for i, (b0, b1) in enumerate(blocks)]
    rule = {"lam": a.lam, "gmin": int(round(0.15 / FR)), "grow": int(round(a.grow / FR)), "split_gap": int(round(a.split_gap / FR)), "grow_unit": int(round(a.grow_unit / FR))}
    shifts, best = plan(B, act, c, n, rule); base = best; splits = []; dbg = []
    if a.splits == "on" and fv.any():
        allc = split_candidates(W / "dubbed_16k.wav", man, L, ref, cores, dbg := [])
        cands = [(f, cls) for f, cls in allc if fv[max(0, f - 25):f + 25].mean() > 0.5]
        for _ in range(a.max_splits):
            trial = None
            for f, cls in cands:
                i = next((k for k, b in enumerate(B) if b[4] + 10 < f < b[5] - 10), None)
                if i is None: continue
                if min(act[B[i][0]:f].sum(), act[f:B[i][1]].sum()) * FR < a.min_piece: continue   # no orphan word: >= min_piece s of speech on both sides
                B2 = B[:i] + [[B[i][0], f, B[i][2], B[i][3], B[i][4], f], [f, B[i][1], ms, "split", f, B[i][5]]] + B[i + 1:]
                s2, c2 = plan(B2, act, c, n, rule)
                if trial is None or c2 < trial[0]: trial = (c2, B2, s2, f, cls)
            if trial is None or best - trial[0] < a.split_gain / FR: break
            best, B, shifts = trial[0], trial[1], trial[2]; splits.append([round(trial[3] * FR, 2), trial[4]]); cands = [q for q in cands if q[0] != trial[3]]
    # render: every block's samples moved by its shift (blocks tile the track; v2: 20-ms equal-power (sin / cos) fades at cut points that move
    # apart, so two blocks pushed together crossfade at constant power and a cut that opens never steps)
    z = np.zeros_like(y); h = int(FR * sr); fd = int(0.02 * sr); ramp = np.sin(np.linspace(0, np.pi / 2, fd))
    for q, ((b0, b1, *_), k) in enumerate(zip(B, shifts)):
        i0, i1 = b0 * h, (min(len(y), b1 * h) if q + 1 < len(B) else len(y)); seg = y[i0:i1].copy()
        if len(seg) > 2 * fd:   # fades only where the neighbour moves differently (a cut point that opens / closes)
            if q and shifts[q - 1] != k: seg[:fd] *= ramp
            if q + 1 < len(B) and shifts[q + 1] != k: seg[-fd:] *= ramp[::-1]
        j0 = i0 + k * h; c0 = max(0, -j0); c1 = max(0, j0 + len(seg) - len(z)); seg = seg[c0:len(seg) - c1]; j0 = max(0, j0)
        z[j0:j0 + len(seg)] += seg
    D0 = act.copy(); D1 = np.zeros(n, bool)
    for (b0, b1, *_), k in zip(B, shifts):
        ix = np.flatnonzero(act[b0:b1]) + b0 + k; D1[ix[(ix >= 0) & (ix < n)]] = True
    err = lambda D: round(float(((D & ~S) | (S & ~D))[fv].sum() * FR), 2)
    for f in W.iterdir():
        if f.name in ("dubbed_24k.wav", "dubbed_16k.wav") or f.name.endswith(".manifest.json"): continue
        (O / f.name).unlink(missing_ok=True); (O / f.name).symlink_to(f)
    sf.write(str(O / "dubbed_24k.wav"), z, sr, subtype="PCM_16")
    import librosa
    sf.write(str(O / "dubbed_16k.wav"), librosa.resample(z, orig_sr=sr, target_sr=16000), 16000, subtype="PCM_16")
    def shift_at(t):   # shift of the block holding time t (r6 timeline)
        f = int(t / FR); k = next((q for q, b in enumerate(B) if b[0] <= f < b[1]), None)
        if k is None: k = int(np.argmin([min(abs(f - b[0]), abs(f - b[1])) for b in B]))
        return shifts[k] * FR
    for u in man["units"]:
        al = u.get("alignment") or {}
        for p in al.get("placements", []):
            if not p.get("placed"): continue
            s0, s1 = shift_at(p["placed"][0] + 0.03), shift_at(p["placed"][1] - 0.03); p["lip_shift_s"] = [round(s0, 2), round(s1, 2)]
            p["placed"] = [round(p["placed"][0] + s0, 3), round(p["placed"][1] + s1, 3)]
            if "track_start" in p: p["track_start"] = round(p["track_start"] + s0, 3)
            for q in (p.get("spread") or {}).get("pauses", []):
                if "t" in q: q["t"] = round(q["t"] + shift_at(q["t"]), 3)
        if "track_start" in al and al.get("placements"): al["track_start"] = round(al["track_start"] + shift_at(al["track_start"] + 0.05), 3)
    d = np.diff(np.concatenate([[0], D1.astype(np.int8), [0]]))
    rep = {"blocks": len(B), "splits": splits, "moved_blocks": int(sum(1 for k in shifts if k)), "shift_abs_mean_s": round(float(np.mean(np.abs(shifts))) * FR, 3),
           "shift_abs_max_s": round(float(np.max(np.abs(shifts))) * FR, 3), "face_visible_frac": fvis,
           "mismatch_face_visible_s_before": err(D0), "mismatch_face_visible_s_after": err(D1), "cost_shift_only": round(base * FR, 2), "cost_new": round(best * FR, 2),
           "params": {k: getattr(a, k) for k in ("max_shift", "split_shift", "lam", "w_off", "grow", "grow_unit", "split_gap", "min_piece", "split_gain", "max_splits", "splits")},
           "shifts": [[round(ca * FR, 2), round(cb * FR, 2), round(k * FR, 2), kind] for (_, _, _, kind, ca, cb), k in zip(B, shifts)],
           "split_boundaries": dbg, "dub_speech": [[round(p * FR, 2), round(q * FR, 2)] for p, q in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]}
    man["lip_place"] = rep; man["dubbed_track"] = {**man.get("dubbed_track", {}), "lip_placed": True}
    (O / pathlib.Path(a.manifest).name).write_text(json.dumps(man, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("shifts", "dub_speech")}))


if __name__ == "__main__":
    main()
