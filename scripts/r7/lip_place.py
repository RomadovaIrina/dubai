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
r8 --pace on (user on final_r7 04: "strange starts and pauses"; scripts/r8/pacing.py lists them): the DP only counted frame overlap, so it pulled
a sentence 0.9 s into the previous sentence's source speech (04 u1 "City of Sochi" glued 0.1 s after u0, then 6.7 s of dub silence) or started
one 1.3-1.5 s late inside a source pause (04 u3 / u4). With --pace on:
  * onset anchor: a sentence's first block pays onset_early per frame it starts > onset_tol before the speaker starts that sentence, onset_late per
    frame > onset_tol after (face-visible onsets only);
  * two sentences keep >= min(their r6 gap, unit_gmin) between them (was 0.15 s);
  * gate blocks: the take's noisy pauses that the remix's dub gate mutes (r3_remix.dub_gate runs) are block boundaries here too (04 3.9-5.7 s:
    a 1.7 s take pause over a 0.8 s source pause could not shrink, it was inside one block);
  * dictation splits: a block may also split at a word boundary with no punctuation (not after a function word) when the source pauses
    >= dsplit_src_pause there; such a cut costs dsplit_cost s and may open up to dsplit_gap s (a speaker who dictates word by word: 04 u1).
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


def blocks_of(L, min_gap_s=0.08, cut_rel_db=-35.0, speech_rel_db=-30.0, quiet=None):
    """blocks [a, b) in frames between real pauses (>= min_gap_s below p95 + cut_rel_db); act = speech frames (p95 + speech_rel_db);
    quiet = frames that count as pause whatever their level (r8 --gate-blocks: the dub gate's runs)"""
    n = len(L); ref = np.percentile(L[L > -100], 95) if (L > -100).any() else 0.0; loud = L > ref + cut_rel_db; act = L > ref + speech_rel_db
    if quiet is not None: loud &= ~quiet; act &= ~quiet
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
    """B = [[a, b, bound_frames, kind_of_gap_before, core_a, core_b, src_onset]] (tile [a, b) is rendered; gaps / limits use the loud core;
    kind: 'r6' / 'unit' / 'split' / 'dsplit'; src_onset = frame where the speaker starts this block's sentence, -1 = not a sentence start);
    returns (shifts in frames, cost)"""
    K = max(b[2] for b in B); sh = np.arange(-K, K + 1); dil = 7   # dub speech dilated by 0.15 s (r4_mouth_audio)
    cost = np.full((len(B), len(sh)), np.inf); arg = np.zeros((len(B), len(sh)), int)
    for i, (a, b, bd, kind, ca, cb, *ex) in enumerate(B):
        m = np.flatnonzero(np.convolve(act[a:b].astype(float), np.ones(2 * dil + 1), "same") > 0) + a
        ok = (np.abs(sh) <= bd) & (ca + sh >= 0) & (cb + sh <= n) & (m.min() + sh >= 0 if len(m) else True) & (m.max() + sh < n if len(m) else True)
        bc = np.array([float(c[m + k].sum()) if o else np.inf for k, o in zip(sh, ok)]) + rule["lam"] * np.abs(sh)
        on = ex[0] if ex else None
        if rule.get("onset") and (on is not None or (kind != "unit" and i and ca - B[i - 1][5] >= rule["onset"][5])):
            # r8 onset anchor (frames, face-visible onsets): a sentence's first block starts at one of ITS source speech onsets (before the first
            # one: onset_early per frame beyond tol; elsewhere: onset_late per frame of distance to the nearest of them beyond tol); a block after
            # a dub pause >= inner_gap starts at the nearest source onset (onset_inner per frame beyond tol, capped at inner_cap)
            we, wl, tol, fw, ons, ig, wi, cap = rule["onset"]; x = ca + sh
            if on is not None:
                first, own = on; dist = np.abs(x[:, None] - own[None, :]).min(1)
                oc = np.where(x < first - tol, we * (first - tol - x), wl * np.maximum(0, dist - tol))
                bc = bc + fw[min(first, len(fw) - 1)] * oc
            elif len(ons):
                dist = np.abs(x[:, None] - ons[None, :]).min(1); bc = bc + fw[np.clip(x, 0, len(fw) - 1)] * wi * np.minimum(cap, np.maximum(0, dist - tol))
        if i == 0: cost[0] = bc; continue
        g0 = ca - B[i - 1][5]
        gmin = 0 if kind in ("split", "dsplit") else min(g0, rule["ugmin"] if kind == "unit" else rule["gmin"])
        gmax = g0 + (rule["split_gap"] if kind == "split" else rule["dsplit_gap"] if kind == "dsplit" else rule["grow"] if kind == "r6" else rule["grow_unit"])
        for j, k in enumerate(sh):
            if not np.isfinite(bc[j]): continue
            lo, hi = max(-K, k + g0 - gmax), min(K, k + g0 - gmin)
            if lo > hi: continue
            seg = cost[i - 1, lo + K:hi + K + 1]; mm = int(np.argmin(seg))
            if np.isfinite(seg[mm]): cost[i, j] = seg[mm] + bc[j]; arg[i, j] = lo + K + mm
    j = int(np.argmin(cost[-1])); best = float(cost[-1, j]); out = [0] * len(B)
    for i in range(len(B) - 1, -1, -1): out[i] = int(sh[j]); j = arg[i, j]
    return out, best


FUNC = {"the", "a", "an", "and", "or", "but", "to", "of", "in", "on", "at", "for", "with", "we", "i", "it", "that", "this", "so", "is", "are", "was", "be",
        "our", "my", "you", "they", "he", "she", "as", "by", "from", "which", "who", "will", "would", "can", "have", "has", "very", "just", "really"}


def split_candidates(dub16, man, L, ref, blocks, dbg=None, quiet_rel_db=-25.0, inside=False):
    """logical English boundaries inside blocks: [(frame of the quietest point between the two words, class)]; inside = also word boundaries
    with no punctuation (class 'INSIDE'), never right after a function word ("the |", "we are |")"""
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
        if st <= 0 and not (inside and pl.norm(S[w2s[i]][0]) not in FUNC): continue
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
    ap.add_argument("--splits", choices=["on", "off"], default="on")
    ap.add_argument("--pace", choices=["on", "off"], default="off", help="r8: onset anchor + unit gap floor + gate blocks + dictation splits (defaults below)")
    ap.add_argument("--onset-early", type=float); ap.add_argument("--onset-late", type=float); ap.add_argument("--onset-tol", type=float, default=0.2)
    ap.add_argument("--unit-gmin", type=float); ap.add_argument("--gate-blocks", choices=["on", "off"]); ap.add_argument("--dsplit", choices=["on", "off"])
    ap.add_argument("--cover-all", choices=["on", "off"], default="off", help="r8 --cover-all on (management on final_r7 02 / 05: the dub keeps disappearing): off-screen source speech counts like face-visible speech")
    ap.add_argument("--onset-inner", type=float); ap.add_argument("--inner-gap", type=float, default=0.3); ap.add_argument("--inner-cap", type=float, default=1.0)
    ap.add_argument("--off-early", type=float, default=0.0, help="r9 (02 'the pauses are long'): OFF screen (face not visible), a unit whose start leaves a dub gap "
                    "longer than the source's own silence there by > 0.3 s starts up to this many seconds earlier (the whole unit, its inner gaps kept; timing only, "
                    "not before the unit gap floor); 0 = off")
    ap.add_argument("--dsplit-src-pause", type=float, default=1.0); ap.add_argument("--dsplit-cost", type=float, default=0.8); ap.add_argument("--dsplit-gap", type=float, default=4.0)
    a = ap.parse_args(); P = a.pace == "on"
    for k, on_v, off_v in (("onset_early", 2.0, 0.0), ("onset_late", 1.0, 0.0), ("onset_inner", 0.5, 0.0), ("unit_gmin", 0.3, 0.15), ("gate_blocks", "on", "off"), ("dsplit", "on", "off")):
        if getattr(a, k) is None: setattr(a, k, on_v if P else off_v)
    W, O = pathlib.Path(a.work).resolve(), pathlib.Path(a.out); O.mkdir(parents=True, exist_ok=True); man = json.loads(pathlib.Path(a.manifest).read_text())
    y, sr = sf.read(str(W / "dubbed_24k.wav"), dtype="float32"); y = y if y.ndim == 1 else y.mean(1)
    L = frames_db(y, sr); n = len(L); quiet = None; gate_runs = []
    if a.gate_blocks == "on":   # the remix's dub gate (same parameters) decides which take pauses are noise to be muted: those are pauses here too
        import r3_remix, tempfile
        with tempfile.TemporaryDirectory() as td:
            gate_runs = r3_remix.dub_gate(W / "dubbed_24k.wav", pathlib.Path(td) / "g24.wav", pathlib.Path(td) / "g16.wav")["runs"]
        quiet = np.zeros(n, bool)
        for g0_, g1_ in gate_runs: quiet[int(np.ceil((g0_ + 0.02) / FR)):int((g1_ - 0.02) / FR)] = True
    blocks, cores, act, ref = blocks_of(L, quiet=quiet)
    S = np.zeros(n, bool)
    for s in man["speech_intervals"]: S[int(s["start"] / FR):int(np.ceil(s["end"] / FR))] = True
    fv, fvis = face_mask(a.face, n, man["source"]["duration_s"]); fv_real = fv.copy()   # r9: the real face mask (--off-early)
    if a.cover_all == "on": fv = np.ones(n, bool)   # r8: off-screen speech placed like on-screen speech (02: face_visible 0, nothing moved before)
    w = np.where(fv, 1.0, a.w_off); c = w * (1 - 2 * S.astype(float))
    pls = [(u["id"], p) for u in man["units"] for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed")]
    def unit_at(f0, f1):
        t0, t1 = f0 * FR, f1 * FR; ov = [(min(t1, p["placed"][1]) - max(t0, p["placed"][0]), uid) for uid, p in pls]; o = max(ov, default=(0, None))
        return o[1] if o[0] > 0 else None
    d_ = np.diff(np.concatenate([[0], S.astype(np.int8), [0]])); ons = np.flatnonzero(d_ == 1)   # source speech onsets (frames)
    src_on = {}   # unit -> (frame where the speaker starts it, every source onset inside its bursts): the first burst onset moved to the next speech
    for u in man["units"]:
        ps = [p for p in (u.get("alignment") or {}).get("placements", []) if p.get("placed")]
        if not ps: continue
        f0 = int(round(ps[0]["burst"][0] / FR)); f1 = int(round(ps[0]["burst"][1] / FR)); fe = int(round(ps[-1]["burst"][1] / FR)); nz = np.flatnonzero(S[f0:f1])
        first = int(f0 + nz[0]) if len(nz) else f0; src_on[u["id"]] = (first, np.unique(np.concatenate([[first], ons[(ons > first) & (ons < fe)]])))
    anchor = bool(a.onset_early or a.onset_late or a.onset_inner)
    mx, ms = int(round(a.max_shift / FR)), int(round(a.split_shift / FR)); B = []
    for i, (b0, b1) in enumerate(blocks):
        uid = unit_at(*cores[i]); kind = "r6" if i and uid == unit_at(*cores[i - 1]) and uid is not None else "unit"
        B.append([b0, b1, mx, kind, *cores[i], src_on.get(uid) if kind == "unit" and anchor else None])
    rule = {"lam": a.lam, "gmin": int(round(0.15 / FR)), "ugmin": int(round(a.unit_gmin / FR)), "grow": int(round(a.grow / FR)), "split_gap": int(round(a.split_gap / FR)),
            "grow_unit": int(round(a.grow_unit / FR)), "dsplit_gap": int(round(a.dsplit_gap / FR)),
            "onset": (a.onset_early, a.onset_late, int(round(a.onset_tol / FR)), w, ons, int(round(a.inner_gap / FR)), a.onset_inner, int(round(a.inner_cap / FR))) if anchor else None}
    shifts, best = plan(B, act, c, n, rule); base = best; splits = []; dbg = []
    if a.splits == "on" and fv.any():
        allc = split_candidates(W / "dubbed_16k.wav", man, L, ref, cores, dbg := [], inside=a.dsplit == "on")
        Sl = np.zeros(n, bool)   # source pauses >= dsplit_src_pause (a dictation split must face one)
        d_ = np.diff(np.concatenate([[0], (~S).astype(np.int8), [0]]))
        for p0, p1 in zip(np.flatnonzero(d_ == 1), np.flatnonzero(d_ == -1)):
            if (p1 - p0) * FR >= a.dsplit_src_pause: Sl[p0:p1] = True
        cands = [(f, cls) for f, cls in allc if fv[max(0, f - 25):f + 25].mean() > 0.5]; pens = 0.0
        for _ in range(a.max_splits):
            trial = None
            for f, cls in cands:
                i = next((k for k, b in enumerate(B) if b[4] + 10 < f < b[5] - 10), None)
                if i is None: continue
                if min(act[B[i][0]:f].sum(), act[f:B[i][1]].sum()) * FR < a.min_piece: continue   # no orphan word: >= min_piece s of speech on both sides
                ds = cls == "INSIDE"; pen = a.dsplit_cost / FR if ds else 0.0
                if ds and not Sl[max(0, f - int(a.max_shift / FR)):f + int(a.dsplit_gap / FR)].any(): continue   # no long source pause near: no dictation split
                B2 = B[:i] + [[B[i][0], f, B[i][2], B[i][3], B[i][4], f, *B[i][6:]], [f, B[i][1], ms, "dsplit" if ds else "split", f, B[i][5], None]] + B[i + 1:]
                s2, c2 = plan(B2, act, c, n, rule); c2 += pens + pen
                if trial is None or c2 < trial[0]: trial = (c2, B2, s2, f, cls, pen)
            if trial is None or best - trial[0] < a.split_gain / FR: break
            best, B, shifts = trial[0], trial[1], trial[2]; pens += trial[5]; splits.append([round(trial[3] * FR, 2), trial[4]]); cands = [q for q in cands if q[0] != trial[3]]
    off_moves = []
    if a.off_early > 0:   # r9: off screen, close a between-unit dub gap that is longer than the source's silence there (the speaker talks, the dub waits)
        E, tolf, ug = int(round(a.off_early / FR)), int(round(0.3 / FR)), rule["ugmin"]
        for i in range(1, len(B)):
            if B[i][3] != "unit": continue
            ca, pe = B[i][4] + shifts[i], B[i - 1][5] + shifts[i - 1]   # this unit's first block start / the previous block's end (dub frames, loud cores)
            if fv_real[max(0, ca - 25):ca + 25].mean() >= 0.5: continue   # on screen: the lips decide
            gap = ca - pe; ex = gap - int((~S[max(0, pe):max(0, ca)]).sum())   # the part of the dub gap where the source speaks
            js = [i] + [j for j in range(i + 1, len(B)) if all(B[q][3] != "unit" for q in range(i + 1, j + 1))]   # the unit's blocks
            mv = min(E, ex - tolf, gap - ug, min(B[j][2] + shifts[j] for j in js))
            if mv <= 0: continue
            for j in js: shifts[j] -= mv
            off_moves.append([round(ca * FR, 2), round(mv * FR, 2)])
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
    rep = {"blocks": len(B), "splits": splits, "off_early_moves": off_moves, "moved_blocks": int(sum(1 for k in shifts if k)), "shift_abs_mean_s": round(float(np.mean(np.abs(shifts))) * FR, 3),
           "shift_abs_max_s": round(float(np.max(np.abs(shifts))) * FR, 3), "face_visible_frac": fvis,
           "mismatch_face_visible_s_before": err(D0), "mismatch_face_visible_s_after": err(D1), "cost_shift_only": round(base * FR, 2), "cost_new": round(best * FR, 2),
           "params": {k: getattr(a, k) for k in ("max_shift", "split_shift", "lam", "w_off", "grow", "grow_unit", "split_gap", "min_piece", "split_gain", "max_splits", "splits", "pace", "onset_early", "onset_late", "onset_inner", "inner_gap", "inner_cap", "onset_tol", "unit_gmin", "gate_blocks", "dsplit", "dsplit_src_pause", "dsplit_cost", "dsplit_gap")}, "gate_runs": gate_runs,
           "shifts": [[round(ca * FR, 2), round(cb * FR, 2), round(k * FR, 2), kind] for (_, _, _, kind, ca, cb, *_), k in zip(B, shifts)],
           "split_boundaries": dbg, "dub_speech": [[round(p * FR, 2), round(q * FR, 2)] for p, q in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1))]}
    man["lip_place"] = rep; man["dubbed_track"] = {**man.get("dubbed_track", {}), "lip_placed": True}
    (O / pathlib.Path(a.manifest).name).write_text(json.dumps(man, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("shifts", "dub_speech")}))


if __name__ == "__main__":
    main()
