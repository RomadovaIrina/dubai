#!/usr/bin/env python3
"""r16 Q 08.11: Irina's watch list. Per scored video (r16q/watch/<id>/{det.json, det.series.npz, eval/, spring.json, status.json} + its job dir):
the 3-5 most suspicious seconds, with what kind and whether the detector fires at the same rate on the eye-accepted golden (= likely FP).
    python3 watch_sum.py ID:FINAL:JOBDIR ... > WATCH.md"""
import json, sys, glob, os, re
import numpy as np

W = "/workspace/dub/r16q/watch"; G = "/workspace/dub/r16q/gold/e2e"
# golden false-positive floor: the max flag_share of each detector over the 4 accepted finals (det_acc_*), also the e2e golden (det_e2e_*)
FLOOR = {}
for f in glob.glob(f"{G}/det_acc_*.json") + glob.glob(f"{G}/det_e2e_*.json"):
    for k, v in json.load(open(f))["det"].items():
        FLOOR[k] = max(FLOOR.get(k, 0.0), float(v.get("flag_share", 0.0)))
KIND = {"black": "black mouth", "slide": "slide (cheeks / upper)", "swap": "swap (other face)", "overlap": "overlap", "texture": "texture",
        "explode": "snap / shape explode", "edge": "edge seam", "flicker": "flicker"}
WEIGHT = {"black": 3, "explode": 3, "swap": 3, "edge": 2, "flicker": 1.5, "slide": 1, "overlap": 1, "texture": 0.5}


def mmss(t): return f"{int(t // 60)}:{t % 60:04.1f}"


def one(spec):
    vid, final, job = spec.split(":"); O = f"{W}/{vid}"; out = {"id": vid, "final": final}
    ev = {}; p = f"{O}/eval/eval.json"
    if os.path.exists(p): ev = json.load(open(p))
    st = json.load(open(f"{O}/status.json")) if os.path.exists(f"{O}/status.json") else {}
    det = json.load(open(f"{O}/det.json")) if os.path.exists(f"{O}/det.json") else {"det": {}}
    sp = json.load(open(f"{O}/spring.json")) if os.path.exists(f"{O}/spring.json") else {}
    dur = det.get("frames", 0) / 25.0
    cand = {}  # second -> list of (score, text)

    def add(t, score, txt):
        s = int(t); cand.setdefault(s, []).append((score, txt))
    rates = {}
    for k, v in det["det"].items():
        share = float(v.get("flag_share", 0.0)); fp = share <= 1.2 * FLOOR.get(k, 0.0) + 1e-9; rates[k] = (share, FLOOR.get(k, 0.0), fp)
        for r in v.get("ranges", [])[:400]:
            t0, t1, sc = r[0], r[1], (r[2] if len(r) > 2 else 1.0)
            add(t0, WEIGHT.get(k, 1) * (0.4 if fp else 1.0) * min(3.0, 1.0 + (t1 - t0)), f"{KIND.get(k, k)} {t0:.2f}-{t1:.2f}s ({sc:.2f})" + (" [golden-FP level]" if fp else ""))
    # chin slide series: top seconds (the chin moves with the mouth: watch only, low weight)
    zp = f"{O}/det.series.npz"
    if os.path.exists(zp):
        z = np.load(zp)
        if "slide_ch" in z.files:
            ch = z["slide_ch"]; n = len(ch) // 25
            sec = np.array([np.nanmean(ch[i * 25:(i + 1) * 25]) for i in range(n)]) if n else np.array([])
            if len(sec) and np.isfinite(sec).any():
                med = np.nanmedian(sec)
                for i in np.argsort(np.nan_to_num(sec, nan=-1))[::-1][:3]:
                    if sec[i] > 2.0 * med: add(i, 1.0, f"chin slide {sec[i]:.2f} (x{sec[i] / med:.1f} of the video median)")
    # sync per second (speaker track, dub speech): low conf at 0, or a clear best offset != 0
    ps = []
    for f in glob.glob(f"{O}/eval/sync_main.json"):
        ps = json.load(open(f)).get("per_second", [])
    lost = []
    SR = []   # on-screen speech of the lip-synced faces (route): off-screen / source-picture seconds are not "face lost"
    rp = f"{job}/e2e/work/route/route.json"
    if os.path.exists(rp):
        for fc in json.load(open(rp)).get("faces", []):
            if fc.get("mode") == "lipsync": SR += [tuple(x) for x in fc.get("speak_ranges", [])]
    onscreen = lambda t: (not SR) or any(a <= t + 0.5 <= b for a, b in SR)
    for p_ in ps:
        if (p_.get("dub_act") or 0) < 0.5 or not onscreen(p_["t"]): continue
        if p_.get("conf_at0") is None:
            lost.append(p_["t"]); continue
        if p_["conf_at0"] < 1.5: add(p_["t"], 2.0, f"low sync conf_at0 {p_['conf_at0']:.2f}")
        if p_.get("offset_best", 0) != 0 and p_["conf_best"] - p_["conf_at0"] > 2.0: add(p_["t"], 2.0, f"sync drift: best offset {p_['offset_best']:+d} fr (+{p_['conf_best'] - p_['conf_at0']:.1f})")
    # shot switches
    cuts = []
    sj = f"{job}/e2e/work/ks/shots.json"
    if os.path.exists(sj):
        try:
            S = json.load(open(sj)); cuts = [(b[0] / 25.0, b[1]) for b in S.get("bounds", [])]
            segs = S.get("segments", [])
            for a, b, t0, t1, kind in segs:
                if kind == "source": add(t0, 1.5, f"shot switch to SOURCE picture {t0:.1f}-{t1:.1f}s (no lip-sync there by design?)")
        except Exception: pass
    for t, why in cuts:
        add(t, 1.2, f"shot switch ({why}) at {t:.2f}s")
    # spring worst seconds
    sp_ok = bool(sp.get("final_metric")) and sp.get("floor_metric", {}).get("spring_scale_pct", [9])[0] < 0.6 and (sp.get("upper_ratio") or 9) < 1.5 and not cuts
    sp["ok"] = sp_ok   # multi-shot / camera-motion videos: the auto region spans cuts, worst seconds = the cuts -> not a spring number
    if sp_ok:
        for s_, v in sp["final_metric"]["worst_seconds"][:2]:
            add(s_, 1.0 + (0.5 if (sp.get("spring_ratio") or 0) > 1.6 else 0), f"cheek spring {v:.2f} %/frame (video mean {sp['final_metric']['spring_scale_pct'][0]:.2f})")
    # merge into 1 s buckets, keep the strongest 5 (avoid neighbours within 2 s)
    tot = sorted(((sum(s for s, _ in v), k) for k, v in cand.items()), reverse=True); pick = []
    for sc, k in tot:
        if all(abs(k - q) > 2 for q in pick): pick.append(k)
        if len(pick) == 5: break
    out.update(dur=dur, ev=ev, st=st, rates=rates, sp=sp, cand=cand, pick=sorted(pick), lost=lost, cuts=cuts)
    return out


def flags_of(st):
    f = st.get("flags", []); w = [x.get("flag") for x in st.get("warnings", [])]
    return f, w


print("# WATCH 08.11: what to look at in the final e2e set (Q)\n")
print("Each video: the 3-5 most suspicious seconds (mm:ss), what fired there, and whether that detector fires at the same rate on the "
      "eye-accepted golden finals (`[golden-FP level]` = probably fine, look only if you have time). Scores: speech@0 = full-length SyncNet at "
      "offset 0 on the speaker track (golden: m03 8.68, m18 7.93, old03 7.55, q02 1.81; >= 5 is good), spring = extra cheek frame-to-frame scale "
      "vs the source (no-beard golden ratio 1.2-1.5, KeySync beards 1.75-1.96).\n")
for spec in sys.argv[1:]:
    r = one(spec); ev = r["ev"]; st = r["st"]; f, w = flags_of(st)
    print(f"## {r['id']}  ({r['dur']:.0f} s)  `{r['final']}`\n")
    print(f"- status **{st.get('status', '?')}**; flags: {', '.join(f) or '-'}; warnings: {', '.join(x for x in w if x) or '-'}; refund {st.get('refund_suggested')}")
    if ev:
        print(f"- speech@0 **{ev.get('speech_at0')}** over {ev.get('n_speech_sec')} s of on-screen dub speech; face lost {ev.get('face_lost_share')}; eval status {ev.get('status')}; global offset {ev.get('av_offset')}")
    if r["sp"].get("spring_ratio") is not None and not r["sp"].get("ok"):
        print(f"- spring: n/a (multi-shot / camera motion: the auto region spans cuts, upper control ratio {r['sp'].get('upper_ratio')}; needs a per-shot region)")
    elif r["sp"].get("spring_ratio") is not None:
        print(f"- spring {r['sp']['final_metric']['spring_scale_pct'][0]:.3f} % (ratio {r['sp']['spring_ratio']}, region {r['sp']['region']})")
    hot = [f"{k} {s * 100:.1f} % (golden {g * 100:.1f} %)" for k, (s, g, fp) in r["rates"].items() if not fp]
    print(f"- detectors ABOVE the golden FP level: {', '.join(hot) or 'none'}")
    if r["lost"]: print(f"- face not found during dub speech: {len(r['lost'])} s, e.g. {', '.join(mmss(t) for t in r['lost'][:6])}")
    print("\n| when | what fired | |\n|---|---|---|")
    for k in r["pick"]:
        items = sorted(r["cand"][k], reverse=True)[:4]
        allfp = all("[golden-FP level]" in t or t.startswith("shot switch") or t.startswith("cheek spring") for _, t in items)
        print(f"| **{mmss(k)}** | {'; '.join(t for _, t in items)} | {'probably FP / by design' if allfp else 'LOOK'} |")
    print()
