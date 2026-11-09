#!/usr/bin/env python3
"""r16 Q defect atlas (brief_r16c_analysis): one row per defect occurrence from the calibrated checks + Irina / Max eye rows.
    python3 atlas.py --a DIR_0811 [--b DIR_NOW] [--sync SYNC.json] --out OUTDIR
DIR/<id>.json        r16q_defects.py (tail / outside / chroma / occlude / seam / mouth)
DIR/<id>.detall.json r16q_detect.py  (black / slide / swap / overlap / texture / explode / edge / flicker)
DIR/<id>.spring.json r16q_spring.py  (single-shot clips)
SYNC.json            {id: {"per_s": [[t, conf_at0, offset], ...]}}  (T's per-second SyncNet; optional)
Writes OUTDIR/ATLAS.md, OUTDIR/atlas_rows.json (for the clip maker), prints the per-class summary."""
import argparse, json, os, pathlib

CLASSES = ["seam jitter", "texture boil", "black hole", "soft mouth", "lip shape", "tail", "mask edge outside the face", "colour",
           "occlusion", "head turn", "off-screen voice", "sync", "flicker / snap"]
# default root cause / owner per class (refined by evidence notes); see the root-cause column notes in ATLAS.md
CAUSE = {"seam jitter": ("paste", "K1"), "texture boil": ("generator", "K1/K2"), "black hole": ("generator", "K2"), "soft mouth": ("generator", "K2"),
         "lip shape": ("generator", "K2"), "tail": ("generator / route", "T"), "mask edge outside the face": ("paste", "K1"), "colour": ("paste", "K1"),
         "occlusion": ("gate", "T/K1"), "head turn": ("gate", "T/K1"), "off-screen voice": ("route", "T"), "sync": ("audio-align", "T"),
         "flicker / snap": ("generator", "K2")}
WHY = {"seam jitter": "the paste mask reaches below the jaw line: generated jaw moves, source neck does not (K1 tight mask cuts it 80-90 %)",
       "texture boil": "raw KeySync on beards (K1 q04 paste arms did not move the spring)",
       "black hole": "KeySync paints a dark void for wide-open mouths",
       "soft mouth": "the mouth is blurrier than the source face (fast student / paste blend)",
       "lip shape": "lip ring thinner / fatter than the source (ref 0 / gate / generator)",
       "tail": "frames after the last keyframe / window are extrapolated, then the paste snaps back to the source",
       "mask edge outside the face": "the mask grabs collar / background / neck; the generator redraws it every frame",
       "colour": "the paste colour match lifts the black floor toward blue in dim warm light (K2 d203a99)",
       "occlusion": "no occlusion gate in 08.11: the generator paints the face over the object",
       "head turn": "no pose gate in 08.11: the crop box edge / background enter the generated region",
       "off-screen voice": "the dub voice is not the on-screen person; the face should stay original",
       "sync": "mouth vs audio (per-second SyncNet)", "flicker / snap": "frame-to-frame jumps of the generated mouth"}
DET_FLOOR = {"black": 0.04, "slide": 0.05, "texture": 0.33, "edge": 0.022, "flicker": 0.11, "explode": 0.003, "swap": 0.005, "overlap": 0.005}
DET_CLASS = {"black": "black hole", "texture": "texture boil", "edge": "mask edge outside the face", "flicker": "flicker / snap",
             "explode": "flicker / snap", "slide": "flicker / snap", "swap": "flicker / snap", "overlap": "flicker / snap"}

# Irina 09.11 / Max eye rows on the 08.11 files (ФИДБЕК_0911): (video, t0, t1, class, note)
EYE = [("f0811_m03", 40.1, 40.4, "tail", "face 'momo' at the end, last 6 of 17 frames broken"),
       ("f0811_m03", 0.0, 40.4, "colour", "'shadow on the cheek' (paste colour at the cheek edge)"),
       ("full_m18", 19.0, 20.3, "tail", "'19 s something terrible with the face': swollen, smeared lower face"),
       ("full_m18", 10.0, 11.0, "lip shape", "lips 'eaten', rolled in; golden opens wide (Max: mouth problems in m18)"),
       ("f0811_q01", 44.0, 45.0, "tail", "smeared wide beard / mouth, jump to the original in the last frame"),
       ("f0811_q01_subs", 44.0, 45.0, "tail", "same tail as q01"),
       ("long5", 108.0, 150.0, "mask edge outside the face", "neck smeared, mask edge at the hair; 'unwatchable after 1:38'"),
       ("long5", 116.0, 117.0, "colour", "1:56 dark-blue mouth ('black hole'), blue spot on the cheek"),
       ("long5", 108.0, 150.0, "occlusion", "hair over the cheek painted over"),
       ("q03_e2e_alpha_full", 0.0, 45.0, "mask edge outside the face", "collar and neck shake ('reject because of the neck')"),
       ("q03_e2e_alpha_full", 29.0, 30.0, "sync", "mouth closed while the voice plays (forgiven: fade-in)"),
       ("q05_e2e_full", 11.5, 13.5, "occlusion", "the cup over the mouth gets smeared into the face"),
       ("q05_e2e_full", 0.0, 45.0, "head turn", "head turns: square corner, window inside the generated region"),
       ("q05_e2e_full", 0.0, 45.0, "texture boil", "'the whole face trembles'"),
       ("full_old03", 64.0, 66.0, "mask edge outside the face", "right cheek edge against the foliage changes width every frame"),
       ("f0811_q04", 0.0, 45.0, "texture boil", "grey beard springs (handheld, low probe confidence)"),
       ("f0811_q02", 34.8, 38.6, "occlusion", "hand at the mouth: mouth drawn over the hand?"),
       ("f0811_q06", 0.0, 45.0, "off-screen voice", "KeySync did not run (profile, W_NO_FACE): original Russian mouth under the English voice")]


def fmt_t(t):
    t = float(t); return f"{int(t // 60)}:{t % 60:04.1f}"


def rows_for(d, setname):
    D = pathlib.Path(d); out = []
    ids = sorted(p.name[:-5] for p in D.glob("*.json") if p.name.count(".") == 1 and not p.name.startswith("CALIB"))
    for vid in ids:
        r = json.load(open(D / f"{vid}.json")); fr = r.get("frames", 0); dur = fr / 25.0
        def add(cls, t0, t1, sev, metric, note=""):
            c, o = CAUSE[cls]; out.append(dict(set=setname, video=vid, t0=round(float(t0), 2), t1=round(float(t1), 2), cls=cls, sev=sev, metric=metric,
                                               cause=c, owner=o, note=note or WHY[cls]))
        t = r["tail"]
        if t.get("flag"):
            lr = t.get("local_max", 0); add("tail", max(0, dur - 1.0), dur, 3 if lr >= 2.0 else 2, f"tail local {lr} (thr 1.5)")
        o = r["outside"]
        for b, nm in (("wob_l", "left cheek"), ("wob_r", "right cheek"), ("wob_n", "neck")):
            v = o.get(b, {}); p95 = v.get("ratio_p95", 0)
            if v.get("flag"):
                runs = v.get("runs_1s") or [[0, dur, p95]]
                for t0, t1, pk in runs[:6]: add("mask edge outside the face", t0, t1, 3 if pk >= 2.0 else 2, f"{nm} wobble p95 {p95} (thr 1.2), run peak {pk}")
        s = r.get("seam")
        if s and s.get("excess_median") is not None and s["excess_median"] > 0:
            em = s["excess_median"]; ps = s.get("per_s") or []
            add("seam jitter", 0, dur, 3 if em >= 1.5 else 2 if em >= 0.8 else 1, f"seam excess median {em} / p95 {s.get('excess_p95')} L*/fr (floor 0.25-0.5; fix bar 0.35)")
            for sec, v in (s.get("worst_s") or [])[:3]:
                if v >= 2.5: add("seam jitter", sec, sec + 1, 3 if v >= 4 else 2, f"seam excess {v} in this second")
        c = r["chroma"]
        for t0, t1, pk in c.get("runs", [])[:6]: add("colour", t0, t1, 2 if pk < 8 else 3, f"dark pixels bluer than source by {pk} b* (thr 4)")
        oc = r["occlude"]
        for t0, t1, pk in oc.get("paint_runs", [])[:6]: add("occlusion", t0, t1, 3 if pk >= 20 else 2, f"jaw landmarks off the source by {pk} % IOD (painted over an object; thr 10)")
        for t0, t1, pk in oc.get("occl_runs", [])[:6]: add("occlusion", t0, t1, 1, f"lower face off cheek colour share {pk}")
        for t0, t1, pk in oc.get("turn_runs", [])[:10]:
            if fr and r.get("seam", {}).get("excess_median", 1) == 0: continue   # no KeySync anywhere (q06): the source is kept
            add("head turn", t0, t1, 2, f"|yaw| {pk} (thr 0.45, >= 0.5 s)")
        m = r.get("mouth") or {}
        for t0, t1, pk in m.get("soft_runs", [])[:6]: add("soft mouth", t0, t1, 2 if -pk > 0.55 else 3, f"mouth sharpness ours / src {-pk:.2f} (thr 0.70)")
        for t0, t1, pk in m.get("lip_thin_runs", [])[:6]: add("lip shape", t0, t1, 2 if -pk > 0.5 else 3, f"lip ring ours / src {-pk:.2f} (thin, thr 0.65)")
        for t0, t1, pk in m.get("lip_fat_runs", [])[:6]: add("lip shape", t0, t1, 2, f"lip ring ours / src {pk:.2f} (fat, thr 1.5)")
        dp = D / f"{vid}.detall.json"
        if dp.exists():
            dj = json.load(open(dp)); det = dj.get("det", {}); n = max(dj.get("frames", fr), 1)
            for k, v in det.items():
                if k not in DET_CLASS or not isinstance(v, dict): continue
                share = v.get("flag_share", 0) or 0; fl = DET_FLOOR.get(k, 0.05)
                if share <= fl: continue   # at or under the golden false-positive floor
                for t0, t1, pk in (v.get("ranges") or [])[:5]:
                    # TRUST_v2: detectors are spot finding only -> capped at sev 2 (3 only when an eye row confirms it)
                    add(DET_CLASS[k], t0, t1, 1 if share < 2 * fl else 2, f"{k} detector, video share {share:.3f} vs golden floor {fl}, peak {pk}")
        sp = D / f"{vid}.spring.json"
        if sp.exists():
            sj = json.load(open(sp)); ratio = sj.get("spring_ratio"); up = sj.get("upper_ratio") or 0
            # 1.7: q04 (Irina: springs) 1.76 vs q01 (Irina: fine) 1.64 -> thin margin; upper_ratio > 1.5 = camera / region unreliable
            if isinstance(ratio, (int, float)) and sj.get("spring_above_floor") and ratio >= 1.7:
                add("texture boil", 0, dur, 2, f"cheek spring {ratio}x floor (thr 1.7; Irina-fine q01 = 1.64){'; upper_ratio ' + str(up) + ' (handheld: region less reliable)' if up > 1.5 else ''}")
    if out and all((r["cls"] != "seam jitter") for r in out if r["video"] == vid) is False: pass
    return out


def drop_nokey(rows, d):
    """videos where KeySync never ran (seam excess 0, e.g. q06): only the off-screen / route rows are meaningful"""
    D = pathlib.Path(d); keep = []
    for r in rows:
        p = D / f"{r['video']}.json"
        if p.exists() and r["cls"] not in ("off-screen voice", "sync"):
            s = json.load(open(p)).get("seam") or {}
            if s.get("excess_median") == 0: continue
        keep.append(r)
    return keep


def _unused():
    out = []
    return out


def sync_rows(path, setname):
    if not path or not os.path.exists(path): return []
    out = []
    for vid, v in json.load(open(path)).items():
        for t, conf, off in v.get("per_s", []):
            if conf is not None and conf < 2.0 and off not in (None, 0):
                c, o = CAUSE["sync"]
                out.append(dict(set=setname, video=vid, t0=t, t1=t + 1, cls="sync", sev=3 if conf < 0.5 else 2, metric=f"SyncNet conf@0 {conf}, best offset {off}",
                                cause=c, owner=o, note=WHY["sync"]))
    return out


K2_VID = {"m03": "f0811_m03", "m18": "full_m18", "old03": "full_old03", "q01": "f0811_q01", "q02": "f0811_q02", "q03": "q03_e2e_alpha_full",
          "q04": "f0811_q04", "q05": "q05_e2e_full", "long5s3": "long5"}
K2_OFF = {"long5s3": 107.8}   # K2's long5 rows are relative to the dark-haired woman's shot


def k2_rows(path):
    """K2_MOUTH_ATLAS_ROWS.md: 'video | t0-t1 | class | sev | metric | root cause | evidence | owner' (no leading pipe)"""
    out = []
    if not path or not os.path.exists(path): return out
    for ln in open(path, encoding="utf-8"):
        c = [x.strip() for x in ln.strip().strip("|").split("|")]
        if len(c) < 8 or c[0] not in K2_VID: continue
        try: t0, t1 = (float(x) for x in c[1].split("-")[:2])
        except ValueError: continue
        cls = c[2].split("(")[0].strip()
        if cls not in CAUSE: continue
        off = K2_OFF.get(c[0], 0.0); cause = c[5].split(":")[0].split("(")[0].strip() or CAUSE[cls][0]
        out.append(dict(set="a", video=K2_VID[c[0]], t0=t0 + off, t1=t1 + off, cls=cls, sev=int(c[3]) if c[3].isdigit() else 2, metric=f"K2: {c[4]}",
                        cause=cause, owner=c[7] or CAUSE[cls][1], note=c[5], src="K2", evidence=f"K2 {c[6]}"))
    return out


def merge(rows):
    """merge same video / class / set runs that touch (gap <= 0.5 s) -> one occurrence"""
    rows = sorted(rows, key=lambda r: (r["set"], r["video"], r["cls"], r["t0"])); out = []
    for r in rows:
        p = out[-1] if out else None
        if p and (p["set"], p["video"], p["cls"]) == (r["set"], r["video"], r["cls"]) and r["t0"] <= p["t1"] + 0.5 and (p["t1"] - p["t0"]) < 30 and (r["t1"] - r["t0"]) < 30:
            p["t1"] = max(p["t1"], r["t1"]); p["sev"] = max(p["sev"], r["sev"])
            if r["metric"] not in p["metric"]: p["metric"] += "; " + r["metric"]
            srcs = set(p.get("src", "Q").split("+")) | set(r.get("src", "Q").split("+")); p["src"] = "+".join(sorted(srcs))
            if r.get("src") in ("K1", "K2") and r.get("note") and r["note"] not in p["note"]: p["note"] = r["note"] + " | " + p["note"]
            if r.get("evidence") and r["evidence"] not in p.get("evidence", ""): p["evidence"] = (p.get("evidence", "") + "; " + r["evidence"]).strip("; ")
        else: out.append(dict(r))
    return out


# windows Irina explicitly called fine (ФИДБЕК_0911): metric rows inside drop to sev 1 ("eye: fine here")
EYE_FINE = [("long5", 0.0, 107.9, "Irina: 'the rest is fine, 1:38 is fine'; bad = the dark-haired woman's shot ~1:48-2:30"),
            ("long5", 150.0, 300.0, "Irina: 'the rest is fine'"),
            ("f0811_q01", 0.0, 43.9, "Irina: q01 'ok, nothing jerks'"), ("f0811_q01_subs", 0.0, 43.9, "Irina: subs ok, only the end bad")]


def eye_match(rows):
    """eye rows (same class, overlapping) -> sev 3 + note; eye rows nothing of that class caught -> an eye-only row listing what else
    fired there; rows inside an eye-fine window -> sev 1"""
    for v, t0, t1, cls, note in EYE:
        hit = [r for r in rows if r["set"] == "a" and r["video"] == v and r["cls"] == cls and r["t0"] < t1 + 0.5 and r["t1"] > t0 - 0.5]
        for r in hit: r["sev"] = 3; r["eye"] = note
        if not hit:
            near = [r for r in rows if r["set"] == "a" and r["video"] == v and r["t0"] < t1 + 0.5 and r["t1"] > t0 - 0.5 and (r["t1"] - r["t0"]) < 20]
            ov = "; ".join(sorted({f"{r['cls']}: {r['metric'].split(';')[0]}" for r in near})[:3])
            c, o = CAUSE[cls]; rows.append(dict(set="a", video=v, t0=t0, t1=t1, cls=cls, sev=3, metric="(eye only: no check of this class fired)" + (f"; overlaps {ov}" if ov else ""),
                                                cause=c, owner=o, note=WHY[cls], eye=note, src="eye"))
    for r in rows:
        if r["set"] != "a" or r.get("eye"): continue
        for v, t0, t1, note in EYE_FINE:
            if r["video"] == v and r["t0"] >= t0 - 0.5 and r["t1"] <= t1 + 0.5: r["sev"] = 1; r["eye_fine"] = note
    return rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--a", required=True); ap.add_argument("--b"); ap.add_argument("--sync-a"); ap.add_argument("--sync-b")
    ap.add_argument("--clips", default="clips"); ap.add_argument("--out", required=True)
    ap.add_argument("--top10", help="n|video|t0|dur|label per line"); ap.add_argument("--k2", help="K2_MOUTH_ATLAS_ROWS.md"); ap.add_argument("--k1", help="k1_rows.json (rows + seam_cause)"); a = ap.parse_args()
    k1 = json.load(open(a.k1)) if a.k1 and os.path.exists(a.k1) else {"rows": [], "seam_cause": {}}
    base = rows_for(a.a, "a")
    for r in base:
        r.setdefault("src", "Q")
        if r["cls"] == "seam jitter" and r["video"] in k1["seam_cause"]: r["note"] = "K1: " + k1["seam_cause"][r["video"]]
    extra = [dict(r, set="a") for r in k1["rows"]] + k2_rows(a.k2)
    rows = merge(drop_nokey(base, a.a) + extra + sync_rows(a.sync_a, "a")); rows = eye_match(rows)
    if a.b: rows += merge(drop_nokey(rows_for(a.b, "b"), a.b) + sync_rows(a.sync_b, "b"))
    for r in rows: r["dur"] = round(r["t1"] - r["t0"], 2)
    top = []
    if a.top10 and os.path.exists(a.top10):
        for ln in open(a.top10):
            n, vid, t, d, lab = ln.strip().split("|"); t, d = float(t), float(d); top.append((n, vid, t, d, lab))
            for r in rows:
                if r["set"] == "a" and r["video"] == vid and r["t0"] < t + d and r["t1"] > t and (r["t1"] - r["t0"]) < 100 and r["sev"] >= 2:
                    r["clip"] = f"clip {a.clips}/{n}_{lab}_*.mp4 ({fmt_t(t)}+{d:g} s)"
    O = pathlib.Path(a.out); O.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (-r["sev"], r["set"], -(r["sev"] == 3 and "eye" in r), r["video"], r["t0"]))
    json.dump(rows, open(O / "atlas_rows.json", "w"), indent=1, ensure_ascii=False)
    L = ["# r16 defect atlas (Q)\n", "Sets: **a** = the delivered 08.11 e2e files (what Max and Irina saw)" + ("; **b** = r16-int head, every fix ON (T)." if a.b else "; b = pending (T's r16-int renders)."),
         "Severity: 3 = eye-confirmed (Irina / Max) or metric >= 2x threshold; 2 = over threshold; 1 = info. Detector rows only where the video's flag share exceeds the golden false-positive floor (TRUST_v2).",
         "Root cause / owner = the class default plus evidence; see the root-cause notes from K1 / K2 for per-occurrence detail.\n"]
    # per-class summary
    L += ["## Per class: occurrences (sev 3 / all)\n", "| class | a: 08.11 | b: now | videos (a) |", "|---|---|---|---|"]
    for c in CLASSES:
        ra = [r for r in rows if r["set"] == "a" and r["cls"] == c]; rb = [r for r in rows if r["set"] == "b" and r["cls"] == c]
        vids = sorted({r["video"] for r in ra})
        L.append(f"| {c} | {sum(r['sev'] == 3 for r in ra)} / {len(ra)} | " + (f"{sum(r['sev'] == 3 for r in rb)} / {len(rb)}" if a.b else "pending") + f" | {', '.join(v.replace('f0811_', '').replace('_e2e_alpha_full', '').replace('_e2e_full', '').replace('full_', '') for v in vids)} |")
    if top:
        L += ["\n## Top-10 worst moments (0.5x clips; panels 08.11 | now | source, audio = 08.11 at half speed)\n", "| # | video | window | what | atlas rows (class: metric) |", "|---|---|---|---|---|"]
        for n, vid, t, d, lab in top:
            rs = [r for r in rows if r["set"] == "a" and r["video"] == vid and r["t0"] < t + d and r["t1"] > t and (r["t1"] - r["t0"]) < 100 and r["sev"] >= 2]
            L.append(f"| {n} | {vid} | {fmt_t(t)}–{fmt_t(t + d)} | {lab.split('_', 2)[-1].replace('_', ' ')} | " +
                     "; ".join(sorted({f"{r['cls']} (sev {r['sev']}): {r['metric'].split(';')[0][:90]}" for r in rs})[:4]) + " |")
    L += ["\n## Atlas (sorted by severity)\n", "| # | set | video | t | class | sev | metric | root cause | evidence | owner | by |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        ev = (f"eye: {r['eye']}; " if r.get("eye") else "") + (f"eye: fine here ({r['eye_fine']}); " if r.get("eye_fine") else "") + \
             (r.get("clip") or r.get("evidence") or f"{r['video']}.mp4 @ {fmt_t(r['t0'])}")
        by = r.get("src", "Q") + ("+eye" if r.get("eye") and r.get("src") != "eye" else "")
        L.append(f"| {i} | {r['set']} | {r['video']} | {fmt_t(r['t0'])}–{fmt_t(r['t1'])} | {r['cls']} | {r['sev']} | {r['metric']} | {r['cause']}: {r['note']} | {ev} | {r['owner']} | {by} |")
    (O / "ATLAS.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[5:5 + 3 + len(CLASSES)]))


if __name__ == "__main__":
    main()
