import json, glob, collections
rows = []
for f in sorted(glob.glob("out_*.json")):
    d = json.load(open(f)); s = d["summary"]; dark_src = s["src_black_share_open"]
    for e in d["events"]:
        r = e["ref"] or {}; cls = e["class"]
        if cls == "black hole":
            if dark_src > 0.6:
                cause = "source (dark mouth in the source; ours opens wider: ours %.2f vs src %.2f)" % (e["ours_open"], e["src_open"])
            elif e["audio_db"] < 45:
                cause = "generator: opens a dark mouth on a near-silent dub (%.0f dB); ref teeth %.2f" % (e["audio_db"], r.get("teeth", 0))
            else:
                cause = "generator: wide opening (%.2f vs src %.2f) on a loud vowel (%.0f dB) with a teeth-free ref (teeth %.2f)" % (e["ours_open"], e["src_open"], e["audio_db"], r.get("teeth", 0))
        elif cls == "soft mouth":
            cause = ("generator (2-step student, low-res source); in a pause (%.0f dB), gate at 2 steps" % e["audio_db"]) if e["audio_db"] < 55 else "generator (2-step student)"
        elif "swollen" in cls:
            if e["video"] == "q05":
                cause = "occlusion (landmarks on the cup)"
            else:
                cause = "generator: rounded-vowel overshoot, width %.2f vs src %.2f, lips %.2fx, f1 band %.2f" % (e["ours_width"], e["src_width"], e["ours_lips_rel"], e["audio_f1"])
        else:
            cause = "generator: thin lips at %.0f dB (ref lips %.2fx median)" % (e["audio_db"], r.get("lips_rel", 1))
        rows.append("%s | %.2f-%.2f | %s | %d | %s %s | %s | out_%s.json frames %d-%d | K2" % (e["video"], e["t0"], e["t1"], cls, e["severity"], e["metric"], e["value"], cause, e["video"], e["frames"][0], e["frames"][1]))
rows.sort(key=lambda x: (-int(x.split(" | ")[3]), x))
hdr = "video | t_start-t_end | class | severity 1-3 | metric value | root cause | evidence | owner\n---|---|---|---|---|---|---|---\n"
open("K2_MOUTH_ATLAS_ROWS.md", "w").write(hdr + "\n".join(rows) + "\n")
print(len(rows), "rows")
for k, v in sorted(collections.Counter((x.split(" | ")[0], x.split(" | ")[2]) for x in rows).items()): print(k, v)
