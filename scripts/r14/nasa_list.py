#!/usr/bin/env python3
"""r14: candidate talking-head clips from the NASA Image and Video Library (images-api.nasa.gov; US public domain, NASA media guidelines).
Searches interview-type queries, keeps videos with a ~medium.mp4 (720p) asset, records URL / title / centre / date / size (HEAD) -> JSON.
    python3 scripts/r14/nasa_list.py --out nasa_candidates.json [--max 40]"""
import argparse, json, urllib.parse, urllib.request

Q = ["preflight interview", "post-flight interview", "astronaut interview", "expedition crew interview", "flight engineer interview"]


def get(u):
    with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "dabai-research/1.0"}), timeout=60) as r: return json.load(r)


def head(u):
    try:
        with urllib.request.urlopen(urllib.request.Request(u, method="HEAD", headers={"User-Agent": "dabai-research/1.0"}), timeout=60) as r: return int(r.headers.get("Content-Length") or 0)
    except Exception: return 0


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--max", type=int, default=40); a = ap.parse_args()
    seen, out = set(), []
    for q in Q:
        d = get("https://images-api.nasa.gov/search?" + urllib.parse.urlencode({"q": q, "media_type": "video", "page_size": 50}))
        for it in d["collection"]["items"]:
            m = it["data"][0]; nid = m["nasa_id"]
            if nid in seen: continue
            seen.add(nid)
            try: files = get(it["href"])
            except Exception: continue
            med = [f for f in files if f.endswith("~medium.mp4")]
            if not med: continue
            u = urllib.parse.quote(med[0], safe=":/~"); sz = head(u)
            if not sz or sz > 400 * 2**20: continue   # skip hour-long tapes; interviews are a few minutes
            out.append({"nasa_id": nid, "title": m.get("title"), "center": m.get("center"), "date": m.get("date_created"), "url": u, "bytes": sz,
                        "page": "https://images.nasa.gov/details/" + urllib.parse.quote(nid), "licence": "US public domain (NASA media usage guidelines)",
                        "attribution": f"NASA / {m.get('center') or 'NASA'}, \"{m.get('title')}\" ({(m.get('date_created') or '')[:10]}), images.nasa.gov"})
            if len(out) >= a.max: break
        if len(out) >= a.max: break
    json.dump(out, open(a.out, "w"), indent=1); print(f"{len(out)} candidates, {sum(o['bytes'] for o in out) / 2**30:.1f} GB")


if __name__ == "__main__":
    main()
