"""Piece-tail / piece-head check (the 05 r7v3a 31.5 s "fart": a TTS hesitation murmur left at a piece's end and cut off into silence):
every placed piece is CTC-aligned to its own text; audible material (> voice p95 - db_below, run >= min_s) after its last word + 0.08 s
or before its first word - 0.06 s is listed.
    python scripts/r7/piece_tails.py --tag r7v3a6 [--prev r7v3a5] [ids]"""
import argparse, json, sys, pathlib, numpy as np, soundfile as sf
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "pilot"))
A = "/workspace/dub/artifacts/r4a"

def tails(al, y, sr, man, db_below=30.0, min_s=0.1):
    h = int(0.01 * sr); db = 20 * np.log10(np.sqrt((y[:len(y) // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-9); ref = np.percentile(db[db > -80], 95)
    out = []
    for u in man["units"]:
        for p in (u.get("alignment") or {}).get("placements") or []:
            if not p.get("placed") or not p.get("text"): continue
            t0, t1 = p["placed"]; seg = y[int(t0 * sr):int(t1 * sr)].astype(np.float32)
            sp = al.align(seg, sr, p["text"]) if len(seg) > sr * 0.1 else None
            if not sp: continue
            for side, a, b in (("tail", t0 + sp[-1]["end"] + 0.08, t1), ("head", t0, t0 + sp[0]["start"] - 0.06)):
                if b - a < min_s: continue
                loud = db[int(a * 100):int(b * 100)] > ref - db_below
                if loud.sum() * 0.01 >= min_s:
                    out.append({"unit": u["id"], "side": side, "t": round(a, 2), "dur": round(b - a, 2), "loud_s": round(loud.sum() * 0.01, 2),
                                "max_db_rel": round(float(db[int(a * 100):int(b * 100)].max() - ref), 1), "text": p["text"], "retake": bool(p.get("r7_retake"))})
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--prev"); ap.add_argument("ids", nargs="*", default=["01", "02", "03", "04", "05"])
    a = ap.parse_args()
    from tts_qa import TtsQA
    from tts_align import CtcAligner
    al = CtcAligner.from_qa(TtsQA())
    for id_ in a.ids:
        for t in [x for x in (a.prev, a.tag) if x]:
            y, sr = sf.read(f"{A}/{t}/{id_}/dubbed_24k.wav"); man = json.load(open(f"{A}/{t}/{id_}/{id_}_r4b.manifest.json"))
            r = tails(al, y, sr, man)
            print(f"{id_} {t}: {len(r)}", " | ".join(f"u{x['unit']} {x['side']} {x['t']}s {x['loud_s']}/{x['dur']}s {x['max_db_rel']} dB{' RETAKE' if x['retake'] else ''} '{x['text'][-25:]}'" for x in r), flush=True)
