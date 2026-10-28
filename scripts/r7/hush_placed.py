"""r7_audio_v3a.1: timing-preserving clean-up of the re-taken pieces on an already lip-placed dub (no re-placement: the face render stays valid).
Every placed piece of a re-taken unit (placements with r7_retake) is CTC-aligned to its own text; non-word TTS material is ducked to -30 dB
with 20-ms sin^2 ramps:
  - inside the piece: inter-word gaps >= 0.25 s after punctuation, up to the next word's acoustic onset (tts_align.hush_gaps; inside a
    phrase a duck would make a pause there: 05 u3 'our / app');
  - the tail: from the last word + 0.08 s to the piece end (05 v3a 31.45-31.98 s: a hesitation murmur cut off into silence, the user's "fart");
  - the head: from the piece start to the first word's onset - 0.03 s (never past its CTC start - 0.06 s).
Writes <out>/ = symlinks of <placed> + the new dubbed_24k.wav / dubbed_16k.wav + hush.json; piece placements and lengths are unchanged.
    python scripts/r7/hush_placed.py --placed <tag>/placed/<id> --out <new tag>/placed/<id>"""
import argparse, json, pathlib, re, sys, numpy as np, soundfile as sf, librosa
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent / "pilot"))
from tts_align import hush_gaps

def duck(y, sr, a, b, floor_db=-30.0, ramp_s=0.02, side="tail"):
    i, j = max(0, int(a * sr)), min(len(y), int(b * sr)); r = int(ramp_s * sr)
    if j - i < 2 * r: return False
    g = 10 ** (floor_db / 20); env = np.full(j - i, g); q = np.sin(np.linspace(0, np.pi / 2, r)) ** 2
    if side in ("tail", "both"): env[:r] = 1 - (1 - g) * q
    if side in ("head", "both"): env[-r:] = g + (1 - g) * q
    y[i:j] *= env; return True

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--placed", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--min-s", type=float, default=0.1); ap.add_argument("--db-below", type=float, default=30.0); a = ap.parse_args()
    P, O = pathlib.Path(a.placed), pathlib.Path(a.out); O.mkdir(parents=True, exist_ok=True)
    for f in P.iterdir():
        if f.name not in ("dubbed_24k.wav", "dubbed_16k.wav", "hush.json") and not (O / f.name).exists(): (O / f.name).symlink_to(f.resolve())
    man = json.loads(next(P.glob("*_r4a.manifest.json")).read_text())
    y, sr = sf.read(P / "dubbed_24k.wav", dtype="float32"); z = y.copy()
    h = int(0.01 * sr); db = 20 * np.log10(np.sqrt((y[:len(y) // h * h].reshape(-1, h) ** 2).mean(1)) + 1e-9); ref = np.percentile(db[db > -80], 95)
    loud = lambda s, e: (db[int(s * 100):int(e * 100)] > ref - a.db_below).sum() * 0.01
    from tts_align import CtcAligner
    al = None; rep = []   # loaded only when the dub has re-taken pieces (wav2vec2 CTC alone: TtsQA's Whisper was 2-3 s of load per video for nothing)
    for u in man["units"]:
        for p in (u.get("alignment") or {}).get("placements") or []:
            if not (p.get("r7_retake") and p.get("placed") and p.get("text")): continue
            t0, t1 = p["placed"]; i0, i1 = int(t0 * sr), int(t1 * sr); seg = z[i0:i1].copy()
            al = al or CtcAligner.standalone()
            sp = al.align(seg, sr, p["text"])
            if not sp: rep.append({"unit": u["id"], "text": p["text"], "skipped": "align"}); continue
            ws = p["text"].split(); after = {k for k, w in enumerate(ws) if re.search(r"[,.;:!?]$", w)} if len(ws) == len(sp) else set()   # phrase boundaries only
            seg, gaps = hush_gaps(seg, sr, sp, after=after); row = {"unit": u["id"], "text": p["text"], "placed": [t0, t1], "gaps": [[round(t0 + x, 2), round(t0 + y_, 2)] for x, y_ in gaps]}
            ta = sp[-1]["end"] + 0.08
            if t1 - t0 - ta >= a.min_s and loud(t0 + ta, t1) >= a.min_s and duck(seg, sr, ta, t1 - t0, side="tail"): row["tail"] = [round(t0 + ta, 2), round(t1, 2)]
            # head: up to the first word's acoustic onset (level within 8 dB of the word) - 0.03, never past CTC start - 0.06
            w = sp[0]; k0, k1 = int(w["start"] * 100), max(int(w["start"] * 100) + 1, int(w["end"] * 100)); lv = db[int(t0 * 100):int(t1 * 100)]
            if k1 <= len(lv):
                k = 0
                while k < int((w["start"] - 0.06) * 100) and lv[k] < np.median(lv[k0:k1]) - 8: k += 1
                tb = k / 100 - 0.03
                if tb >= a.min_s and loud(t0, t0 + tb) >= a.min_s and duck(seg, sr, 0.0, tb, side="head"): row["head"] = [round(t0, 2), round(t0 + tb, 2)]
            z[i0:i1] = seg; rep.append(row)
    sf.write(O / "dubbed_24k.wav", z, sr, subtype=sf.info(P / "dubbed_24k.wav").subtype)
    sf.write(O / "dubbed_16k.wav", librosa.resample(z, orig_sr=sr, target_sr=16000), 16000, subtype="PCM_16")
    (O / "hush.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False)); print(json.dumps(rep, ensure_ascii=False))

if __name__ == "__main__": main()
