"""r15 T: CPU tests of the streamed generator stage (lazy_frames.py + the process_video / assemble patches). No GPU.
    python scripts/r15/test_gen_stream.py <master_25fps.mp4> <work dir>
1. read_frames (now list(iter_frames)) == the pre-r15 implementation, LS_CMS_READ=1 and 0
2. LazyFrames == the list for a segment-like access pattern + backward reads + slices + iteration; the cache stays bounded
3. PatchFrames rebuilds every frame exactly (mouth-box edits, unchanged frames, a whole-frame change, a shape change)
4. assemble_frames: list master + full replacement frames == lazy master + PatchFrames (decoded frames AND file bytes)"""
import hashlib, os, pathlib, random, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "scripts" / "pilot"))
master, work = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); work.mkdir(parents=True, exist_ok=True)


def old_read_frames(p):   # the pre-r15 read_frames, verbatim
    import cv2
    if os.environ.get("LS_CMS_READ") == "1":
        import av
        c = av.open(str(p)); fr = []
        for f_ in c.decode(c.streams.video[0]):
            cs = getattr(f_, "colorspace", None); cs = "ITU709" if cs == 1 else ("ITU601" if cs in (5, 6) else ("ITU709" if f_.height >= 720 else "ITU601"))
            fr.append(f_.reformat(format="bgr24", src_colorspace=cs, src_color_range="MPEG").to_ndarray())
        c.close(); return fr
    cap = cv2.VideoCapture(str(p)); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f)
    cap.release(); return fr


import face_aware_latentsync as fa, lazy_frames as lf, face_aware_latentsync_accel as faa
ok_all = True
def check(name, cond):
    global ok_all; ok_all &= bool(cond); print(("PASS " if cond else "FAIL ") + name, flush=True)

# 1
for cms in ("1", "0"):
    os.environ["LS_CMS_READ"] = cms
    A = old_read_frames(master); B = fa.read_frames(master)
    check(f"1 read_frames == pre-r15 (LS_CMS_READ={cms}, {len(A)} frames)", len(A) == len(B) and all(np.array_equal(x, y) for x, y in zip(A, B)))
os.environ["LS_CMS_READ"] = "1"; ref = old_read_frames(master); n = len(ref)

# 2
L = lf.LazyFrames(master, back=64); ok = len(L) == n
seg = [(0, 40), (55, 120), (130, 131), (150, min(n, 230)), (20, 60), (n - 30, n)]   # forward segments + a backward read (split) + the tail
for a, b in seg:
    L.release_before(a); ok &= all(np.array_equal(L[i], ref[i]) for i in range(a, b)); ok &= all(np.array_equal(L[i], ref[i]) for i in range(a, b))
random.seed(1); ok &= all(np.array_equal(L[i], ref[i]) for i in sorted(random.sample(range(n), min(n, 40))))
ok &= all(np.array_equal(x, y) for x, y in zip(L[10:20], ref[10:20])) and np.array_equal(L[-1], ref[-1])
it_ok = sum(1 for _ in L) == n and all(np.array_equal(x, y) for x, y in zip(iter(L), ref))
check(f"2 LazyFrames == list ({L.stats()}), iteration", ok and it_ok and L.peak_cached <= 64 + 200)

# 3
P = lf.PatchFrames(n=12); rng = np.random.default_rng(0); outs = []
for k in range(10):
    f = ref[k].copy()
    if k % 3 == 0: f[300 + k:420, 200:330] = rng.integers(0, 255, (120 - k, 130, 3), dtype=np.uint8)   # a mouth-box edit
    if k == 4: f = 255 - f                                                                             # the whole frame
    outs.append(f); P.put(k, f, ref[k])
P.put(10, np.zeros((8, 8, 3), np.uint8), ref[10])   # a shape change -> stored whole
check("3 PatchFrames exact", all(np.array_equal(P.get(k, ref[k]), outs[k]) for k in range(10)) and P.get(10, ref[10]).shape == (8, 8, 3)
      and P.get(11, ref[11], "d") == "d" and len(P) == 12 and P.get(1, ref[1]) is ref[1])

# 4
segs = []; pos = 0; idx = 0
for ln, act in [(30, "PASS"), (60, "LATENT_SYNC"), (25, "PASS"), (40, "LATENT_SYNC"), (n - 155, "PASS")]:
    if ln <= 0: continue
    segs.append({"index": idx, "start_frame": pos, "end_frame": pos + ln, "frames": ln, "action": act}); pos += ln; idx += 1
segs[1]["crossfade"] = {"in": 4, "out": 4}; segs[3]["crossfade"] = {"in": 4, "out": 0}
full, patched = {}, {}
for s in segs:
    if s["action"] != "LATENT_SYNC": continue
    m = s["frames"] - (3 if s["index"] == 3 else 0)   # seg 3: LatentSync returned fewer frames (padded with the original)
    frs = []
    for k in range(m):
        f = ref[s["start_frame"] + k].copy(); f[350:470, 240:380] = (f[350:470, 240:380].astype(np.int16) + 37 + k).clip(0, 255).astype(np.uint8); frs.append(f)
    full[s["index"]] = frs; pf = lf.PatchFrames(n=len(frs))
    for k in range(min(len(frs), s["frames"])): pf.put(k, frs[k], ref[s["start_frame"] + k])
    patched[s["index"]] = pf
aud = work / "t_audio.m4a"
fa.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", str(n / 25), "-c:a", "aac", str(aud)])
o1, o2 = work / "t_list.mp4", work / "t_lazy.mp4"
r1 = faa.assemble_frames(ref, aud, [dict(s) for s in segs], full, o1, work)
r2 = faa.assemble_frames(lf.LazyFrames(master, back=64), aud, [dict(s) for s in segs], patched, o2, work)
d1, d2 = old_read_frames(o1), old_read_frames(o2)
same_frames = len(d1) == len(d2) == n and all(np.array_equal(x, y) for x, y in zip(d1, d2))
h1, h2 = hashlib.md5(o1.read_bytes()).hexdigest(), hashlib.md5(o2.read_bytes()).hexdigest()
check(f"4 assemble list == lazy+patches (stats {r1 == r2}: {r2}; decoded frames identical {same_frames}; file bytes identical {h1 == h2})", same_frames and r1 == r2)
for p in (o1, o2, aud): p.unlink(missing_ok=True)
print("GEN_STREAM_TESTS_OK" if ok_all else "GEN_STREAM_TESTS_FAILED")
sys.exit(0 if ok_all else 1)
