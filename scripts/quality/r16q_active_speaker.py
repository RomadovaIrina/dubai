#!/usr/bin/env python3
"""r16 Q: active-speaker check. Does any VISIBLE face actually speak each diarization turn in the SOURCE?

SyncNet (the same S3FD tracks / crop / syncnet_v2 as scripts/quality/sync_perframe.py) runs on the SOURCE video + SOURCE audio,
every face track separately. For each diarization turn (and pooled per speaker), each face track gets the SyncNet confidence
over the turn's frames: conf = median(d) - min(d) over offsets, with the best offset within OFF_WIN of 0
(a real speaker is in sync with its own voice, while the wrong face can still score at a random lag).
ON SCREEN when conf >= CONF_MIN with >= MIN_FR frames of that face inside the turn(s); turns < 1.5 s inherit their speaker's decision.
Otherwise OFF SCREEN = keep the source face for its units (no lip-sync); the dub audio still plays.
A mouth-landmark correlation with the speech envelope was tried first and fails (0.07 even on SyncNet-synced KeySync output).

Output JSON:
  tracks: [{id, frames, box_center_xy, box_size}] (to match T's MediaPipe tracks by centre)
  speakers: {sp: {track, conf, offset, frames, on_screen}}
  turns: [{start, end, speaker, track, conf, offset, on_screen, from}]
  hook: {off_screen_speakers, keep_source_all}
Usage:
  python r16q_active_speaker.py --video SRC_25fps.mp4 [--audio SRC.wav] --manifest dub_manifest.json --out as.json [--device cpu|cuda]"""
import argparse, json, os, pathlib, shutil, subprocess, sys, tempfile
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import sync_perframe as spf

CONF_MIN = float(os.environ.get("AS_CONF_MIN", "2.0")); OFF_WIN = int(os.environ.get("AS_OFF_WIN", "3"))
MIN_FR = int(os.environ.get("AS_MIN_FR", "25")); FPS = 25.0; RESCUE = float(os.environ.get("AS_RESCUE", "1.0"))

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True, help="the SOURCE video on the 25 fps grid"); ap.add_argument("--audio", default=None, help="default: the video's own track")
ap.add_argument("--manifest", default=None); ap.add_argument("--out", default=None); ap.add_argument("--device", default="cpu")
ap.add_argument("--prep-only", default=None, help="r19 speed: write the detector's dub-independent CPU inputs (its 25 fps mpeg4 copy, the jpg frames, the wav, the scene list) "
                "into this dir and exit: no GPU, no manifest (e2e.sh runs it during the audio stage)")
ap.add_argument("--prep", default=None, help="r19 speed: a finished --prep-only dir of the SAME --video: the detector reads it instead of making those files again")
a = ap.parse_args()
if not a.prep_only and not (a.manifest and a.out): ap.error("--manifest and --out are required (unless --prep-only)")


def file_sig(path, chunk=4 << 20):   # size + md5 of the first and last 4 MB
    import hashlib
    n = os.path.getsize(path); h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(chunk))
        if n > chunk: f.seek(max(chunk, n - chunk)); h.update(f.read(chunk))
    return f"{n}:{h.hexdigest()}"


def detector_prep(video, out):
    """r19 speed: the first, dub-independent CPU steps of SyncNetDetector.__call__ (third_party/latentsync/eval/syncnet_detect.py), the SAME commands:
    <out>/video/video.mp4 (25 fps mpeg4, qscale 2), <out>/frames/%06d.jpg, <out>/video/audio.wav, + its scene_detect() result as frame numbers
    (<out>/scenes.json). No GPU, no model: e2e.sh runs it beside the audio stage; detector_run() then starts at the face detection."""
    LS = spf._ls_root(); sys.path.insert(0, str(LS))
    from eval.syncnet_detect import SyncNetDetector
    out = pathlib.Path(out); vd, fd = out / "video", out / "frames"
    for d in (vd, fd): shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    (out / "done").unlink(missing_ok=True); v = str(pathlib.Path(video).resolve())
    subprocess.run(f"ffmpeg -y -nostdin -loglevel error -i {v} -qscale:v 2 -async 1 -r 25 {vd / 'video.mp4'}", shell=True, stdout=None)
    subprocess.run(f"ffmpeg -y -nostdin -loglevel error -i {vd / 'video.mp4'} -qscale:v 2 -f image2 {fd / '%06d.jpg'}", shell=True, stdout=None)
    subprocess.run(f"ffmpeg -y -nostdin -loglevel error -i {vd / 'video.mp4'} -ac 1 -vn -acodec pcm_s16le -ar 16000 {vd / 'audio.wav'}", shell=True, stdout=None)
    sc = SyncNetDetector.scene_detect(None, str(vd))   # (the method does not use self)
    json.dump({"video": v, "sig": file_sig(v), "scenes": [[int(x[0].frame_num), int(x[1].frame_num)] for x in sc]}, open(out / "scenes.json", "w"))
    (out / "done").write_text("1")


class _AheadCv2:
    """r19 speed: the cv2 module as eval.syncnet_detect sees it, with imread() of the detector's numbered jpg frames decoded AHEAD in a thread pool
    (detect_face / crop_video read them one by one, in order: the jpg decode was 11 s of the 25 s on m03). The same arrays, read earlier."""
    def __init__(s, cv, workers=8, ahead=24):
        import concurrent.futures, threading
        s._cv, s._ex, s._ahead, s._tl, s._names = cv, concurrent.futures.ThreadPoolExecutor(workers), ahead, threading.local(), {}
    def __getattr__(s, k): return getattr(s._cv, k)
    def imread(s, fname, *x):
        d, b = os.path.split(fname); stem, ext = os.path.splitext(b)
        if x or not stem.isdigit(): return s._cv.imread(fname, *x)
        if d not in s._names: s._names[d] = set(os.listdir(d))
        q = s._tl.__dict__.setdefault("q", {}); i = int(stem)   # per reader thread: {file name: future}
        for j in range(i, i + s._ahead):
            bj = f"{j:0{len(stem)}d}{ext}"
            if bj not in q and bj in s._names[d]: q[bj] = s._ex.submit(s._cv.imread, os.path.join(d, bj))
        for k in [k for k in q if k < b]: del q[k]   # (the readers only move forward)
        fut = q.pop(b, None)
        return fut.result() if fut is not None else s._cv.imread(fname)


def detector_run(det, video_path, min_track, prep):
    """= SyncNetDetector.__call__(video_path, min_track) with the files of detector_prep() (same commands, made earlier) instead of making them here;
    the jpg frames are decoded ahead (_AheadCv2) and the face tracks are cropped side by side (each with its own temp dir): the same files."""
    import concurrent.futures, eval.syncnet_detect as sd
    R = pathlib.Path(det.detect_results_dir); crop_dir, temp_dir = R / "crop", R / "temp"; prep = pathlib.Path(prep)
    shutil.rmtree(R, ignore_errors=True); crop_dir.mkdir(parents=True); temp_dir.mkdir()
    (R / "video").symlink_to(prep / "video"); (R / "frames").symlink_to(prep / "frames")
    cv_ = sd.cv2; sd.cv2 = _AheadCv2(cv_)
    try:
        faces = det.detect_face(str(R / "frames")); alltracks = []
        for f0, f1 in json.load(open(prep / "scenes.json"))["scenes"]:
            if f1 - f0 >= min_track: alltracks.extend(det.track_face(faces[f0:f1], min_track=min_track))
        def crop(ii):
            td = temp_dir / ("%05d" % ii); td.mkdir()
            det.crop_video(alltracks[ii], os.path.join(str(crop_dir), "%05d" % ii), str(R / "frames"), 25, str(td), str(R / "video"))
        with concurrent.futures.ThreadPoolExecutor(max(1, min(4, len(alltracks)))) as ex: list(ex.map(crop, range(len(alltracks))))
    finally: sd.cv2 = cv_
    shutil.rmtree(temp_dir)


def prep_ok(prep, video):   # the prep was made from this very file (e2e.sh hands the route a COPY of the master the prep read)
    try: return (pathlib.Path(prep) / "done").exists() and json.load(open(pathlib.Path(prep) / "scenes.json"))["sig"] == file_sig(video)
    except Exception: return False


def tracks_with_boxes(video, audio, device):
    """= spf.syncnet_dists, plus each S3FD track's per-frame box (the detector's track dict is captured in crop_video)."""
    import cv2, torch
    if device == "cpu": torch.set_num_threads(int(os.environ.get("AS_THREADS", "32")))
    LS = spf._ls_root(); sys.path.insert(0, str(LS))
    work = pathlib.Path(tempfile.mkdtemp(prefix="asq_")); (work / "checkpoints").symlink_to(LS / "checkpoints")
    old = os.getcwd(); os.chdir(work)
    try:
        v = pathlib.Path(video).resolve()
        if audio is not None:
            v = work / "in.mp4"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", str(pathlib.Path(video).resolve()), "-i", str(pathlib.Path(audio).resolve()),
                            "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest", str(v)], check=True)
        from eval.syncnet import SyncNetEval
        from eval.syncnet_detect import SyncNetDetector
        from python_speech_features import mfcc
        from scipy.io import wavfile
        det = SyncNetDetector(device=device, detect_results_dir="detect_results"); info = {}
        orig = det.crop_video

        def crop_video(track, cropfile, *x, **k):
            fr = np.asarray(track["frame"]); bb = np.asarray(track["bbox"], np.float32)
            info[pathlib.Path(cropfile).stem] = (int(fr[0]), fr, bb); return orig(track, cropfile, *x, **k)
        det.crop_video = crop_video
        if a.prep and audio is None and prep_ok(a.prep, v): detector_run(det, str(v), MIN_FR, a.prep); print(f"AS_PREP used: {a.prep}", flush=True)
        else: det(video_path=str(v), min_track=MIN_FR)
        net = SyncNetEval(device=device); net.loadParameters(str(LS / "checkpoints/auxiliary/syncnet_v2.model")); S = getattr(net, "__S__"); S.eval()
        out = []
        for cf in sorted((work / "detect_results/crop").glob("*.mp4")):
            cap = cv2.VideoCapture(str(cf)); ims = []
            while True:
                ok, fr = cap.read()
                if not ok: break
                ims.append(cv2.resize(fr, (224, 224)))
            cap.release()
            wav = work / "a.wav"
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-nostdin", "-i", str(cf), "-ac", "1", "-vn", "-acodec", "pcm_s16le", "-ar", "16000", str(wav)], check=True)
            sr, au = wavfile.read(str(wav)); cc = np.stack([np.array(i) for i in zip(*mfcc(au, sr))])
            im = np.stack(ims, 3)[None].transpose(0, 3, 4, 1, 2).astype(np.float32); cct = cc[None, None].astype(np.float32)
            n = min(len(ims), len(au) // 640) - 5
            if n < 10: continue
            imf, ccf = [], []
            with torch.no_grad():
                for i in range(0, n, 64):
                    j = range(i, min(n, i + 64))
                    imf.append(S.forward_lip(torch.from_numpy(np.concatenate([im[:, :, f:f + 5] for f in j])).to(device)).cpu())
                    ccf.append(S.forward_aud(torch.from_numpy(np.concatenate([cct[:, :, :, f * 4:f * 4 + 20] for f in j])).to(device)).cpu())
            imf, ccf = torch.cat(imf), torch.cat(ccf)
            ccp = torch.nn.functional.pad(ccf, (0, 0, spf.VSHIFT, spf.VSHIFT)); W = 2 * spf.VSHIFT + 1
            d = torch.stack([torch.nn.functional.pairwise_distance(imf[[i]].repeat(W, 1), ccp[i:i + W]) for i in range(n)]).numpy()
            f0, frs, bb = info.get(cf.stem, (0, None, None))
            out.append({"f0": f0, "frame": f0 + np.arange(n) + 2, "d": d, "bbox": bb, "bframes": frs})
        return out
    finally:
        os.chdir(old); shutil.rmtree(work, ignore_errors=True)


def conf_of(dd):
    """-> (conf, offset) with the minimum searched only within OFF_WIN of 0."""
    md = dd.mean(0); c = spf.VSHIFT; win = md[c - OFF_WIN:c + OFF_WIN + 1]; k = int(win.argmin())
    return float(np.median(md) - win[k]), int(OFF_WIN - k)


if a.prep_only:
    detector_prep(a.video, a.prep_only); print("AS_PREP_DONE " + a.prep_only); sys.exit(0)
tr = tracks_with_boxes(a.video, a.audio, a.device)
turns = json.load(open(a.manifest))["diarization"]["turns"]
res = {"thr": {"conf": CONF_MIN, "off_win": OFF_WIN, "min_frames": MIN_FR}, "tracks": [], "speakers": {}, "turns": []}
for i, t in enumerate(tr):
    bb = t["bbox"]; c = None if bb is None else [round(float(np.median((bb[:, 0] + bb[:, 2]) / 2)), 1), round(float(np.median((bb[:, 1] + bb[:, 3]) / 2)), 1)]
    s = None if bb is None else round(float(np.median(bb[:, 2] - bb[:, 0])), 1)
    cw, ow = conf_of(t["d"]); t["cw"], t["ow"] = cw, ow
    res["tracks"].append({"id": i, "f0": int(t["f0"]), "frames": int(len(t["d"])), "box_center_xy": c, "box_size": s,
                          "conf_whole": round(cw, 3), "offset_whole": ow})


def speaks(b):
    """conf >= CONF_MIN, or a weaker turn (>= RESCUE) at exactly the offset of a track that speaks over the whole clip
    (low-res / occluded faces: q08 480p turns 1.0-1.8 all at -3; a silent face scores at random offsets: q02 Vera)."""
    t = tr[b["track"]]
    return b["conf"] >= CONF_MIN or (t["cw"] >= CONF_MIN and b["offset"] == t["ow"] and b["conf"] >= RESCUE)


def best_track(mask_fn):
    best = None
    for i, t in enumerate(tr):
        m = mask_fn(t["frame"])
        if m.sum() < MIN_FR: continue
        c, o = conf_of(t["d"][m])
        if best is None or c > best["conf"]: best = {"track": i, "conf": round(c, 3), "offset": o, "frames": int(m.sum())}
    return best


spk = {}
for t in turns: spk.setdefault(t["speaker"], []).append((t["start"], t["end"]))
for sp, iv in spk.items():
    b = best_track(lambda fr: np.any([(fr >= s * FPS) & (fr < e * FPS) for s, e in iv], axis=0))
    res["speakers"][sp] = (b or {"track": None}) | {"speech_s": round(sum(e - s for s, e in iv), 2), "on_screen": bool(b and speaks(b))}
for t in turns:
    b = best_track(lambda fr: (fr >= t["start"] * FPS) & (fr < t["end"] * FPS)) if t["end"] - t["start"] >= 1.5 else None
    if b: row = b | {"on_screen": bool(speaks(b)), "from": "turn"}
    else: row = {"track": res["speakers"][t["speaker"]]["track"], "on_screen": res["speakers"][t["speaker"]]["on_screen"], "from": "speaker"}
    res["turns"].append({"start": t["start"], "end": t["end"], "speaker": t["speaker"]} | row)
off = sorted(sp for sp, v in res["speakers"].items() if not v["on_screen"])
res["hook"] = {"off_screen_speakers": off, "keep_source_all": len(off) == len(res["speakers"]),
               "note": "mute the off-screen speakers' units in every face's generator input (T: ks_e2e_prep units filter); keep_source_all -> skip the face stage"}
json.dump(res, open(a.out, "w"), indent=1, default=float)
print("ACTIVE " + json.dumps({"tracks": [(x["id"], x["frames"], x["conf_whole"]) for x in res["tracks"]],
                              "speakers": {sp: {k: v.get(k) for k in ("track", "conf", "offset", "speech_s", "on_screen")} for sp, v in res["speakers"].items()},
                              "turns_off": sum(not x["on_screen"] for x in res["turns"]), "turns": len(res["turns"])}))
