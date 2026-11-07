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
ap.add_argument("--manifest", required=True); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cpu")
a = ap.parse_args()


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
        det(video_path=str(v), min_track=MIN_FR)
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
