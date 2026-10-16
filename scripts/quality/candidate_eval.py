#!/usr/bin/env python3
"""Package and score one video-only candidate against its baseline run: mux the candidate frames with the run's OWN final audio (copied,
so the audio stream must be bit-identical), write a candidate manifest (the run manifest with `output` replaced), then SyncNet on the
final mix and with the dub voice only, mouth_jitter_hr (vs the run's 25 fps master, with the run's segments), face_temporal, and the
md5 of the muxed audio stream vs the baseline output's.

    python scripts/quality/candidate_eval.py --manifest RUN/final/03_final.manifest.json --frames CAND_lossless.mp4 --tag C2T --out DIR
"""
from __future__ import annotations
import argparse, json, pathlib, subprocess, sys, tempfile
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pilot"))


def audio_md5(p: str) -> str:
    return subprocess.run(["ffmpeg", "-loglevel", "error", "-i", p, "-map", "0:a:0", "-c", "copy", "-f", "md5", "-"], capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True); ap.add_argument("--frames", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--no-face-temporal", action="store_true")
    a = ap.parse_args()
    man = json.loads(pathlib.Path(a.manifest).read_text()); work = pathlib.Path(man["work_dir"]); vid = pathlib.Path(man["input"]).stem
    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True); mp4 = out / f"{vid}_{a.tag}.mp4"
    mix = work / "final_mix_aac.m4a" if (work / "final_mix_aac.m4a").exists() else pathlib.Path(man["dubbed_track"]["aac"])
    tmp = out / f".{vid}_{a.tag}.tmp.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", a.frames, "-i", str(mix), "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-crf", "13",
                    "-pix_fmt", "yuv420p", "-c:a", "copy", str(tmp)], check=True)
    tmp.rename(mp4)
    cm = {**man, "output": str(mp4), "candidate": {"tag": a.tag, "frames": a.frames, "base_manifest": a.manifest}}
    cmp = out / f"{vid}_{a.tag}.manifest.json"; cmp.write_text(json.dumps(cm, indent=1, ensure_ascii=False))
    res = {"id": vid, "tag": a.tag, "output": str(mp4)}
    import cv2
    n_c = int(cv2.VideoCapture(str(mp4)).get(cv2.CAP_PROP_FRAME_COUNT)); n_b = int(cv2.VideoCapture(man["output"]).get(cv2.CAP_PROP_FRAME_COUNT))
    res["frames"] = {"candidate": n_c, "baseline": n_b, "equal": n_c == n_b}
    res["audio_md5"] = {"candidate": audio_md5(str(mp4)), "baseline": audio_md5(man["output"])}; res["audio_md5"]["identical"] = res["audio_md5"]["candidate"] == res["audio_md5"]["baseline"]
    if (man.get("lipsync") or {}).get("by_action", {}).get("LATENT_SYNC", {}).get("frames", 0):
        from syncnet_common import eval_syncnet
        o = eval_syncnet(mp4); res["syncnet_mix"] = {"confidence": o["confidence"], "av_offset": o["av_offset_frames"]}
        t = pathlib.Path(tempfile.mkdtemp()) / f"{vid}_dubonly.mp4"
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp4), "-i", man["dubbed_track"]["aac"], "-map", "0:v:0", "-map", "1:a:0", "-c", "copy", str(t)], check=True)
        o = eval_syncnet(t); res["syncnet_dub_only"] = {"confidence": o["confidence"], "av_offset": o["av_offset_frames"]}; t.unlink()
        py = sys.executable
        subprocess.run([py, str(HERE / "mouth_jitter_hr.py"), "--video", str(mp4), "--reference", str(work / "master_25fps.mp4"), "--manifest", str(cmp),
                        "--out", str(out / f"{vid}_{a.tag}.mouth_jitter_hr.json"), "--tag", f"{vid}_{a.tag}"], check=True)
        if not a.no_face_temporal:
            subprocess.run([py, str(HERE / "face_temporal.py"), "--manifest", str(cmp), "--out", str(out / "face_temporal"), "--tag", a.tag, "--clips", "0", "--sheets", "0"], check=True)
    (out / f"{vid}_{a.tag}.eval.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
