"""Shared wrapper around LatentSync's official eval/eval_sync_conf.py."""
from __future__ import annotations
import os, pathlib, re, shutil, subprocess, time
from pilot_common import THIRD_PARTY

LS = THIRD_PARTY / "latentsync"
DEFAULT_MODEL = LS / "checkpoints/auxiliary/syncnet_v2.model"


def eval_syncnet(video: pathlib.Path, *, keep_work: bool=False) -> dict:
    video = video.resolve()
    if not video.exists():
        raise FileNotFoundError(video)
    detect = LS / "detect_results"
    temp = LS / "temp_pilot_syncnet"
    if detect.exists(): shutil.rmtree(detect, ignore_errors=True)
    if temp.exists(): shutil.rmtree(temp, ignore_errors=True)
    env=os.environ.copy()
    env["PYTHONPATH"] = str(LS) + (":"+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd=["/venv/dabai/bin/python","-m","eval.eval_sync_conf",
         "--initial_model",str(DEFAULT_MODEL),"--video_path",str(video),"--temp_dir",str(temp)]
    t0=time.perf_counter()
    p=subprocess.run(cmd,cwd=str(LS),env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    dt=time.perf_counter()-t0
    out=p.stdout
    m1=re.search(r"SyncNet confidence:\s*([-+0-9.]+)",out)
    m2=re.search(r"AV offset:\s*([-+0-9]+)",out)
    result={"video":str(video),"returncode":p.returncode,"seconds":round(dt,3),
            "confidence":float(m1.group(1)) if m1 else None,
            "av_offset_frames":int(m2.group(1)) if m2 else None,
            "stdout_tail":"\n".join(out.splitlines()[-30:])}
    if not keep_work:
        shutil.rmtree(detect,ignore_errors=True); shutil.rmtree(temp,ignore_errors=True)
    if p.returncode!=0 or result["confidence"] is None:
        raise RuntimeError(f"SyncNet evaluation failed for {video}:\n{result['stdout_tail']}")
    return result
