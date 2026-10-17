"""Shared wrapper around LatentSync's official eval/eval_sync_conf.py."""
from __future__ import annotations
import os, pathlib, re, shutil, subprocess, time
from pilot_common import THIRD_PARTY

LS = (THIRD_PARTY / "latentsync").resolve()
if not LS.exists():   # DUB_ROOT unset: the repo this file lives in
    LS = pathlib.Path(__file__).resolve().parents[2] / "third_party" / "latentsync"
DEFAULT_MODEL = LS / "checkpoints/auxiliary/syncnet_v2.model"


def eval_syncnet(video: pathlib.Path, *, keep_work: bool=False) -> dict:
    """SyncNet confidence / AV offset of `video` (LatentSync eval/eval_sync_conf.py). Runs in a PRIVATE working directory (2026-10-17):
    the upstream script writes detect_results/ and its temp dir relative to the cwd, so two evaluations started from the LatentSync tree at the
    same time overwrote each other's frames / crops (errors, or a score computed on another video's frames)."""
    import tempfile
    video = video.resolve()
    if not video.exists():
        raise FileNotFoundError(video)
    work = pathlib.Path(tempfile.mkdtemp(prefix="syncnet_"))
    (work / "checkpoints").symlink_to(LS / "checkpoints")          # s3fd loads checkpoints/auxiliary/sfd_face.pth relative to the cwd
    temp = work / "temp_pilot_syncnet"
    env=os.environ.copy()
    env["PYTHONPATH"] = str(LS) + (":"+env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    cmd=["/venv/dabai/bin/python","-m","eval.eval_sync_conf",
         "--initial_model",str(DEFAULT_MODEL),"--video_path",str(video),"--temp_dir",str(temp)]
    t0=time.perf_counter()
    p=subprocess.run(cmd,cwd=str(work),env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    dt=time.perf_counter()-t0
    out=p.stdout
    m1=re.search(r"SyncNet confidence:\s*([-+0-9.]+)",out)
    m2=re.search(r"AV offset:\s*([-+0-9]+)",out)
    result={"video":str(video),"returncode":p.returncode,"seconds":round(dt,3),
            "confidence":float(m1.group(1)) if m1 else None,
            "av_offset_frames":int(m2.group(1)) if m2 else None,
            "stdout_tail":"\n".join(out.splitlines()[-30:]), "workdir": str(work)}
    if not keep_work:
        shutil.rmtree(work,ignore_errors=True)
    if p.returncode!=0 or result["confidence"] is None:
        raise RuntimeError(f"SyncNet evaluation failed for {video}:\n{result['stdout_tail']}")
    return result
