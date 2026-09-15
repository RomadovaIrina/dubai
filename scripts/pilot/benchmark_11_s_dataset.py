#!/usr/bin/env python3
"""Pilot 0.11 — run the existing S measurement across a dataset.

Delegates each file to measure_stages.py so the stage definition remains single-source:
S = VAD+split + final concat + subtitle burn.
"""
from __future__ import annotations
import argparse, pathlib, subprocess, sys

EXT={'.mp4','.mov','.mkv','.avi','.webm'}

def main()->int:
    ap=argparse.ArgumentParser(); ap.add_argument('videos')
    ap.add_argument('--repeat',type=int,default=1); ap.add_argument('--asr',action='store_true')
    ap.add_argument('--split-mode',choices=['reencode','copy'],default='reencode')
    a=ap.parse_args(); root=pathlib.Path(__file__).resolve().parents[2]
    p=pathlib.Path(a.videos).resolve(); vids=[p] if p.is_file() else sorted(x for x in p.iterdir() if x.suffix.lower() in EXT)
    if not vids: raise SystemExit('no videos found')
    fail=0
    for i,v in enumerate(vids,1):
        print(f"\n===== 0.11 {i}/{len(vids)} {v.name} =====",flush=True)
        cmd=['/venv/dabai/bin/python',str(root/'scripts/pilot/measure_stages.py'),str(v),'--repeat',str(a.repeat),'--split-mode',a.split_mode]
        if a.asr: cmd.append('--asr')
        rc=subprocess.run(cmd,cwd=str(root)).returncode; fail += (rc!=0)
    subprocess.run(['/venv/dabai/bin/python',str(root/'scripts/pilot/measure_stages.py'),'--report'],cwd=str(root))
    return 1 if fail else 0
if __name__=='__main__': raise SystemExit(main())
