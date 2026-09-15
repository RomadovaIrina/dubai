#!/usr/bin/env python3
"""Summarize pilot 0.1–0.11 from reports/pilot without inventing missing results."""
from __future__ import annotations
import json, pathlib, sys
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT

SPECS={
 '0.1':('Blackwell image','0.1_blackwell.json'),
 '0.2':('LatentSync compatibility','0.2_latentsync.json'),
 '0.3':('FlashAttention sm_120','0.3_flashattention_build.json'),
 '0.4':('VRAM residency','0.4_residency.json'),
 '0.5':('GPU-min/video-min','0.5_gpu_coefficient.json'),
 '0.6':('CodeFormer timing','0.6_codeformer.json'),
 '0.7':('CodeFormer vs SyncNet','0.7_codeformer_syncnet.json'),
 '0.8':('SyncNet real content','0.8_syncnet_real_content.json'),
 '0.9':('Qwen Q8 vs Q5','0.9_qwen_quant.json'),
 '0.10':('Cold start','0.10_cold_start.json'),
}

def load(p:pathlib.Path):
    try:return json.loads(p.read_text())
    except Exception:return None

def status_011():
    rows=sorted(PILOT.glob('stages_*.json'))
    good=[]
    for p in rows:
        x=load(p)
        if x and x.get('median',{}).get('S') is not None: good.append(x)
    return {'closed':bool(good),'runs':len(good),'files':[x.get('input') for x in good],
            'verdict':'MEASURED' if good else 'MISSING'}

def main()->int:
    rows=[]; closed=0
    for k,(title,name) in SPECS.items():
        p=PILOT/name; x=load(p)
        if not x: rows.append((k,title,'MISSING','-',name)); continue
        c=bool(x.get('closed')); closed+=c
        verdict=x.get('verdict') or ('CLOSED' if c else 'INCOMPLETE')
        if x.get('provisional') and not c: verdict='PROVISIONAL_'+str(verdict)
        rows.append((k,title,'CLOSED' if c else 'OPEN',verdict,name))
    x=status_011(); closed+=bool(x['closed']); rows.append(('0.11','Sequential S','CLOSED' if x['closed'] else 'OPEN',x['verdict'],f"{x['runs']} run(s)"))
    print(f"Pilot: {closed}/11 formally closed\n")
    print(f"{'ID':<5} {'STATE':<8} {'TASK':<28} VERDICT")
    print('-'*90)
    for k,t,s,v,_ in rows: print(f"{k:<5} {s:<8} {t:<28} {v}")
    md=['# Pilot status','',f'**{closed}/11 formally closed**','',
        '| ID | state | task | verdict/evidence |','|---|---|---|---|']
    md += [f'| {k} | {s} | {t} | {v} |' for k,t,s,v,_ in rows]
    (PILOT/'PILOT_STATUS.md').write_text('\n'.join(md)+'\n')
    return 0
if __name__=='__main__':raise SystemExit(main())
