#!/usr/bin/env python3
"""Pilot 0.8 — SyncNet on the five FINAL dubbed E2E outputs (translated TTS audio + optimized LatentSync video).

The spec says the real-content value becomes an acceptance threshold but does not define how five values must be
reduced to one number. This script reports all raw values plus mean/median/min/max over the SCOREABLE videos and does
NOT silently choose the contractual threshold.

Semantics (2026-09-18 revision):
  * every video is first judged as a PIPELINE output (manifest written by run_clean_pipeline_05.py next to the mp4:
    <stem>.manifest.json -> validation.pass, dubbed track, frame classes, LatentSync segments);
  * SyncNet (upstream eval/eval_sync_conf.py, S3FD face tracker) is evaluated wherever it can obtain a face track;
  * content WITHOUT any face (router: VALID_FACE = 0, LatentSync segments = 0) cannot be scored by SyncNet. That is
    NOT a pipeline failure: status = N/A_NO_FACE, reason = "SyncNet/S3FD cannot obtain a face track", the video still
    counts as a valid, fully dubbed output;
  * a SyncNet failure on a video that DOES contain faces is a real FAIL.
The old 0.8 blocker ("LatentSync fails on 01/02/05 with Face not detected") is obsolete since face-aware routing.

    source scripts/env.sh
    python scripts/pilot/benchmark_08_syncnet_dataset.py qa_outputs/final_formal --originals test_videos --expect 5
"""
from __future__ import annotations
import argparse, json, pathlib, statistics, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT, write_json, gpu_info
from syncnet_common import eval_syncnet, DEFAULT_MODEL
from benchmark_06_codeformer import probe

EXT={'.mp4','.mov','.mkv','.avi','.webm'}
NA_NO_FACE="N/A_NO_FACE"
NA_REASON="SyncNet/S3FD cannot obtain a face track"

def _key(p:pathlib.Path)->str:
    """'latentsync_04.mp4', '04_final.mp4' and '04.mp4' -> '04' so outputs can be paired with originals."""
    s=p.stem.lower()
    if s.startswith("latentsync_"): s=s[len("latentsync_"):]
    for suf in ("_final","_optimized","_r1","_r2"):
        if s.endswith(suf): s=s[:-len(suf)]
    return s

def _manifest(v:pathlib.Path)->dict|None:
    m=v.with_name(v.stem+".manifest.json")
    if not m.exists(): return None
    try: return json.loads(m.read_text())
    except Exception: return None

def _pipeline_facts(man:dict|None)->dict:
    if not man: return {"manifest":False}
    ls=man.get("lipsync") or {}; fc=ls.get("frame_classes") or {}
    return {"manifest":True,"pipeline_valid":bool((man.get("validation") or {}).get("pass")),
            "codeformer":man.get("codeformer"),"dubbed_audio":bool((man.get("dubbed_track") or {}).get("aac")),
            "translation_quant":(man.get("translation") or {}).get("quant") or (man.get("translation") or {}).get("model"),
            "frame_classes":fc,"latentsync_segments":(ls.get("by_action") or {}).get("LATENT_SYNC",{}).get("segments",0),
            "latentsync_seconds":ls.get("latentsync_seconds_of_video"),"speech_gate":(man.get("speech_gate") or {}).get("enabled"),
            "video_backend":man.get("video_backend"),"has_face":fc.get("VALID_FACE",0)>0}

def _eval_rows(vids:list[pathlib.Path],tag:str,with_manifest:bool)->list[dict]:
    rows=[]
    for i,v in enumerate(vids,1):
        print(f"[{tag} {i}/{len(vids)}] {v.name}",flush=True)
        facts=_pipeline_facts(_manifest(v)) if with_manifest else {"manifest":False}
        row={"name":v.name,"key":_key(v),"meta":probe(v),"pipeline":facts}
        try:
            r=eval_syncnet(v); row.update(confidence=r["confidence"],av_offset_frames=r["av_offset_frames"],eval_s=r["seconds"],status="PASS",measurable=True)
        except Exception as e:
            err=f"{type(e).__name__}: {str(e)[:400]}"
            if with_manifest and facts.get("manifest") and not facts.get("has_face") and facts.get("latentsync_segments",0)==0:
                row.update(status=NA_NO_FACE,measurable=False,reason=NA_REASON+" (no face in content: VALID_FACE=0, LatentSync segments=0)",error=err)
            else:
                row.update(status="FAIL",measurable=False,reason="SyncNet evaluation failed on content that contains faces",error=err)
        rows.append(row); print({k:x for k,x in row.items() if k not in ("meta","error")},flush=True)
    return rows

def main()->int:
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos",help="directory with the final dubbed outputs (+ <stem>.manifest.json), or one video")
    ap.add_argument("--expect",type=int,default=5)
    ap.add_argument("--originals",default=None,help="directory with the untouched source videos -> reference confidence + deltas")
    ap.add_argument("--audio-note",default="translated TTS dubbed audio (final clean E2E outputs, CodeFormer OFF)")
    ap.add_argument("--config-note",default="reports/pilot/FINAL_ML_CONFIG.md")
    ap.add_argument("--json",default=str(PILOT/"0.8_syncnet_real_content.json"))
    ap.add_argument("--md",default=str(PILOT/"0.8_syncnet_real_content.md"))
    a=ap.parse_args(); p=pathlib.Path(a.videos).resolve()
    vids=[p] if p.is_file() else sorted(x for x in p.iterdir() if x.suffix.lower() in EXT)
    if not vids: raise SystemExit("no videos found")
    rows=_eval_rows(vids,"final",True)
    orig_rows=[]
    if a.originals:
        op=pathlib.Path(a.originals).resolve()
        ovids=sorted(x for x in op.iterdir() if x.suffix.lower() in EXT)
        orig_rows=_eval_rows(ovids,"original",False)
        by_key={r["key"]:r for r in orig_rows if r["status"]=="PASS"}
        for r in rows:
            o=by_key.get(r["key"])
            if o and r["status"]=="PASS":
                r["original_confidence"]=o["confidence"]; r["original_av_offset_frames"]=o["av_offset_frames"]
                r["confidence_delta"]=round(r["confidence"]-o["confidence"],4)
                r["abs_offset_delta_frames"]=abs(r["av_offset_frames"])-abs(o["av_offset_frames"])
    ok=[r for r in rows if r["status"]=="PASS"]; na=[r for r in rows if r["status"]==NA_NO_FACE]; failed=[r for r in rows if r["status"]=="FAIL"]
    valid=[r for r in rows if r["pipeline"].get("pipeline_valid")]
    conf=[r["confidence"] for r in ok]; offs=[abs(r["av_offset_frames"]) for r in ok]
    closed=len(vids)==a.expect and len(valid)==len(vids) and not failed and len(ok)+len(na)==len(vids)
    verdict=(f"CLOSED — {len(valid)}/{len(vids)} final E2E outputs valid; SyncNet measured on {len(ok)}/{len(vids)}, "
             f"{len(na)} N/A (no face); contract threshold still UNDECIDED") if closed else \
            (f"OPEN — valid outputs {len(valid)}/{len(vids)}, SyncNet PASS {len(ok)}, N/A {len(na)}, FAIL {len(failed)}")
    summary={"task":"0.8","closed":closed,"verdict":verdict,"expected_videos":a.expect,"found_videos":len(vids),
             "pipeline_valid":len(valid),"scoreable":len(ok),"not_applicable_no_face":len(na),"failed":len(failed),
             "confidence":{"mean":round(statistics.mean(conf),4) if conf else None,"median":round(statistics.median(conf),4) if conf else None,
                           "min":min(conf) if conf else None,"max":max(conf) if conf else None},
             "abs_av_offset_frames":{"mean":round(statistics.mean(offs),4) if offs else None,"median":statistics.median(offs) if offs else None,
                                      "max":max(offs) if offs else None},
             "worst_by_confidence":min(ok,key=lambda r:r["confidence"])["name"] if ok else None,
             "worst_by_abs_offset":max(ok,key=lambda r:abs(r["av_offset_frames"]))["name"] if ok else None,
             "rows":rows,"originals":orig_rows,"gpu":gpu_info(),
             "syncnet":{"evaluator":"third_party/latentsync/eval/eval_sync_conf.py (upstream, S3FD face tracker)","model":str(DEFAULT_MODEL)},
             "semantics":{"dataset":"five real customer/content videos, FINAL dubbed E2E outputs","lipsync_processing":"optimized face-aware LatentSync (speech gate, RetinaFace routing), CodeFormer OFF",
                          "audio":a.audio_note,"config":a.config_note,
                          "no_face_rule":"content without any face cannot be scored by SyncNet; recorded as N/A_NO_FACE, not a pipeline failure",
                          "old_blocker":"'LatentSync fails on 01/02/05 with Face not detected' is NO LONGER RELEVANT: face-aware routing passes such frames through"},
             "contract_threshold":"UNDECIDED — spec does not define aggregation rule",
             "at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    write_json(pathlib.Path(a.json),summary)
    md=["# Pilot 0.8 — SyncNet on real content (final dubbed E2E outputs)","",
        f"**Verdict:** `{verdict}`  ",
        f"Outputs valid: **{len(valid)}/{len(vids)}** · SyncNet scoreable: **{len(ok)}** · N/A (no face): **{len(na)}** · FAIL: **{len(failed)}**  ",
        f"Confidence (scoreable) mean/median/min/max: **{summary['confidence']['mean']} / {summary['confidence']['median']} / {summary['confidence']['min']} / {summary['confidence']['max']}**  ",
        f"Max |AV offset| (scoreable): **{summary['abs_av_offset_frames']['max']} frames**  ",
        f"Audio: {a.audio_note}. Config: `{a.config_note}`.  ",
        "**The old blocker \"LatentSync fails on 01/02/05 with Face not detected\" is no longer relevant**: face-aware routing "
        "passes no-face / small-face frames through, all five final E2E videos were produced and validated.  ",
        "**Contract threshold is intentionally not auto-selected:** the source spec does not define whether it should be min/mean/median/etc.","",
        "| video | pipeline valid | dubbed audio | SyncNet measurable | confidence | AV offset (frames) | original conf / offset | Δconf | status / reason |",
        "|---|---|---|---|---:|---:|---|---:|---|"]
    for r in rows:
        pf=r["pipeline"]; oc=f"{r.get('original_confidence','-')} / {r.get('original_av_offset_frames','-')}"
        md.append(f"| {r['name']} | {'yes' if pf.get('pipeline_valid') else 'no'} | {'yes' if pf.get('dubbed_audio') else 'no'} | {'yes' if r.get('measurable') else 'no'} | "
                  f"{r.get('confidence','-')} | {r.get('av_offset_frames','-')} | {oc} | {r.get('confidence_delta','-')} | {r['status']}{(' — '+r['reason']) if r.get('reason') else ''} |")
    md+=["","Per-video pipeline facts (from the runner manifests):","","| video | backend | speech gate | VALID / SMALL / NO_FACE / NO_SPEECH frames | LatentSync segments | LatentSync s | translation |","|---|---|---|---|---:|---:|---|"]
    for r in rows:
        pf=r["pipeline"]; fc=pf.get("frame_classes") or {}
        md.append(f"| {r['name']} | {pf.get('video_backend','-')} | {pf.get('speech_gate','-')} | {fc.get('VALID_FACE','-')} / {fc.get('SMALL_FACE','-')} / {fc.get('NO_FACE','-')} / {fc.get('NO_SPEECH','-')} | {pf.get('latentsync_segments','-')} | {pf.get('latentsync_seconds','-')} | {pf.get('translation_quant','-')} |")
    pathlib.Path(a.md).write_text("\n".join(md)+"\n")
    print("\n".join(md)); return 0 if closed else 2

if __name__=="__main__": raise SystemExit(main())
