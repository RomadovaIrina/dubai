#!/usr/bin/env python3
"""Pilot 0.9 — Qwen2.5-7B Q8_0 vs Q5_K_M on CPU.

Dataset JSONL fields:
  id, source_lang, target_lang, source, reference(optional)
Formal closure requires at least 3 distinct target languages and an agreed quality method.
If sacrebleu is installed and references exist, this script reports chrF/ BLEU. Otherwise
it writes every translation to CSV for manual review and deliberately leaves quality open.
"""
from __future__ import annotations
import argparse, csv, glob, json, pathlib, statistics, sys, time
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
from pilot_common import PILOT, THREADS, write_json


def first_shard(pattern:str)->str:
    f=sorted(glob.glob(pattern));
    if not f: raise FileNotFoundError(pattern)
    return f[0]


def load_data(path:pathlib.Path)->list[dict]:
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip() and not x.lstrip().startswith('#')]


def bench(model_path:str,quant:str,data:list[dict],ctx:int,max_tokens:int)->list[dict]:
    from llama_cpp import Llama
    print(f"loading {quant}: {model_path}",flush=True); t0=time.perf_counter()
    llm=Llama(model_path=model_path,n_ctx=ctx,n_threads=THREADS,n_threads_batch=THREADS,
              n_gpu_layers=0,verbose=False)
    print(f"loaded in {time.perf_counter()-t0:.2f}s",flush=True)
    rows=[]
    for i,x in enumerate(data,1):
        msgs=[{"role":"system","content":"You are a professional translator. Return only the translation, with no explanation."},
              {"role":"user","content":f"Translate from {x['source_lang']} to {x['target_lang']}:\n{x['source']}"}]
        t=time.perf_counter(); r=llm.create_chat_completion(msgs,max_tokens=max_tokens,temperature=0.0); dt=time.perf_counter()-t
        txt=r['choices'][0]['message']['content'].strip(); u=r.get('usage',{}); ct=int(u.get('completion_tokens') or 0)
        row={**x,"quant":quant,"translation":txt,"seconds":round(dt,4),"completion_tokens":ct,
             "tokens_per_s":round(ct/dt,3) if dt and ct else None}
        rows.append(row); print(f"[{i}/{len(data)}] {quant} {row['tokens_per_s']} tok/s -> {txt[:100]}")
    del llm
    return rows


def add_reference_metrics(rows:list[dict])->tuple[bool,str]:
    refs=[r.get('reference') for r in rows]
    if not all(refs): return False,"references missing; manual quality review required"
    try:
        import sacrebleu
    except Exception:
        return False,"sacrebleu not installed; outputs saved for manual review"
    for r in rows:
        r['chrf']=round(sacrebleu.sentence_chrf(r['translation'],[r['reference']]).score,3)
        r['bleu']=round(sacrebleu.sentence_bleu(r['translation'],[r['reference']]).score,3)
    return True,"sacrebleu sentence chrF/BLEU"


def main()->int:
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',required=True)
    ap.add_argument('--q5',default='/workspace/dub/models/qwen2.5-7b-instruct-gguf/*q5_k_m*.gguf')
    ap.add_argument('--q8',default='/workspace/dub/models/qwen2.5-7b-instruct-gguf/*q8_0*.gguf')
    ap.add_argument('--ctx',type=int,default=4096); ap.add_argument('--max-tokens',type=int,default=256)
    ap.add_argument('--quality-tolerance-chrf',type=float,default=0.0,
                    help='Q5 may trail Q8 by at most this many chrF points; use only if team accepts chrF')
    a=ap.parse_args(); data=load_data(pathlib.Path(a.dataset))
    langs=sorted({x['target_lang'] for x in data})
    if len(langs)<3: print(f"WARNING: only {len(langs)} target languages: {langs}",file=sys.stderr)
    rows=[]
    for quant,pat in [('q5_k_m',a.q5),('q8_0',a.q8)]: rows += bench(first_shard(pat),quant,data,a.ctx,a.max_tokens)
    metric_ok,metric_note=add_reference_metrics(rows)
    outcsv=PILOT/'0.9_qwen_outputs.csv'; fields=[]
    for r in rows:
        for k in r:
            if k not in fields: fields.append(k)
    with outcsv.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    by={}
    for q in ('q5_k_m','q8_0'):
        rr=[r for r in rows if r['quant']==q]; t=[r['tokens_per_s'] for r in rr if r.get('tokens_per_s')]
        c=[r.get('chrf') for r in rr if r.get('chrf') is not None]
        by[q]={"mean_tokens_per_s":round(statistics.mean(t),3) if t else None,
               "median_tokens_per_s":round(statistics.median(t),3) if t else None,
               "mean_chrf":round(statistics.mean(c),3) if c else None}
    quality_not_worse=None
    if metric_ok: quality_not_worse=by['q5_k_m']['mean_chrf'] >= by['q8_0']['mean_chrf']-a.quality_tolerance_chrf
    rep={"task":"0.9","languages":langs,"samples":len(data),"metrics":by,"quality_method":metric_note,
         "q5_not_worse":quality_not_worse,
         "recommendation":('Q5_K_M' if quality_not_worse else 'Q8_0' if quality_not_worse is False else 'MANUAL_REVIEW_REQUIRED'),
         "closed":len(langs)>=3 and quality_not_worse is not None,"outputs_csv":str(outcsv)}
    write_json(PILOT/'0.9_qwen_quant.json',rep)
    md=["# Pilot 0.9 — Qwen Q5_K_M vs Q8_0","",
        f"Languages: **{', '.join(langs)}**  ",f"Quality method: **{metric_note}**  ",
        f"Recommendation: **{rep['recommendation']}**","",
        "| quant | mean tok/s | median tok/s | mean chrF |","|---|---:|---:|---:|"]
    for q in ('q5_k_m','q8_0'):
        x=by[q]; md.append(f"| {q} | {x['mean_tokens_per_s']} | {x['median_tokens_per_s']} | {x['mean_chrf']} |")
    (PILOT/'0.9_qwen_quant.md').write_text('\n'.join(md)+'\n',encoding='utf-8'); print('\n'.join(md))
    return 0 if rep['closed'] else 2
if __name__=='__main__': raise SystemExit(main())
