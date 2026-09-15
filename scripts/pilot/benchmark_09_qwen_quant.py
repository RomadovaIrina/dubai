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


SYSTEM_PROMPT="You are a professional translator. Return only the translation, with no explanation."

def _messages(x:dict)->list[dict]:
    return [{"role":"system","content":SYSTEM_PROMPT},
            {"role":"user","content":f"Translate from {x['source_lang']} to {x['target_lang']}:\n{x['source']}"}]


def timed_completion(llm,msgs:list[dict],max_tokens:int)->dict:
    """One streamed chat completion. Time-to-first-token ~ prompt evaluation; the remaining chunks give the
    pure generation rate (model load is never inside these numbers)."""
    t0=time.perf_counter(); t_first=None; parts=[]; n=0
    for ch in llm.create_chat_completion(msgs,max_tokens=max_tokens,temperature=0.0,stream=True):
        d=ch['choices'][0].get('delta') or {}
        if 'content' in d and d['content'] is not None:
            n+=1
            if t_first is None: t_first=time.perf_counter()
            parts.append(d['content'])
    t_end=time.perf_counter()
    prompt_tokens=len(llm.tokenize(("".join(m['content'] for m in msgs)).encode('utf-8')))
    gen_s=(t_end-t_first) if t_first else 0.0
    return {"translation":"".join(parts).strip(),"completion_tokens":n,"prompt_tokens_approx":prompt_tokens,
            "wall_s":round(t_end-t0,4),"ttft_s":round((t_first-t0) if t_first else 0.0,4),
            "prompt_tok_per_s":round(prompt_tokens/(t_first-t0),2) if t_first else None,
            "gen_tok_per_s":round((n-1)/gen_s,3) if n>1 and gen_s>0 else None}


def bench(model_path:str,quant:str,data:list[dict],ctx:int,max_tokens:int,threads:int,warmup:int,repeats:int)->tuple[list[dict],dict]:
    from llama_cpp import Llama
    print(f"loading {quant}: {model_path}",flush=True); t0=time.perf_counter()
    llm=Llama(model_path=model_path,n_ctx=ctx,n_threads=threads,n_threads_batch=threads,
              n_gpu_layers=0,verbose=False)
    load_s=round(time.perf_counter()-t0,2)
    size_gib=round(sum(pathlib.Path(p).stat().st_size for p in glob.glob(model_path.replace("-00001-of-","-*-of-")) or [model_path])/2**30,3)
    print(f"loaded in {load_s}s, {size_gib} GiB, threads={threads}",flush=True)
    for _ in range(warmup):
        timed_completion(llm,_messages(data[0]),max_tokens)   # warm-up, not recorded
    rows=[]
    for i,x in enumerate(data,1):
        runs=[timed_completion(llm,_messages(x),max_tokens) for _ in range(repeats)]
        g=[r["gen_tok_per_s"] for r in runs if r["gen_tok_per_s"]]; p=[r["prompt_tok_per_s"] for r in runs if r["prompt_tok_per_s"]]
        row={**x,"quant":quant,"translation":runs[0]["translation"],
             "outputs_identical_across_repeats":len({r["translation"] for r in runs})==1,
             "completion_tokens":runs[0]["completion_tokens"],"prompt_tokens_approx":runs[0]["prompt_tokens_approx"],
             "wall_s_mean":round(statistics.mean(r["wall_s"] for r in runs),4),
             "ttft_s_mean":round(statistics.mean(r["ttft_s"] for r in runs),4),
             "prompt_tok_per_s_mean":round(statistics.mean(p),2) if p else None,
             "gen_tok_per_s_runs":g,"gen_tok_per_s_mean":round(statistics.mean(g),3) if g else None,
             "gen_tok_per_s_median":round(statistics.median(g),3) if g else None}
        rows.append(row); print(f"[{i}/{len(data)}] {quant} gen {row['gen_tok_per_s_mean']} tok/s ({row['completion_tokens']} tok) -> {row['translation'][:90]}",flush=True)
    del llm
    return rows,{"model_path":model_path,"size_gib":size_gib,"load_s":load_s,"threads":threads,"n_ctx":ctx,"max_tokens":max_tokens,
                 "n_gpu_layers":0,"warmup":warmup,"repeats":repeats,"temperature":0.0}


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
    ap.add_argument('--threads',type=int,default=THREADS,help='n_threads = n_threads_batch; pick from the cgroup quota, not os.cpu_count()')
    ap.add_argument('--warmup',type=int,default=1); ap.add_argument('--repeats',type=int,default=3)
    a=ap.parse_args(); data=load_data(pathlib.Path(a.dataset))
    langs=sorted({x['target_lang'] for x in data}); directions=sorted({x.get('direction',f"{x['source_lang']}->{x['target_lang']}") for x in data})
    if len(langs)<3: print(f"WARNING: only {len(langs)} target languages: {langs}",file=sys.stderr)
    rows=[]; cfg={}
    for quant,pat in [('q5_k_m',a.q5),('q8_0',a.q8)]:
        rr,c=bench(first_shard(pat),quant,data,a.ctx,a.max_tokens,a.threads,a.warmup,a.repeats); rows+=rr; cfg[quant]=c
    metric_ok,metric_note=add_reference_metrics(rows)
    # format / completeness checks that need no reference: non-empty, not truncated, no explanation preamble, target script
    import re
    def fmt(r):
        t=r['translation']; tl=r['target_lang']
        cyr=len(re.findall(r'[А-Яа-яЁё]',t)); lat=len(re.findall(r'[A-Za-zÄÖÜäöüß]',t)); letters=cyr+lat or 1
        script_ok=(cyr/letters>0.8) if tl=='Russian' else (lat/letters>0.8)
        return {"non_empty":bool(t),"not_truncated":r['completion_tokens']<a.max_tokens,
                "no_preamble":not re.match(r'^(translation|перевод|übersetzung)\s*:',t,re.I),
                "single_answer":t.count('\n\n')==0,"target_script_ok":script_ok,
                "no_cjk":not re.search(r'[぀-ヿ㐀-鿿가-힯]',t)}  # Qwen can drift into Chinese mid-sentence
    for r in rows: r.update({f"fmt_{k}":v for k,v in fmt(r).items()}); r['format_ok']=all(fmt(r).values())
    outcsv=PILOT/'0.9_qwen_translations.csv'
    # paired CSV: one line per sentence with both quant outputs side by side
    q5={r['id']:r for r in rows if r['quant']=='q5_k_m'}; q8={r['id']:r for r in rows if r['quant']=='q8_0'}
    with outcsv.open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(["id","direction","category","source","q5_k_m_output","q8_0_output","q5_format_ok","q8_format_ok","q5_gen_tok_s","q8_gen_tok_s","q5_tokens","q8_tokens","identical_outputs","reference"])
        for x in data:
            a5,a8=q5[x['id']],q8[x['id']]
            w.writerow([x['id'],x.get('direction'),x.get('category'),x['source'],a5['translation'],a8['translation'],a5['format_ok'],a8['format_ok'],
                        a5['gen_tok_per_s_mean'],a8['gen_tok_per_s_mean'],a5['completion_tokens'],a8['completion_tokens'],a5['translation']==a8['translation'],x.get('reference','')])
    by={}
    for q in ('q5_k_m','q8_0'):
        rr=[r for r in rows if r['quant']==q]
        g=[v for r in rr for v in r['gen_tok_per_s_runs']]; p=[r['prompt_tok_per_s_mean'] for r in rr if r.get('prompt_tok_per_s_mean')]
        c=[r.get('chrf') for r in rr if r.get('chrf') is not None]
        by[q]={**cfg[q],"gen_tok_per_s_mean":round(statistics.mean(g),3) if g else None,
               "gen_tok_per_s_median":round(statistics.median(g),3) if g else None,
               "prompt_tok_per_s_mean":round(statistics.mean(p),2) if p else None,
               "completion_tokens_total":sum(r['completion_tokens'] for r in rr),
               "wall_s_total_measured":round(sum(r['wall_s_mean']*a.repeats for r in rr),1),
               "format_ok":sum(r['format_ok'] for r in rr),"format_total":len(rr),"truncated":sum(not r['fmt_not_truncated'] for r in rr),
               "mean_chrf":round(statistics.mean(c),3) if c else None,
               "per_direction":{d:{"format_ok":sum(r['format_ok'] for r in rr if r.get('direction')==d),"n":sum(1 for r in rr if r.get('direction')==d),
                                   "gen_tok_per_s_mean":round(statistics.mean(v for r in rr if r.get('direction')==d for v in r['gen_tok_per_s_runs']),3)} for d in directions}}
    quality_not_worse=None
    if metric_ok: quality_not_worse=by['q5_k_m']['mean_chrf'] >= by['q8_0']['mean_chrf']-a.quality_tolerance_chrf
    speed_winner='Q5_K_M' if by['q5_k_m']['gen_tok_per_s_median']>by['q8_0']['gen_tok_per_s_median'] else 'Q8_0'
    identical=sum(1 for x in data if q5[x['id']]['translation']==q8[x['id']]['translation'])
    quality_verdict=('Q5_NOT_WORSE' if quality_not_worse else 'Q8_BETTER') if metric_ok else 'MANUAL_REVIEW_REQUIRED'
    selected='Q5_K_M' if quality_not_worse else ('Q8_0' if quality_not_worse is False else 'UNDECIDED')
    rep={"task":"0.9","languages":langs,"directions":directions,"samples":len(data),"metrics":by,"quality_method":metric_note,
         "identical_outputs_q5_vs_q8":identical,"q5_not_worse":quality_not_worse,
         "speed_winner":speed_winner,"quality_verdict":quality_verdict,"selected_quant":selected,
         "recommendation":selected,
         "measurement_complete":True,"closed":len(langs)>=3 and quality_not_worse is not None,"outputs_csv":str(outcsv),
         "at":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
    write_json(PILOT/'0.9_qwen_quant.json',rep)
    md=["# Pilot 0.9 — Qwen Q5_K_M vs Q8_0","",
        f"Languages: **{', '.join(langs)}**  ",f"Quality method: **{metric_note}**  ",
        f"Speed winner: **{speed_winner}**  Quality verdict: **{quality_verdict}**  Selected quant: **{selected}**","",
        "| quant | size GiB | load sec | gen tok/s mean | median | prompt tok/s | format ok | quality |","|---|---:|---:|---:|---:|---:|---:|---|"]
    for q in ('q5_k_m','q8_0'):
        x=by[q]; md.append(f"| {q} | {x['size_gib']} | {x['load_s']} | {x['gen_tok_per_s_mean']} | {x['gen_tok_per_s_median']} | {x['prompt_tok_per_s_mean']} | {x['format_ok']}/{x['format_total']} | {x['mean_chrf'] if x['mean_chrf'] is not None else 'MANUAL_REVIEW_REQUIRED'} |")
    md+=["","| direction | Q5 quality | Q8 quality | winner/tie |","|---|---|---|---|"]
    for d in directions:
        f5=by['q5_k_m']['per_direction'][d]; f8=by['q8_0']['per_direction'][d]
        md.append(f"| {d} | format {f5['format_ok']}/{f5['n']}, {'chrF n/a' if not metric_ok else ''} | format {f8['format_ok']}/{f8['n']}, {'chrF n/a' if not metric_ok else ''} | {'MANUAL_REVIEW_REQUIRED' if not metric_ok else ''} |")
    (PILOT/'0.9_qwen_quant.md').write_text('\n'.join(md)+'\n',encoding='utf-8'); print('\n'.join(md))
    return 0 if rep['closed'] else 2
if __name__=='__main__': raise SystemExit(main())
