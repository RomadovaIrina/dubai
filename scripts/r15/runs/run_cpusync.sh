cd /workspace/dub/dubai-f; . scripts/env.sh >/dev/null; B=/workspace/dub/artifacts/r4m; O=/workspace/dub/r15f3/cpusync; mkdir -p $O
AUD=/workspace/dub/artifacts/r4a/r9_q2_m03/q2_m03/dubbed_16k.wav
for t in q2_m03fu q2_m03fub q2_m03fubla q2_m03fublac q2_m03fublam q2_m03fultc q2_m03fublab5 q2_m03fublab5sp; do V=$(ls $B/r9f_$t/*_r9f_$t.mp4 | head -1); T0=$(date +%s)
  ffmpeg -loglevel error -y -i $V -vf fps=25 -c:v libx264 -crf 12 -an $O/$t.v25.mp4
  SYNC_DEVICE=cpu CUDA_VISIBLE_DEVICES= nice -n 19 python scripts/quality/sync_perframe.py --video $O/$t.v25.mp4 --audio $AUD --out $O/$t.sync.json > $O/$t.log 2>&1
  python3 -c "import json,sys; ps=json.load(open(sys.argv[1]))[\"per_second\"]; s=[x[\"conf_at0\"] for x in ps if x.get(\"conf_at0\") is not None and (x.get(\"dub_act\") or 0)>=0.5]; print(sys.argv[2].ljust(18), \"conf_speech\", round(sum(s)/len(s),3), \"n\", len(s))" $O/$t.sync.json $t || tail -3 $O/$t.log
  echo "   $(( $(date +%s) - T0 )) s"; rm -f $O/$t.v25.mp4; done; echo CPUSYNC_DONE
