cd /workspace/dub/dubai-f; . scripts/env.sh >/dev/null; B=/workspace/dub/artifacts/r4m; O=/workspace/dub/r15f3/cpusync; F=/workspace/dub/r15f3/flash
AUD=/workspace/dub/artifacts/r4a/r9_q2_m03/q2_m03/dubbed_16k.wav; SRC=$(ls /workspace/dub/r14d/eval/q2_m03/e2e/work/in/q2_m03.* | head -1); SL=$B/r9g_q2_m03fu/timing/q2_m03_source_lm_F.npz
arm(){ sx=$1; lt=$2; ex=$3; tag=$4; LCA=0.45 GT=r9g_q2_m03fu LT=$lt LT_EXTRA="$ex" LT_TAG=$tag nice -n 19 bash scripts/r15/comp_f2.sh q2_m03 $sx off > /dev/null 2>&1
  grep -h "lip_temporal" /workspace/dub/r15f2/logs/comp_q2_m03$sx.log | grep -o "median_gated_frames_gt05[^,]*"
  V=$B/r9f_q2_m03$sx/q2_m03_r9f_q2_m03$sx.mp4
  CUDA_VISIBLE_DEVICES= nice -n 19 python scripts/quality/r3_flash.py --video $V --src $SRC --lm $SL --out $F/q2_m03$sx.flash.json > /dev/null 2>&1
  python3 -c "import json,sys; d=json.load(open(sys.argv[1])); print(sys.argv[2], \"flash\", d[\"flash_frames\"], \"strong\", sum(1 for e in d[\"flash_list\"] if (e.get(\"mc_ratio\") or 0)>=2 and (e.get(\"mc_vs_src\") or 0)>=2))" $F/q2_m03$sx.flash.json $sx
  ffmpeg -loglevel error -y -i $V -vf fps=25 -c:v libx264 -crf 12 -an $O/$sx.v25.mp4
  SYNC_DEVICE=cpu CUDA_VISIBLE_DEVICES= nice -n 19 python scripts/quality/sync_perframe.py --video $O/$sx.v25.mp4 --audio $AUD --out $O/$sx.sync.json > /dev/null 2>&1; rm -f $O/$sx.v25.mp4
  python3 -c "import json,sys; ps=json.load(open(sys.argv[1]))[\"per_second\"]; s=[x[\"conf_at0\"] for x in ps if x.get(\"conf_at0\") is not None and (x.get(\"dub_act\") or 0)>=0.5]; print(sys.argv[2], \"sync\", round(sum(s)/len(s),3))" $O/$sx.sync.json $sx; }


arm fublcLg15 both "--radius 3 --clamp 4 7 7 --median-gate 1.5" Lg15
arm fublcL color "--radius 3 --clamp 4 7 7" L4
arm fublcL2 color "--radius 3 --clamp 2 7 7" L2
for t in fublamg15 fublacg15 fublcLg15 fublcL fublcL2; do bash scripts/r15/lf.sh q2_m03$t /workspace/dub/r15f3/lipflick_w.jsonl | grep -o "\"W_excess_dL\": [0-9.]*, \"W_jumps_p99\": [0-9]*" | sed "s/^/$t /"; done
echo G2_DONE
