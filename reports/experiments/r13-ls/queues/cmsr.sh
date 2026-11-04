cd /workspace/dub/dubai
until grep -q EVEN2_DONE /workspace/dub/r12/logs/even2.out; do sleep 20; done
export FC_CMS=1 LS_CMS_READ=1 QUICK=1
B=/workspace/dub/artifacts/r4m
for id in m03 old03; do
  FC_EXTRA_ADD="--sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.5" bash scripts/r12/ls_arm.sh $id cmsr ""
  ( . scripts/env.sh >/dev/null; SRC=$(ls /workspace/dub/r10/runs/$id/work/in/$id.* | head -1); V=$B/r9f_${id}cmsr/${id}_r9f_${id}cmsr.mp4
    python scripts/r12/lab_drift.py --video $V --src $SRC --src-lm $B/r9g_$id/timing/${id}_source_lm_F.npz --label "$id cmsr"
    python scripts/quality/r3_flash.py --video $V --src $SRC --lm $B/r9g_$id/timing/${id}_source_lm_F.npz --out /workspace/dub/r12/flash_${id}cmsr.json > /dev/null 2>&1
    python3 -c "import json;d=json.load(open(\"/workspace/dub/r12/flash_${id}cmsr.json\"));print(\"== flash ${id} cmsr strong\", sum(1 for e in d[\"flash_list\"] if (e.get(\"mc_ratio\") or 0)>=2 and (e.get(\"mc_vs_src\") or 0)>=2))" ) &
done; wait; echo CMSR_DONE
