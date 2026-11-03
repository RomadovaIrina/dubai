cd /workspace/dub/dubai; export FC_CMS=1 LS_CMS_READ=1 QUICK=1 FC_EXTRA_ADD="--sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.5"
B=/workspace/dub/artifacts/r4m
for id in m03 old03; do
  bash scripts/r12/ls_arm.sh $id e34q "" LS_RATE=3/4 LS_RATE_EVEN=1
  ( . scripts/env.sh >/dev/null; SRC=$(ls /workspace/dub/r10/runs/$id/work/in/$id.* | head -1); V=$B/r9f_${id}e34q/${id}_r9f_${id}e34q.mp4; SL=$B/r9g_$id/timing/${id}_source_lm_F.npz
    python scripts/r12/mouth_pop.py --video $V --src $SRC --src-lm $SL --label "$id e34q @6.25" --cad-hz 6.25
    python scripts/r12/mouth_pop.py --video $B/r9f_${id}cmsr/${id}_r9f_${id}cmsr.mp4 --src $SRC --src-lm $SL --label "$id cmsr @6.25" --cad-hz 6.25
    python scripts/quality/r3_flash.py --video $V --src $SRC --lm $SL --out /workspace/dub/r12/flash_${id}e34q.json > /dev/null 2>&1
    python3 -c "import json;d=json.load(open(\"/workspace/dub/r12/flash_${id}e34q.json\"));print(\"== flash ${id} e34q strong\", sum(1 for e in d[\"flash_list\"] if (e.get(\"mc_ratio\") or 0)>=2 and (e.get(\"mc_vs_src\") or 0)>=2))"
    bash scripts/r12/clip_ls.sh $id e34q "k5 + q2 + even 3/4" | tail -2 ) &
done; wait; echo E34Q_DONE
