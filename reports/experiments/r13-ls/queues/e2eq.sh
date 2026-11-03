cd /workspace/dub/dubai; export E2E_METRICS=off DUB_ALLOW_NC=1 LS_CMS_READ=1 FC_CMS=1 FC_EXTRA="--pre-threads=12 --sharpen-luma on --sharpen-auto 1.05 --sharpen-lips-max 0.2"
M=/workspace/dub/inputs/mamakris; T=/workspace/dub/inputs/test_videos_raw; O=/workspace/dub/r13/e2e; mkdir -p $O
declare -A F=([m03]=$M/story-1774249919482-367803614.MOV [m18]=$M/story-1774250817106-616905094.mp4 [old03]=$T/03.mp4)
for id in m03 old03 m18; do
  E2E_FX=1 E2E_SYNC=1 ID=${id}q2 bash scripts/r9/e2e.sh ${F[$id]} $O/${id}q2 > $O/${id}q2.out 2>&1
  echo "== ${id}q2 $(grep -h "NVML GPU-busy" $O/${id}q2/logs/e2e.log | tail -1) | $(grep -h "SUCCESS flags" $O/${id}q2/logs/e2e.log | tail -1 | cut -c1-240)"
done; echo E2EQ_DONE
