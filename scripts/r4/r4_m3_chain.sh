cd /workspace/dub/dubai; . scripts/env.sh >/dev/null 2>&1; B=/workspace/dub/artifacts/r4m
until grep -q R4_E2E_DONE $B/e2e.out; do sleep 30; done
bash $B/e2e_snapshot.sh 03 01 02 > $B/e2e2.out 2>&1
for id in 04 03 05 01 02; do python scripts/r4/r4_gate.py score --label v4repro --dir $B/v4repro --ids $id > /dev/null 2>&1; done
python scripts/r4/r4_gate.py compare --new $B/m3:m3 --prev $B/v4repro:v4repro --ids 04,03,05,01,02 --audio-info --out $B/gate_m3_all.md > /dev/null 2>&1
for id in 04 03 05 01 02; do
  VNAME=r4_merged_m3 GATE_ARGS=--audio-info bash scripts/r4/r4_publish.sh merged m3 $B/m3:m3 $B/v4repro:v4repro $id > $B/pub_m3_$id.out 2>&1 && echo "PUBLISHED $id" || echo "GATE FAIL $id"
done
echo M3_CHAIN_DONE
