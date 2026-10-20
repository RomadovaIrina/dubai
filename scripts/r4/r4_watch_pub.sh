# publish m2 WATCH files for each id as soon as both renders are done
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null 2>&1; B=/workspace/dub/artifacts/r4m; OB=/workspace/dub/outbox; F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf; done_ids="04 03"
for id in 05 01 02; do
  until grep -q "m2 $id\|FAIL m2 $id" $B/run.out && grep -q "v4repro $id\|FAIL v4repro $id" $B/run.out; do sleep 20; done
  [ -f $B/m2/${id}_m2.mp4 ] || { echo "no m2 $id"; continue; }; done_ids="$done_ids $id"
  VM=$B/m2/work_${id}_m2/${id}_m2_25fps.manifest.json
  python scripts/quality/r3_joints.py --video $B/m2/${id}_m2.mp4 --manifest $VM --lm /workspace/dub/artifacts/r3d/timing/${id}_source_lm_F.npz --out $B/m2/${id}_m2.joints_full.json > /dev/null 2>&1
  cp $B/m2/${id}_m2.mp4 $OB/${id}_r4_merged_m2_WATCH.mp4
  ffmpeg -loglevel error -y -i /workspace/dub/reference/v4/${id}_v4.mp4 -i $B/m2/${id}_m2.mp4 -filter_complex "[0:v]drawtext=fontfile=$F:text='V4':x=12:y=12:fontsize=36:fontcolor=yellow:box=1:boxcolor=black@0.6[a];[1:v]drawtext=fontfile=$F:text='M2 (WATCH, not gate-passing)':x=12:y=12:fontsize=36:fontcolor=yellow:box=1:boxcolor=black@0.6[b];[a][b]hstack=inputs=2,scale='min(2560,iw)':-2[v]" -map "[v]" -map 1:a:0 -c:v libx264 -crf 20 -preset veryfast -pix_fmt yuv420p -c:a copy $OB/${id}_r4_merged_m2_WATCH_sbs.mp4
  python scripts/r4/r4_gate.py compare --new $B/m2:m2 --prev $B/v4repro:v4repro --ids $(echo $done_ids | tr ' ' ',') --out $B/gate_m2_all.md > /dev/null 2>&1
  python3 scripts/r4/r4_watch_note.py --new $B/m2:m2 --prev $B/v4repro:v4repro --gate $B/gate_m2_all.md --ids $done_ids --out $OB/r4_merged_m2_WATCH.md > /dev/null
  cat $B/watch_extra.md >> $OB/r4_merged_m2_WATCH.md
  echo "== WATCH $id $(date +%T)"
done
echo WATCH_PUB_DONE
