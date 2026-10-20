cd /workspace/dub/dubai; . scripts/env.sh >/dev/null 2>&1; B=/workspace/dub/artifacts/r4m; OB=/workspace/dub/outbox; F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf; done_ids="05 04"
INTRO=$(sed -n 3p $OB/r4_merged_m3_WATCH.md)
for id in 03 01 02; do
  until grep -q "m3 $id\|FAIL .* $id" $B/e2e2.out 2>/dev/null; do sleep 30; done
  [ -f $B/m3/${id}_m3.mp4 ] || { echo "no m3 $id"; continue; }; done_ids="$done_ids $id"
  python scripts/r4/r4_gate.py score --label v4repro --dir $B/v4repro --ids $id > /dev/null 2>&1
  VM=$B/m3/work_${id}_m3/${id}_m3_25fps.manifest.json
  python scripts/quality/r3_joints.py --video $B/m3/${id}_m3.mp4 --manifest $VM --lm /workspace/dub/artifacts/r3d/timing/${id}_source_lm_F.npz --out $B/m3/${id}_m3.joints_full.json > /dev/null 2>&1
  cp $B/m3/${id}_m3.mp4 $OB/${id}_r4_merged_m3_WATCH.mp4
  ffmpeg -loglevel error -y -i /workspace/dub/reference/v4/${id}_v4.mp4 -i $B/m3/${id}_m3.mp4 -filter_complex "[0:v]drawtext=fontfile=$F:text='V4':x=12:y=12:fontsize=36:fontcolor=yellow:box=1:boxcolor=black@0.6[a];[1:v]drawtext=fontfile=$F:text='M3 (WATCH)':x=12:y=12:fontsize=36:fontcolor=yellow:box=1:boxcolor=black@0.6[b];[a][b]hstack=inputs=2,scale='min(2560,iw)':-2[v]" -map "[v]" -map 1:a:0 -c:v libx264 -crf 20 -preset veryfast -pix_fmt yuv420p -c:a copy $OB/${id}_r4_merged_m3_WATCH_sbs.mp4
  python scripts/r4/r4_gate.py compare --new $B/m3:m3 --prev $B/v4repro:v4repro --ids $(echo $done_ids | tr ' ' ',') --audio-info --out $B/gate_m3_watch.md > /dev/null 2>&1
  python3 scripts/r4/r4_watch_note.py --name m3 --intro "$INTRO" --new $B/m3:m3 --prev $B/v4repro:v4repro --gate $B/gate_m3_watch.md --ids $done_ids --out $OB/r4_merged_m3_WATCH.md > /dev/null
  echo "== m3 WATCH $id $(date +%T)"
done
echo M3_WATCH_DONE
