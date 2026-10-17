#!/bin/bash
# Round 2 track B: work dir for <id> from the canonical baseline (audio symlinked, regenerated 25 fps master, manifest with patched paths).
set -eu
id=$1; W=/workspace/dub/artifacts/restoration-temporal/r2/work; C=/workspace/dub/canonical/artifacts/baseline/$id
mkdir -p $W/work_$id; cd $W/work_$id
for f in $C/work/work_$id/*; do b=$(basename $f); [ "$b" = master_25fps.mp4 ] && continue; ln -sfn $f $b; done
src=$(python3 -c "import json;print(json.load(open('$C/final/${id}_final.manifest.json'))['input'])")
[ -f master_25fps.mp4 ] || ffmpeg -loglevel error -y -nostdin -i $src -r 25 -crf 18 -an master_25fps.mp4
python3 - <<PY
import json, subprocess
m = json.load(open('$C/final/${id}_final.manifest.json')); old = m['work_dir']
m = json.loads(json.dumps(m).replace(old, '$W/work_$id')); m['output'] = m['output'] + ' (not backed up)'
n = int(subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=nb_read_frames','-of','csv=p=0','master_25fps.mp4'],capture_output=True,text=True).stdout)
assert n == m['lipsync']['master_frames'], (n, m['lipsync']['master_frames'])
json.dump(m, open('$W/${id}_final.manifest.json', 'w'), indent=1); print('$id', 'master frames', n, '== manifest; work_dir', m['work_dir'])
PY
