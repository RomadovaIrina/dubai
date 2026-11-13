#!/bin/bash
# r18, INSIDE the image (r18.sh repaste): re-paste a finished job (no generation, CPU) with the KS_* env of this container, then exactly e2e's
# tail: hi-res merge (if the face work was downscaled), mux with the SAME accepted mix by stream copy. (= r16d_repost.sh without its eval, with
# an env that may hold spaces.) KS_POST_REUSE=1 (r18.sh default): the finished post layer is reused, only the paste differs.
#   repaste_in.sh <job id> <variant>  ->  /r18/out/<id>/<variant>.mp4 (+ .paste.txt: the paste line of every face / shot)
set -u
ID=$1; VAR=$2; R=/workspace/dub/dubai; cd $R; J=/data/jobs/$ID; W=$J/e2e/work; KW=$W/ks; AD=/r18/dubs/$ID; O=/r18/out/$ID; mkdir -p $O; PY=/venv/dabai/bin/python; s=$(date +%s)
[ -f $KW/shots.json ] || { echo "REPASTE $ID $VAR failed: no job work dir ($KW)"; exit 2; }
RT=""; [ -f $W/route/route.json ] && RT=$W/route/route.json
for K in $KW/kit $W/shots/s*/ks/kit; do [ -d $K ] && cp scripts/keysync/*.py scripts/keysync/r15/*.py scripts/keysync/r16/*.py $K/ 2>/dev/null; done   # the kit dates from the job's prep
NSEG=$($PY -c "import json; print(len(json.load(open('$KW/shots.json'))['segments']))"); NEW=$W/r18_face_$VAR.mp4; rm -f $NEW
echo "REPASTE $ID $VAR: segments $NSEG, env $(env | grep -E '^KS_(MASK|PLAIN|SEAM|POST|TAIL|OCCL|PAUSE|CAP|CMS|CM_)' | sort | tr '\n' ' ')"
if [ "$NSEG" -gt 1 ]; then KS_SHOTS_REPOST=1 bash scripts/keysync/r16/ks_shots.sh $ID $AD $W $NEW "$RT" $KW > $W/r18_face_$VAR.log 2>&1
else KS_PHASE=post bash scripts/keysync/r16/ks_face.sh $ID x $AD $W $NEW $RT > $W/r18_face_$VAR.log 2>&1; fi
rc=$?; grep -h -E "KS_POST_REUSE|KS_PASTE|ks_seam|KS_SHOTS_REPOST|KS_SHOT [0-9]+ .*(fail|source)|KS_FLAG" $W/r18_face_$VAR.log $W/shots/s*/face_post.log 2>/dev/null | cut -c1-260 | tee $O/$VAR.paste.txt
[ $rc = 0 ] && [ -s $NEW ] || { echo "REPASTE $ID $VAR failed: face post rc=$rc"; tail -5 $W/r18_face_$VAR.log; exit 1; }
VID=$NEW
if ls $W/hires_src.* > /dev/null 2>&1; then HI=$W/hires_25fps.mp4
  [ -s $HI ] || ffmpeg -loglevel error -y -i $(ls $W/hires_src.* | head -1) -map 0:v:0 -an -vf fps=25 -c:v libx264 -preset veryfast -crf 12 -pix_fmt yuv420p $HI
  $PY scripts/r15/hires_merge.py --hi $HI --lo-src $KW/master_25fps.mp4 --lo-comp $VID --out $W/r18_hires_$VAR.mp4 --json $W/r18_hires_$VAR.json --threads ${HIRES_THREADS:-16} > $W/r18_hires_$VAR.log 2>&1 \
    && VID=$W/r18_hires_$VAR.mp4 || { echo "REPASTE $ID $VAR failed: hires merge"; tail -3 $W/r18_hires_$VAR.log; exit 1; }; fi
ffmpeg -loglevel error -y -i $VID -i $AD/final_mix_aac.m4a -map 0:v:0 -map 1:a:0 -c copy -movflags +faststart $O/$VAR.mp4 || { echo "REPASTE $ID $VAR failed: mux"; exit 1; }
rm -f $NEW $W/r18_hires_$VAR.mp4
echo "REPASTE_DONE $ID $VAR $(( $(date +%s) - s )) s -> $O/$VAR.mp4 ($(stat -c %s $O/$VAR.mp4) B)"
