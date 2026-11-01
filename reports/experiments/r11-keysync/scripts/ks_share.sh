#!/bin/bash
# KeySync share pack: full-length full-frame side-by-side, r9+fix (LatentSync) vs KeySync raw paste; blind (A/B, random side, KEY.txt) + labelled
set -e; O=/workspace/dub/outbox/r11_keysync/share; mkdir -p $O; B=/workspace/dub/artifacts/r4m; A=/workspace/dub/artifacts/r4a
F=$(fc-match -f "%{file}" "DejaVu Sans:bold"); echo "font $F"
echo "Blind A/B key (KeySync = raw paste of KeySync 512 crop, its per-video AV offset corrected: m18 delayed 1 frame, old03 advanced 1 frame; LatentSync = r9+fix, the current pipeline). $(date -u +%F)" > $O/KEY.txt
for v in m18:q18 old03:qo3; do n=${v%%:*}; id=${v#*:}
  L=$B/r9f_$id/${id}_r9f_$id.mp4; K=$B/ks_${id}d/${id}_ks_${id}d.mp4; M=$A/r9_${id}_htd/$id/final_mix_aac.m4a
  W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width -of csv=p=0 $L); PW=$(( W > 720 ? 720 : W ))
  lab() { echo "drawtext=fontfile=$F:text='$1':fontsize=h/26:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=12:x=(w-tw)/2:y=h/40"; }
  mk() {  # mk <left> <right> <left label> <right label> <out>
    ffmpeg -v error -y -i $1 -i $2 -i $M -filter_complex "[0:v]scale=$PW:-2:flags=lanczos,setsar=1,$(lab "$3")[a];[1:v]scale=$PW:-2:flags=lanczos,setsar=1,$(lab "$4")[b];[a][b]hstack=shortest=1[v]" \
      -map "[v]" -map 2:a -c:v libx264 -crf ${CRF:-20} -preset slow -pix_fmt yuv420p -profile:v high -movflags +faststart -c:a aac -b:a 160k -shortest $5
    s=$(stat -c %s $5); [ $s -le 25000000 ] || { echo "over 25 MB ($s), crf+3"; CRF=$(( ${CRF:-20} + 3 )) mk "$@"; }
  }
  FIRST=${FIRST:-$(( RANDOM % 2 ))}; NV=$(( ${NV:-0} + 1 )); SIDE=$(( NV == 1 ? FIRST : 1 - FIRST ))   # FIRST: 0 = KeySync on A for the first video; the second video gets the other side
  if [ $SIDE = 0 ]; then mk $K $L A B $O/blind_$n.mp4; echo "$n: A = KeySync, B = LatentSync (now)" >> $O/KEY.txt
  else mk $L $K A B $O/blind_$n.mp4; echo "$n: A = LatentSync (now), B = KeySync" >> $O/KEY.txt; fi
  mk $L $K "сейчас (LatentSync)" "KeySync" $O/labelled_$n.mp4
done
ls -la $O; cat $O/KEY.txt; for f in $O/*.mp4; do ffprobe -v error -show_entries stream=codec_name,width,height,nb_frames -show_entries format=duration -of compact $f | tr "\n" " "; echo; done
