#!/bin/bash
# r9 audio AB for the ear (r8_watch.sh format, phone-sized): <id>_r9fast-audio_WATCH_AB_prev-then-new.mp4 = the source video with the PREV
# htdemucs (demucs) final mix, a 1-s black "B" card, then the same video with the NEW demucs mix; <= 25 MB each (video bitrate from the length);
# + <id>_r9fast-audio_listen.md (coverage / pacing spans PREV vs NEW with both timecodes, scripts/r8/listen.py on the Spleeter run dirs: same dub)
#   bash scripts/r9/r9_audio_ab.sh <out dir> <id>:<prev tag>:<new tag> ...     (tags in /workspace/dub/artifacts/r4a; the demucs mix = <tag>_htd)
set -u
cd /workspace/dub/dubai; . scripts/env.sh >/dev/null 2>&1; export DUB_ROOT=/workspace/dub/dubai
OUT=$1; shift; A=/workspace/dub/artifacts/r4a; mkdir -p $OUT; V=r9fast-audio; MB=${AB_MB:-24}
for spec in "$@"; do
  IFS=: read -r id P N <<< "$spec"
  d=/workspace/dub/r9/runs/$id/work/in; [ -d $d ] || d=/workspace/dub/inputs/test_videos_raw; src=$(ls $d/$id.* | head -1)
  dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $src)
  vk=$(python3 -c "print(max(300, int($MB * 8 * 1000 / (2 * $dur + 1)) - 140))")   # kbit/s video so the whole file stays under MB megabytes (128k audio + muxing)
  t=$(mktemp -d); enc="-c:v libx264 -preset medium -b:v ${vk}k -maxrate ${vk}k -bufsize $((2 * vk))k -pix_fmt yuv420p -c:a aac -b:a 128k -ar 48000 -ac 2"
  for s in a:$P:A b:$N:B; do k=${s%%:*}; r=${s#*:}; tg=${r%%:*}; lb=${r#*:}
    ffmpeg -loglevel error -y -i $src -i $A/${tg}_htd/$id/final_mix_aac.m4a -map 0:v:0 -map 1:a:0 \
      -vf "scale=-2:854,fps=25,drawtext=text='$lb  $([ $lb = A ] && echo baseline || echo r9fast-audio)':x=20:y=20:fontsize=30:fontcolor=white:box=1:boxcolor=black@0.6" $enc -shortest $t/$k.mp4
  done
  W=$(ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=p=0 $t/a.mp4)
  ffmpeg -loglevel error -y -f lavfi -i "color=black:s=${W/,/x}:r=25:d=1" -f lavfi -i "anullsrc=r=48000:cl=stereo" -t 1 \
    -vf "drawtext=text='B  $V':x=(w-tw)/2:y=(h-th)/2:fontsize=56:fontcolor=white" $enc -shortest $t/card.mp4
  printf "file '%s'\n" $t/a.mp4 $t/card.mp4 $t/b.mp4 > $t/l.txt
  o=$OUT/${id}_${V}_WATCH_AB_prev-then-new.mp4; ffmpeg -loglevel error -y -f concat -safe 0 -i $t/l.txt -c copy -movflags +faststart $o
  off=$(python3 -c "import subprocess;d=float(subprocess.check_output(['ffprobe','-v','error','-show_entries','format=duration','-of','csv=p=0','$t/a.mp4']));print(round(d+1,2))")
  fc=$A/sep/$id/source_mediapipe.npz; [ -f $fc ] || fc=$(ls $A/${N}/../sep/$id/source_mediapipe.npz 2>/dev/null | head -1)
  python scripts/r8/listen.py --id $id --ver $V --prev $A/$P/cov/cov_$id.json --new $A/$N/cov/cov_$id.json --pace-prev $A/$P/cov/pace_$id.json --pver baseline \
    --pace-new $A/$N/cov/pace_$id.json --offset $off --cuts-prev $A/$P/cov/cuts_$id.json --cuts-new $A/$N/cov/cuts_$id.json \
    --man-prev $A/$P/$id/${id}_r4b.manifest.json --man-new $A/$N/$id/${id}_r4b.manifest.json > $OUT/${id}_${V}_listen.md 2> $t/listen.err || { echo "WARN listen $id"; tail -2 $t/listen.err; }
  mb=$(python3 -c "import os;print(round(os.path.getsize('$o')/1e6,1))")
  echo "AB $id: $o  ($mb MB, A = $P, B = $N, B starts at $off s)"; rm -rf $t
done
