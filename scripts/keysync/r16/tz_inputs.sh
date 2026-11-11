#!/bin/bash
# r16 TZ: the KeySync test inputs (CPU, ffmpeg) -> /workspace/dub/r16tz/in: the r13 edge set (gen_edge.sh: 4K vertical, silence, music only,
# three faces, long_10min, ...) + what the r16 brief adds: one face 20 s, 2 shots with a hard cut, no face, a tiny face, 2 faces side by side.
set -u
R=$(cd "$(dirname "$0")/../../.." && pwd); T=/workspace/dub/inputs/test_videos_raw; M=/workspace/dub/inputs/mamakris; O=/workspace/dub/r16tz/in; mkdir -p $O
M18=$M/story-1774250817106-616905094.mp4; M03=$M/story-1774249919482-367803614.MOV; O03=$T/03.mp4
X="-c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p -c:a aac -b:a 160k"; F="-loglevel error -y"
ff(){ ffmpeg $F "$@" || echo "FAILED $*"; }
bash $R/scripts/r13/gen_edge.sh > $O/gen_edge.log 2>&1; for f in /workspace/dub/r13s/edge/in/*; do ln -sfn $f $O/$(basename $f); done   # the r13 set (incl. long_10min.mp4)
ff -i $M18 -t 20 $X $O/m18_20s.mp4                                                                                 # one face, 20 s (the fault suite's single-shot input)
V="scale=720:1280:force_original_aspect_ratio=decrease,pad=720:1280:(ow-iw)/2:(oh-ih)/2,setsar=1,fps=30"; A="aresample=48000,aformat=channel_layouts=stereo"
ff -i $M18 -i $O03 -filter_complex "[0:v]trim=0:10,setpts=PTS-STARTPTS,$V[v0];[0:a]atrim=0:10,asetpts=PTS-STARTPTS,$A[a0];[1:v]trim=0:10,setpts=PTS-STARTPTS,$V[v1];[1:a]atrim=0:10,asetpts=PTS-STARTPTS,$A[a1];[v0][a0][v1][a1]concat=n=2:v=1:a=1[v][a]" \
   -map "[v]" -map "[a]" $X $O/two_shots_cut.mp4                                                                   # 2 shots, a hard cut at 10 s (2 different people)
ff -i $M18 -t 20 -vf "crop=720:300:0:980,scale=720:1280,setsar=1" $X $O/no_face.mp4                                # no face (the lower part of the frame), the speech kept
ff -i $M18 -t 20 -vf "scale=180:320,pad=720:1280:270:480:black,setsar=1" $X $O/tiny_face.mp4                       # the whole frame at 1/4 size: a tiny face
ff -i $M18 -i $M03 -t 20 -filter_complex "[0:v]scale=-2:1280,setsar=1[a];[1:v]scale=-2:1280,setsar=1[b];[a][b]hstack=inputs=2[v]" -map "[v]" -map 0:a $X $O/two_faces.mp4   # 2 faces (m18 speaks)
for f in $O/*.mp4; do echo "$(basename $f) $(ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of csv=p=0 $f 2>/dev/null) $(ffprobe -v error -show_entries format=duration -of csv=p=0 $f 2>/dev/null)"; done
echo TZ_INPUTS_DONE
