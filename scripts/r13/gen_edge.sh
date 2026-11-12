#!/bin/bash
# r13 round 3: edge-case inputs for the robustness suite, all made from our own videos (ffmpeg, CPU). -> /workspace/dub/r13s/edge/in/<case>.<ext>
set -u
T=${EDGE_T:-/workspace/dub/inputs/test_videos_raw}; M=${EDGE_M:-/workspace/dub/inputs/mamakris}; O=${EDGE_O:-/workspace/dub/r13s/edge/in}; mkdir -p $O; cd $O   # r17: paths overridable (the VM)
B=$M/story-1774250817106-616905094.mp4      # m18: 720x1280 30 fps, one face, 20 s of speech
B3=$T/03.mp4
F="-loglevel error -y"
X="-c:v libx264 -crf 18 -preset fast -pix_fmt yuv420p -c:a aac -b:a 160k"
ff(){ ffmpeg $F "$@" || echo "FAILED $*"; }
ff -i $M/story-1773579769284-320619146.MP4 -t 20 -c copy fps60_real.mp4                                   # a real 60 fps phone clip (1072x1904)
ff -i $B -vf "fps=60" $X fps60_conv.mp4                                                                     # 30 -> 60 fps (frames doubled)
ff -i $B -vf "scale=2160:3840:flags=lanczos" -c:v libx264 -crf 20 -preset fast -pix_fmt yuv420p -c:a copy res4k_vertical.mp4
ff -i $B -vf "scale=-2:1080,pad=1920:1080:(ow-iw)/2:0" $X horizontal_1920x1080.mp4                           # landscape (pillarbox)
ff -i $B -vf "crop=720:720:0:200" $X square_720.mp4                                                           # square, the face centred
ff -i $B -vf "transpose=2" $X rot_tmp.mp4 && ff -display_rotation:v:0 -90 -i rot_tmp.mp4 -c copy rotated_meta90.mp4 && rm -f rot_tmp.mp4   # coded landscape + display matrix -90 (phone; r15: ffmpeg 6.1 ignores -metadata rotate)
ff -i $B3 -t 20 -vf "select='not(eq(mod(n\,4)\,0))'" -fps_mode vfr $X vfr.mp4                                 # variable frame rate (every 4th frame dropped, timestamps kept)
ff -i $B -c:v libx265 -pix_fmt yuv420p10le -x265-params "colorprim=bt2020:transfer=arib-std-b67:colormatrix=bt2020nc" -tag:v hvc1 -c:a copy hdr_hlg_10bit.mp4
ff -ss 2 -t 3 -i $B3 $X short_3s.mp4
ff -i $B -an -c:v copy no_audio.mp4
ff -i $B -f lavfi -i "sine=frequency=220:sample_rate=48000,volume=0.2[a0];sine=frequency=330:sample_rate=48000,volume=0.15[a1];[a0][a1]amix=inputs=2,apulsator=hz=2" -map 0:v -map 1:a -t 20 $X music_only.mp4 2>/dev/null \
  || ff -i $B -f lavfi -i "aevalsrc='0.2*sin(2*PI*220*t)+0.15*sin(2*PI*330*t)*(1+sin(2*PI*2*t))':s=48000" -map 0:v -map 1:a -t 20 $X music_only.mp4
ff -i $B -f lavfi -i "anullsrc=r=48000:cl=stereo" -map 0:v -map 1:a -t 20 $X silence.mp4
ff -i $B -i $M/story-1774250573471-666527943.MP4 -i $B3 -filter_complex "[0:v]scale=-2:640,setsar=1[a];[1:v]scale=-2:640,setsar=1[b];[2:v]scale=-2:640,setsar=1[c];[a][b][c]hstack=inputs=3[v]" \
  -map "[v]" -map 0:a -t 20 $X three_faces.mp4                                                                 # 3 people side by side (m18 speaks)
ff -i $B -vf "crop=720:1280:330:0:keep_aspect=0,pad=720:1280:0:0" $X face_half_out.mp4 2>/dev/null || ff -i $B -vf "crop=400:1280:320:0,pad=720:1280:0:0" $X face_half_out.mp4
ff -i $B -vf "eq=brightness=-0.32:gamma=0.6" $X very_dark.mp4
# 10+ min: 01 + 02 + 03 + 04 + 05 + 02 + 01 (480x848 / 464x848 -> 464x848, 30 fps) = 10.5 min
ff -i $T/01.MP4 -i $T/02.mp4 -i $T/03.mp4 -i $T/04.mp4 -i $T/05.mp4 -i $T/02.mp4 -i $T/01.MP4 -filter_complex \
  "$(for k in 0 1 2 3 4 5 6; do printf "[$k:v]scale=464:848,setsar=1,fps=30[v$k];[$k:a]aresample=48000,aformat=channel_layouts=stereo[a$k];"; done)[v0][a0][v1][a1][v2][a2][v3][a3][v4][a4][v5][a5][v6][a6]concat=n=7:v=1:a=1[v][a]" \
  -map "[v]" -map "[a]" $X long_10min.mp4
cp $M/story-1774250832175-746471562.mp4 odd_fps_29.58_real.mp4                                                # a real odd-rate phone clip (355/12 fps)
for f in *; do echo "$f $(ffprobe -v error -select_streams v:0 -show_entries stream=codec_name,width,height,r_frame_rate,avg_frame_rate,pix_fmt,color_transfer:stream_side_data=rotation -of csv=p=0 $f 2>/dev/null | head -1) a=$(ffprobe -v error -select_streams a -show_entries stream=codec_name -of csv=p=0 $f | head -1) dur=$(ffprobe -v error -show_entries format=duration -of csv=p=0 $f)"; done
echo GEN_EDGE_DONE
