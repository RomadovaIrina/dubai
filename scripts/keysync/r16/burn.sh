#!/bin/bash
# burn.sh IN OUT LABEL : full-length copy with label + time burnt in, audio stream copied untouched
F=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf
ffmpeg -v error -y -i "$1" -vf "drawtext=fontfile=$F:text='$3 %{pts\:hms}':x=10:y=10:fontsize=h/40:fontcolor=yellow:box=1:boxcolor=black@0.5" -c:v libx264 -crf 17 -preset medium -pix_fmt yuv420p -c:a copy "$2"
