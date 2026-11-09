#!/bin/bash
# q clips from Wikimedia Commons originals, cut like Q (CLIPS.md): native size/fps, H.264 + AAC 48 kHz
while read -r id ss dur file; do
  [ -s $id.webm ] || curl -sL -A "dabai-r16-research/1.0" -o $id.webm "https://commons.wikimedia.org/wiki/Special:FilePath/$file"
  echo "$id webm $(stat -c %s $id.webm) $(ffprobe -v error -show_entries format=duration -of csv=p=0 $id.webm)"
  ffmpeg -nostdin -v error -y -ss $ss -i $id.webm -t $dur -c:v libx264 -crf 16 -preset medium -pix_fmt yuv420p -c:a aac -ar 48000 -b:a 192k $id.mp4
  echo "$id clip $(ffprobe -v error -show_entries stream=width,height,r_frame_rate -show_entries format=duration -of csv=p=0 $id.mp4 | tr "\n" " ")"
done <<LIST
q01_amir 10 45 WIKITONGUES-_Amir_speaking_Russian_and_Hebrew.webm
q02_vera 92 45 The_Juhuri_language,_casually_spoken_-_Vera_speaking_Judeo-Tat_and_Russian_-_Wikitongues.webm
q03_priest 60 45 Страстная_седмица._Что_означает_каждый_день_недели-.webm
q04_farikh 20 45 Михаил_Фарих_Экспедиция_По_следам_двух_капитанов.webm
q05_rozhkov 100 45 Андрей_Рожков_рассказывает_о_лечении_в_клинике_НАРАН.webm
q06_suvorov 10 45 Suvorov_rodion2000.webm
LIST
