# r16 Q: open-licence test clips (07.11.2026)

All clips come from Wikimedia Commons, under licences that allow reuse with attribution: CC0, CC BY 3.0 / 4.0 and CC BY-SA 4.0. CC BY-SA also requires derivatives to keep the same licence, so the dubbed versions of the BY-SA clips must stay CC BY-SA if they are ever published.
Many of the CC BY 3.0 files are YouTube CC-BY uploads that Commons reviewed.
- Originals were downloaded one at a time at a low rate: server `/workspace/dub/r16q/src/c<N>.webm`.
- Cuts are re-encoded to H.264 / AAC 48 kHz, at the native size and fps: `/workspace/dub/r16q/clips/<id>.mp4`.
- Cut list: `/workspace/dub/r16q/cuts.txt` (id, source, start s, duration s).
- The Russian speech in each segment was checked with a Whisper language scan.

| id | segment | size / fps | covers | licence | author (attribution) | source |
|---|---|---|---|---|---|---|
| q01_amir | 10-55 s | 1080x606 23.98 | beard, low light (dark room, warm key), bald head, frontal | CC BY-SA 4.0 | Bogreudell / Wikitongues | https://commons.wikimedia.org/wiki/File:WIKITONGUES-_Amir_speaking_Russian_and_Hebrew.webm |
| q02_vera | 92-137 s | 1920x1080 30 | older face, glasses, hands near the face, close-up | CC BY-SA 4.0 | Wikitongues | https://commons.wikimedia.org/wiki/File:The_Juhuri_language,_casually_spoken_-_Vera_speaking_Judeo-Tat_and_Russian_-_Wikitongues.webm |
| q03_priest | 60-105 s | 1920x1080 29.97 | beard / moustache, frontal studio, a cutaway at the end | CC BY 3.0 | Информационное агентство БелТА | https://commons.wikimedia.org/wiki/File:Страстная_седмица._Что_означает_каждый_день_недели-.webm |
| q04_farikh | 20-65 s | 1920x1080 29.97 | older face, beard, partial profile (3/4), outdoors, snow-bright | CC BY-SA 4.0 | Ольга Суховарова | https://commons.wikimedia.org/wiki/File:Михаил_Фарих_Экспедиция_По_следам_двух_капитанов.webm |
| q05_rozhkov | 100-145 s | 1920x1080 25 | head turns, a hand / cup near the face, stubble, fast speech, a cutaway (product shot) | CC BY 3.0 | Наран Клиника тибетской медицины | https://commons.wikimedia.org/wiki/File:Андрей_Рожков_рассказывает_о_лечении_в_клинике_НАРАН.webm |
| q06_suvorov | 10-55 s | 1920x1080 25 | LOW / warm light, older face, partial profile, gestures; burnt-in subtitles | CC BY 4.0 | Arsenysob | https://commons.wikimedia.org/wiki/File:Suvorov_rodion2000.webm |
| q07_altapress | 3-50 s | 1280x720 30 | TWO+ faces (a crowd of reporters around), glasses, microphones near the mouth, fast speech, outdoors | CC BY 3.0 | altapress.ru | https://commons.wikimedia.org/wiki/File:Altapress.ru-_Мария_Мотузная_рассказывает_о_втором_заседании_суда.webm |
| q08_albats | 120-165 s | 640x480 14.97 | a HAND (cigarette) at the mouth, head on hand, low resolution / low fps | CC BY 3.0 | Alexandr Plushev | https://commons.wikimedia.org/wiki/File:Интервью_Евгении_Альбац_Александру_Плющеву,_2010.webm |
| q09_vertical | 20-60 s | 1080x1920 29.97 | VERTICAL 1080p phone selfie, frontal; speech is Tachelhit (Berber), NOT Russian | CC BY 3.0 | Abderrahman Ait Ali / Wikitongues | https://commons.wikimedia.org/wiki/File:WIKITONGUES-_Abderrahman_speaking_Tachelhit_(vertical_original).webm |

Candidates looked at and left out:
- Theodor Tezhik (CC BY-SA 4.0; glasses + a lens at the mouth): nearly silent, mean −55 dB.
- Tom Scott grizzly selfie (PD): no talking-head segment.
- Kythnos 4K oral story (CC BY-SA 4.0): download pending (rate-limited); it would be the only 4K clip, in Greek.
- No vertical 4K Russian clip was found on Commons. T has `res4k_vertical.mp4` for #23.

Coverage gaps:
- vertical 4K (q09 is vertical 1080p, not Russian);
- a true side profile (q04 / q06 are 3/4 only).
