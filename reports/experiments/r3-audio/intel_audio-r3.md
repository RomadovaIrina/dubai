# Intelligibility guard: audio-r3

Whisper large-v3 WER and wav2vec2 CTC CER of every dub unit against its English text.
Unit mean per video; `bad` = CTC CER ≥ 0.3.

| video | units | WER (Whisper) | CER (CTC) | CER (Whisper) | bad | extra-sound units | last word lost | max speed-up |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 25 | 0.017 | 0.068 | 0.008 | 0 | 0 | 0 | x1.15 |
| 02 | 18 | 0.075 | 0.140 | 0.039 | 1 | 0 | 0 | x1.138 |
| 03 | 10 | 0.010 | 0.053 | 0.002 | 0 | 0 | 0 | x1.15 |
| 04 | 6 | 0.009 | 0.049 | 0.003 | 0 | 0 | 0 | x1.025 |
| 05 | 7 | 0.011 | 0.141 | 0.005 | 1 | 0 | 0 | x1.0 |

All 66 units: WER mean 0.030, units with WER > 0: 12; CTC CER mean 0.091.

## Worst 20 units (clips in `listen/`, names start with `audio-r3_worst`)

| # | clip | expected (English text) | Whisper heard | CTC heard | WER | CER ctc | hints |
|---:|---|---|---|---|---:|---:|---|
| 1 | `audio-r3_worst01_video02_1m46s_unclear.wav` | Hamam and all this space, indeed. | Our mom and all these spaces, indeed. | amaminall dispace indeed | 0.67 | 0.23 | slowed x0.875, first word differs (onset?) |
| 2 | `audio-r3_worst02_video01_0m26s_unclear.wav` | Such wallpapers withstand up to ten repaintings. Floor. | Such wallpapers withstand up to ten repaintings floor to floor. | such pool papers withstand uctor tenry paintings floor | 0.25 | 0.13 |  |
| 3 | `audio-r3_worst03_video02_1m23s_unclear.wav` | And the hamam is also present here. | and hammam is also present here | and the hammam is also present here | 0.29 | 0.04 | slowed x0.889 |
| 4 | `audio-r3_worst04_video02_0m15s_unclear.wav` | Here are our various things here zone for changing clothes barrel chaise longues. | Here are our warbiest things here, zone for changing clothes, barrel chaise, lungs | here a a wabies things here zon for changing clothes bawl chaise lons | 0.15 | 0.23 | slowed x0.875 |
| 5 | `audio-r3_worst05_video02_0m26s_unclear.wav` | In this place, we have a Finnish sauna. | in this place we have a finished sauna | in this place we have a finish sorner | 0.12 | 0.17 | slowed x0.875 |
| 6 | `audio-r3_worst06_video01_1m11s_unclear.wav` | Sink with a high-strength brass faucet... Balcony... | A sink with a high strength brass faucet balcony. | a sink with high strength brass fossit balcony | 0.12 | 0.15 | first word differs (onset?) |
| 7 | `audio-r3_worst07_video02_1m14s_fast.wav` | Well, this definitely won't break the bank. | Well, this definitely won't break the bank. | wo with defandiwan break the bank | 0.00 | 0.40 | sped up x1.12, slowed x0.888, CTC only (accent / rare word, Whisper fine) |
| 8 | `audio-r3_worst08_video02_0m41s_unclear.wav` | The Russian bath should have a stove burning wood. | The Russian bus should have a stove burning wood. | de russian bats would have a stow burning wood | 0.11 | 0.15 | slowed x0.875 |
| 9 | `audio-r3_worst09_video05_0m36s_unclear.wav` | So let there be such a video extra she definitely won't. | so let there be such a video extra she definitely won't | so led ther wer such a vider ex trusshin defon the wond | 0.00 | 0.36 | slowed x0.875, CTC only (accent / rare word, Whisper fine) |
| 10 | `audio-r3_worst10_video01_0m50s_unclear.wav` | Bathroom... Matte wide-format ceramic tile six hundred by six hundred with a thickness of nine millimeters.... | Bathroom, matte white format ceramic tile, 600 by 600 with a thickness of 9 millimeters. | bassroom mach white format seramic tyle six hundred by six hundred with the thickness of nine million metrs | 0.06 | 0.17 |  |
| 11 | `audio-r3_worst11_video01_0m33s_fast.wav` | Egger with a substrate. | Egger with a substrate | eger with sub street | 0.00 | 0.26 | sped up x1.15 |
| 12 | `audio-r3_worst12_video05_0m21s_unclear.wav` | Here it just wants to see how our application will handle this format. | here just wants to see how our application will handle this format | her just wants to see howr application will handle this foremat | 0.08 | 0.10 | slowed x0.875 |
| 13 | `audio-r3_worst13_video03_0m18s_unclear.wav` | Today is Friday, August twenty-eighth, and we will consider this the start of our cooperation. | Today is Friday, August 28, and we will consider this the start of our cooperation. | to day is friday august twenty eight and we will consider this distap of our corporation | 0.06 | 0.10 | slowed x0.875 |
| 14 | `audio-r3_worst14_video05_0m02s_unclear.wav` | Let's test another such format for video shooting. | let's test another such format for video shooting | le'st test a not a such format forvido chuting | 0.00 | 0.21 | slowed x0.875 |
| 15 | `audio-r3_worst15_video02_0m30s_unclear.wav` | I will show the Russian banya. | I will Show the Russian banya | i wi  show the russian mane | 0.00 | 0.21 |  |
| 16 | `audio-r3_worst16_video02_1m43s_fast.wav` | Here is such an interesting sauna we have. | Here is such an interesting sauna we have. | here is such an interesting this orner we have | 0.00 | 0.21 | sped up x1.14 |
| 17 | `audio-r3_worst17_video02_0m45s_unclear.wav` | So it all looks like this. | so it all looks like this | so it alros like this | 0.00 | 0.20 |  |
| 18 | `audio-r3_worst18_video01_1m18s_unclear.wav` | Plaster with white matte water-resistant paint. | plaster with white matte water-resistant paint | plaster with white mat water resisting pin | 0.00 | 0.15 | slowed x0.875 |
| 19 | `audio-r3_worst19_video01_0m17s_unclear.wav` | Walls... Wallpaper on a fiber-based substrate for painting... | Walls. Wallpaper on a fiber-based substrate for painting. | walls wall paper on the fibre bas substrate for painting | 0.00 | 0.15 | slowed x0.875 |
| 20 | `audio-r3_worst20_video04_0m11s_unclear.wav` | Sochi city today, August twenty-eight, and we are starting the working processes for creating our magical application. | Sochi City today, August 28th, and we are starting the working processes for creating our magical application. | zolki city to day august twenty eight and we are starting the working processes for creating our magical application | 0.06 | 0.03 | slowed x0.875 |
