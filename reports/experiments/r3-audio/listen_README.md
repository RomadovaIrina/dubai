<!-- audio-r2 -->
## Worst 20 units (clips in `listen/`, names start with `audio-r2_worst`)

| # | clip | expected (English text) | Whisper heard | CTC heard | WER | CER ctc | hints |
|---:|---|---|---|---|---:|---:|---|
| 1 | `audio-r2_worst01_video02_1m46s_unclear.wav` | Hamam and all this space. | Tamam and all the space | amam and all the space | 0.40 | 0.15 | slowed x0.875, first word differs (onset?) |
| 2 | `audio-r2_worst02_video01_0m50s_number.wav` | Bathroom... Matte wide-format ceramic tile 600x600 with a thickness of 9 mm... | bathroom matte white format ceramic tile 600 x 67 with a thickness of 9 mm | ba through mut white form at seromic tile six hundred ixix seventh with a thickness of nine milimit | 0.33 | 0.28 | number in text, extra sound: mm |
| 3 | `audio-r2_worst03_video02_1m23s_unclear.wav` | And the hamam is also available here. | And hammam is also available here | and the hummum is also vilable here | 0.29 | 0.17 |  |
| 4 | `audio-r2_worst04_video01_0m26s_number.wav` | Such wallpapers withstand up to 10 repaintings.. Floor.. | Such wallpapers would stand up to 10 repaintings floor, | such wall peapers with stand up to tenry paintings floor | 0.25 | 0.04 | number in text |
| 5 | `audio-r2_worst05_video02_0m15s_unclear.wav` | Here are our various things here zone for changing clothes barrel chaise longues. | Here are our warbiest things here, zone for changing clothes, barrel chaise, lungs | here a a wabies things here zon for changing clothes bawl chaise lons | 0.15 | 0.23 | slowed x0.875 |
| 6 | `audio-r2_worst06_video02_0m47s_unclear.wav` | What is such a nice smell here? | what is such a nice smell here | what is that you're nice tmill here | 0.00 | 0.46 | slowed x0.875, CTC only (accent / rare word, Whisper fine) |
| 7 | `audio-r2_worst07_video02_0m26s_unclear.wav` | In this place, we have a Finnish sauna. | in this place we have a finished sauna | in this place we have a finish sorner | 0.12 | 0.17 | slowed x0.875 |
| 8 | `audio-r2_worst08_video01_1m11s_unclear.wav` | Sink with a high-strength brass faucet... Balcony... | A sink with a high strength brass faucet balcony. | a sink with high strength brass fossit balcony | 0.12 | 0.15 | first word differs (onset?) |
| 9 | `audio-r2_worst09_video02_1m14s_fast.wav` | Well, this definitely won't break the bank. | Well, this definitely won't break the bank. | wo with defandiwan break the bank | 0.00 | 0.40 | sped up x1.12, slowed x0.888, CTC only (accent / rare word, Whisper fine) |
| 10 | `audio-r2_worst10_video05_0m36s_unclear.wav` | So let there be such a video extra she definitely won't. | so let there be such a video extra she definitely won't | so led ther wer such a vider ex trusshin defon the wond | 0.00 | 0.36 | slowed x0.875, CTC only (accent / rare word, Whisper fine) |
| 11 | `audio-r2_worst11_video02_0m41s_unclear.wav` | The Russian bath should have a stove burning wood. | The Russian bus should have a stove burning wood. | de russian bats would have a stowe burning wood | 0.11 | 0.12 | slowed x0.875 |
| 12 | `audio-r2_worst12_video02_0m30s_unclear.wav` | I will show the Russian sauna. | I will Show the Russian sauna | iven show the russian tona | 0.00 | 0.29 |  |
| 13 | `audio-r2_worst13_video05_0m21s_unclear.wav` | Here it just wants to see how our application will handle this format. | here just wants to see how our application will handle this format | her just wants to see howr application will handle this foremat | 0.08 | 0.10 | slowed x0.875 |
| 14 | `audio-r2_worst14_video05_0m02s_unclear.wav` | Let's test another such format for video shooting. | let's test another such format for video shooting | le'st test a not a such format forvido chuting | 0.00 | 0.21 | slowed x0.875 |
| 15 | `audio-r2_worst15_video05_0m41s_unclear.wav` | We will thoroughly test all variants that it is possible to imagine. | we will thoroughly test all variants that it is possible to imagine | boy will throug it test or variens that it is possible to imagine | 0.00 | 0.21 | slowed x0.875 |
| 16 | `audio-r2_worst16_video01_0m33s_fast.wav` | Egger with a substrate.. | Egger with a substrate | eggor with the substrate | 0.00 | 0.21 | sped up x1.14 |
| 17 | `audio-r2_worst17_video02_0m45s_unclear.wav` | So it all looks like this. | so it all looks like this | so it alros like this | 0.00 | 0.20 |  |
| 18 | `audio-r2_worst18_video02_1m43s_fast.wav` | Here is such an interesting sauna that we have. | Here is such an interesting sauna that we have. | there is such an interesting sshonour that we have | 0.00 | 0.18 | sped up x1.15, slowed x0.875 |
| 19 | `audio-r2_worst19_video04_0m11s_number.wav` | Sochi city today, August 28, and we are starting the working processes for creating our magical application. | Saki City Today August 28 And we are starting the working processes for creating our magical application | sulky city to day august twenty eight and we are starting the working processes for creating our magical application | 0.06 | 0.04 | slowed x0.875, number in text, first word differs (onset?) |
| 20 | `audio-r2_worst20_video03_0m11s_unclear.wav` | Probably more in general what I want to say is that I am glad to meet you and glad to cooperate. | probably more in general what I want to say is that I am glad to meet you and let to cooperate | probably more in general what i want to say is that i am glad to meet you and lad to coorporate | 0.05 | 0.04 | slowed x0.875 |
<!-- audio-r2 -->

<!-- bed-AB -->
## Background bed: Spleeter TF (original) vs the PyTorch GPU port, video 03, 0:40–1:10
- `bed_spleeter_AB_video03_0m40s-1m10s_1-TF-original.wav`: the separated background as the TF 2.9 CPU venv makes it (audio-r2).
- `…_2-torch-GPU-port.wav`: the same weights run by `spleeter_torch.py` on the GPU (audio-r3).
- `…_3-difference-x1000.wav`: their difference, amplified 1000× (+60 dB). It is still −97.5 dBFS, i.e. silence.

Both beds are −39.2 dBFS. The difference before amplification is −157.5 dBFS, so the SNR is 118 dB. They should sound identical.
<!-- bed-AB -->
<!-- audio-r3 -->
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
<!-- audio-r3 -->
