# Intelligibility guard: audio-r2

Whisper large-v3 WER and wav2vec2 CTC CER of every dub unit against its English text.
Unit mean per video; `bad` = CTC CER ≥ 0.3.

| video | units | WER (Whisper) | CER (CTC) | CER (Whisper) | bad | extra-sound units | last word lost | max speed-up |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 25 | 0.028 | 0.057 | 0.008 | 0 | 1 | 0 | x1.15 |
| 02 | 18 | 0.060 | 0.164 | 0.032 | 2 | 0 | 0 | x1.15 |
| 03 | 10 | 0.008 | 0.038 | 0.005 | 0 | 0 | 0 | x1.15 |
| 04 | 6 | 0.009 | 0.037 | 0.005 | 0 | 0 | 0 | x1.11 |
| 05 | 7 | 0.011 | 0.149 | 0.005 | 1 | 0 | 0 | x1.0 |

All 66 units: WER mean 0.030, units with WER > 0: 12; CTC CER mean 0.091.

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
