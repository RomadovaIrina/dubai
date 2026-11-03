# Edge-case robustness suite (r13 round 3, 03–04.11)

- **How:** `scripts/r13/edge_suite.py` (+ `gen_edge.sh` builds the synthetic inputs from our own clips). Each case runs one full job through `dabai_job.py` (the backend interface) and checks the status, the flags and the output format against the expectation.
- **Run:** `r13s/edge/suite.log` + `suite2.log` on r13-t.
- **Result: 15 / 16 run cases OK.** The 10-minute case (`long_10min`) was not run: paused for the student round, then the server was wrapped up.

| case | status | flags | verdict | wall s |
|---|---|---|---|---:|
| short_3s | SUCCESS | low_res | OK | 111 |
| no_audio | ERROR | no_audio, no_usable_input | OK (expected ERROR) | 3 |
| silence | ERROR | no_speech | OK (expected ERROR) | 36 |
| music_only | ERROR | no_speech | OK (expected ERROR) | 18 |
| fps60_conv | SUCCESS | high_fps | OK | 372 |
| fps60_real | SUCCESS | high_fps | OK | 621 |
| odd_fps_29.58_real | SUCCESS | odd_fps | OK | 270 |
| vfr | SUCCESS | odd_fps, variable_fps, low_res | OK after the fix (first run: face_comp_failed → new CFR bake stage in e2e.sh) | 222 |
| rotated_meta90 | SUCCESS | rotated_source, face_partly_absent | OK (the face_partly_absent flag is cosmetic: the precheck probes the coded, unrotated frames) | 303 |
| horizontal_1920x1080 | SUCCESS | landscape | OK | 333 |
| square_720 | SUCCESS | – | OK | 273 |
| **res4k_vertical** | **PARTIAL_SUCCESS** | **face_gen_failed** | **FAIL**: the face generator fails on 2160×3840; the user still gets the dubbed video with the original face. Generator code = LS agent territory, not touched | 363 |
| hdr_hlg_10bit | SUCCESS | hdr_or_10bit | OK | 312 |
| three_faces | SUCCESS | landscape, two_faces | OK (only the speaker is animated) | 243 |
| face_half_out | SUCCESS | – | OK | 300 |
| very_dark | SUCCESS | – | OK | 252 |
| long_10min | – | – | not run | – |

## Open
1. **4K vertical face_gen_failed:** needs the face agent (likely a downscale before the generator). Ask the coordinator before anyone edits face code.
2. **long_10min:** run when a server is up (about 1 h wall).
3. **Cosmetic:** the rotated clip's `face_partly_absent` flag; the precheck should probe the display-rotated frames.
