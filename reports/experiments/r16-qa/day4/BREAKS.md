# now_1011: tail-like breaks over ALL seconds (Q, day 4)

Rule: a per-frame signal jumps vs the 2 s before it (4-frame mean / median before, the rule that catches the m03 / m18 / q01 tails).
Signals: res_out / res_low (lower-face residual vs source), jaw_dev / lm_dev (ours' jaw / stable landmarks vs the source's, % IOD),
mouth_sharp, lip_thick, seam (jaw-band wobble), ghost (second chin). rank = score x (1 + 0.5 per extra signal at the same moment).

## Worst 10

| # | video | t (s) | window | rank | classes | context | vs 08.11 | signals (value, was) | png |
|---|---|---|---|---|---|---|---|---|---|
| 1 | old03 | 47.72 | 47.00-48.00 | 32.88 | ghost chin / lip shape spike / tail-like break | - | also 08.11 (r21.2) | ghost 4.111 (was -0.274), ghost 2.117 (was 0.034), jaw_dev 6.25 (was 2.56), lip_thick_fat 1.573 (was 0.94) | w01_old03_47.72.jpg |
| 2 | m18 | 7.76 | 6.76-8.08 | 17.88 | ghost chin / seam spike / tail-like break | - | also 08.11 (r6.5) | ghost 1.49 (was 0.051), jaw_dev 10.2 (was 2.1), seam 3.82 (was 0.94), res_low 17.31 (was 9.36) | w02_m18_7.76.jpg |
| 3 | old03 | 73.20 | 72.48-73.52 | 15.78 | ghost chin / tail-like break | - | also 08.11 (r1.9) | ghost 2.63 (was 0.326), jaw_dev 5.2 (was 2.18) | w03_old03_73.20.jpg |
| 4 | q02 | 41.04 | 40.96-42.68 | 8.0 | seam spike / tail-like break | in pause, cut @40.92 | also 08.11 (r2.0) | res_low 9.97 (was 1.66), seam 4.62 (was 0.05), res_out 4.84 (was 1.43), res_out 5.47 (was 2.74) | w04_q02_41.04.jpg |
| 5 | q01 | 30.80 | 30.56-31.08 | 6.72 | ghost chin / seam spike | - | also 08.11 (r7.1) | ghost 1.12 (was -0.171), seam 4.44 (was 1.06) | w05_q01_30.80.jpg |
| 6 | m18 | 13.08 | 13.04-13.84 | 6.38 | ghost chin / tail-like break | - | also 08.11 (r4.5) | ghost 1.063 (was 0.052), jaw_dev 8.06 (was 2.68) | w06_m18_13.08.jpg |
| 7 | q01 | 34.20 | 33.68-34.64 | 5.34 | ghost chin / lip shape spike | - | also 08.11 (r1.5) | ghost 0.89 (was -0.322), lip_thick 0.569 (was 1.227) | w07_q01_34.20.jpg |
| 8 | q02 | 44.72 | 44.40-45.00 | 5.14 | tail-like break | in pause, video end | new in now | jaw_dev 5.05 (was 1.31), res_out 9.14 (was 4.47), res_low 15.52 (was 8.77) | w08_q02_44.72.jpg |
| 9 | m03 | 39.40 | 37.92-39.92 | 3.38 | tail-like break | pause @38.34; pause @39.60, in pause, video end | also 08.11 (r2.8) | res_out 25.45 (was 7.54), jaw_dev 6.51 (was 2.14) | w09_m03_39.40.jpg |
| 10 | m03 | 3.96 | 3.88-4.16 | 2.07 | tail-like break | - | also 08.11 (r2.0) | jaw_dev 5.02 (was 1.61) | w10_m03_3.96.jpg |

## Per video (moments with rank >= 1.2)


### m03 (40.52 s, ghost median 0.01, p95 0.079; pauses 14, cuts 0)

| t | window | rank | classes | context | vs 08.11 | signals |
|---|---|---|---|---|---|---|
| 3.96 | 3.88-4.16 | 2.07 | tail-like break | - | also 08.11 (r2.0) | jaw_dev 5.02 (was 1.61) |
| 11.44 | 11.40-11.64 | 1.39 | seam spike | - | also 08.11 (r2.8) | seam 3.93 (was 1.14) |
| 24.52 | 24.44-24.72 | 1.62 | tail-like break | - | also 08.11 (r1.7) | jaw_dev 7.06 (was 2.91) |
| 27.72 | 27.52-27.88 | 1.26 | seam spike | - | new in now | seam 3.96 (was 1.45) |
| 29.32 | 29.20-29.56 | 1.45 | seam spike | - | also 08.11 (r2.1) | seam 4.84 (was 1.94) |
| 34.04 | 33.96-34.28 | 1.32 | tail-like break | pause @34.00, in pause | new in now | jaw_dev 6.86 (was 3.48) |
| 39.40 | 37.92-39.92 | 3.38 | tail-like break | pause @38.34; pause @39.60, in pause, video end | also 08.11 (r2.8) | res_out 25.45 (was 7.54), jaw_dev 6.51 (was 2.14) |

### m18 (20.04 s, ghost median 0.026, p95 0.495; pauses 4, cuts 0)

| t | window | rank | classes | context | vs 08.11 | signals |
|---|---|---|---|---|---|---|
| 1.84 | 1.76-2.00 | 1.82 | ghost chin | - | also 08.11 (r6.6) | ghost 0.455 (was -0.018) |
| 4.36 | 4.32-4.72 | 1.51 | ghost chin | - | also 08.11 (r4.0) | ghost 0.379 (was -0.007) |
| 7.76 | 6.76-8.08 | 17.88 | ghost chin / seam spike / tail-like break | - | also 08.11 (r6.5) | ghost 1.49 (was 0.051), jaw_dev 10.2 (was 2.1), seam 3.82 (was 0.94), res_low 17.31 (was 9.36) |
| 10.56 | 10.48-10.84 | 2.71 | lip shape spike / tail-like break | - | also 08.11 (r6.1) | jaw_dev 6.01 (was 2.21), lip_thick 0.555 (was 0.992) |
| 11.80 | 11.72-12.00 | 1.53 | seam spike | - | also 08.11 (r4.5) | seam 3.71 (was 0.66) |
| 13.08 | 13.04-13.84 | 6.38 | ghost chin / tail-like break | - | also 08.11 (r4.5) | ghost 1.063 (was 0.052), jaw_dev 8.06 (was 2.68) |
| 15.88 | 15.80-16.16 | 2.08 | ghost chin | pause @16.02, in pause | also 08.11 (r3.4) | ghost 0.52 (was -0.096) |
| 17.44 | 17.32-17.68 | 1.36 | ghost chin | - | new in now | ghost 0.339 (was 0.044) |
| 19.84 | 19.80-20.04 | 1.24 | ghost chin | in pause, video end | also 08.11 (r2.9) | ghost 0.309 (was -0.114) |

### old03 (90.32 s, ghost median -0.194, p95 2.539; pauses 3, cuts 0)

| t | window | rank | classes | context | vs 08.11 | signals |
|---|---|---|---|---|---|---|
| 1.32 | 1.20-1.60 | 9.57 | ghost chin | - | also 08.11 (r14.6) | ghost 2.392 (was -0.086) |
| 2.52 | 2.44-3.40 | 9.0 | ghost chin | - | also 08.11 (r16.0) | ghost 2.251 (was -0.107), ghost 1.366 (was -0.107) |
| 4.36 | 4.12-4.52 | 3.32 | ghost chin | - | new in now | ghost 0.83 (was -0.099) |
| 6.60 | 6.52-6.80 | 5.29 | ghost chin | - | new in now | ghost 1.323 (was -0.315) |
| 8.12 | 8.04-8.44 | 14.58 | ghost chin | - | new in now | ghost 3.645 (was -0.251) |
| 13.24 | 13.24-13.68 | 4.18 | ghost chin | - | also 08.11 (r10.6) | ghost 1.044 (was -0.785) |
| 14.52 | 14.44-15.16 | 3.24 | ghost chin | - | also 08.11 (r1.8) | ghost 0.809 (was -1.951) |
| 17.56 | 17.52-17.76 | 4.48 | ghost chin | - | also 08.11 (r5.5) | ghost 1.119 (was -0.77) |
| 19.32 | 18.84-20.00 | 14.33 | ghost chin / tail-like break | - | also 08.11 (r5.5) | ghost 2.388 (was -0.624), jaw_dev 5.28 (was 1.2), jaw_dev 4.82 (was 1.15) |
| 21.36 | 21.24-21.72 | 7.93 | ghost chin | - | new in now | ghost 1.982 (was -0.708) |
| 24.48 | 24.40-25.00 | 8.88 | ghost chin | - | new in now | ghost 2.221 (was -0.466) |
| 26.04 | 25.92-26.52 | 5.2 | ghost chin | - | new in now | ghost 1.301 (was -0.448) |
| 29.24 | 29.12-29.40 | 2.73 | ghost chin | - | also 08.11 (r9.3) | ghost 0.683 (was -0.162) |
| 36.08 | 35.48-37.88 | 11.31 | ghost chin | - | new in now | ghost 2.828 (was -0.154), ghost 1.434 (was -0.06), ghost 1.019 (was 0.174) |
| 41.68 | 41.48-42.52 | 10.44 | ghost chin / lip shape spike | - | also 08.11 (r6.6) | ghost 1.74 (was -0.41), lip_thick_fat 1.503 (was 0.953) |
| 44.32 | 43.68-45.44 | 10.61 | ghost chin | - | also 08.11 (r1.3) | ghost 2.653 (was -0.126), ghost 1.158 (was -0.029) |
| 47.72 | 47.00-48.00 | 32.88 | ghost chin / lip shape spike / tail-like break | - | also 08.11 (r21.2) | ghost 4.111 (was -0.274), ghost 2.117 (was 0.034), jaw_dev 6.25 (was 2.56), lip_thick_fat 1.573 (was 0.94) |
| 49.16 | 48.84-49.56 | 7.2 | ghost chin | - | also 08.11 (r2.6) | ghost 1.801 (was -0.025), ghost 0.286 (was -0.098) |
| 50.32 | 50.24-50.60 | 8.4 | ghost chin | - | new in now | ghost 2.101 (was -0.033) |
| 53.96 | 53.88-54.96 | 11.59 | ghost chin | - | new in now | ghost 2.897 (was -0.901), ghost 1.005 (was -1.043) |
| 55.84 | 55.16-56.72 | 12.21 | ghost chin | - | new in now | ghost 3.052 (was 0.219), ghost 2.556 (was 0.464) |
| 58.12 | 57.92-58.36 | 12.93 | ghost chin / lip shape spike | - | also 08.11 (r4.9) | ghost 2.154 (was -1.785), lip_thick 0.815 (was 1.288) |
| 64.72 | 64.64-65.00 | 1.24 | tail-like break | - | new in now | res_low 13.16 (was 7.07) |
| 66.84 | 66.60-67.20 | 4.06 | ghost chin | - | also 08.11 (r2.5) | ghost 1.014 (was -0.697) |
| 67.92 | 67.48-69.44 | 15.07 | ghost chin | - | also 08.11 (r2.5) | ghost 3.768 (was -0.53), ghost 2.127 (was -0.002), ghost 1.743 (was -0.017) |
| 71.04 | 70.96-71.72 | 3.53 | ghost chin | - | new in now | ghost 0.881 (was -1.161), ghost 0.387 (was -1.064) |
| 73.20 | 72.48-73.52 | 15.78 | ghost chin / tail-like break | - | also 08.11 (r1.9) | ghost 2.63 (was 0.326), jaw_dev 5.2 (was 2.18) |
| 74.16 | 74.04-74.40 | 14.64 | ghost chin | - | also 08.11 (r1.5) | ghost 3.659 (was 0.459) |
| 75.40 | 75.32-78.12 | 8.83 | ghost chin | - | new in now | ghost 2.206 (was -0.271), ghost 2.202 (was 0.313), ghost 2.187 (was 0.131), ghost 1.698 (was 0.167) |
| 79.00 | 78.96-79.44 | 1.58 | ghost chin | - | also 08.11 (r6.7) | ghost 0.395 (was -0.029) |
| 80.08 | 79.76-81.04 | 12.53 | ghost chin | - | also 08.11 (r6.7) | ghost 3.134 (was -0.072) |
| 82.20 | 81.64-82.68 | 6.92 | ghost chin | - | also 08.11 (r15.4) | ghost 1.729 (was 0.394), ghost 0.798 (was 0.346) |
| 86.28 | 84.20-86.92 | 11.28 | ghost chin / tail-like break | - | also 08.11 (r4.1) | ghost 1.88 (was 0.423), ghost 1.688 (was -0.224), ghost 0.594 (was -0.37), res_low 13.88 (was 8.2) |
| 89.28 | 89.24-89.52 | 3.95 | ghost chin | video end | also 08.11 (r2.4) | ghost 0.988 (was -0.17) |
| 90.12 | 90.00-90.32 | 1.39 | ghost chin | video end | also 08.11 (r2.4) | ghost 0.348 (was -0.664) |

### q01 (45.0 s, ghost median -0.185, p95 0.561; pauses 0, cuts 0)

| t | window | rank | classes | context | vs 08.11 | signals |
|---|---|---|---|---|---|---|
| 2.60 | 2.48-3.00 | 1.49 | seam spike | - | also 08.11 (r1.4) | seam 4.02 (was 1.03) |
| 14.88 | 14.84-15.12 | 1.3 | seam spike | - | new in now | seam 4.62 (was 2.02) |
| 18.04 | 17.96-18.24 | 1.47 | seam spike | - | also 08.11 (r1.9) | seam 4.58 (was 1.65) |
| 20.36 | 19.72-20.60 | 2.8 | ghost chin | - | also 08.11 (r2.3) | ghost 0.7 (was -0.036) |
| 21.12 | 21.08-21.36 | 2.15 | ghost chin | - | also 08.11 (r2.3) | ghost 0.538 (was -0.001) |
| 23.64 | 23.56-23.84 | 2.23 | ghost chin | - | new in now | ghost 0.557 (was -0.547) |
| 25.44 | 24.80-25.80 | 4.36 | ghost chin | - | also 08.11 (r12.2) | ghost 1.091 (was -0.105), ghost 0.884 (was -0.706) |
| 26.72 | 26.68-26.92 | 1.74 | ghost chin | - | also 08.11 (r1.9) | ghost 0.434 (was 0.182) |
| 30.80 | 30.56-31.08 | 6.72 | ghost chin / seam spike | - | also 08.11 (r7.1) | ghost 1.12 (was -0.171), seam 4.44 (was 1.06) |
| 32.52 | 32.44-32.72 | 2.44 | ghost chin | - | also 08.11 (r1.2) | ghost 0.611 (was -0.145) |
| 34.20 | 33.68-34.64 | 5.34 | ghost chin / lip shape spike | - | also 08.11 (r1.5) | ghost 0.89 (was -0.322), lip_thick 0.569 (was 1.227) |
| 35.84 | 35.76-36.12 | 5.26 | ghost chin | - | new in now | ghost 1.316 (was -0.064) |
| 39.16 | 39.12-39.44 | 1.69 | ghost chin | - | new in now | ghost 0.422 (was -0.189) |
| 44.20 | 44.00-44.40 | 1.75 | ghost chin | video end | also 08.11 (r1.4) | ghost 0.438 (was -0.786) |

### q02 (45.0 s, ghost median -0.001, p95 0.067; pauses 12, cuts 1)

| t | window | rank | classes | context | vs 08.11 | signals |
|---|---|---|---|---|---|---|
| 41.04 | 40.96-42.68 | 8.0 | seam spike / tail-like break | in pause, cut @40.92 | also 08.11 (r2.0) | res_low 9.97 (was 1.66), seam 4.62 (was 0.05), res_out 4.84 (was 1.43), res_out 5.47 (was 2.74) |
| 43.40 | 43.28-43.68 | 2.37 | seam spike / tail-like break | pause @43.70 | also 08.11 (r1.3) | seam 4.1 (was 0.93), res_out 5.66 (was 3.23) |
| 44.72 | 44.40-45.00 | 5.14 | tail-like break | in pause, video end | new in now | jaw_dev 5.05 (was 1.31), res_out 9.14 (was 4.47), res_low 15.52 (was 8.77) |

### q04 (45.0 s, ghost median -0.037, p95 0.033; pauses 0, cuts 0)

| t | window | rank | classes | context | vs 08.11 | signals |
|---|---|---|---|---|---|---|
| – | | | none | | | |

### q06 (45.0 s, ghost median 0.0, p95 0.0; pauses 10, cuts 2)

| t | window | rank | classes | context | vs 08.11 | signals |
|---|---|---|---|---|---|---|
| – | | | none | | | |
