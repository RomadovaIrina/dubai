# Defect checks vs Irina's verdicts (09.11)

Agreement: tail 8/8, outside 9/9, chroma 7/7, occlude 7/7


## tail

| video | Irina | check | detail | |
|---|---|---|---|---|
| f0811_m03 | flag 40.1-40.4 | FLAG | local res_out 2.5 @39.64, jaw_dev 1.38 @39.64; global z 2.12 | OK |
| full_m18 | flag 19-20.3 | FLAG | local res_out 1.16 @19.72, jaw_dev 2.09 @19.96; global z 0.46 | OK |
| f0811_q01 | flag last 1 s | FLAG | local res_out 1.08 @44.28, jaw_dev 2.14 @44.64; global z 1.24 | OK |
| f0811_q01_subs | flag last 1 s | FLAG | local res_out 1.08 @44.28, jaw_dev 1.92 @44.68; global z 1.12 | OK |
| f0811_q04 | fine  | - | local res_out 1.03 @44.16, jaw_dev 0.96 @44.72; global z 0.58 | OK |
| full_old03 | fine  | - | local res_out 1.13 @89.68, jaw_dev 0.99 @89.44; global z 1.78 | OK |
| long5 | False  | – | not scored | |
| q03_e2e_alpha_full | fine  | - | local ; global z -inf | OK |
| q05_e2e_full | ?  | - | local res_out 0.16 @44.68, jaw_dev 0.19 @44.2; global z 1.41 |  |
| f0811_q02 | ?  | FLAG | local res_out 1.33 @44.88, jaw_dev 2.56 @44.84; global z 0.17 |  |
| f0811_q06 | fine no KeySync | - | local ; global z 0.0 | OK |

## outside

| video | Irina | check | detail | |
|---|---|---|---|---|
| q03_e2e_alpha_full | flag neck / collar, whole video | FLAG | l p95 0.65 []; r p95 1.08 []; n p95 2.58 [[0.0, 12.28], [12.72, 23.4], [29.48, 32.36]] | OK |
| full_old03 | flag right cheek ~64-66 | FLAG | l p95 0.8 []; r p95 1.29 [[0.0, 0.52]]; n p95 1.21 [] | OK |
| long5 | True neck 108-150 | – | not scored | |
| q05_e2e_full | flag window on head turns | FLAG | l p95 1.51 [[11.56, 13.56], [20.88, 21.88]]; r p95 1.23 [[11.56, 13.24]]; n p95 2.03 [[3.44, 4.0], [11.56, 15.48], [20.88, 21.88]] | OK |
| f0811_m03 | fine  | - | l p95 1.07 []; r p95 0.91 []; n p95 0.8 [] | OK |
| f0811_q01 | fine  | - | l p95 0.77 []; r p95 0.88 []; n p95 1.02 [] | OK |
| f0811_q01_subs | fine  | - | l p95 0.85 []; r p95 0.95 []; n p95 1.11 [] | OK |
| f0811_q04 | fine beard boil is inside the face | - | l p95 0.77 []; r p95 0.49 []; n p95 0.61 [] | OK |
| full_m18 | fine  | - | l p95 0.79 []; r p95 0.92 []; n p95 0.97 [] | OK |
| f0811_q06 | fine no KeySync | - | l p95 0.0 []; r p95 0.0 []; n p95 0.0 [] | OK |
| f0811_q02 | ?  | - | l p95 0.89 []; r p95 0.81 []; n p95 0.84 [] |  |

## chroma

| video | Irina | check | detail | |
|---|---|---|---|---|
| long5 | True 108-150, 116 blue mouth | – | not scored | |
| f0811_m03 | ? shadow on the cheek | FLAG | 5.5% median 3.55 [[3.72, 3.96], [20.36, 20.8], [38.24, 39.12]] |  |
| f0811_q01 | fine  | - | 0.0% median 0.76 [] | OK |
| f0811_q04 | fine  | - | 0.0% median -2.5 [] | OK |
| full_old03 | fine  | - | 0.0% median 4.42 [] | OK |
| full_m18 | fine  | - | 0.0% median -0.18 [] | OK |
| q03_e2e_alpha_full | fine  | - | 0.0% median -1.32 [] | OK |
| q05_e2e_full | ?  | - | 0.0% median 0.07 [] |  |
| f0811_q06 | fine  | - | 0.0% median 0.0 [] | OK |
| f0811_q01_subs | fine  | - | 0.0% median 0.62 [] | OK |
| f0811_q02 | ?  | - | 0.0% median -0.42 [] |  |

## occlude

| video | Irina | check | detail | |
|---|---|---|---|---|
| q05_e2e_full | flag cup ~12 s + turns | FLAG | paint [[11.12, 13.2]] occl [] turn [[0.0, 0.52], [11.8, 13.32], [17.2, 17.84]] | OK |
| f0811_q02 | flag hand 34.8-38.6 | FLAG | paint [] occl [[14.08, 14.32], [29.48, 29.72]] turn [[8.64, 9.16], [10.64, 11.32], [18.08, 18.8], [18.96, 19.52]] | OK |
| long5 | True hair over the cheek 108-150 | – | not scored | |
| f0811_m03 | fine  | - | paint [] occl [] turn [] | OK |
| f0811_q01 | fine  | - | paint [] occl [] turn [] | OK |
| full_m18 | fine  | - | paint [] occl [] turn [] | OK |
| full_old03 | fine  | - | paint [] occl [] turn [] | OK |
| f0811_q04 | fine  | - | paint [] occl [] turn [] | OK |
