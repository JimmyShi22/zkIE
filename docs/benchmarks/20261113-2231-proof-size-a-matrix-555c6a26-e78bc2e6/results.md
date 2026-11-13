# Per-run matrix results

All values are from `matrix/matrix.json`; timings are stage wall times in ms,
memory is VmHWM in KiB. The proof bytes column is the proof artifact directory
total. All 20 runs pass every stage.

## gpt2-seq16

| run | commit ms | prove ms | verify ms | proof bytes | prove HWM KiB | verify HWM KiB | output digest | outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| cold-0 | 3373 | 14301 | 71627 | 2190159492 | 4496132 | 4119288 | 135c4d1c47aee4fd... (16 elem) | PASS PASS PASS |
| warm-0 |  | 14260 | 71609 | 2190159492 | 4495224 | 4120820 | 135c4d1c47aee4fd... (16 elem) | PASS PASS |
| warm-1 |  | 14295 | 71442 | 2190159492 | 4496672 | 4100276 | 135c4d1c47aee4fd... (16 elem) | PASS PASS |
| warm-2 |  | 14286 | 72047 | 2190159492 | 4497476 | 4103228 | 135c4d1c47aee4fd... (16 elem) | PASS PASS |

Warm medians: prove 14286 ms, verify 71609 ms, prove HWM 4496672 KiB, verify HWM 4103228 KiB.

## gpt2-seq512

| run | commit ms | prove ms | verify ms | proof bytes | prove HWM KiB | verify HWM KiB | output digest | outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| cold-0 | 3372 | 60973 | 318670 | 11127652972 | 27377888 | 18867560 | eac0949d6463927e... (512 elem) | PASS PASS PASS |
| warm-0 |  | 60963 | 312451 | 11127652972 | 27377740 | 18868356 | eac0949d6463927e... (512 elem) | PASS PASS |
| warm-1 |  | 60738 | 318716 | 11127652972 | 27378368 | 18869104 | eac0949d6463927e... (512 elem) | PASS PASS |
| warm-2 |  | 61088 | 313535 | 11127652972 | 27377872 | 18849952 | eac0949d6463927e... (512 elem) | PASS PASS |

Warm medians: prove 60963 ms, verify 313535 ms, prove HWM 27377872 KiB, verify HWM 18868356 KiB.

## gemma3-seq16

| run | commit ms | prove ms | verify ms | proof bytes | prove HWM KiB | verify HWM KiB | output digest | outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| cold-0 | 9308 | 35302 | 182674 | 6223297337 | 12718380 | 10947536 | 8683f5e22e51d140... (16 elem) | PASS PASS PASS |
| warm-0 |  | 35114 | 182596 | 6223297337 | 12718396 | 10946876 | 8683f5e22e51d140... (16 elem) | PASS PASS |
| warm-1 |  | 35317 | 183594 | 6223297337 | 12718664 | 10951568 | 8683f5e22e51d140... (16 elem) | PASS PASS |
| warm-2 |  | 35244 | 188254 | 6223297337 | 12718464 | 10965588 | 8683f5e22e51d140... (16 elem) | PASS PASS |

Warm medians: prove 35244 ms, verify 183594 ms, prove HWM 12718464 KiB, verify HWM 10951568 KiB.

## gemma3-seq512

| run | commit ms | prove ms | verify ms | proof bytes | prove HWM KiB | verify HWM KiB | output digest | outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| cold-0 | 9529 | 113996 | 487948 | 16578332025 | 52503072 | 30528432 | 6d1df6b50c20d02d... (512 elem) | PASS PASS PASS |
| warm-0 |  | 114221 | 481342 | 16578332025 | 52503080 | 30492948 | 6d1df6b50c20d02d... (512 elem) | PASS PASS |
| warm-1 |  | 114604 | 487503 | 16578332025 | 52503644 | 30509692 | 6d1df6b50c20d02d... (512 elem) | PASS PASS |
| warm-2 |  | 114028 | 486512 | 16578332025 | 52503076 | 30512024 | 6d1df6b50c20d02d... (512 elem) | PASS PASS |

Warm medians: prove 114221 ms, verify 486512 ms, prove HWM 52503080 KiB, verify HWM 30509692 KiB.

## timesfm-16patch

| run | commit ms | prove ms | verify ms | proof bytes | prove HWM KiB | verify HWM KiB | output digest | outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| cold-0 | 5390 | 23738 | 117847 | 3824905648 | 8206040 | 4187768 | cc72805b7a845cb2... (32768 elem) | PASS PASS PASS |
| warm-0 |  | 23668 | 116243 | 3824905648 | 8208312 | 4194684 | cc72805b7a845cb2... (32768 elem) | PASS PASS |
| warm-1 |  | 23767 | 116086 | 3824905648 | 8207804 | 4177420 | cc72805b7a845cb2... (32768 elem) | PASS PASS |
| warm-2 |  | 23619 | 116370 | 3824905648 | 8208524 | 4185692 | cc72805b7a845cb2... (32768 elem) | PASS PASS |

Warm medians: prove 23668 ms, verify 116243 ms, prove HWM 8208312 KiB, verify HWM 4185692 KiB.
