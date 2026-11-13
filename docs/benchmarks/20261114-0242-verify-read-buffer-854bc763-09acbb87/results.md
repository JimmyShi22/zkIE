# Production CLI verify, per run

All values are from the verify `result.json` in each run directory; timings are
stage wall times in ms, memory is VmHWM in KiB. Proof files are unchanged from
the five-case acceptance matrix.

## gpt2-seq16

| run | outcome | stage wall ms | VmHWM KiB | output digest | elements |
| --- | --- | ---: | ---: | --- | ---: |
| cold-0 | PASS | 17626 | 4102992 | 135c4d1c47aee4fd1985fa862afefbedde9d59beaf112bd2cf0dcfff03339280 | 16 |
| warm-0 | PASS | 17322 | 4105044 | 135c4d1c47aee4fd1985fa862afefbedde9d59beaf112bd2cf0dcfff03339280 | 16 |
| warm-1 | PASS | 17595 | 4102432 | 135c4d1c47aee4fd1985fa862afefbedde9d59beaf112bd2cf0dcfff03339280 | 16 |
| warm-2 | PASS | 17562 | 4102648 | 135c4d1c47aee4fd1985fa862afefbedde9d59beaf112bd2cf0dcfff03339280 | 16 |

Warm medians: 17562 ms, 4102648 KiB.

## gpt2-seq512

| run | outcome | stage wall ms | VmHWM KiB | output digest | elements |
| --- | --- | ---: | ---: | --- | ---: |
| cold-0 | PASS | 39963 | 18849552 | eac0949d6463927eb9a961d61d618fb703e4b02169e89db10dbada94a5f0dfdd | 512 |
| warm-0 | PASS | 38784 | 18849800 | eac0949d6463927eb9a961d61d618fb703e4b02169e89db10dbada94a5f0dfdd | 512 |
| warm-1 | PASS | 39596 | 18867820 | eac0949d6463927eb9a961d61d618fb703e4b02169e89db10dbada94a5f0dfdd | 512 |
| warm-2 | PASS | 39902 | 18850544 | eac0949d6463927eb9a961d61d618fb703e4b02169e89db10dbada94a5f0dfdd | 512 |

Warm medians: 39596 ms, 18850544 KiB.

## gemma3-seq16

| run | outcome | stage wall ms | VmHWM KiB | output digest | elements |
| --- | --- | ---: | ---: | --- | ---: |
| cold-0 | PASS | 29875 | 10952880 | 8683f5e22e51d1405fe3efcc7487a9a4df2e694c5d9f0bed64e921810d77c562 | 16 |
| warm-0 | PASS | 30113 | 10985800 | 8683f5e22e51d1405fe3efcc7487a9a4df2e694c5d9f0bed64e921810d77c562 | 16 |
| warm-1 | PASS | 29846 | 10947524 | 8683f5e22e51d1405fe3efcc7487a9a4df2e694c5d9f0bed64e921810d77c562 | 16 |
| warm-2 | PASS | 29895 | 10949652 | 8683f5e22e51d1405fe3efcc7487a9a4df2e694c5d9f0bed64e921810d77c562 | 16 |

Warm medians: 29895 ms, 10949652 KiB.

## gemma3-seq512

| run | outcome | stage wall ms | VmHWM KiB | output digest | elements |
| --- | --- | ---: | ---: | --- | ---: |
| cold-0 | PASS | 75425 | 30529980 | 6d1df6b50c20d02db715fd14b12b9134623739375c44a029dd15b6b30bd9344c | 512 |
| warm-0 | PASS | 75334 | 30511812 | 6d1df6b50c20d02db715fd14b12b9134623739375c44a029dd15b6b30bd9344c | 512 |
| warm-1 | PASS | 75809 | 30511796 | 6d1df6b50c20d02db715fd14b12b9134623739375c44a029dd15b6b30bd9344c | 512 |
| warm-2 | PASS | 75554 | 30509568 | 6d1df6b50c20d02db715fd14b12b9134623739375c44a029dd15b6b30bd9344c | 512 |

Warm medians: 75554 ms, 30511796 KiB.

## timesfm-16patch

| run | outcome | stage wall ms | VmHWM KiB | output digest | elements |
| --- | --- | ---: | ---: | --- | ---: |
| cold-0 | PASS | 22355 | 4190544 | cc72805b7a845cb2b8505d6092dbb79477d083394a03c99cdac7f5665de56ba3 | 32768 |
| warm-0 | PASS | 22215 | 4187160 | cc72805b7a845cb2b8505d6092dbb79477d083394a03c99cdac7f5665de56ba3 | 32768 |
| warm-1 | PASS | 22303 | 4189212 | cc72805b7a845cb2b8505d6092dbb79477d083394a03c99cdac7f5665de56ba3 | 32768 |
| warm-2 | PASS | 22460 | 4188216 | cc72805b7a845cb2b8505d6092dbb79477d083394a03c99cdac7f5665de56ba3 | 32768 |

Warm medians: 22303 ms, 4188216 KiB.
