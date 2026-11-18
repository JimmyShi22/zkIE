# Rollback baseline summary (generated)

Evidence: `20261118-0842-weight-batch-rollback-31825ece`
Driver: zkie/tests/acceptance_rounds.rs, 4 rounds per case (round 1 cold, rounds 2-4 warm)

## gpt2-seq16

| case | round | commit ms | commit HWM KiB | prove ms | prove HWM KiB | proof artifact B | proof.bin deleted B | verify ms | verify HWM KiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2-seq16 | 1 | 246143 | 27295172 | 93412 | 5824820 | 1881723822 | 1879423254 | 78790 | 7439004 |
| gpt2-seq16 | 2 | 243577 | 27295320 | 93136 | 5826744 | 1881741581 | 1879423254 | 78253 | 7439360 |
| gpt2-seq16 | 3 | 246616 | 27295580 | 93356 | 5822952 | 1881762067 | 1879423254 | 78823 | 7439308 |
| gpt2-seq16 | 4 | 243225 | 27295368 | 92856 | 5824620 | 1881759600 | 1879423254 | 78556 | 7439288 |

## gemma3-seq16

| case | round | commit ms | commit HWM KiB | prove ms | prove HWM KiB | proof artifact B | proof.bin deleted B | verify ms | verify HWM KiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gemma3-seq16 | 1 | 9309 | 1086108 | 34031 | 12166596 | 5725337524 | 5725333550 | 29742 | 10382932 |
| gemma3-seq16 | 2 | 9296 | 1086100 | 33840 | 12166876 | 5725337524 | 5725333550 | 29507 | 10383084 |
| gemma3-seq16 | 3 | 9272 | 1086108 | 33786 | 12166456 | 5725337524 | 5725333550 | 30071 | 10381216 |
| gemma3-seq16 | 4 | 9503 | 1086096 | 34386 | 12166540 | 5725337524 | 5725333550 | 29682 | 10385628 |

## timesfm-16patch

| case | round | commit ms | commit HWM KiB | prove ms | prove HWM KiB | proof artifact B | proof.bin deleted B | verify ms | verify HWM KiB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| timesfm-16patch | 1 | 5506 | 175928 | 23275 | 7860524 | 3647375061 | 3647107068 | 23314 | 4039996 |
| timesfm-16patch | 2 | 5385 | 176348 | 23342 | 7859444 | 3647375061 | 3647107068 | 23254 | 4040912 |
| timesfm-16patch | 3 | 5406 | 175600 | 23395 | 7859432 | 3647375061 | 3647107068 | 23340 | 4037228 |
| timesfm-16patch | 4 | 5397 | 177064 | 23783 | 7860872 | 3647375061 | 3647107068 | 23092 | 4026852 |

## Main table (cold / warm median)

| case | proof artifact cold / warm median (B) | commit cold | commit warm med | prove cold | prove warm med | prove HWM warm med | verify cold | verify warm med | verify HWM warm med |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt2-seq16 | 1,881,723,822 / 1,881,759,600 | 246,143 | 243,577 | 93,412 | 93,136 | 5,824,620 | 78,790 | 78,556 | 7,439,308 |
| gemma3-seq16 | 5,725,337,524 / 5,725,337,524 | 9,309 | 9,296 | 34,031 | 33,840 | 12,166,540 | 29,742 | 29,682 | 10,383,084 |
| timesfm-16patch | 3,647,375,061 / 3,647,375,061 | 5,506 | 5,397 | 23,275 | 23,395 | 7,859,444 | 23,314 | 23,254 | 4,037,228 |

## Comparison against the asset-consume closure baseline (gpt2-seq16)

- baseline prove warm median 92,652 ms; this run warm median 93136 ms (delta +0.5224%); ±3% band [89872, 95432] ms -> WITHIN
- baseline proof artifact 1,881,722,126 B; this run warm median 1881759600 B (delta +0.0020%); ±3% band [1825270462, 1938173789] B -> WITHIN
- baseline proof artifact 1,881,722,126 B; this run cold 1881723822 B (delta +0.0001%) -> WITHIN

## Prior-batch warm medians for reference (20261117-0251)

| case | commit | prove | verify | proof artifact B |
|---|---:|---:|---:|---:|
| gpt2-seq16 | 242,468 | 92,652 | 78,126 | 1,881,742,222 |
| gemma3-seq16 | 9,309 | 33,511 | 29,504 | 5,725,337,524 |
| timesfm-16patch | 5,376 | 23,082 | 23,176 | 3,647,375,061 |

ALL_PASS: YES
