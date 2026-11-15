# Proof plaintext segment account

- Purpose: account the current production proof bytes by the actual serialized
  fields, using the five archived production artifacts. No code change, no
  protocol change, no case rerun.
- Measured artifacts: the after-side proofs of
  `20261115-1727-public-tables-062c1520-61c8d2ab` (gpt2/gemma3 seq16,
  timesfm 16-patch) and `20261115-1741-public-tables-seq512-062c1520-61c8d2ab`
  (gpt2/gemma3 seq512). Measured code `062c1520`; baseline `61c8d2ab`.
- Method: each `proof.bin` is deserialized with the production trace type and
  every field and sub-field is re-serialized with the same encoder. The
  top-level segments are mutually exclusive and sum exactly to the file
  length; the whole-artifact `result.json` `proof_bytes` is listed separately
  and is never mixed into the same total. Each asset opening is token
  independent, so the asset segment is identical at 16 and 512 positions.

## Top-level segments

| case | proof.bin bytes | outer Vec prefixes (asset + step) | asset openings | share | relation receipts | share | other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2 seq16 | 1,980,401,278 | 16 | 1,753,111,896 | 88.52% | 227,289,230 | 11.48% | 152 |
| gemma3 seq16 | 6,013,555,254 | 16 | 5,710,016,064 | 94.95% | 303,539,038 | 5.05% | 152 |
| timesfm 16-patch | 3,740,667,388 | 16 | 3,498,978,744 | 93.54% | 241,424,428 | 6.45% | 264,216 |
| gpt2 seq512 | 10,917,923,358 | 16 | 1,753,111,896 | 16.06% | 9,164,807,342 | 83.94% | 4,120 |
| gemma3 seq512 | 16,368,601,254 | 16 | 5,710,016,064 | 34.88% | 10,658,581,070 | 65.12% | 4,120 |

The `outer Vec prefixes` column is the 8-byte length prefix of the asset
list plus the 8-byte length prefix of the step list (16 B total); those
bytes are included inside the asset-opening and relation-receipt figures,
not added on top of them. `other` is tokens, argmax and the recorded output
together. `tokens + assets + steps + argmax + output` equals `proof.bin` for
every case; the raw segment files carry the check (`top_sum` == `file_len`).
The sub-tables below exclude the 16-byte outer prefixes; their rows sum to
the corresponding segment minus 8 bytes.

## Asset openings

| case | row value payload | value length prefixes | row paths | link proofs | identity bookkeeping |
| --- | ---: | ---: | ---: | ---: | ---: |
| gpt2 seq16/512 | 1,750,319,104 | 898,000 | 898,000 | 81,536 | 915,248 |
| gemma3 seq16/512 | 5,699,987,320 | 3,302,256 | 3,302,256 | 100,672 | 3,323,552 |
| timesfm 16-patch | 3,494,722,576 | 1,378,736 | 1,378,736 | 99,008 | 1,399,680 |

The per-opening parts above exclude the 8-byte outer asset-list prefix;
the outer prefix is shown in the top-level table. Asset classes (opening
bytes per class; each subtotal also excludes the outer asset-list prefix):

| case | weights | embedding / output head | norm | RoPE | extension | structure / config |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2 seq16/512 | 1,211,572,800 | 541,103,088 | 436,000 | 0 | 0 | 0 |
| gemma3 seq16/512 | 1,287,059,472 | 4,301,284,336 | 729,296 | 2,148,320 | 118,794,632 | 0 |
| timesfm 16-patch | 3,498,355,520 | 16,912 | 338,240 | 0 | 0 | 268,064 |

The four trusted public tables are absent from every number above: their
openings were removed in the previous round and are not counted again.

## Relation receipts

The content figures below exclude the 8-byte outer step-list prefix (shown
in the top-level table). The relation `inputs` field length includes empty
inner containers; it is a serialized field length, not input plaintext, and
field lengths are not the same quantity as value payloads.

| case | relation proofs | roots | config | external indices | port wiring | opcode tags |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2 seq16 | 225,901,536 | 1,112,456 | 105,120 | 35,712 | 125,474 | 8,924 |
| gemma3 seq16 | 302,825,168 | 574,432 | 55,568 | 18,760 | 60,922 | 4,180 |
| timesfm 16-patch | 238,979,176 | 2,094,952 | 128,416 | 56,568 | 151,942 | 13,366 |
| gpt2 seq512 | 9,164,259,808 | 346,888 | 76,832 | 22,400 | 95,810 | 5,596 |
| gemma3 seq512 | 10,658,258,512 | 211,488 | 43,408 | 14,088 | 50,554 | 3,012 |

Relation proof fields (bytes):

| case | inputs | outputs | raw | quotients | helpers | aux | lengths |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt2 seq16 | 104,952 | 131,204,096 | 92,319,552 | 35,696 | 35,696 | 1,798,472 | 403,072 |
| gemma3 seq16 | 52,632 | 134,312,896 | 167,142,944 | 16,720 | 16,720 | 1,085,056 | 198,200 |
| timesfm 16-patch | 160,688 | 141,011,888 | 93,117,144 | 53,464 | 53,464 | 3,927,688 | 654,840 |
| gpt2 seq512 | 61,048 | 5,037,602,688 | 4,118,882,752 | 22,384 | 22,384 | 7,459,528 | 209,024 |
| gemma3 seq512 | 34,648 | 4,717,651,648 | 5,930,789,664 | 12,048 | 12,048 | 9,641,408 | 117,048 |

Oracle-opening value payload inside the relation proofs: gpt2 seq16
212,514,304; gemma3 seq16 272,427,904; timesfm 188,469,784; gpt2 seq512
8,838,889,472; gemma3 seq512 9,736,908,800. The remaining oracle bytes are
length prefixes and index bookkeeping.

## Whole-artifact metric, listed separately

| case | result.json proof_bytes | proof.bin | difference |
| --- | ---: | ---: | ---: |
| gpt2 seq16 | 1,980,405,259 | 1,980,401,278 | 3,981 |
| gemma3 seq16 | 6,013,559,226 | 6,013,555,254 | 3,972 |
| timesfm 16-patch | 3,740,935,379 | 3,740,667,388 | 267,991 |
| gpt2 seq512 | 10,917,933,294 | 10,917,923,358 | 9,936 |
| gemma3 seq512 | 16,368,611,194 | 16,368,601,254 | 9,940 |

The difference is the statement and public-input files plus the artifact
counting of the whole proof product; it is not part of the segment totals.

## Observations

- Short sequences: asset openings dominate the proof (88.5% gpt2, 95.0%
  gemma3, 93.5% timesfm). Long sequences: relation receipt plaintext
  dominates (81% of gpt2 seq512 and 59% of gemma3 seq512 is oracle-opening
  value payload).
- The gemma3 extension table is the largest remaining non-weight asset
  plaintext (118.8 MB); the RoPE tables are 2.1 MB; the token embedding and
  output head are the largest asset class (4.3 GB).
- No new deletion candidate follows from this account: every remaining asset
  value is still consumed from its carried opening in the production path,
  and no alternative production verification executes for it. The original
  checks stay; the conclusion is not that the size goal is complete.

## Boundaries

- Measurement only; no workspace rerun, no case rerun, no code change. The
  hard gate stays at 45,080 B. Not pushed.
