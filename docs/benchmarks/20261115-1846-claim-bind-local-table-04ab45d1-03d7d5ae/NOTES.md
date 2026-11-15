# Independent local encoded table for the projection claim chain

- Scheme: `claim-bind`, local table identity group. The table is an
  independent local commitment object; it is not the registered production
  table and this record must not be read as a registered-table closure.
- Code: `04ab45d1`; previous commit `03d7d5ae`; branch `refactor/v2`.
- Change: the diagnostic encoded table is now built with the existing private
  randomness mechanism. `sample_asset_table_randomness` produces the salts
  and the row tails once per process; every table built in the process reuses
  that instance, and the commit uses a fixed local configuration
  `Digest([0x5B; 32])`, distinct from the production table configuration.
  The transposed layout stays: table row `c` is weight column `c`, the
  message index is the input row (`message[input] = W[input*1024 + c]`). The
  evidence line records the local table configuration, root and record so the
  verifier's identity comes from the trusted local table object, not from a
  proof value.
- Table parameters: code `{log_n 12, ell 2048, r 1024}`, `v = 10`, 1024
  rows, 40-byte salts, one-time distance derived from the code; layout
  `message_len 2048`, `tail_len 1024`, `pads 3`, `valid_elements 1,048,576`,
  zero padding. The previous fixture's fixed salts and all-zero tails are
  gone.
- Evidence (single run, release, local first+shared chains only):
  `SHARED SEAL EVIDENCE (independent local table, not the registered table; local first+shared chains, not R5): limbs=3 discovery=26.702223209s table_config=5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b5b table_root=f007fafd29a7f132610576fcf27e22e7bddd77673a53426654a7a867c0e75055 table_record=aa6ac93e9e9dd097f0e46947930d13072bfa97bd68fef87dd7ef99566198b8a9 seal_bytes=3400720 seal_prove=33.536400549s seal_verify=17.115277598s negatives=5/5 count_before_rerun=387 count_after_rerun=516 lifetime=1024 VmHWM:	8945204 kB`
- The five shared-seal negatives are rejected; the stable ledger continues
  from 387 to 516 with lifetime 1024; discovery is reported separately.
- Boundaries: the second chain is outside this path; the real claim/maps and
  the raw-to-requantized output relation are not wired in this group; the
  other 11 heads and the whole asset opening keep their original plaintext
  verification; no production CLI, registration, anchor or policy change; no
  weight plaintext deletion; the hard gate stays at 45,080 B; not pushed.
- Logs: `local-table-evidence.log` (full evidence run).
