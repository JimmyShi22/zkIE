# lm_head asset opening coverage for the limited-key table

## Scheme

- Scheme: `asset-opening-coverage`.
- Code: `40bf940f`.

## Purpose

Check whether the verified production trace's lm_head asset opening covers the
limited-key table mapping `(table row r, matrix row y, low j) -> W[y, 64r+j]`
over 1024 rows and 65536 columns. No proof is produced; this is a coverage
measurement that decides whether table entries can be tied to the existing
opening without a new commitment.

## Method

- Decode the production `gpt2-seq16` trace proof (4,161,782,574 B) as a shared
  trace.
- Select the output-head asset by its trusted asset id and print the opening's
  row count, row range, per-row value widths, a sample of missing rows and a
  sample of short rows.
- The probe was a temporary example and was removed after the measurement.

## Results

- Spec: asset id 192, 67,108,864 elements, row width 65,536.
- Opening: 1024 rows, row count 1024, rows 0..1023, every row width 65,536.
- Missing row sample: none. Short row sample: none.
- Conclusion: full coverage of 1024 x 65,536. The table mapping needs no new
  commitment to use this opening.
- Limit: only opened table columns can be checked against the opening; the
  unopened entries remain covered by the table root alone.

## Logs

- `coverage.txt`: the decoded values.
