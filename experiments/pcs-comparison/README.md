# pcs-comparison

Standalone, isolated experiment workspace comparing two polynomial commitment
schemes on the same *integer-valued* MLE tables. This is benchmark-only code;
it is **not** part of the main zkIE workspace (`/data/jimmyshi/ie`), it changes
nothing in production GKR code, and it does not establish any connection
between the two PCS implementations or to the GKR/recursion stack.

- `kzg`: KZG-style multilinear PC over **BN254** — `ark-poly-commit 0.5`
  `multilinear_pc::MultilinearPC` (pairing-based, XZZPD19-style, `setup` /
  `trim` / `commit` / `open` / `check`). The two tables are committed and
  opened **SEPARATELY** (one commitment + one proof per table; this crate's
  multilinear PC has no multi-open API).
- `whir`: upstream **p3-whir 0.7** over the **Goldilocks base field** with a
  **quadratic extension challenge field** (`BinomialExtensionField<Goldilocks, 2>`),
  mirroring `p3-whir/examples/whir.rs`. The two tables are **batched into ONE
  WHIR witness** and opened at two prescribed points with **ONE FRI proof**
  and ONE commitment. Target configuration: 90-bit security with a PoW budget
  of 0. If p3-whir rejects that construction, the rejection reason is recorded
  as-is and security is **not** weakened. A testing-parameter run (32-bit, 10
  PoW bits) is included as an **explicitly insecure** functional baseline.
  Transcript contract: every open/verify pair starts a fresh challenger at
  `[domain_separator, commitment]` on both sides (upstream requirement — the
  commitment binds each opening transcript).

Run (from this directory):

    cargo test --release
    cargo run --release --bin kzg
    cargo run --release --bin whir

Env controls: `PCS_N` (comma-separated table arities, default `6,12`) and
`PCS_ROUNDS` (default `5`). Both bins print the rayon thread count.

## Benchmarked scenario

For `n = 6` and `n = 12`: two tables of `2^n` integer values each (values
0..99 from the SHARED deterministic generator `tables::values`, so both
schemes run on the identical integer inputs), 5 rounds, one fresh random
challenge vector per table per round. Setup/SRS generation is measured and
reported separately; commit, open, verify, and serialized proof bytes are
reported. KZG verify timings exclude the caller-side O(N) MLE evaluation;
WHIR timings exclude the independent extension-field MLE evaluation, which is
checked against the returned opening values outside the timed regions.

## Field mismatch — read this carefully

The two runs evaluate MLEs over **different fields**:

- Goldilocks: `p = 2^64 - 2^32 + 1`
- BN254 scalar field: `p = 21888242871839275222246405745257275088548364400416034343698204186575808495617`

The "integer-valued" tables embed the same integers, but as elements of
different fields. **It is NOT claimed that a BN254 MLE evaluation equals a
Goldilocks MLE evaluation**, nor that any GKR claim over one field maps to the
other. Timing is comparable only at the level of PCS operation counts
(commit/open/verify calls on same-sized tables), never as "the same field
computation".

## SRS note (KZG side)

`MultilinearPC::setup` draws its trapdoor from **OS randomness**, locally, for
this benchmark run only. The returned parameters contain **group encodings of
trapdoor-derived values** (`g^{t_i}` in `g_mask`, eq-based powers of `g` and
`h`) — not the raw trapdoor — and dropping them at the end of the run is NOT
a secure-erasure claim. This is a benchmark-only, locally generated setup with
**no production ceremony**; the parameter contents are never printed.

## Security-scope note

The 90-bit figure is the **configured PCS opening target** for the WHIR
experiment. It is **not a claim about total GKR/sumcheck soundness** of any
composed proof — this workspace contains no GKR/recursion integration at all.

## Limitations

- No custom cryptography: both sides use the published library PCS as-is.
- No aggregation beyond what each library provides (WHIR: one batched proof;
  KZG: one proof per table, summed byte counts reported).
- The KZG commitment here is deterministic (no hiding randomness).
- Timings are single-process, release-mode, measured with `std::time::Instant`;
  they are indicative, not rigorous benchmarks.
- WHIR proof bytes are measured with `postcard` (LEB128 varints — an upper
  bound vs a fixed-width encoding); KZG bytes via `CanonicalSerialize`
  compressed form.

## Measured results (rerun 2026-10-07, release, rayon threads: 64)

KZG BN254 (`2 tables, opened separately; proof/commitment bytes = sum over both`):

| n | setup | commit | open (avg) | verify (avg, PCS-only) | proof | commitment | SRS |
|---|---|---|---|---|---|---|---|
| 6  | 4.4 ms | 0.35 ms | 5.4 ms | 6.7 ms | 784 B | 80 B | 12.5 KB |
| 12 | 51.1 ms | 3.9 ms | 37.9 ms | 8.0 ms | 1552 B | 80 B | 787 KB |

Wrong-value and wrong-commitment proofs rejected: yes (both n).

WHIR Goldilocks+quadratic-ext (`2 tables batched in ONE witness, ONE proof`):

| n | config | queries | PoW max | setup | dft_init | commit | open (avg) | verify (avg) | proof |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 90-bit, pow 0 | 97 | 0 | 0.015 ms | 0.004 ms | 0.212 ms | 1.654 ms | 0.124 ms | 2417 B |
| 6 | 32-bit, pow 10 (insecure baseline) | 24 | 10 | 0.003 ms | 0.002 ms | 0.137 ms | 1.730 ms | 0.136 ms | 2417 B |
| 12 | 90-bit, pow 0 | 19 | 0 | 0.003 ms | 0.007 ms | 0.783 ms | 3.823 ms | 2.004 ms | 49758 B |
| 12 | 32-bit, pow 10 (insecure baseline) | 5 | 10 | 0.003 ms | 0.007 ms | 0.591 ms | 3.078 ms | 0.688 ms | 17126 B |

Commitment (batched): 65 B. Wrong-point, wrong-commitment, and mutated-value
proofs rejected: yes (all rows). The 90-bit/pow-0 construction is accepted by
p3-whir (final_queries 97 / 19, max_pow_bits 0).

## n = 19 measurements (2026-10-07)

WHIR, 90-bit target, `PCS_POW_BUDGET=4` (RAYON_NUM_THREADS=16, 3 rounds):

```
/usr/bin/time -v timeout 120 env RAYON_NUM_THREADS=16 PCS_N=19 PCS_ROUNDS=3 \
  PCS_POW_BUDGET=4 PCS_TESTING_BASELINE=0 cargo run --release --bin whir
```

- pow_budget=0 is REJECTED by p3-whir with `PowBitsExceedBudget { required: 4,
  budget: 0 }` — recorded, not weakened.
- pow_budget=4 is accepted: final_queries=10, max_pow_bits=4.
- setup 0.019 ms | dft_init 0.578 ms | commit 52.735 ms | open 80.667 ms
  (avg/round) | verify 3.962 ms (avg/round) | proof 107932 B (batched, both
  tables) | commitment 65 B. All rejections true.
- Process peak RSS (whole process, incl. rejection-test overhead and the
  Cargo launcher, NOT per-proof): 190588 KiB; wall 0.75 s.

KZG BN254, n = 19 (RAYON_NUM_THREADS=16, 3 rounds):

```
timeout 120 env RAYON_NUM_THREADS=16 PCS_N=19 PCS_ROUNDS=3 cargo run --release --bin kzg
```

- setup 5500.837 ms | commit 341.651 ms | open 2990.380 ms (avg/round) |
  verify 8.412 ms (avg/round, PCS-only) | proof 2448 B (sum, 2 tables) |
  commitment 80 B (sum) | SRS 100664144 B.
- Process peak RSS 767304 KiB, total 16.80 s. All rejections true.
- These are different fields and different batching (WHIR: 2 tables in one
  witness/one proof; KZG: 2 separate commitments/proofs), so numbers compare
  at the PCS-operation level only, as stated above.
