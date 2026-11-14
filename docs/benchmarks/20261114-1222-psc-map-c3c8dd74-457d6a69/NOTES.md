# A single-map summand for the general sumcheck

## Scheme

- Scheme: `psc-map`.
- Current code: `c3c8dd74`, following `457d6a69`.
- Previous code: `457d6a69`.

## Purpose

Add the summand form the weight merge needs: one shown map of degree one,
summed over the Boolean cube. The inner sumcheck supplies the map's unmasked
round polynomials over a dense cube; each round is linear with the two sums at
the variable's points. A focused test proves a map's cube sum, verifies it, and
shows that a changed round leaves another output claim, which the caller binds
to committed messages.

## Measurement basis

- Same machine, debug build for the focused and workspace runs; release for
  the hard-gate test.
- Focused: `cargo test -p zkie-ops --lib psc_general`.
- Workspace: `cargo test --workspace`.
- Hard gate: the norm-free toy output-head proof, dumped and removed.

## Key numbers

| figure | value |
|---|---:|
| focused tests | 9 passed |
| workspace | 51 suites, 1032 passed, 0 failed, 19 ignored |
| norm-free toy proof | 45,080 B |
| toy proof sha256 | `7a5da6ebcbfdb61d5ee88ab60e67f45613906959b0c5de998cb870f5e1de4cfb` |

## Four-column mapping

| requirement | anchor | change | evidence |
|---|---|---|---|
| The merge runs a sumcheck whose summand is one map of degree one | masked-sumcheck summand forms | `Summand::Map` (one shown map, degree one) | focused test proves and verifies the map's cube sum |
| The map's rounds are plain sums over the cube | plain-sum variable weight | `MapInner` rounds and binding | focused test; a changed round leaves another output claim |

## Conclusions

- The one-map summand and its dense-cube inner sumcheck are in place.
- No session or proof message is touched; the toy proof is unchanged.

## Open items

- The merge transcript (claims fixed, batch challenge, start, rounds, point,
  end value with its mask, leftover claim) is not wired.
- The leftover claim's chain check and the source-layer reads follow.
